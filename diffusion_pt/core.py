import torch
import torch.nn as nn
from tqdm import tqdm

# We will need to convert these helpers to PyTorch as well
from .helpers import extract, apply_conditioning, mse_loss, cosine_beta_schedule


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module, # Now we pass an instantiated nn.Module
        horizon: int,
        observation_dim: int,
        action_dim: int,
        n_timesteps: int = 1000,
        clip_denoised: bool = True,
        predict_epsilon: bool = False,
    ):
        """
        Implements Gaussian diffusion as a PyTorch nn.Module.

        Args:
            model (nn.Module): The denoising model (e.g., TemporalUnet).
            horizon (int): Time horizon of each trajectory.
            observation_dim (int): Dimensionality of observations.
            action_dim (int): Dimensionality of actions.
            n_timesteps (int): Number of diffusion steps.
            clip_denoised (bool): Whether to clip predicted x_0 to [-1, 1].
            predict_epsilon (bool): Whether model predicts noise instead of x_0.
        """
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model # The model is now an attribute of this class
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.n_timesteps = n_timesteps

        # --- Precompute diffusion schedule constants ---
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register constants as buffers to move them to the correct device automatically
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Clipped to avoid log(0)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Predicts x_0 from x_t and noise. """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise # If model predicts x_0 directly

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Computes the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0). """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_mean_variance(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Computes the mean and variance of the reverse process p(x_{t-1} | x_t). """
        # The model is called directly, no params argument needed
        noise_pred = self.model(x, cond, t)

        x_recon = self.predict_start_from_noise(x, t, noise=noise_pred)
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ A single step of the reverse diffusion process. """
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        model_std = torch.exp(0.5 * model_log_variance)
        noise = torch.randn_like(x)
        
        # No noise at the final step
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * model_std * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, cond: torch.Tensor) -> torch.Tensor:
        """ The full reverse diffusion loop from noise to a sample. """
        device = self.betas.device
        batch_size = shape[0]
        
        # Start with random noise
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        # Iterate backwards through time
        for i in tqdm(reversed(range(0, self.n_timesteps)), desc='Sampling loop', total=self.n_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t)
            x = apply_conditioning(x, cond, self.action_dim)
        return x

    def conditional_sample(self, cond: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """ Generates a sample conditioned on the observation. """
        horizon = horizon or self.horizon
        shape = (cond.shape[0], horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond)

    # --------------------------- Training ---------------------------

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """ Samples from the forward diffusion process q(x_t | x_0). """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_loss(self, x_start: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """ Calculates the loss for training. """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # 1. Sample a random timestep for each example in the batch
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device).long()
        
        # 2. Sample noise and create the noisy input x_t
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        # 3. Get the model's prediction
        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        # 4. Calculate the loss
        if mask is None:
            mask = torch.ones_like(x_start, dtype=torch.bool)
        
        if self.predict_epsilon:
            loss = mse_loss(x_recon, noise, ~mask)
        else:
            loss = mse_loss(x_recon, x_start, ~mask)
            
        return loss

    # Alias for the trainer class, making it more idiomatic
    def forward(self, x_start: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self.p_loss(x_start, cond, mask)