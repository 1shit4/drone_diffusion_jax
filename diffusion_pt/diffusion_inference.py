import logging
import sys
import torch
from pathlib import Path

# Import the PyTorch versions of our modules
from .core import GaussianDiffusion
from .model import TemporalUnet
from data_pt.dataset import DiffusionTrajDataset
from utils_pt.config import load_config, ConfigError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Policy:
    """
    A policy wrapper that loads a trained PyTorch-based Gaussian Diffusion model
    and exposes it as a callable for use in a controller.
    """

    def __init__(
        self,
        config_path: str,
        use_ema: bool = True,
        device: str = "cuda",
        seed: int = 0,
    ):
        """
        Args:
            config_path (str): Path to the configuration file.
            use_ema (bool): Whether to use EMA weights or the raw model weights.
            device (str): The device to run inference on ('cuda' or 'cpu').
            seed (int): Random seed for deterministic sampling.
        """
        logger.info("Initializing policy...")
        
        # --- 1. Set up device and random seed ---
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        logger.info(f"Using device: {self.device}")

        # --- 2. Load configuration ---
        try:
            config = load_config(config_path)
            config.workdir = str(Path(config.workdir).expanduser().resolve())
        except ConfigError as e:
            print(f"[ERROR] Failed to load configuration: {e}", file=sys.stderr)
            sys.exit(1)

        # --- 3. Load dataset for normalization stats ---
        ds = DiffusionTrajDataset(
            dataset_path=config.dataset_path,
            horizon=config.horizon,
            max_path_length=config.max_path_length,
        )
        self.action_dim = ds.action_dim
        self.normalizer = ds.normalizer
        # The normalizer needs to work with torch tensors on the correct device
        self.normalizer.to_torch(self.device)

        # --- 4. Instantiate the model and diffusion process ---
        model = TemporalUnet(
            horizon=config.horizon,
            transition_dim=ds.transition_dim,
            cond_dim=ds.obs_dim,
            dim=config.dim,
            dim_mults=config.dim_mults,
            attention=config.attention,
        )

        self.diffusion = GaussianDiffusion(
            model=model,
            horizon=config.horizon,
            observation_dim=ds.obs_dim,
            action_dim=ds.action_dim,
            n_timesteps=config.n_timesteps,
            clip_denoised=config.clip_denoised,
            predict_epsilon=config.predict_epsilon,
        ).to(self.device)
        
        # Set model to evaluation mode
        self.diffusion.eval()

        # --- 5. Load model checkpoint ---
        ckpt_path = Path(config.workdir) / "checkpoint_latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
        logger.info(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        if use_ema:
            logger.info("Using EMA model weights.")
            state_dict = checkpoint['ema_model_state_dict']
        else:
            logger.info("Using raw model weights.")
            state_dict = checkpoint['model_state_dict']
            
        self.diffusion.load_state_dict(state_dict)
        step = checkpoint.get('step', 'unknown')
        logger.info(f"Successfully loaded model from step {step}.")

    @torch.no_grad()
    def __call__(self, obs: 'np.ndarray') -> 'np.ndarray':
        """
        Performs a conditional sample to generate a trajectory of actions.

        Args:
            obs (np.ndarray): The current observation from the environment.

        Returns:
            np.ndarray: The planned sequence of actions.
        """
        # 1. Convert numpy observation to a torch tensor on the correct device
        obs_tensor = torch.from_numpy(obs).to(self.device, dtype=torch.float32)

        # 2. Normalize the observation
        obs_norm = self.normalizer.normalize_observation(obs_tensor)
        
        # 3. Add a batch dimension
        obs_norm = obs_norm.unsqueeze(0)

        # 4. Perform the conditional sampling
        # The model is stateful, so we don't need to pass params or rng
        sample = self.diffusion.conditional_sample(
            cond=obs_norm,
            horizon=self.diffusion.horizon,
        )
        
        # 5. Extract and unnormalize the action part of the trajectory
        # Remove batch dimension
        sample = sample.squeeze(0) 
        # Slice out the action dimensions
        action_sample = sample[:, :self.action_dim]
        action_unnorm = self.normalizer.unnormalize_action(action_sample)
        
        # 6. Convert the final tensor back to a numpy array for the controller
        return action_unnorm.cpu().numpy()