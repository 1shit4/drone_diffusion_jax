import torch
import torch.nn.functional as F


######################################################################################
######################## Index Extractions and Conditioning ##########################
######################################################################################
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extracts values from `a` at indices `t`, and reshapes to match `x_shape`
    for broadcasting.

    Args:
        a (torch.Tensor): [T] or [T, ...] tensor from which to extract.
        t (torch.Tensor): [B] tensor of time indices.
        x_shape (tuple): The shape of the tensor `x` that will be broadcasted to.

    Returns:
        torch.Tensor: Extracted values reshaped to [B, 1, ..., 1].
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t) # `gather` is the PyTorch equivalent of array indexing
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def apply_conditioning(
    x: torch.Tensor, conditions: torch.Tensor, action_dim: int
) -> torch.Tensor:
    """
    Overwrites the observation portion of the first timestep in each trajectory.

    Args:
        x (torch.Tensor): [B, H, D] - full trajectory (actions + observations).
        conditions (torch.Tensor): [B, obs_dim] - conditioning observations.
        action_dim (int): Index where observation features start in x.

    Returns:
        torch.Tensor: Modified trajectory with conditions applied at t=0.
    """
    # In PyTorch, tensors are mutable, so we can use direct slicing and assignment.
    # This is more idiomatic than JAX's immutable `.at[...].set(...)` syntax.
    x[:, 0, action_dim:] = conditions
    return x


######################################################################################
######################## Diffusion Schedules #########################################
######################################################################################


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Cosine beta schedule from "Improved DDPM" paper:
    https://openreview.net/forum?id=-NEXDKk8gZ

    Args:
        timesteps (int): Number of diffusion steps.
        s (float): Small offset for stability.
        dtype (torch.dtype): Output dtype.

    Returns:
        torch.Tensor: Beta values for each timestep [T].
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=dtype) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


#######################################################################################
######################### Loss Functions ##############################################
#######################################################################################


def mse_loss(pred: torch.Tensor, targ: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Squared Error loss, applying a mask.

    Args:
        pred (torch.Tensor): [B, H, D] - Predicted values.
        targ (torch.Tensor): [B, H, D] - Target values.
        mask (torch.Tensor): [B, H] or [B, H, D] - Boolean mask. Where True, the loss is kept.
    """
    # Using PyTorch's built-in MSE loss is efficient and clear.
    # `reduction='none'` gives us the loss for each element.
    loss = F.mse_loss(pred, targ, reduction='none')
    
    # If mask is [B, H], expand it to [B, H, D] to match the loss shape
    if mask.dim() == loss.dim() - 1:
        mask = mask.unsqueeze(-1)

    # Apply the mask. We multiply by the mask (converted to float)
    # because `where` can be less efficient.
    loss = loss * mask.float()
    
    # We want the mean of the *unmasked* elements.
    # So, we sum the loss and divide by the number of True elements in the mask.
    return loss.sum() / mask.sum()