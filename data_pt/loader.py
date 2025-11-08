from dataclasses import dataclass
import torch

# The custom, threaded DataLoader class is no longer needed.
# We will use torch.utils.data.DataLoader instead, as shown in the converted train.py.

# -------------------------------------------------------------
# Dataclass for batch data
# -------------------------------------------------------------

@dataclass
class Batch:
    """
    A dataclass container to hold a batch of data. This makes the training
    loop code cleaner and more readable.

    Attributes:
        trajectories (torch.Tensor): Batched trajectories of shape (B, H, D).
        conditions (torch.Tensor): Batched conditions of shape (B, obs_dim).
        mask (torch.Tensor): Early termination mask of shape (B, H).
    """

    trajectories: torch.Tensor
    conditions: torch.Tensor
    mask: torch.Tensor

    def to(self, device: str) -> 'Batch':
        """
        Moves all tensors in the batch to the specified device.

        Args:
            device (str): The target device (e.g., 'cpu', 'cuda').

        Returns:
            Batch: A new Batch object with all tensors on the specified device.
        """
        return Batch(
            trajectories=self.trajectories.to(device),
            conditions=self.conditions.to(device),
            mask=self.mask.to(device),
        )