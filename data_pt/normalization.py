from typing import Tuple, Union

import torch
import numpy as np

# The Array type hint is updated to include torch.Tensor
Array = Union[np.ndarray, torch.Tensor]


# -------------------------------------------------------------
# Limit Normalization
# -------------------------------------------------------------
class LimitNormalizer:
    """
    Normalizes and unnormalizes actions and observations to and from [-1, 1] range
    using known min/max limits for each.

    Supports both NumPy and PyTorch tensors.
    """

    def __init__(
        self, action_max: Array, action_min: Array, obs_max: Array, obs_min: Array
    ):
        """
        Initializes the normalizer with bounds for actions and observations.
        The bounds are stored as NumPy arrays by default.

        Args:
            action_max (Array): Maximum action values (D_a,).
            action_min (Array): Minimum action values (D_a,).
            obs_max (Array): Maximum observation values (D_o,).
            obs_min (Array): Minimum observation values (D_o,).
        """
        # Store limits as numpy arrays initially
        self.action_max = np.array(action_max)
        self.action_min = np.array(action_min)
        self.observation_max = np.array(obs_max)
        self.observation_min = np.array(obs_min)

    def normalize_action(self, action: Array) -> Array:
        """
        Normalizes the action to the range [-1, 1].

        Args:
            action (Array): Action to normalize.

        Returns:
            Array: Normalized action.
        """
        return (action - self.action_min) / (self.action_max - self.action_min) * 2 - 1

    def unnormalize_action(self, action: Array) -> Array:
        """
        Unnormalizes the action back to its original range.

        Args:
            action (Array): Action to unnormalize.

        Returns:
            Array: Unnormalized action.
        """
        return (action + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    def normalize_observation(self, observation: Array) -> Array:
        # Same as normalize_action but for observations
        return (observation - self.observation_min) / (
            self.observation_max - self.observation_min
        ) * 2 - 1

    def unnormalize_observation(self, observation: Array) -> Array:
        # Same as unnormalize_action but for observations
        return (observation + 1) / 2 * (
            self.observation_max - self.observation_min
        ) + self.observation_min

    def normalize(self, action: Array, observation: Array) -> Tuple[Array, Array]:
        """
        Normalizes both actions and observations.

        Args:
            action (Array): Raw action.
            observation (Array): Raw observation.

        Returns:
            Tuple[Array, Array]: Normalized (action, observation).
        """
        return self.normalize_action(action), self.normalize_observation(observation)

    def unnormalize(self, action: Array, observation: Array) -> Tuple[Array, Array]:
        """
        Unnormalizes both actions and observations.

        Args:
            action (Array): Normalized action.
            observation (Array): Normalized observation.

        Returns:
            Tuple[Array, Array]: Original-scale (action, observation).
        """
        return self.unnormalize_action(action), self.unnormalize_observation(
            observation
        )

    def to_torch(self, device: str = "cpu") -> "LimitNormalizer":
        """
        Converts all internal limits to PyTorch tensors and moves them to the specified device.
        This is crucial for GPU-based training and inference.

        Args:
            device (str): The target device (e.g., 'cpu', 'cuda:0').

        Returns:
            LimitNormalizer: Self with torch.Tensor-converted bounds.
        """
        self.action_max = torch.tensor(self.action_max, dtype=torch.float32, device=device)
        self.action_min = torch.tensor(self.action_min, dtype=torch.float32, device=device)
        self.observation_max = torch.tensor(self.observation_max, dtype=torch.float32, device=device)
        self.observation_min = torch.tensor(self.observation_min, dtype=torch.float32, device=device)
        return self