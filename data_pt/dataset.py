import logging
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import LimitNormalizer


# The segment function is a pure numpy utility and does not need to be changed.
def segment(
    observations: np.ndarray, terminals: np.ndarray, max_path_length: int
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    # ... (This function remains exactly the same)
    assert len(observations) == len(
        terminals
    ), "Observations and terminals must have the same length."
    obs_dim = observations.shape[1]
    trajs = [[]]
    for obs, term in zip(observations, terminals):
        trajs[-1].append(obs)
        if term.squeeze():
            trajs.append([])
    if len(trajs[-1]) == 0:
        trajs.pop()
    trajs = [np.stack(traj, axis=0) for traj in trajs]
    n_paths = len(trajs)
    path_lengths = [len(traj) for traj in trajs]
    pad_trajs = np.zeros((n_paths, max_path_length, obs_dim), dtype=observations.dtype)
    early_term = np.zeros((n_paths, max_path_length), dtype=bool)
    for i, traj in enumerate(trajs):
        L = path_lengths[i]
        pad_trajs[i, :L] = traj
        early_term[i, L:] = True
    return pad_trajs, early_term, path_lengths


# -------------------------------------------------------------
# PyTorch compatible Dataset class
# -------------------------------------------------------------
class DiffusionTrajDataset(Dataset):
    def __init__(
        self,
        dataset_path: str = "data",
        horizon: int = 16,
        max_path_length: int = 1000,
    ):
        """
        Initializes the dataset for use with a PyTorch DataLoader.

        Args:
            dataset_path (str): Path to the dataset file.
            horizon (int): The number of timesteps in each trajectory slice.
            max_path_length (int): Maximum length of paths for padding.
        """
        logging.info(
            f"[Dataset] horizon: {horizon}, max_path_length: {max_path_length}"
        )
        self.horizon = horizon
        self.max_path_length = max_path_length

        # --- Load and process raw data ---
        try:
            with open(os.path.join(dataset_path, "data.pkl"), "rb") as f:
                ds = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}/data.pkl")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        ds = {k: v.astype(np.float32) for k, v in ds.items()}
        obs = ds["states"]
        acts = ds["actions"]
        terms = ds["terminal"]
        if acts.ndim == 1:
            acts = acts[:, None]

        # --- Normalization ---
        a_max, a_min = acts.max(axis=0), acts.min(axis=0)
        o_max, o_min = obs.max(axis=0), obs.min(axis=0)
        self.normalizer = LimitNormalizer(a_max, a_min, o_max, o_min)
        acts_n, obs_n = self.normalizer.normalize(acts, obs)
        
        self.obs_dim = obs.shape[1]
        self.action_dim = acts.shape[1]
        self.transition_dim = self.obs_dim + self.action_dim

        # --- Segment into trajectories and create valid indices ---
        joined = np.concatenate([acts_n, obs_n], axis=-1)
        joined_seg, term_flags, path_lengths = segment(joined, terms, max_path_length)

        # Build flat indices of all valid (path, start_time) tuples
        indices = []
        for p, L in enumerate(path_lengths):
            for i in range(L - horizon + 1):
                indices.append((p, i))
        self.indices = np.array(indices) # Shape: (N_samples, 2)

        # Store the segmented trajectories to be sliced in __getitem__
        self.joined_seg = joined_seg # Shape: (N_paths, T_max, D)
        self.term_flags = term_flags # Shape: (N_paths, T_max)

    def __len__(self) -> int:
        """ Returns the total number of valid trajectory slices. """
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single trajectory slice from the dataset.
        This method is called by the PyTorch DataLoader.
        """
        # Get the path and start time for this index
        path_idx, start_time = self.indices[idx]

        # Slice out the trajectory window and the corresponding mask
        end_time = start_time + self.horizon
        trajectory = self.joined_seg[path_idx, start_time:end_time]
        mask = self.term_flags[path_idx, start_time:end_time]

        # The condition is the observation part of the first step
        condition = trajectory[0, self.action_dim:]

        # Convert to PyTorch tensors
        # Note: Dtype conversion is important
        return {
            'trajectories': torch.from_numpy(trajectory).float(),
            'conditions': torch.from_numpy(condition).float(),
            'mask': torch.from_numpy(mask).bool(),
        }