# In utils/config.py

import yaml
from dataclasses import dataclass
from typing import List, Any

# Custom exception for configuration errors
class ConfigError(Exception):
    pass

@dataclass
class Config:
    # -------------------------------------
    # Dataset and Loader Configuration
    # -------------------------------------
    dataset_path: str
    horizon: int
    max_path_length: int
    batch_size: int

    # -------------------------------------
    # Model Configuration
    # -------------------------------------
    dim: int
    attention: bool
    dim_mults: List[int]

    # -------------------------------------
    # Diffusion Configuration
    # -------------------------------------
    n_timesteps: int
    predict_epsilon: bool
    clip_denoised: bool

    # -------------------------------------
    # Training Configuration
    # -------------------------------------
    total_steps: int
    learning_rate: float
    grad_accum_steps: int
    ema_decay: float
    update_ema_every: int
    warmup_steps: int
    prefetch_batches: int
    seed: int

    # -------------------------------------
    # Logging & Checkpointing Configuration
    # -------------------------------------
    workdir: str
    log_every: int
    save_every: int
    resume: bool


def load_config(file_path: str) -> Config:
    """
    Loads a YAML configuration file into a Config dataclass.
    
    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Config: A dataclass object with the configuration parameters.
    
    Raises:
        ConfigError: If the file is not found, cannot be parsed, or is invalid.
    """
    try:
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        if not isinstance(config_dict, dict):
            raise ConfigError("YAML file must contain a dictionary of key-value pairs.")
            
        # The dataclass constructor handles the type checking for us
        return Config(**config_dict)

    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found at: {file_path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML file: {e}")
    except TypeError as e:
        # This will catch errors like missing keys or wrong types
        raise ConfigError(f"Error creating config from YAML. Check for missing keys or wrong data types: {e}")