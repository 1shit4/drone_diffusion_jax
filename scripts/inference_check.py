import argparse
import logging
import time
import numpy as np
import torch
from pathlib import Path

from diffusion.diffusion_inference import Policy  # adjust import to your project layout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def measure_inference_time(policy, obs_shape, runs=100, warmup=5):
    """
    Measures average inference time for the diffusion model.

    Args:
        policy (Policy): The loaded Policy object.
        obs_shape (tuple): Shape of the observation input (obs_dim,).
        runs (int): Number of timed runs.
        warmup (int): Number of warm-up runs to ignore in timing.
    """
    logger.info(f"Running {warmup} warm-up passes...")
    dummy_obs = np.random.randn(*obs_shape).astype(np.float32)
    
    # Warm-up (e.g., trigger CUDA kernel loading, graph setup, etc.)
    for _ in range(warmup):
        _ = policy(dummy_obs)

    # torch.cuda.synchronize(policy.device) if policy.device.type == 'cuda' else None

    logger.info(f"Measuring inference time over {runs} runs...")
    times = []
    for i in range(runs):
        start = time.time()
        _ = policy(dummy_obs)
        # torch.cuda.synchronize(policy.device) if policy.device.type == 'cuda' else None
        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"Run {i+1}/{runs}: {elapsed*1000:.2f} ms")

    avg_time = np.mean(times)
    std_time = np.std(times)
    logger.info(f"\nAverage Inference Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"\n✅ Average Inference Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Check diffusion model inference time")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up runs")
    args = parser.parse_args()

    # Load policy
    policy = Policy(config="experiment_jax.yaml", use_ema=True, rng=42)

    # Determine observation dimension from dataset normalizer
    obs_dim = 21
    obs_shape = (obs_dim,)

    measure_inference_time(policy, obs_shape, runs=args.runs, warmup=args.warmup)


if __name__ == "__main__":
    main()
