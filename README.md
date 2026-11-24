# Diffusion-Based Residual Dynamics Learning for Aerial Manipulation

**JAX and PyTorch experiments in learning and compensating for unmodeled quadrotor and aerial-manipulator dynamics.**

This repository extends the [DroneDiffusion](https://github.com/avirupdas55/DroneDiffusion) framework from quadrotor dynamics estimation to experiments with a **quadrotor carrying a two-joint manipulator**. The objective is to estimate residual dynamics from recorded state and control trajectories, then use those estimates as compensation terms alongside nominal feedback control. The aerial-manipulator experiments incorporate arm configuration and joint motion in addition to vehicle states.

The repository includes dataset-processing code, conditional diffusion models, JAX and PyTorch training variants, saved JAX checkpoints, and ROS 2/MAVROS controller-integration experiments.

## Residual Dynamics Learning

A nominal dynamics model does not capture every force and moment acting on a flying robot. Model uncertainty becomes more pronounced with aerial manipulation because arm motion changes the coupled dynamics and can introduce additional disturbances. **Residual dynamics learning** seeks to model the difference between recorded system behavior and a chosen nominal model.

The implementation uses a **conditional denoising diffusion model** to generate a short sequence of residual estimates from the current robot state. A temporal U-Net operates over trajectory windows; the diffusion sampler progressively reconstructs a trajectory from noise while conditioning on the first observation. The generated residual terms are intended to augment the underlying flight and arm controllers rather than replace the complete feedback-control system.

Two data formulations are implemented:

| System | Conditioning state | Predicted residual channels |
| --- | --- | --- |
| **Quadrotor** | **15 dimensions:** position (3), velocity (3), acceleration (3), roll/pitch/yaw (3), and previous control values (3) | **3:** vehicle-control residual components |
| **Aerial manipulator** | **21 dimensions:** vehicle position, velocity, acceleration, and roll/pitch/yaw (12); two joint positions and velocities (4); previous vehicle and joint control values (5) | **5:** three vehicle-control and two joint-control residual components |

The aerial-manipulator preprocessing script also constructs a nominal **two-link joint-space inertia model** to form the joint residual targets. See [`data/prepare_data_manipulator.py`](data/prepare_data_manipulator.py) for the precise logged-field mapping and target computation.

## Diffusion Model and Training

The diffusion implementation uses a **temporal U-Net with residual 1D convolutional blocks**, timestep embeddings, observation conditioning, and optional attention. The training pipeline includes trajectory segmentation, normalization, padded horizon windows, an Adam optimizer, gradient accumulation, and exponential moving average (EMA) model weights.

The provided JAX experiment configuration specifies:

| Setting | Value |
| --- | --- |
| Trajectory horizon | 16 steps |
| Diffusion denoising steps | 20 |
| Training steps | 100,000 |
| Batch size | 256 |
| Learning rate | 0.0002 |
| EMA decay | 0.995 |
| Training target | Denoised trajectory (`predict_epsilon: false`) |

The archive contains completed **100,000-step JAX training checkpoints** under `checkpoints_drone/`, `checkpoints_manip/`, and `checkpoints_new/`, plus training logs. A corresponding PyTorch implementation is included in `diffusion_pt/` and `data_pt/`, with its own training log.

## Controller Integration

The ROS 2/MAVROS scripts explore how diffusion-generated corrections can be included in a feedback controller running with PX4–Gazebo:

- **Quadrotor control:** vehicle-state feedback and residual compensation are combined to generate an acceleration reference, then converted to a desired attitude and collective-thrust command.
- **Aerial-manipulator control:** the policy input is extended with arm states and previous control commands. Predicted residuals are separated into vehicle and joint components, with additional joint-effort control logic for the manipulator.
- **Inference variants:** separate JAX and PyTorch wrappers load diffusion models and generate horizon-length residual sequences. The repository also contains a combined vehicle/arm control-node experiment and trajectory-reference generation utilities.

These scripts are experimental integration components; they use external ROS 2, PX4, MAVROS, and Gazebo installations and vehicle-specific controller parameters.

## Code Structure

| Path | Contents |
| --- | --- |
| [`data/`](data/) | Quadrotor and aerial-manipulator log preprocessing, trajectory datasets, normalization, and JAX data loading |
| [`diffusion/`](diffusion/) | JAX/Flax temporal U-Net, Gaussian diffusion, and inference wrapper |
| [`training/`](training/) | JAX training state, optimizer updates, EMA handling, and checkpoint saving |
| [`diffusion_pt/`](diffusion_pt/) and [`data_pt/`](data_pt/) | PyTorch model, diffusion, inference, and data utilities |
| [`scripts/train.py`](scripts/train.py) | JAX training entry point |
| [`scripts/drone_controller.py`](scripts/drone_controller.py) | Quadrotor feedback-controller experiment |
| [`scripts/drone_controller_manip_og.py`](scripts/drone_controller_manip_og.py), [`scripts/drone_controller_manip_pt.py`](scripts/drone_controller_manip_pt.py) | Aerial-manipulator inference/control variants |
| [`scripts/comb_controller.py`](scripts/comb_controller.py), [`scripts/drone_controller_manip.py`](scripts/drone_controller_manip.py), [`scripts/manip_asmc.py`](scripts/manip_asmc.py) | Separated vehicle and manipulator controller logic |
| [`scripts/trajectory_publisher.py`](scripts/trajectory_publisher.py) | Hover, line, circle, figure-eight, vertical, and helical references |
| [`config/`](config/) | JAX and PyTorch experiment configurations |
| `checkpoints_drone/`, `checkpoints_manip/`, `checkpoints_new/` | Saved JAX model checkpoints and training logs |

## Training Entry Point

Recorded trajectory CSVs are processed into a dataset containing `states`, `actions`, and `terminal` arrays. The preprocessing scripts use dataset-directory names specified in their source; update them to match the location of your logs and set the matching `dataset_path` in the experiment configuration.

With a compatible JAX/Flax environment and the prepared dataset:

```bash
python data/prepare_data.py
python scripts/train.py --config config/experiment_jax.yaml
```

For aerial-manipulator data, use `data/prepare_data_manipulator.py` and point the experiment configuration to the resulting dataset. **Raw trajectory datasets are not included in this repository**; saved checkpoints alone do not supply the data-dependent normalization required by the current inference wrapper.

## Upstream Work and Attribution

This work builds on **[DroneDiffusion: Robust Quadrotor Dynamics Learning with Diffusion Models](https://arxiv.org/abs/2409.11292)** and its [JAX implementation](https://github.com/avirupdas55/DroneDiffusion). The base diffusion modeling approach and associated software originated in that project. The code here explores its adaptation to aerial-manipulator dynamics, including joint-state and joint-residual data preparation, alternative training implementations, and ROS 2 control integration.

The upstream **Apache License 2.0** and copyright notice are retained in [`LICENSE`](LICENSE).

*Research experiments conducted at the Robotics Research Center, IIIT Hyderabad.*
