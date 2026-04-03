# ROS2 Autonomous Explorer

Autonomous exploration robot using Next-Best-View (NBV) planning, multi-sensor fusion, SLAM, and Nav2 navigation — simulated in Ignition Gazebo 6.

## Overview

A 4WD Mecanum holonomic robot that explores an unknown maze environment autonomously. It fuses LiDAR, radar, and RGB-D depth into a single scan, builds a 2D occupancy map with SLAM Toolbox, and continuously computes the next best viewpoint to maximise frontier coverage.

An optional RL local planner replaces Nav2's DWB controller with a PPO-trained policy, trained headlessly in MuJoCo at 10–40× Gazebo speed.

### System Data Flow

![Data Flow](src/docs/data_flow.png)

### Simulation Environment

![Gazebo Simulation](src/docs/gazebo.png)

### SLAM + RViz Visualization

![RViz Visualization](src/docs/Rviz.png)

## Project Structure

```
src/
├── autonomous_explorer/
│   ├── autonomous_explorer/    # Python library (nbv_utils, localization, mapping)
│   ├── scripts/                # ROS2 node executables
│   │   ├── obstacle_cluster_node.py   — LiDAR obstacle clustering
│   │   └── nbv_goal_provider_node.py  — NBV mission controller
│   ├── launch/
│   │   ├── nav2_exploration.launch.py — top-level orchestration
│   │   ├── localization.launch.py
│   │   ├── mapping.launch.py
│   │   └── navigation.launch.py
│   ├── config/                 # YAML parameters for all subsystems
│   └── urdf/                   # Robot model + MuJoCo/Gazebo worlds
├── rl_local_planner/
│   ├── rl_local_planner/       # Python library (envs, reward, feature extractor, curriculum)
│   ├── scripts/                # Training, testing, and ONNX export
│   ├── launch/                 # Training + inference launch files
│   ├── config/                 # RL hyperparameters + controller params
│   └── models/                 # Trained ONNX models
└── sensor_fusion/              # Multi-sensor fusion node

models/
└── explorer_ppo.onnx           # Exported ONNX policy for inference

rl_best_model/
└── best_model.zip              # Best SB3 checkpoint (by eval reward)

rl_checkpoints/
├── ppo_step_*.zip              # Periodic checkpoints
├── ppo_final.zip               # Final checkpoint
└── vec_normalize.pkl           # VecNormalize obs/reward stats

scripts/
├── test_mujoco_model.py        # Headless or visual evaluation script
├── benchmark.py                # Speed benchmark
└── monitor_training.sh         # TensorBoard + log tail helper
```

## Quick Start

### Prerequisites
- ROS2 Humble
- Ignition Gazebo 6 (Fortress)
- `ros-humble-nav2-*`, `ros-humble-slam-toolbox`, `ros-humble-robot-localization`

### Install Dependencies

```bash
cd ~/ros2-autonomous-explorer

# ROS2 dependencies
rosdep install --from-paths src --ignore-src -r -y

# Python dependencies
pip install -r requirements.txt
```

### Build

```bash
colcon build --symlink-install
source install/setup.bash
```

### Run

```bash
# Full system: Gazebo + SLAM + Nav2 + NBV exploration
ros2 launch autonomous_explorer nav2_exploration.launch.py

# With RViz
ros2 launch autonomous_explorer nav2_exploration.launch.py use_rviz:=true
```

Subsystems start in sequence:

| Time | Component |
|------|-----------|
| T+0s | Gazebo, sensor fusion, EKF localization |
| T+12s | SLAM Toolbox (waits for EKF TF) |
| T+17s | Nav2 stack (waits for `/map`) |
| T+22s | NBV goal provider (waits for Nav2 action server) |

## NBV Planning Library

Core algorithms in `autonomous_explorer/nbv_utils.py` (pure Python, no ROS2 deps):

| Class | Role |
|-------|------|
| `OccupancyMapper` | Bayesian log-odds grid synced from SLAM Toolbox |
| `OutlineExtractor` | Polar-sector frontier detection via jump edges |
| `CandidateGenerator` | NBV candidate placement at frontiers + uniform fallback |
| `NBVScorer` | Scores candidates by visibility (ray-casting), distance, and orientation |

## Robot Model

4WD Mecanum holonomic base (0.5 × 0.3 m, 3.0 kg)

| Sensor | Spec |
|--------|------|
| Hokuyo GPU LiDAR | 16-beam, 360°, 0.08–18 m, 40 Hz |
| 9-DOF IMU | Accelerometer + gyroscope + magnetometer |
| RGB-D Camera | 0.3–6 m depth range |
| Forward Radar | 90° FOV, 50 m range |

## RL Local Planner

A PPO-trained reinforcement learning policy that replaces Nav2's DWB controller as the local planner. The RL agent learns to navigate toward waypoints while avoiding obstacles. Training uses a **MuJoCo backend** for 10–40× faster simulation than Gazebo; the trained policy deploys back into ROS2 via ONNX inference.

### Architecture

The RL controller is a standalone ROS2 node that publishes `/cmd_vel` directly. When enabled, Nav2's DWB controller output is remapped to a dead topic — the BT navigator still orchestrates goals, the global planner still computes paths, but the RL policy drives the robot.

```
NBV goal provider → Nav2 BT → SmacPlanner2D → /plan (global path)
                                                  ↓
                              RL controller → /cmd_vel (local control)
```

### Training with MuJoCo (recommended)

No Gazebo or ROS2 required — runs headlessly.

```bash
python3 src/rl_local_planner/scripts/train_ppo.py --sim mujoco
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./tb_logs/
```

### Training with Gazebo (ROS2 required)

```bash
# Terminal 1 — launch simulation
ros2 launch rl_local_planner train.launch.py

# Terminal 2 — run training
python3 src/rl_local_planner/scripts/train_ppo.py \
    --config src/rl_local_planner/config/training_config.yaml
```

### Curriculum

Training uses 4-stage curriculum learning with automatic promotion based on success rate:

| Stage | Goal Distance | Max Steps | Conditions | Advances when |
|-------|--------------|-----------|------------|---------------|
| 0 (bootstrap) | 0.3–0.8 m | 200 | Fixed spawn cluster | Success rate > 80% |
| 1 (easy) | 0.8–2 m | 200 | All spawn points | Success rate > 70% |
| 2 (medium) | 2–4 m | 300 | All spawns, full noise | Success rate > 60% |
| 3 (hard) | 3–6 m | 400 | All spawns + dynamics randomization | — |

Domain randomization (stage 3): friction (0.7–1.3×), velocity scale (0.8–1.2×), action delay (0–2 steps), LiDAR noise, odometry noise.

### Trained Model Performance

Evaluated on `best_model.zip` (1.5 M training timesteps):

| Stage | Goal Range | Success Rate |
|-------|-----------|-------------|
| 1 — easy | 0.8–2 m | ~90% |
| 2 — medium | 2–4 m | ~87% |
| 3 — hard | 3–6 m | ~70% |

### Export to ONNX

```bash
python3 src/rl_local_planner/scripts/export_onnx.py \
    --checkpoint ./rl_best_model/best_model.zip \
    --output ./models/explorer_ppo.onnx
```

The script exports, then runs a verification pass with ONNX Runtime and prints the output shape.

### Test the Trained Model

**Headless (stats only):**

```bash
python3 scripts/test_mujoco_model.py \
    --model rl_best_model/best_model.zip \
    --stage 3 --episodes 20
```

**Visual simulation (MuJoCo viewer window):**

```bash
python3 scripts/test_mujoco_model.py \
    --model rl_best_model/best_model.zip \
    --stage 3 --episodes 5 --render
```

A window opens showing the robot navigating the maze. A green sphere marks the current goal. Use `--delay` to control playback speed (default `0.05` s/step = real-time; `0` = maximum speed).

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `rl_best_model/best_model.zip` | Checkpoint to evaluate |
| `--stage` | `2` | Curriculum stage (0–3) |
| `--episodes` | `5` | Number of episodes to run |
| `--max-steps` | curriculum default | Override episode step limit |
| `--render` | off | Open MuJoCo viewer window |
| `--delay` | `0.05` | Sleep between steps when rendering (seconds) |

### Deploy with ROS2

```bash
ros2 launch rl_local_planner rl_exploration.launch.py \
    use_rl_controller:=true
```

The default (`use_rl_controller:=false`) launches the original DWB system — no changes to existing behavior.

### RL Components

| Module | Role |
|--------|------|
| `obs_builder.py` | Shared observation construction (costmap 84×84, LiDAR 360-ray, goal vector, velocity) |
| `reward.py` | Normalized reward: goal progress, collision penalty, proximity shaping, smoothness |
| `feature_extractor.py` | Custom CNN (costmap) + MLP (scan) feature extractor for SB3 |
| `curriculum.py` | 4-stage curriculum with success-rate triggers and per-stage step limits |
| `mujoco_env.py` | Gymnasium env backed by MuJoCo — identical interface to GazeboExplorerEnv |
| `mujoco_sim.py` | Ray-cast LiDAR simulation and costmap construction inside MuJoCo |
| `gym_env.py` | Gymnasium wrapper around Gazebo (scan-gated sync, costmap collision detection) |
| `onnx_inference.py` | ONNX model loading and inference with graceful degradation |
| `rl_controller_node.py` | ROS2 inference node: carrot-point extraction, safety layer, 10 Hz cmd_vel |

### Configuration

| File | Purpose |
|------|---------|
| `config/training_config.yaml` | PPO hyperparameters, reward weights, curriculum thresholds |
| `config/rl_params.yaml` | Inference node parameters (velocity limits, safety thresholds) |

## Tech Stack

- ROS2 Humble · Ignition Gazebo 6 · Nav2 · SLAM Toolbox
- robot_localization EKF · imu_filter_madgwick
- Python 3.10 · NumPy · SciPy
- MuJoCo · PyTorch · Stable-Baselines3 · Gymnasium · ONNX Runtime
