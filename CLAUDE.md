# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## Build & Run

```bash
# Install Python deps (one-time)
pip install -r requirements.txt
rosdep install --from-paths src --ignore-src -r -y

# Build ROS2 packages
colcon build --symlink-install
source install/setup.bash

# Run full system (Gazebo + SLAM + Nav2 + NBV exploration)
ros2 launch autonomous_explorer nav2_exploration.launch.py use_rviz:=true

# Run with RL local planner instead of DWB
ros2 launch rl_local_planner rl_exploration.launch.py use_rl_controller:=true
```

## RL Training & Evaluation

```bash
# Train PPO in MuJoCo (recommended — no ROS2 needed, 10-40× faster)
python3 src/rl_local_planner/scripts/train_ppo.py --sim mujoco

# Monitor training
tensorboard --logdir ./tb_logs/
bash scripts/monitor_training.sh

# Evaluate a checkpoint (headless)
python3 scripts/test_mujoco_model.py --model rl_best_model/best_model.zip --stage 3 --episodes 20

# Evaluate with visual viewer
python3 scripts/test_mujoco_model.py --model rl_best_model/best_model.zip --stage 3 --episodes 5 --render

# Export trained model to ONNX for ROS2 inference
python3 src/rl_local_planner/scripts/export_onnx.py \
    --checkpoint ./rl_best_model/best_model.zip \
    --output ./models/explorer_ppo.onnx

# Hyperparameter sweep (Optuna)
python3 src/rl_local_planner/scripts/sweep_optuna.py
```

## Tests & Linting

```bash
# Run tests
pytest src/rl_local_planner/tests/

# Run a single test file
pytest src/rl_local_planner/tests/test_reward.py -v

# Lint
ruff check src/
ruff format src/
```

Test files live in `src/rl_local_planner/tests/` and cover `reward.py`, `curriculum.py`, and `obs_builder.py`.

## Architecture

### Two-Package ROS2 Workspace

**`src/autonomous_explorer/`** — Classical autonomy stack:
- `autonomous_explorer/nbv_utils.py` — Pure-Python NBV library: `OccupancyMapper` (Bayesian log-odds grid), `OutlineExtractor` (frontier detection via polar-sector jump edges), `CandidateGenerator`, `NBVScorer` (ray-cast visibility scoring).
- `scripts/nbv_goal_provider_node.py` — Mission controller node: runs the NBV loop and sends goals to Nav2's BT navigator.
- `scripts/obstacle_cluster_node.py` — LiDAR obstacle clustering node.
- `scripts/path_speed_limiter_node.py` — Limits path speed near obstacles.
- Launch sequence (staggered with `TimerAction`): Gazebo → EKF → SLAM Toolbox → Nav2 → NBV goal provider.

**`src/rl_local_planner/`** — PPO-trained local planner:
- `rl_local_planner/mujoco_env.py` — Gymnasium env backed by MuJoCo physics. Identical observation/action interface to the Gazebo env (`gym_env.py`).
- `rl_local_planner/mujoco_sim.py` — Ray-cast LiDAR simulation and costmap construction inside MuJoCo (360 `mj_ray()` calls per step).
- `rl_local_planner/reward.py` — Reward components: progress, heading, near-goal shaping, goal bonus, collision, proximity, smoothness, step cost.
- `rl_local_planner/curriculum.py` — 4-stage curriculum (0.3–6 m goals) with automatic promotion based on success rate.
- `rl_local_planner/obs_builder.py` — Shared observation construction: costmap 84×84, LiDAR 360-ray, goal vector, velocity.
- `rl_local_planner/feature_extractor.py` — Custom SB3 feature extractor: CNN for costmap + MLP for scan/goal/velocity.
- `rl_local_planner/onnx_inference.py` — ONNX model loading and inference with graceful degradation.
- `scripts/rl_controller_node.py` — ROS2 node: extracts carrot point from global path, runs ONNX inference at 10 Hz, publishes `/cmd_vel` with a safety e-stop layer.

**`src/sensor_fusion/`** — Fuses LiDAR, radar, and RGB-D depth into a single `/scan` topic.

### RL Data Flow

```
Train: MuJoCoExplorerEnv (8 parallel) → PPO (SB3) → best_model.zip → export_onnx.py → explorer_ppo.onnx
Deploy: /plan (global path) → rl_controller_node → ONNX inference → /cmd_vel
```

The robot model is a `<freejoint/>` body in MuJoCo — velocity is written directly to `data.qvel` each step (no wheel joints). Box sizes in the MuJoCo XML (`urdf/worlds/mujoco_maze.xml`) are **half-extents**.

### Observation Space

```
{
  'costmap':     (84, 84, 1) uint8  — local obstacle grid around robot
  'scan':        (360,)      float  — normalised LiDAR ranges
  'goal_vector': (2,)        float  — direction + distance to goal in robot frame
  'velocity':    (3,)        float  — current vx, vy, yaw_rate
}
action: (3,) float  — [vx, vy, vyaw] normalised to [-1, 1]
```

### Key Config Files

| File | Purpose |
|------|---------|
| `src/rl_local_planner/config/rl_params.yaml` | Inference node: velocity limits, safety thresholds, carrot radius |
| `src/autonomous_explorer/config/nav2_params.yaml` | Nav2 stack: planner, controller, costmap params |
| `src/autonomous_explorer/config/slam_toolbox_params.yaml` | SLAM Toolbox online async mapper |
| `src/autonomous_explorer/config/ekf.yaml` | robot_localization EKF: sensor fusion for odometry |
| `src/autonomous_explorer/config/exploration_bt.xml` | Behavior tree for NBV exploration |

### Checkpoints & Models

- `rl_best_model/best_model.zip` — Best SB3 checkpoint (by eval reward); used for evaluation and ONNX export.
- `rl_checkpoints/ppo_step_*.zip` — Periodic checkpoints saved every 50k steps.
- `rl_checkpoints/vec_normalize.pkl` — VecNormalize running obs/reward statistics; must be loaded alongside any checkpoint for correct inference.
- `models/explorer_ppo.onnx` — Exported policy for ROS2 deployment.

### Training Config Schema

`src/rl_local_planner/rl_local_planner/config_schema.py` defines Pydantic models that validate `training_config.yaml` at load time — a typo in a reward weight will raise at startup rather than silently using a default.
