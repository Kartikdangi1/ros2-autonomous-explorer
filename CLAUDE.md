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
# One-time setup
pip install -r requirements.txt
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install
source install/setup.bash

# Classical stack — DWB local planner (default)
ros2 launch autonomous_explorer nav2_exploration.launch.py use_rviz:=true

# RL local planner — PPO policy replaces DWB (single launch file, one arg)
ros2 launch autonomous_explorer nav2_exploration.launch.py controller:=rl use_rviz:=true

# Startup delays are configurable (defaults: 12/17/22 s)
ros2 launch autonomous_explorer nav2_exploration.launch.py slam_delay:=15.0 nav2_delay:=20.0 nbv_delay:=25.0

# Switch from DWB to MPPI: edit nav2_params.yaml controller_plugins to ["MPPIFollowPath"]
```

## RL Training & Evaluation

```bash
# Train PPO — Hydra config, all params overridable on CLI
python3 src/rl_local_planner/scripts/train_ppo.py                        # defaults (mujoco, seed=42)
python3 src/rl_local_planner/scripts/train_ppo.py sim=mujoco num_envs=8  # parallel envs
python3 src/rl_local_planner/scripts/train_ppo.py reward_proximity=-1.0  # override any param
python3 src/rl_local_planner/scripts/train_ppo.py --cfg job              # print config, no training

# Resume from checkpoint
python3 src/rl_local_planner/scripts/train_ppo.py resume_from=./rl_checkpoints/ppo_step_500000

# Monitor
tensorboard --logdir ./tb_logs/
bash scripts/monitor_training.sh

# Evaluate a checkpoint
python3 scripts/test_mujoco_model.py --model rl_best_model/best_model.zip --stage 3 --episodes 20
python3 scripts/test_mujoco_model.py --model rl_best_model/best_model.zip --stage 3 --episodes 5 --render

# Export to ONNX for ROS2 deployment
python3 src/rl_local_planner/scripts/export_onnx.py \
    --checkpoint ./rl_best_model/best_model.zip \
    --output ./models/explorer_ppo.onnx

# Hyperparameter sweep (Optuna)
python3 src/rl_local_planner/scripts/sweep_optuna.py
```

## Tests & Linting

```bash
# Run all tests (anyio plugin conflicts with older pytest — disable it)
pytest src/rl_local_planner/tests/ -p no:anyio

# Single test file
pytest src/rl_local_planner/tests/test_reward.py -v -p no:anyio

# Lint
python3 -m ruff check src/
python3 -m ruff format src/
```

Test files cover `reward.py`, `curriculum.py`, and `obs_builder.py`.

## Architecture

### Three-Package ROS2 Workspace

**`src/autonomous_explorer/`** — Classical autonomy stack:
- `autonomous_explorer/nbv_utils.py` — Pure-Python NBV library with no ROS2 deps: `OccupancyMapper` (Bayesian log-odds grid), `OutlineExtractor` (frontier detection via polar-sector jump edges), `CandidateGenerator`, `NBVScorer` (ray-cast visibility scoring). Also exports `_normalize_angle` — used by `path_speed_limiter_node.py`.
- `scripts/nbv_goal_provider_node.py` — Mission controller: runs the NBV loop, sends goals to Nav2 BT navigator.
- `scripts/path_speed_limiter_node.py` — Curvature-based DWB speed limiter; imports `_normalize_angle` from `nbv_utils`.
- Launch stagger (all configurable via args): Gazebo (T+0) → SLAM (T+`slam_delay`) → Nav2 (T+`nav2_delay`) → NBV + speed limiter (T+`nbv_delay`).

**`src/rl_local_planner/`** — PPO-trained local planner:
- `rl_local_planner/mujoco_env.py` — Primary training env (MuJoCo). Identical obs/action interface to `gym_env.py` (Gazebo). Robot is a `<freejoint/>` body; velocity is written directly to `data.qvel` each step.
- `rl_local_planner/mujoco_sim.py` — Simulates 360-ray LiDAR via `mj_ray()` and builds the 84×84 costmap.
- `rl_local_planner/reward.py` — `RewardWeights` dataclass + `compute_reward()`. All weights are configurable via `training_config.yaml`.
- `rl_local_planner/curriculum.py` — 4-stage curriculum (0.3–6 m goals). Stage 3 adds domain randomisation (velocity scaling, action delay, scan noise).
- `rl_local_planner/obs_builder.py` — Shared observation builder used by all three envs (MuJoCo, Gazebo, Point2D). `MAX_VEL_X/Y` constants here **must match** `config/rl_params.yaml` — both are 0.5 m/s.
- `rl_local_planner/config_schema.py` — Pydantic `TrainingConfig` model. Single source of truth for all training hyperparameters and reward weights. **`extra = "forbid"`** — a typo in `training_config.yaml` raises at startup.
- `scripts/rl_controller_node.py` — ROS2 inference node: extracts carrot point from `/plan`, runs ONNX at 10 Hz, publishes `/cmd_vel` with e-stop.

**`src/sensor_fusion/`** — Fuses LiDAR, radar, and RGB-D depth into `/fused_scan`.

### RL Data Flow

```
Train: MuJoCoExplorerEnv × 8 (SubprocVecEnv)
         → VecNormalize → VecTransposeImage
         → PPO (SB3)
         → best_model.zip + vec_normalize.pkl
         → export_onnx.py
         → explorer_ppo.onnx

Deploy: /plan (Nav2 global path)
         → rl_controller_node (carrot extraction + ONNX inference)
         → /cmd_vel
```

`vec_normalize.pkl` must be loaded alongside any checkpoint during evaluation — it holds the running obs/reward statistics.

### Observation & Action Space

```
Inputs:
  costmap:     (84, 84, 1) uint8   — local obstacle grid (Nav2 costmap values: 0=free, 254=lethal)
  scan:        (360,)      float32 — normalised LiDAR ranges [0, 1]
  goal_vector: (2,)        float32 — [cos θ, sin θ] direction + normalised distance in robot frame
  velocity:    (3,)        float32 — [vx, vy, yaw_rate] normalised by MAX_VEL constants

Action:
  (3,) float32 — [vx, vy, vyaw] in [-1, 1], scaled by obs_builder.scale_action()
```

VecTransposeImage converts costmap from HWC → CHW for the CNN. This wrapper is applied in both `train_ppo.py` and `sweep_optuna.py`.

### Config System

`conf/config.yaml` is the canonical training config, validated at load time by `config_schema.py` (`TrainingConfig` dataclass registered with Hydra ConfigStore). Fields:

| Group | Key fields |
|-------|-----------|
| PPO | `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm` |
| Architecture | `features_dim`, `net_arch_pi`, `net_arch_vf`, `policy` |
| Reward | `reward_progress`, `reward_goal_reached`, `reward_collision`, `reward_proximity`, `reward_proximity_threshold`, `reward_smoothness`, `reward_step_cost`, `reward_heading`, `reward_near_goal`, `reward_near_goal_radius` |
| Shared | `goal_tolerance` (0.3 m) — must match `rl_params.yaml` and `nav2_params.yaml xy_goal_tolerance` |

`rl_params.yaml` is for the inference node only (velocity limits, safety thresholds, carrot radius). It does not participate in training.

### STUCK Detection

`STUCK_WINDOW = 60` steps, `STUCK_THRESHOLD = 0.15` m — identical in `gym_env.py`, `mujoco_env.py`, and `point2d_env.py`. If you change one, change all three.

### Key Config Files

| File | Purpose |
|------|---------|
| `src/rl_local_planner/config/training_config.yaml` | Training hyperparams + reward weights (validated by schema) |
| `src/rl_local_planner/config/rl_params.yaml` | Inference node: velocity limits (0.5/0.5 m/s), safety, carrot radius |
| `src/autonomous_explorer/config/nav2_params.yaml` | Nav2: planner, DWB controller, costmap params |
| `src/autonomous_explorer/config/slam_toolbox_params.yaml` | SLAM Toolbox online async mapper |
| `src/autonomous_explorer/config/ekf.yaml` | robot_localization EKF |
| `src/autonomous_explorer/config/exploration_bt.xml` | Behavior tree for NBV mission |

### Checkpoints & Models

| Path | Notes |
|------|-------|
| `rl_best_model/best_model.zip` | Best checkpoint by eval reward; used for ONNX export |
| `rl_checkpoints/ppo_step_*.zip` | Periodic saves every 50 k steps |
| `rl_checkpoints/vec_normalize.pkl` | Running normalisation stats — required for correct evaluation |
| `models/explorer_ppo.onnx` | Deployed policy for ROS2 |

### Box Sizes in MuJoCo XML

`urdf/worlds/mujoco_maze.xml` uses MJCF format where box `size` is **half-extents** (a `size="1 1 1"` box is 2 × 2 × 2 m).
