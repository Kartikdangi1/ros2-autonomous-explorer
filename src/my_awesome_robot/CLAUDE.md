# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# From workspace root: ~/ros2-vision-obstacle-avoidance
colcon build --packages-select my_awesome_robot sensor_fusion_preprocess --symlink-install
source install/setup.bash

# Launch full system (Gazebo + Nav2 + SLAM + NBV exploration)
ros2 launch my_awesome_robot nav2_exploration.launch.py

# Optional: enable RViz
ros2 launch my_awesome_robot nav2_exploration.launch.py use_rviz:=true
```

Build both packages together since `my_awesome_robot` depends on `sensor_fusion_preprocess`.

## Architecture

**4-tier pipeline:**
```
Simulation (Ignition Gazebo 6)
  → Sensor Fusion (LiDAR + Radar + RGB-D → /fused_scan)
  → Localization & Mapping (EKF + SLAM Toolbox → /map, TF)
  → Exploration & Navigation (NBV planner → Nav2 NavigateToPose)
```

### Node Responsibilities

| Node | File | Role |
|------|------|------|
| `sensor_fusion_preprocess_node` | `sensor_fusion_preprocess/` package | Merges LiDAR + radar + RGB-D depth into `/fused_scan` using per-bin priority (LiDAR > Depth > Radar) |
| `frontend_node` | `scripts/frontend_node.py` | Clusters `/scan` into `/detected_obstacles` and `/obstacle_markers` |
| `nbv_goal_provider_node` | `scripts/nbv_goal_provider_node.py` | Mission controller: reads `/map`, computes Next-Best-View candidates, sends `NavigateToPose` goals to Nav2 |
| SLAM Toolbox (`async_slam_toolbox_node`) | external | Publishes `/map`, owns `map→odom` TF |
| robot_localization EKF | external | Fuses `/odom` + `/imu` → `odometry/filtered`, sole publisher of `odom→base_link` TF |

**TF tree ownership (critical — do not duplicate):**
- `map → odom`: SLAM Toolbox
- `odom → base_link`: robot_localization EKF (30 Hz)
- `base_link → *`: robot_state_publisher from URDF (100 Hz)

### NBV Planning Library (`my_awesome_robot/nbv_utils.py`)

Core algorithms live in the Python package (not `scripts/`), importable as `from my_awesome_robot.nbv_utils import ...`:

- `OccupancyMapper` — Bayesian log-odds grid, synced from SLAM Toolbox `/map`
- `OutlineExtractor` — Polar-sector frontier detection; produces jump edges (range discontinuities > 1.0 m)
- `CandidateGenerator` — Places NBV candidates at jump edges + uniform sampling (hybrid strategy)
- `NBVScorer` — Scores candidates by visibility (ray-casting), travel distance, and orientation cost

`nbv_goal_provider_node.py` orchestrates these classes: it subscribes to `/map` and `/fused_scan`, generates/scores candidates, blacklists failed goals (30 s TTL), and re-plans continuously during execution.

### Startup Timing (in `nav2_exploration.launch.py`)

Components have explicit delays to respect dependency ordering:
- T+0s: Gazebo, robot_state_publisher, ros_gz_bridge, sensor fusion, IMU filter, EKF
- T+12s: SLAM Toolbox (waits for EKF TF to be available)
- T+17s: Nav2 stack (waits for `/map`)
- T+22s: NBV goal provider (waits for Nav2 action server)

The robot model is **statically embedded in `maze_world.sdf`** (not dynamically spawned) to avoid a MecanumDrive plugin race condition.

### Configuration Files (`config/`)

| File | Purpose |
|------|---------|
| `nav2_params.yaml` | Global/local costmaps, SmacPlanner2D, DWB controller, `inflation_radius=0.35 m` |
| `slam_toolbox_params.yaml` | Ceres solver backend, `resolution=0.05 m`, loop closure enabled |
| `ekf.yaml` | State fusion: `frequency=30 Hz`, `two_d_mode=true`, fuses odom + IMU |
| `sensor_fusion_params.yaml` | Sensor range limits for fusion node |
| `robot_params.yaml` | `ros_gz_bridge` topic/type mappings |
| `exploration_bt.xml` | Nav2 behavior tree with recovery escalation (spin → backup → wait) |

## Key Topics

| Topic | Type | Publisher |
|-------|------|-----------|
| `/scan` | LaserScan | Gazebo bridge (primary LiDAR) |
| `/fused_scan` | LaserScan | sensor_fusion_preprocess |
| `/imu` | Imu | Gazebo bridge |
| `/odom` | Odometry | Gazebo bridge (wheel odometry) |
| `/map` | OccupancyGrid | SLAM Toolbox |
| `/detected_obstacles` | PoseArray | frontend_node |
| `/detected_obstacles_fusion` | PoseArray | sensor_fusion_preprocess |

## Robot Model

4WD Mecanum holonomic base (0.5 × 0.3 m, 3.0 kg) with Hokuyo GPU LiDAR (16-beam, 360°, 40 Hz), 9-DOF IMU, magnetometer, RGB-D camera, and forward radar proxy. Mecanum wheels use anisotropic friction (`mu=1.0`, `mu2=0.01`) for holonomic motion.

## Code Style

- Python nodes: class inherits `Node`, `declare_parameter()` for all params, `use_sim_time: True`
- `nbv_utils.py` is a library module (no `main()`); node scripts import from it via `from my_awesome_robot.nbv_utils import ...`
- ROS2 Humble + Ignition Gazebo 6 (not Classic)
