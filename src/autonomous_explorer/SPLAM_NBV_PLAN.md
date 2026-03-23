# 6D SPLAM / NBV Exploration Node - Implementation Plan

## Context

The project needs a node that performs **Simultaneous Planning, Localization, And Mapping** (SPLAM) — tightly coupling 3D lidar odometry, occupancy mapping, and Next-Best-View exploration into a single unified system. This replaces the current decoupled approach (separate SLAM + separate frontier explorer) with an integrated pipeline where planning decisions inform where to map next, and mapping results inform the next plan.

The robot has a 16-beam 3D LiDAR (360x16, 0.5-20m range, 40Hz on `/points`), IMU (100Hz on `/imu`), and wheel odometry (`/odom`).

---

## Architecture: Single Unified Node

**File:** `scripts/nbv_splam_node.py` (main node)
**File:** `scripts/nbv_utils.py` (algorithms/data structures)
**File:** `config/nbv_splam_params.yaml` (parameters)
**File:** `launch/nbv_splam.launch.py` (launch file)

**Modified:** `CMakeLists.txt` (add new scripts to install list)
**Modified:** `launch/all.launch.py` (option to launch SPLAM node)

### Why a single node?
SPLAM requires tight coupling: the planner needs the map, the mapper needs the pose, and the pose estimation uses the latest scan. Keeping them in one node avoids latency and synchronization issues.

---

## Pipeline (10 Steps)

### Step 1: 3D LiDAR Odometry (Localization)

**Input:** `/points` (PointCloud2), `/imu` (Imu), `/odom` (Odometry)
**Output:** 6DOF robot pose estimate

- Parse PointCloud2 into numpy array (N,3) using `sensor_msgs_py`
- Filter by height (remove ground < -0.3m, ceiling > 2.0m)
- Voxel-grid downsample (0.1m) for efficiency
- **ICP scan-to-scan matching** (reuse SVD approach from existing `slam_node.py:232-286`):
  - Extend to 3D: 3-point SVD for (dx, dy, dz, roll, pitch, yaw)
  - Use scipy KDTree for correspondence finding
  - Max 20 iterations, convergence threshold 1e-5
- Fuse with IMU for roll/pitch stabilization (complementary filter)
- Fuse with wheel odometry as motion prior for ICP initial guess
- Publish corrected pose, broadcast TF `map -> odom`

**Reuse:** `slam_node.py` patterns for ICP, quaternion utilities, TF broadcasting, `utils.py` MathUtils

### Step 2: 3D-to-2D Projection (for planning)

- Project filtered 3D points to 2D ground plane (drop z)
- This 2D representation drives the exploration polygon extraction
- Keep 3D cloud for map accumulation

### Step 3: Occupancy Grid Mapping

**Output:** `/map` (OccupancyGrid)

- Log-odds Bayesian occupancy grid (reuse approach from `slam_node.py:386-426`)
- Grid: 800x800 cells, 0.05m resolution (40m x 40m)
- Bresenham ray tracing for free-space carving (reuse `slam_node.py:549-571`)
- Update with each processed scan at the corrected SLAM pose
- Publish at 2Hz with TRANSIENT_LOCAL QoS

**Reuse:** `slam_node.py` `update_map()`, `publish_map()`, `bresenham_line()`, `world_to_grid()`

### Step 4: Outline Polygon Extraction

- Convert 2D scan points to polar coordinates centered on robot
- Sort by polar angle, bin into 1-degree sectors
- For each sector: select the max-range point as the outline vertex
- Connect consecutive vertices into edges
- **Jump edge detection:** where gap between consecutive vertices exceeds threshold (0.5m), that edge is a "jump edge" = frontier to unexplored space
- For sectors with no return: insert artificial vertex at sensor max range (boundary ray)
- Result: `OutlinePolygon` with `vertices`, `edges`, `jump_edges`

### Step 5: Candidate NBV Pose Generation

Two strategies combined (hybrid):

**A) Jump Edge Placement (primary):**
- For each jump edge: place candidate at `midpoint + normal * 0.5m` (offset into free space)
- Orient candidate to face the jump edge
- Validate against occupancy grid (must be in free space)

**B) Uniform Sampling (fallback):**
- Grid sample (0.5m spacing) inside outline polygon bounding box
- Point-in-polygon test (ray casting)
- Filter: free cells only, within exploration radius (8m)

### Step 6: Visibility/Coverage Scoring — L(p)

For each candidate p:
- Raycast 360 virtual beams (1-degree resolution) from p
- For each ray: check intersection with jump edges (2D line-segment intersection)
- Also check ray blocked by occupied cells in occupancy grid (Bresenham)
- `L(p) = count of rays intersecting jump edges / total rays`
- Higher L(p) = candidate can observe more unexplored frontier

### Step 7: Travel Cost — ||r - p||

- Euclidean distance from current robot pose r to candidate p
- Normalized by exploration radius

### Step 8: Orientation Cost — |θ(r) - θ(p)|

- Angular difference between current heading and required heading at candidate
- Normalized by π

### Step 9: Candidate Selection + Path Planning

**Score:** `total = w_vis * L(p) - w_dist * dist_cost - w_orient * orient_cost`
**Weights (default):** w_vis=3.0, w_dist=1.0, w_orient=0.5

- Select `p* = argmax(total_score)`
- Plan path using **A*** on inflated occupancy grid:
  - Inflate obstacles by robot radius (0.3m)
  - 8-connected grid search
  - Smooth path (Douglas-Peucker)
- If heading error > 30°: prepend turn-in-place segment
- If path planning fails: blacklist candidate, try next best

### Step 10: Execute + Replan Loop

**State machine:**
```
INITIALIZING → COMPUTING_NBV → PLANNING_PATH → EXECUTING → (loop back to COMPUTING_NBV)
                     ↑              ↓                ↓
                     └── RECOVERY ←─┘                └→ EXPLORATION_COMPLETE
```

- **Proportional controller** for path following (kp_linear=1.0, kp_angular=2.0)
- **Continuous replanning** strategy (replan every 5s while moving, don't stop to scan — 40Hz lidar provides data while moving)
- **Stuck detection:** no progress for 10s → RECOVERY (rotate 360°, backup 0.5m)
- **Completion:** no jump edges with L(p) > 0.1 → EXPLORATION_COMPLETE
- Publish `/cmd_vel` at 10Hz

---

## ROS2 Interface

### Subscriptions
| Topic | Type | Purpose |
|-------|------|---------|
| `/points` | PointCloud2 | 3D LiDAR (primary sensor) |
| `/imu` | Imu | Roll/pitch stabilization |
| `/odom` | Odometry | Motion prior for ICP |
| `/detected_obstacles` | PoseArray | Real-time obstacle avoidance |

### Publications
| Topic | Type | Purpose |
|-------|------|---------|
| `/cmd_vel` | Twist | Motion commands (10Hz) |
| `/map` | OccupancyGrid | SPLAM-built map (2Hz) |
| `/splam_pose` | PoseStamped | Corrected 6DOF pose |
| `/nbv_goal` | PoseStamped | Current NBV target |
| `/nbv_candidates` | MarkerArray | Candidate visualization |
| `/outline_polygon` | Marker | Exploration boundary (LINE_STRIP) |
| `/jump_edges` | MarkerArray | Frontier edges (red lines) |
| `/nbv_path` | Path | Planned trajectory |

### TF Published
- `map → odom` (correction transform, discontinuous)

---

## File Details

### `scripts/nbv_splam_node.py` (~400 lines)
Main ROS2 node class `NBVSPLAMNode(Node)`:
- Subscribers, publishers, timers
- Callback routing
- State machine orchestration
- Instantiates all algorithm modules from `nbv_utils.py`

### `scripts/nbv_utils.py` (~600 lines)
Algorithm classes:
- `PointCloudProcessor` — parse, filter, downsample, transform
- `LidarOdometry` — 3D ICP with IMU fusion
- `OccupancyMapper` — log-odds grid mapping (adapted from slam_node.py)
- `OutlineExtractor` — polar sort, jump edge detection
- `CandidateGenerator` — hybrid jump-edge + sampling
- `NBVScorer` — raycasting visibility + travel/orientation costs
- `AStarPlanner` — A* on inflated grid + path smoothing
- `TrajectoryController` — proportional path follower
- Data classes: `OutlinePolygon`, `Edge`, `JumpEdge`, `Candidate`, `ScoredCandidate`

### `config/nbv_splam_params.yaml`
All tunable parameters organized by module (see Step details for defaults).

### `launch/nbv_splam.launch.py`
Launches: NBV SPLAM node + obstacle_cluster_node (obstacle detection). Does NOT launch separate SLAM nodes (SPLAM node handles its own localization + mapping).

### `CMakeLists.txt` changes
Add to `install(PROGRAMS ...)`:
```
scripts/nbv_splam_node.py
scripts/nbv_utils.py
```

### `launch/all.launch.py` changes
Add launch argument `use_splam:=true` to switch between old exploration and new SPLAM node.

---

## Dependencies

- **Already available:** numpy, scipy (KDTree, SVD), sensor_msgs, nav_msgs, geometry_msgs, visualization_msgs, tf2_ros
- **Needs install:** `ros-${ROS_DISTRO}-sensor-msgs-py` (for PointCloud2 parsing — check if already available)
- No new pip packages required

## Verification

1. `colcon build --packages-select autonomous_explorer --symlink-install`
2. Launch: `ros2 launch autonomous_explorer nbv_splam.launch.py`
3. In RViz: verify outline polygon (green), jump edges (red), candidates (blue arrows), selected goal (magenta), planned path (orange) all visualize correctly
4. Observe robot autonomously exploring the maze world
5. Check `/map` builds correctly as robot moves
6. Compare coverage time vs old `autonomous_exploration_node.py`
