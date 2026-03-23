# 3D Gaussian Splatting (3DGS) Dense SLAM — Implementation Plan

## Context

The current system uses SLAM Toolbox for 2D occupancy grid mapping. This plan adds a parallel
3DGS Dense SLAM layer that builds a photorealistic 3D Gaussian map from the robot's RGB-D camera
while the existing 2D Nav2 stack runs unchanged. The 3DGS system lives entirely in **separate new
files** (separate launch file, config, library module, and ROS2 node) — no existing files are
modified except `CMakeLists.txt` and `package.xml` (minimal additions).

**Design decisions:**
- **Parallel** to SLAM Toolbox — 3DGS does not own TF or `/map`; Nav2 is unaffected
- **Separate launch file** (`gsplat_slam.launch.py`) — can be run standalone or alongside the main launch
- **GPU path** via `gsplat` library (`pip install gsplat torch`); graceful CPU fallback skips rendering
- **Tracking:** uses existing TF tree (`map → camera_optical_link`) — no duplicate VO needed
- **Full pipeline:** keyframe selection → RGBD seeding → Adam optimization → rendered image output

---

## Architecture

```
Existing (unchanged):
  /rgbd/image + /rgbd/depth_image + /rgbd/camera_info  (10 Hz, Gazebo)
  /odometry/filtered  (EKF 30 Hz)
  TF: map → odom → base_link → camera_optical_link

New (parallel, read-only from existing topics):
  gsplat_slam_node
  ├── ApproximateTimeSynchronizer (rgb + depth, slop=0.05s)
  ├── CameraInfo subscriber (cached separately — latched in Gazebo)
  ├── TF lookup: map → camera_optical_link  (no new TF publishing)
  ├── KeyframeSelector: dist >0.3m OR angle >15° OR time >5s
  ├── Background thread: GaussianInitializer → GaussianOptimizer
  └── Publishers:
        /gsplat/gaussians   PointCloud2  (Gaussian centers + RGBA)
        /gsplat/render      Image        (photorealistic novel view)
        /gsplat/keyframes   PoseArray    (camera trajectory)
        /gsplat/map_stats   String       (JSON diagnostics, 5 Hz)
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/autonomous_explorer/autonomous_explorer/gsplat_slam.py` | Core library: `GaussianMap`, `GaussianInitializer`, `GaussianOptimizer`, `GaussianRenderer`, helpers |
| `src/autonomous_explorer/scripts/gsplat_slam_node.py` | ROS2 node executable |
| `src/autonomous_explorer/config/gsplat_params.yaml` | All parameters |
| `src/autonomous_explorer/launch/gsplat_slam.launch.py` | Standalone launch (includes Gazebo + EKF + SLAM Toolbox + 3DGS) |

## Files to Modify

| File | Change |
|------|--------|
| `src/autonomous_explorer/CMakeLists.txt` | Add `scripts/gsplat_slam_node.py` to `install(PROGRAMS)` |
| `src/autonomous_explorer/package.xml` | Add `<exec_depend>python3-torch</exec_depend>` comment block for gsplat pip dep |

---

## Library Module: `gsplat_slam.py`

### Data Classes
```python
@dataclass
class CameraIntrinsics:
    fx, fy, cx, cy: float   # pixels
    width, height: int
    depth_min, depth_max: float  # meters (0.3, 6.0 from robot URDF)

@dataclass
class Keyframe:
    frame_id: int
    stamp: float              # seconds
    pose: np.ndarray          # (4,4) SE3 camera-in-world
    rgb: np.ndarray           # (H, W, 3) uint8
    depth: np.ndarray         # (H, W) float32 meters
    intrinsics: CameraIntrinsics

@dataclass
class GaussianMapStats:
    num_gaussians, num_keyframes, optimizer_iters: int
    gpu_available: bool
    last_update_s: float
```

### Classes

**`GaussianMap`** — thread-safe storage (torch tensors on GPU / numpy on CPU)
- Gaussian params: `positions (N,3)`, `colors (N,3)`, `opacities (N,1)` logit, `scales (N,3)` log, `rotations (N,4)` quat
- Methods: `add_gaussians()`, `prune_by_mask()`, `update_params()`, `get_*_numpy()`, `__len__()`
- Uses `threading.Lock()` (same pattern as `OccupancyMapper._lock` in `nbv_utils.py`)

**`GaussianInitializer`** — seeds Gaussians from RGBD keyframe
- `initialize_from_keyframe(keyframe, gaussian_map, existing_positions)` → int (added count)
- Internals: `_voxel_downsample()` (numpy grid hash), `_estimate_scales()` (scipy KDTree k-NN)
- Filters: depth outside [0.3, 6.0], NaN; max 5000 pts/frame via stride; voxel dedup at 0.05m

**`GaussianOptimizer`** — Adam-based online optimization
- `optimize_keyframe(keyframe, gaussian_map, renderer)` → dict with losses + densify/prune counts
- Adam lr: positions=1.6e-4, colors=2.5e-3, opacities=0.05, scales=5e-3, rotations=1e-3
- Loss: `0.8*L1 + 0.2*(1-SSIM)` on RGB + `0.1*L1` on depth (valid pixels only)
- Densification every 50 iters (clone high-grad small Gaussians); pruning every 100 iters
- Hard cap: `max_gaussians=200_000`

**`GaussianRenderer`** — wraps `gsplat.rasterization()`
- `render(gaussian_map, camera_pose)` → `(rgb_np, depth_np)` or `(None, None)` on CPU
- `render_torch(...)` → tensors (avoids GPU↔CPU round-trip for optimizer)
- `available = GSPLAT_AVAILABLE and use_gpu`

### Helper Functions
```python
rgbd_to_pointcloud(rgb, depth, intrinsics, stride=1) -> (points, colors)
pose_to_camera_matrix(pose_4x4) -> viewmat_4x4   # viewmat = inv(pose)
build_intrinsics_matrix(intrinsics) -> K_3x3
ros_camera_info_to_intrinsics(msg) -> CameraIntrinsics  # reads K[0,4,2,5]
transform_stamped_to_se3(tf_msg) -> ndarray(4,4)  # scipy Rotation.from_quat([x,y,z,w])
quaternion_to_rotation_matrix(q_wxyz) -> R_3x3
```

---

## ROS2 Node: `gsplat_slam_node.py`

**Key implementation notes:**
- **`use_sim_time`**: do NOT `declare_parameter('use_sim_time', ...)` — it is set globally by the launch file via `SetParameter`. Declaring it again raises `ParameterAlreadyDeclaredException` (see `nbv_goal_provider_node.py` comment).
- **CameraInfo**: subscribe separately (not in synchronizer) — it's effectively latched in Gazebo. Cache on first receipt; use for lazy SLAM component init.
- **Image decode (no cv_bridge)**: `np.frombuffer(msg.data, dtype=np.uint8).reshape(H,W,C)` for RGB; `dtype=np.float32` for `32FC1` depth, `dtype=np.uint16 / 1000.0` for `16UC1`.
- **Background thread**: push keyframes to `queue.Queue(maxsize=4)`; daemon thread consumes. Prevents ROS callback thread blocking on GPU compute.
- **TF lookup**: `map → camera_optical_link` at `rgb_msg.header.stamp` with `tf_timeout_s=0.1`.

**Keyframe selection (`_is_new_keyframe`):**
- Accept if: first frame, OR translation > 0.3 m, OR rotation > 15°, OR time gap > 5 s

**Parameter names** (all declared with `self.declare_parameter()`):
- TF: `map_frame`, `camera_frame`, `tf_timeout_s`
- Keyframe: `kf_dist_threshold_m`, `kf_angle_threshold_deg`, `kf_max_time_gap_s`
- Initializer: `voxel_size`, `init_opacity_logit`, `init_scale_factor`, `k_neighbors`, `max_points_per_frame`
- Optimizer: `lr_positions`, `lr_colors`, `lr_opacities`, `lr_scales`, `lr_rotations`, `num_iters_per_kf`, `densify_interval`, `prune_interval`, `opacity_prune_threshold`, `scale_prune_max`, `max_gaussians`, `lambda_depth`
- Renderer: `near_plane`, `far_plane`, `use_gpu`
- Output: `publish_render`, `publish_gaussians`, `stats_publish_hz`

---

## Config: `gsplat_params.yaml`

```yaml
gsplat_slam:
  ros__parameters:
    use_sim_time: true
    map_frame: map
    camera_frame: camera_optical_link   # matches URDF camera_optical_joint child
    tf_timeout_s: 0.10
    kf_dist_threshold_m: 0.30
    kf_angle_threshold_deg: 15.0
    kf_max_time_gap_s: 5.0
    voxel_size: 0.05
    init_opacity_logit: -2.197          # sigmoid(-2.197) ≈ 0.10
    init_scale_factor: 1.0
    k_neighbors: 4
    max_points_per_frame: 5000
    lr_positions: 0.00016
    lr_colors: 0.0025
    lr_opacities: 0.05
    lr_scales: 0.005
    lr_rotations: 0.001
    num_iters_per_kf: 100
    densify_interval: 50
    prune_interval: 100
    opacity_prune_threshold: 0.005
    scale_prune_max: 1.0
    max_gaussians: 200000
    lambda_depth: 0.1
    near_plane: 0.3
    far_plane: 6.0
    use_gpu: true
    publish_render: true
    publish_gaussians: true
    stats_publish_hz: 5.0
```

---

## Launch: `gsplat_slam.launch.py`

Standalone launch that starts only what 3DGS needs:
- Includes Gazebo + robot_state_publisher + ros_gz_bridge + sensor_fusion + EKF + SLAM Toolbox
- Adds `gsplat_slam_node` at T+15s (after TF tree is stable)
- Does NOT start Nav2 or NBV (can be combined with `nav2_exploration.launch.py` if desired)

```python
# Key additions in launch_setup():
GSPLAT_PARAMS = os.path.join(PKG, 'config', 'gsplat_params.yaml')

gsplat_node = TimerAction(period=15.0, actions=[Node(
    package='autonomous_explorer',
    executable='gsplat_slam_node.py',
    name='gsplat_slam',
    parameters=[GSPLAT_PARAMS],
    output='screen'
)])
```

---

## CMakeLists.txt Change

```cmake
install(PROGRAMS
  scripts/obstacle_cluster_node.py
  scripts/nbv_goal_provider_node.py
  scripts/gsplat_slam_node.py          # ← add this line
  DESTINATION lib/${PROJECT_NAME}
)
```

(`gsplat_slam.py` is auto-installed by the existing `ament_python_install_package(${PROJECT_NAME})` call on line 8.)

---

## Implementation Phases

### Phase 1 — Scaffolding (CPU, no rendering)
1. Create `gsplat_slam.py` with all data classes + helper functions + `GaussianMap` (numpy)
2. Create `gsplat_slam_node.py` with parameter declarations, TF setup, 2-topic synchronizer (RGB+depth), `CameraInfo` cache, `_decode_rgb/depth()`, `_is_new_keyframe()`, `_publish_stats()`
3. Create `gsplat_params.yaml`
4. Create `gsplat_slam.launch.py`
5. Update `CMakeLists.txt` + `package.xml`
6. **Test:** `colcon build` → `ros2 launch autonomous_explorer gsplat_slam.launch.py` → verify node starts, logs "Keyframe N accepted"

### Phase 2 — Map Accumulation
1. Implement `GaussianInitializer` (numpy, scipy KDTree)
2. Wire `_rgbd_callback` → keyframe queue → `GaussianInitializer.initialize_from_keyframe()`
3. Implement `_publish_gaussians()` PointCloud2 builder (fields: x,y,z,r,g,b,a)
4. Implement `_publish_keyframes()` PoseArray builder
5. **Test:** RViz PointCloud2 on `/gsplat/gaussians` grows as robot explores; colored maze walls visible

### Phase 3 — GPU Optimization + Rendering
1. `pip install gsplat torch` (if not already installed)
2. Implement `GaussianRenderer.render_torch()` using `gsplat.rasterization()`
3. Implement `GaussianOptimizer` (Adam loop, SSIM loss, densify/prune)
4. Wire background thread to call optimizer after initializer
5. Implement `_publish_render()` for rendered images
6. **Test:** RViz Image display on `/gsplat/render` shows photorealistic maze walls improving over time

---

## Verification

```bash
# Build
colcon build --packages-select autonomous_explorer --symlink-install

# Import check
python3 -c "from autonomous_explorer.gsplat_slam import GaussianMap, rgbd_to_pointcloud; print('OK')"

# Node starts
ros2 launch autonomous_explorer gsplat_slam.launch.py
ros2 node list | grep gsplat          # → /gsplat_slam
ros2 topic list | grep gsplat         # → /gsplat/gaussians /gsplat/render /gsplat/keyframes /gsplat/map_stats

# Stats (JSON with num_gaussians growing)
ros2 topic echo /gsplat/map_stats

# Non-interference check
ros2 topic hz /map                    # → ~3 Hz (SLAM Toolbox unchanged)
ros2 topic hz /odometry/filtered      # → ~30 Hz (EKF unchanged)

# TF check — gsplat_slam must NOT appear as TF publisher
ros2 run tf2_tools view_frames
```

---

## Critical Notes

1. **`camera_optical_link` frame**: URDF defines `camera_optical_joint` with `rpy="-π/2 0 -π/2"` converting ROS camera convention → URDF. The TF lookup `map → camera_optical_link` already encodes this rotation; `pose_to_camera_matrix()` can use it directly.
2. **No TF publishing**: `gsplat_slam_node` must never call `tf2_ros.TransformBroadcaster`. TF ownership is documented in `CLAUDE.md` and must not be violated.
3. **`use_sim_time`**: Set globally by launch via `SetParameter` — do NOT declare it in the node's `__init__`.
4. **Thread safety**: `GaussianMap._lock` serializes ROS callback thread vs. background optimizer thread. Pattern mirrors `OccupancyMapper._lock` in `nbv_utils.py:108`.
5. **CameraInfo in synchronizer**: Subscribe separately and cache; don't include in `ApproximateTimeSynchronizer` (Gazebo publishes it infrequently/latched, causing sync to stall).
