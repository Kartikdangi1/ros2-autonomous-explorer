"""Gymnasium environment wrapping the Gazebo simulation for RL training.

Interfaces with a *running* Gazebo + ROS2 stack via rclpy.  The training
launch file (train.launch.py) must be running before this env is created.

Step synchronisation:  gates on /fused_scan arrival (20 Hz).  Costmap may
be 1-2 steps stale — this mirrors real-world sensor asynchrony and the
policy learns to handle it during training.

Collision detection:  checks whether the robot footprint overlaps lethal
cells in the local costmap (cost ≥ 253) **or** min LiDAR range drops
below a tight threshold (0.15 m, well inside the chassis).
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections import deque

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.srv import ClearEntireCostmap
from sensor_msgs.msg import LaserScan

from rl_local_planner.obs_builder import (
    COSTMAP_OBS_SIZE, SCAN_DIM, RawSensorData, build_observation, scale_action,
)
from rl_local_planner.reward import RewardWeights, RewardState, compute_reward
from rl_local_planner.curriculum import CurriculumManager


# ── Constants ────────────────────────────────────────────────────────────────
COLLISION_LIDAR_THRESHOLD = 0.15   # metres (inside chassis = certain collision)
COLLISION_COSTMAP_LETHAL = 253     # Nav2 inscribed/lethal threshold
STUCK_WINDOW = 25                  # steps
STUCK_THRESHOLD = 0.05             # metres
MAX_STEPS = 200                    # per episode (20 s at 10 Hz)
STEP_TIMEOUT = 0.15                # seconds to wait for new scan
RESET_SETTLE_TIME = 0.5            # seconds after teleport (EKF reset handles sync)
COSTMAP_CLEAR_TIMEOUT = 2.0        # seconds to wait for costmap clear service
COLLISION_GRACE_STEPS = 5          # steps after reset where costmap collision is skipped
SPAWN_MIN_CLEARANCE = 0.30         # metres — min LiDAR reading after spawn; below = in a wall
MAX_SPAWN_RETRIES = 10             # max resamples before giving up and using the last attempt

# Robot footprint half-extents (metres) — from nav2_params.yaml
FOOTPRINT_HALF_X = 0.25
FOOTPRINT_HALF_Y = 0.15


class GazeboExplorerEnv(gym.Env):
    """Gymnasium env for training an RL local planner in Gazebo."""

    metadata = {'render_modes': []}

    def __init__(
        self,
        curriculum: CurriculumManager | None = None,
        reward_weights: RewardWeights | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._curriculum = curriculum or CurriculumManager()
        self._reward_weights = reward_weights or RewardWeights()

        # ── Observation & action spaces ──────────────────────────────────
        self.observation_space = gym.spaces.Dict({
            'costmap': gym.spaces.Box(
                0, 255, shape=(COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1), dtype=np.uint8,
            ),
            'scan': gym.spaces.Box(0.0, 1.0, shape=(SCAN_DIM,), dtype=np.float32),
            'goal_vector': gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            'velocity': gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        # ── ROS2 node ────────────────────────────────────────────────────
        # Use a unique node name to avoid rosout publisher collision warnings.
        # Each env gets its own SingleThreadedExecutor so multiple envs
        # (train + eval) can spin concurrently without executor conflicts.
        if not rclpy.ok():
            rclpy.init()
        node_name = f'rl_gym_env_{id(self):x}'
        self._node = Node(node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._node.get_logger().info('GazeboExplorerEnv: initialising ROS2 subscriptions')

        # Sensor data container
        self._raw = RawSensorData()
        self._scan_event = threading.Event()
        self._lock = threading.Lock()

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Subscriptions
        self._sub_scan = self._node.create_subscription(
            LaserScan, '/fused_scan', self._scan_cb, sensor_qos)
        self._sub_costmap = self._node.create_subscription(
            OccupancyGrid, 'local_costmap/costmap_raw', self._costmap_cb, sensor_qos)
        self._sub_odom = self._node.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, sensor_qos)

        # Publisher
        self._pub_cmd = self._node.create_publisher(Twist, '/cmd_vel', 10)

        # EKF pose reset publisher — resets robot_localization state after each teleport
        # so that /odometry/filtered reflects the Ignition world-frame position.
        # frame_id='odom' matches ekf.yaml world_frame: odom.
        self._pub_set_pose = self._node.create_publisher(
            PoseWithCovarianceStamped, '/set_pose', 10)

        # Service client — clears the local costmap after each teleport so stale
        # obstacle data from previous episodes doesn't cause immediate false collisions.
        self._clear_costmap_client = self._node.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')

        # Service client — clears the global costmap so old obstacle markings
        # from previous episode positions don't persist between episodes.
        self._clear_global_costmap_client = self._node.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')

        # Spin in a background thread
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # ── Episode state ────────────────────────────────────────────────
        self._step_count = 0
        self._reward_state = RewardState()
        self._position_history: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW)
        self._dynamics: dict = {}
        self._action_buffer: deque[np.ndarray] = deque()
        self._collision_grace_remaining = 0  # steps remaining where costmap check is skipped

    # ── ROS2 callbacks ───────────────────────────────────────────────────

    def _spin(self) -> None:
        while rclpy.ok():
            self._executor.spin_once(timeout_sec=0.01)

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._lock:
            self._raw.scan_ranges = np.array(msg.ranges, dtype=np.float32)
            self._raw.scan_age = 0
        self._scan_event.set()

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._raw.costmap = np.array(msg.data, dtype=np.uint8)
            self._raw.costmap_width = msg.info.width
            self._raw.costmap_height = msg.info.height
            self._raw.costmap_resolution = msg.info.resolution
            self._raw.costmap_origin_x = msg.info.origin.position.x
            self._raw.costmap_origin_y = msg.info.origin.position.y
            self._raw.costmap_age = 0

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._raw.robot_x = msg.pose.pose.position.x
            self._raw.robot_y = msg.pose.pose.position.y
            # Quaternion → yaw
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._raw.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
            self._raw.robot_vx = msg.twist.twist.linear.x
            self._raw.robot_vy = msg.twist.twist.linear.y
            self._raw.robot_vyaw = msg.twist.twist.angular.z

    # ── Collision detection ──────────────────────────────────────────────

    def _check_collision(self) -> bool:
        """Ground-truth collision via costmap footprint overlap + tight LiDAR.

        Costmap check is skipped for COLLISION_GRACE_STEPS steps after each
        reset so that a freshly-cleared, repopulating costmap doesn't produce
        false lethal-cell hits at the new spawn position.
        """
        with self._lock:
            # 1. Tight LiDAR check — always active (geometry can't lie)
            if self._raw.scan_ranges is not None:
                min_range = float(np.nanmin(self._raw.scan_ranges))
                if min_range < COLLISION_LIDAR_THRESHOLD:
                    return True

            # 2. Costmap footprint check — suppressed during grace period
            if self._collision_grace_remaining > 0:
                self._collision_grace_remaining -= 1
                return False

            if self._raw.costmap is not None:
                return self._footprint_in_lethal()
        return False

    def _footprint_in_lethal(self) -> bool:
        """Check if any cell under the robot footprint is lethal."""
        r = self._raw
        if r.costmap is None:
            return False

        grid = r.costmap.reshape(r.costmap_height, r.costmap_width)
        res = r.costmap_resolution

        # Robot footprint corners in world frame
        cos_y = math.cos(r.robot_yaw)
        sin_y = math.sin(r.robot_yaw)
        corners_body = [
            (FOOTPRINT_HALF_X, FOOTPRINT_HALF_Y),
            (FOOTPRINT_HALF_X, -FOOTPRINT_HALF_Y),
            (-FOOTPRINT_HALF_X, -FOOTPRINT_HALF_Y),
            (-FOOTPRINT_HALF_X, FOOTPRINT_HALF_Y),
        ]

        # Find bounding box of footprint in grid coordinates
        grid_cols = []
        grid_rows = []
        for bx, by in corners_body:
            wx = r.robot_x + cos_y * bx - sin_y * by
            wy = r.robot_y + sin_y * bx + cos_y * by
            col = int((wx - r.costmap_origin_x) / res)
            row = int((wy - r.costmap_origin_y) / res)
            grid_cols.append(col)
            grid_rows.append(row)

        min_col = max(0, min(grid_cols))
        max_col = min(r.costmap_width - 1, max(grid_cols))
        min_row = max(0, min(grid_rows))
        max_row = min(r.costmap_height - 1, max(grid_rows))

        if min_col > max_col or min_row > max_row:
            return False

        patch = grid[min_row:max_row + 1, min_col:max_col + 1]
        return bool(np.any(patch >= COLLISION_COSTMAP_LETHAL))

    # ── Costmap clearing ─────────────────────────────────────────────────

    def _clear_local_costmap(self) -> None:
        """Call Nav2's clear_entirely_local_costmap service synchronously.

        Removes all stale obstacle markings so the robot doesn't instantly
        get flagged as a collision after teleportation to a new spawn.
        Falls back gracefully if the service is unavailable.
        """
        if not self._clear_costmap_client.wait_for_service(timeout_sec=0.5):
            self._node.get_logger().warn(
                'clear_entirely_local_costmap service not available — '
                'relying on grace steps for collision suppression')
            return
        future = self._clear_costmap_client.call_async(ClearEntireCostmap.Request())
        # Spin until complete (max COSTMAP_CLEAR_TIMEOUT s)
        deadline = time.monotonic() + COSTMAP_CLEAR_TIMEOUT
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

    def _reset_ekf_pose(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Reset robot_localization EKF state to (x, y, yaw) in the odom frame.

        Must be called AFTER _teleport_robot() so the physics engine has already
        moved the robot.  The EKF immediately resets its state estimate, so that
        subsequent /odometry/filtered messages are correctly anchored to the
        Ignition world-frame spawn position.

        Non-zero covariance diagonal is required — robot_localization rejects
        a set_pose with all-zero covariance.
        """
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        # 6×6 row-major covariance [x, y, z, roll, pitch, yaw]
        # Small values signal high confidence without causing the EKF to blend
        cov = [0.0] * 36
        cov[0]  = 0.01   # x
        cov[7]  = 0.01   # y
        cov[14] = 0.001  # z (locked in two_d_mode)
        cov[21] = 0.001  # roll (locked)
        cov[28] = 0.001  # pitch (locked)
        cov[35] = 0.01   # yaw
        msg.pose.covariance = cov
        self._pub_set_pose.publish(msg)

    def _clear_global_costmap(self) -> None:
        """Call Nav2's clear_entirely_global_costmap service synchronously.

        Removes accumulated obstacle markings from previous episode positions so
        the global costmap doesn't retain stale obstacles from old robot poses.
        Falls back gracefully if the service is unavailable.
        """
        if not self._clear_global_costmap_client.wait_for_service(timeout_sec=0.5):
            self._node.get_logger().warn(
                'clear_entirely_global_costmap service not available — '
                'global costmap from previous episode may persist')
            return
        future = self._clear_global_costmap_client.call_async(ClearEntireCostmap.Request())
        deadline = time.monotonic() + COSTMAP_CLEAR_TIMEOUT
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

    # ── Teleport ─────────────────────────────────────────────────────────

    def _teleport_robot(self, x: float, y: float) -> None:
        """Teleport the robot via Ignition's set_pose service."""
        req = (
            f'name: "robot", position: {{x: {x}, y: {y}, z: 0.15}}, '
            f'orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        )
        try:
            subprocess.run(
                [
                    'ign', 'service',
                    '-s', '/world/maze_world/set_pose',
                    '--reqtype', 'ignition.msgs.Pose',
                    '--reptype', 'ignition.msgs.Boolean',
                    '--timeout', '2000',
                    '--req', req,
                ],
                capture_output=True,
                timeout=5.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self._node.get_logger().warn(f'Teleport failed: {e}')

    # ── Spawn validation ─────────────────────────────────────────────────

    def _sample_valid_spawn(self) -> tuple[float, float]:
        """Teleport to a sampled spawn and verify the robot isn't inside a wall.

        Uses the LiDAR minimum range as ground truth: if min_range < SPAWN_MIN_CLEARANCE
        after settling, the spawn is inside or flush against a wall and we resample.
        Retries up to MAX_SPAWN_RETRIES times; uses the last attempt regardless so
        training always continues.

        Returns the accepted (x, y) spawn position.
        """
        for attempt in range(MAX_SPAWN_RETRIES):
            candidate = self._curriculum.sample_spawn(self._rng)

            # Teleport to candidate
            self._teleport_robot(candidate[0], candidate[1])
            self._pub_cmd.publish(Twist())

            # Reset EKF AFTER teleport, BEFORE sleep — aligns /odometry/filtered
            # with the Ignition world-frame position so goal vectors are correct.
            # Must be before sleep so the EKF has the full settle window to flush
            # corrected odometry messages.
            self._reset_ekf_pose(candidate[0], candidate[1], yaw=0.0)

            # Null stale costmap so old data can't trigger false collisions
            with self._lock:
                self._raw.costmap = None

            # Clear Nav2 local costmap
            self._clear_local_costmap()

            # Clear Nav2 global costmap so obstacle cells from old episode
            # positions don't persist
            self._clear_global_costmap()

            # Settle + wait for fresh scan
            time.sleep(RESET_SETTLE_TIME)
            self._scan_event.clear()
            self._scan_event.wait(timeout=2.0)

            # Validate: check LiDAR clearance at this spawn
            with self._lock:
                scan = self._raw.scan_ranges

            if scan is not None:
                min_range = float(np.nanmin(scan))
                if min_range >= SPAWN_MIN_CLEARANCE:
                    if attempt > 0:
                        self._node.get_logger().info(
                            f'Valid spawn found at {candidate} after {attempt + 1} attempts '
                            f'(min_range={min_range:.2f} m)')
                    return candidate
                else:
                    self._node.get_logger().warn(
                        f'Spawn {candidate} rejected: min_range={min_range:.2f} m '
                        f'< {SPAWN_MIN_CLEARANCE} m (in wall). '
                        f'Attempt {attempt + 1}/{MAX_SPAWN_RETRIES}')
            else:
                self._node.get_logger().warn(
                    f'Spawn {candidate}: no scan received after settle. '
                    f'Attempt {attempt + 1}/{MAX_SPAWN_RETRIES}')

        # Exhausted retries — use last candidate anyway so training doesn't hang
        self._node.get_logger().error(
            f'All {MAX_SPAWN_RETRIES} spawn attempts failed clearance check. '
            'Using last candidate. Consider auditing spawn points for this world.')
        return candidate

    # ── Gymnasium interface ──────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a valid spawn — retry if the robot lands inside a wall
        spawn = self._sample_valid_spawn()
        goal = self._curriculum.sample_goal(spawn[0], spawn[1], self._rng)

        # Set goal
        with self._lock:
            self._raw.goal_x = goal[0]
            self._raw.goal_y = goal[1]
            self._raw.costmap_age = 0
            self._raw.scan_age = 0

        # Compute initial goal distance
        dx = goal[0] - spawn[0]
        dy = goal[1] - spawn[1]
        init_dist = math.hypot(dx, dy)

        # Reset episode state
        self._step_count = 0
        self._reward_state = RewardState(
            initial_goal_distance=max(init_dist, 0.1),
            prev_goal_distance=init_dist,
            prev_action=None,
        )
        self._position_history.clear()
        self._dynamics = self._curriculum.sample_dynamics(self._rng)
        self._action_buffer.clear()
        # Suppress costmap-based collision for the first few steps while the
        # freshly-cleared costmap repopulates with scan data from the new pose.
        self._collision_grace_remaining = COLLISION_GRACE_STEPS

        obs = self._get_obs()
        info = {
            'spawn': spawn,
            'goal': goal,
            'initial_distance': init_dist,
            'curriculum_stage': self._curriculum.current_stage,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # ── Domain randomization: action delay ───────────────────────────
        delay = self._dynamics.get('action_delay_steps', 0)
        if delay > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > delay:
                action = self._action_buffer.popleft()
            else:
                action = np.zeros(3, dtype=np.float32)

        # ── Scale and apply velocity command ─────────────────────────────
        vx, vy, vyaw = scale_action(action)

        # Domain randomization: velocity scaling
        vel_scale = self._dynamics.get('vel_scale', 1.0)
        vx *= vel_scale
        vy *= vel_scale
        vyaw *= vel_scale

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.angular.z = vyaw
        self._pub_cmd.publish(cmd)

        # ── Wait for next scan (gate on scan arrival) ────────────────────
        self._scan_event.clear()
        self._scan_event.wait(timeout=STEP_TIMEOUT)

        # Age tracking for stale observations
        with self._lock:
            self._raw.costmap_age += 1
            # scan_age is reset to 0 in _scan_cb when new scan arrives

        # ── Apply sensor noise (domain randomization) ────────────────────
        scan_noise = self._dynamics.get('scan_noise_sigma', 0.0)
        if scan_noise > 0 and self._raw.scan_ranges is not None:
            with self._lock:
                noise = self._rng.normal(0, scan_noise, size=self._raw.scan_ranges.shape)
                self._raw.scan_ranges = self._raw.scan_ranges + noise.astype(np.float32)

        # ── Compute observation ──────────────────────────────────────────
        obs = self._get_obs()

        # ── Compute goal distance ────────────────────────────────────────
        with self._lock:
            goal_dx = self._raw.goal_x - self._raw.robot_x
            goal_dy = self._raw.goal_y - self._raw.robot_y
            robot_yaw = self._raw.robot_yaw
            goal_dist = math.hypot(goal_dx, goal_dy)
            min_scan = float(np.nanmin(self._raw.scan_ranges)) \
                if self._raw.scan_ranges is not None else 999.0

        # ── Check termination conditions ─────────────────────────────────
        goal_reached = goal_dist < self._reward_weights.goal_tolerance
        collision = self._check_collision()

        # Stuck detection
        with self._lock:
            self._position_history.append((self._raw.robot_x, self._raw.robot_y))
        stuck = False
        if len(self._position_history) >= STUCK_WINDOW:
            oldest = self._position_history[0]
            newest = self._position_history[-1]
            displacement = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
            stuck = displacement < STUCK_THRESHOLD

        timeout = self._step_count >= self._curriculum.config.max_steps

        terminated = goal_reached or collision
        truncated = timeout or stuck

        # ── Compute reward ───────────────────────────────────────────────
        reward, reward_info = compute_reward(
            curr_goal_distance=goal_dist,
            min_scan_range=min_scan,
            action=action,
            goal_reached=goal_reached,
            collision=collision,
            state=self._reward_state,
            weights=self._reward_weights,
            robot_yaw=robot_yaw,
            goal_dx=goal_dx,
            goal_dy=goal_dy,
        )

        info = {
            'goal_reached': goal_reached,
            'collision': collision,
            'stuck': stuck,
            'timeout': timeout,
            'goal_distance': goal_dist,
            'min_range': min_scan,
            'step': self._step_count,
            'curriculum_stage': self._curriculum.current_stage,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        with self._lock:
            return build_observation(self._raw)

    def set_curriculum_stage(self, stage: int) -> None:
        """Propagate curriculum stage from the main process to this worker."""
        self._curriculum.current_stage = stage
        self._curriculum._successes.clear()

    def close(self) -> None:
        try:
            self._pub_cmd.publish(Twist())  # stop robot
        except Exception:
            pass  # rclpy context may already be shut down
        try:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
            self._executor.shutdown()
        except Exception:
            pass
