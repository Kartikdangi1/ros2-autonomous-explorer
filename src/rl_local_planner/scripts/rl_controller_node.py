#!/usr/bin/env python3
"""ROS2 RL controller node — drop-in replacement for DWB local planner.

Subscribes to the global path from Nav2's planner_server plus sensor
data, runs ONNX policy inference at 10 Hz, and publishes cmd_vel.

Safety layer:
  - 3-zone reactive wall escape (blend/override RL when min LiDAR range < escape_blend_start)
  - Velocity clamping to configured limits
  - Acceleration limiting (smooth output between steps)

Graceful degradation:
  - If ONNX model is missing / fails to load → zero velocity (no crash)
  - If no /plan available → zero velocity
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from nav2_msgs.srv import ClearEntireCostmap

from rl_local_planner.obs_builder import (
    COSTMAP_OBS_SIZE, MAX_VEL_X, MAX_VEL_Y, MAX_VEL_THETA,
    RawSensorData, build_observation, scale_action,
)
from rl_local_planner.onnx_inference import OnnxPolicy


class RLControllerNode(Node):
    """Runs the trained RL policy and publishes velocity commands."""

    def __init__(self):
        super().__init__('rl_controller')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('carrot_radius', 3.0)
        self.declare_parameter('carrot_min_scale', 0.35)
        self.declare_parameter('carrot_narrow_threshold', 1.5)
        self.declare_parameter('max_vel_x', 0.5)
        self.declare_parameter('max_vel_y', 0.5)
        self.declare_parameter('max_vel_theta', 1.0)
        self.declare_parameter('safety_min_range', 0.18)
        self.declare_parameter('escape_blend_start', 0.28)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('costmap_size', 84)
        self.declare_parameter('max_accel_x', 2.5)
        self.declare_parameter('max_accel_y', 2.5)
        self.declare_parameter('max_accel_theta', 3.2)
        self.declare_parameter('stuck_window', 60)
        self.declare_parameter('stuck_threshold', 0.15)
        self.declare_parameter('stuck_recovery_dist', 1.5)
        self.declare_parameter('plan_max_age_sec', 5.0)

        model_path = self.get_parameter('model_path').value
        rate = self.get_parameter('inference_rate').value
        self._carrot_radius = self.get_parameter('carrot_radius').value
        self._carrot_min_scale = self.get_parameter('carrot_min_scale').value
        self._carrot_narrow_threshold = self.get_parameter('carrot_narrow_threshold').value
        self._max_vx = self.get_parameter('max_vel_x').value
        self._max_vy = self.get_parameter('max_vel_y').value
        self._max_vyaw = self.get_parameter('max_vel_theta').value
        self._safety_range = self.get_parameter('safety_min_range').value
        self._escape_blend_start = self.get_parameter('escape_blend_start').value
        self._goal_tol = self.get_parameter('goal_tolerance').value
        self._max_ax = self.get_parameter('max_accel_x').value
        self._max_ay = self.get_parameter('max_accel_y').value
        self._max_ayaw = self.get_parameter('max_accel_theta').value
        self._dt = 1.0 / rate
        self._stuck_window = self.get_parameter('stuck_window').value
        self._stuck_threshold = self.get_parameter('stuck_threshold').value
        self._stuck_recovery_dist = self.get_parameter('stuck_recovery_dist').value
        self._plan_max_age = self.get_parameter('plan_max_age_sec').value

        # ── Load ONNX model ──────────────────────────────────────────────
        self._policy = OnnxPolicy(model_path)
        if not self._policy.is_loaded:
            self.get_logger().error(
                f'ONNX model not loaded from "{model_path}". '
                'Publishing zero velocity. Check model_path parameter.')

        # ── State ────────────────────────────────────────────────────────
        self._raw = RawSensorData()
        self._lock = threading.Lock()
        self._plan: list[tuple[float, float]] | None = None
        self._plan_lock = threading.Lock()
        self._plan_stamp: float = 0.0
        self._plan_stale_logged: bool = False
        self._prev_cmd = (0.0, 0.0, 0.0)  # for acceleration limiting

        # ── Stuck detection ───────────────────────────────────────────────
        self._position_history: deque[tuple[float, float]] = deque(
            maxlen=self._stuck_window)
        self._stuck_warn_logged: bool = False

        # ── Sensor health ─────────────────────────────────────────────────
        self._scan_nan_warn_time: float = 0.0

        # ── QoS ──────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── Subscriptions ────────────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/fused_scan', self._scan_cb, sensor_qos)
        self.create_subscription(
            OccupancyGrid, 'local_costmap/costmap_raw', self._costmap_cb, sensor_qos)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, sensor_qos)
        self.create_subscription(
            Path, '/plan', self._plan_cb, 10)

        # ── Publishers ───────────────────────────────────────────────────
        self._pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self._pub_stuck = self.create_publisher(Bool, '/rl/stuck', 10)

        # ── Costmap clear service clients ────────────────────────────────
        self._clear_local_client = self.create_client(
            ClearEntireCostmap,
            '/local_costmap/clear_entirely_local_costmap')
        self._clear_global_client = self.create_client(
            ClearEntireCostmap,
            '/global_costmap/clear_entirely_global_costmap')

        # ── Inference timer ──────────────────────────────────────────────
        self._timer = self.create_timer(self._dt, self._inference_tick)

        self.get_logger().info(
            f'RL controller started (rate={rate} Hz, model={model_path})')

    # ── Sensor callbacks ─────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._lock:
            self._raw.scan_ranges = np.array(msg.ranges, dtype=np.float32)

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._raw.costmap = np.array(msg.data, dtype=np.uint8)
            self._raw.costmap_width = msg.info.width
            self._raw.costmap_height = msg.info.height
            self._raw.costmap_resolution = msg.info.resolution
            self._raw.costmap_origin_x = msg.info.origin.position.x
            self._raw.costmap_origin_y = msg.info.origin.position.y

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._raw.robot_x = msg.pose.pose.position.x
            self._raw.robot_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._raw.robot_yaw = math.atan2(siny, cosy)
            self._raw.robot_vx = msg.twist.twist.linear.x
            self._raw.robot_vy = msg.twist.twist.linear.y
            self._raw.robot_vyaw = msg.twist.twist.angular.z
            self._position_history.append(
                (self._raw.robot_x, self._raw.robot_y))

    def _plan_cb(self, msg: Path) -> None:
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._plan_lock:
            self._plan = points
            self._plan_stamp = time.monotonic()
            self._plan_stale_logged = False

    # ── Carrot-point extraction ──────────────────────────────────────────

    def _extract_carrot(self) -> tuple[float, float] | None:
        """Find the best local goal from the global path.

        Algorithm: find the furthest point on the path within carrot_radius
        of the robot that is reachable via line-of-sight on the local costmap.
        """
        with self._plan_lock:
            plan = self._plan
        if not plan:
            return None

        with self._lock:
            rx, ry = self._raw.robot_x, self._raw.robot_y
            latest_scan = (self._raw.scan_ranges.copy()
                           if self._raw.scan_ranges is not None else None)

        # Adaptive carrot radius — shrink in tight corridors so the robot
        # doesn't try to target a point on the far side of a wall.
        if latest_scan is not None:
            min_range = float(np.min(latest_scan))
            if min_range < self._carrot_narrow_threshold:
                corridor_scale = max(
                    self._carrot_min_scale,
                    min_range / self._carrot_narrow_threshold)
            else:
                corridor_scale = 1.0
        else:
            corridor_scale = 1.0
        radius = self._carrot_radius * corridor_scale

        best = None
        best_idx = -1

        # Search from end of path backward — we want the furthest reachable point
        for i in range(len(plan) - 1, -1, -1):
            px, py = plan[i]
            dist = math.hypot(px - rx, py - ry)
            if dist <= radius:
                if self._line_of_sight(rx, ry, px, py):
                    best = (px, py)
                    best_idx = i
                    break

        if best is not None:
            return best

        # Fallback: closest path point within radius that still has LOS.
        # The original fallback had no LOS check and searched the entire plan —
        # this could target a point on the far side of a wall when no nearby
        # point is visible, driving the robot directly into the obstacle.
        min_dist = float('inf')
        los_fallback = None
        for px, py in plan:
            dist = math.hypot(px - rx, py - ry)
            if dist <= radius and dist < min_dist:
                if self._line_of_sight(rx, ry, px, py):
                    min_dist = dist
                    los_fallback = (px, py)

        if los_fallback is not None:
            return los_fallback

        # No LOS-valid path point within radius — robot is surrounded by walls or
        # has drifted off the path entirely.  Return None so the caller publishes
        # zero velocity and Nav2's progress checker triggers recovery.
        return None

    def _line_of_sight(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Ray-march through the local costmap to check line-of-sight."""
        with self._lock:
            r = self._raw
            if r.costmap is None:
                return True  # no costmap → assume clear

            grid = r.costmap.reshape(r.costmap_height, r.costmap_width)
            res = r.costmap_resolution
            ox, oy = r.costmap_origin_x, r.costmap_origin_y
            w, h = r.costmap_width, r.costmap_height

        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 0.01:
            return True

        steps = max(int(dist / res), 1)
        for i in range(steps + 1):
            t = i / steps
            wx = x0 + t * (x1 - x0)
            wy = y0 + t * (y1 - y0)
            col = int((wx - ox) / res)
            row = int((wy - oy) / res)
            if 0 <= col < w and 0 <= row < h:
                if grid[row, col] >= 253:  # lethal / inscribed
                    return False
        return True

    # ── Reactive wall escape ─────────────────────────────────────────────

    def _compute_escape_velocity(
        self, scan_ranges: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute a body-frame escape velocity using LiDAR repulsion.

        Sums repulsion vectors from all rays closer than escape_blend_start.
        Each ray contributes a unit vector pointing AWAY from the obstacle,
        weighted by proximity (closer = stronger).

        For a holonomic robot this maps directly to (vx, vy) body commands.
        vyaw is always 0 — rotating against a wall worsens the situation for
        a rectangular footprint; lateral strafe is the right primitive.

        Bilateral deadlock (walls cancel to near-zero net vector, e.g. the
        robot is centred in a corridor) falls back to pure reverse (-vx).
        """
        escape_speed = 0.20   # m/s — slow, deliberate escape
        n = len(scan_ranges)
        fx, fy = 0.0, 0.0
        for i, r in enumerate(scan_ranges):
            if not math.isfinite(r) or r >= self._escape_blend_start:
                continue
            angle = 2.0 * math.pi * i / n        # body-frame ray angle
            weight = (self._escape_blend_start - r) / self._escape_blend_start
            fx -= weight * math.cos(angle)        # repulsion = opposite of ray
            fy -= weight * math.sin(angle)

        mag = math.hypot(fx, fy)
        if mag < 0.05:
            # Bilateral block (walls cancel): rotate toward the less-obstructed
            # side while reversing slowly — breaks corridor symmetry.
            left_sum = sum(
                (self._escape_blend_start - r) / self._escape_blend_start
                for i, r in enumerate(scan_ranges)
                if math.isfinite(r) and r < self._escape_blend_start and i < n // 2
            )
            right_sum = sum(
                (self._escape_blend_start - r) / self._escape_blend_start
                for i, r in enumerate(scan_ranges)
                if math.isfinite(r) and r < self._escape_blend_start and i >= n // 2
            )
            vyaw = -self._max_vyaw * 0.5 if left_sum > right_sum + 0.1 else (
                self._max_vyaw * 0.5 if right_sum > left_sum + 0.1 else self._max_vyaw * 0.5
            )
            return (-escape_speed * 0.5, 0.0, vyaw)

        scale = escape_speed / mag
        vx = float(np.clip(fx * scale, -self._max_vx, self._max_vx))
        vy = float(np.clip(fy * scale, -self._max_vy, self._max_vy))
        return (vx, vy, 0.0)

    # ── Stuck detection & recovery ───────────────────────────────────────

    def _check_and_recover_stuck(self) -> None:
        """Detect if the robot hasn't moved and trigger costmap clear + escape."""
        with self._lock:
            history = list(self._position_history)

        if len(history) < self._stuck_window:
            return

        oldest_x, oldest_y = history[0]
        newest_x, newest_y = history[-1]
        displacement = math.hypot(newest_x - oldest_x, newest_y - oldest_y)

        if displacement < self._stuck_threshold:
            if not self._stuck_warn_logged:
                self.get_logger().warning(
                    f'Robot stuck: moved only {displacement:.3f} m in '
                    f'{self._stuck_window} steps — triggering recovery')
                self._stuck_warn_logged = True

            self._pub_stuck.publish(Bool(data=True))

            # Clear both costmaps to remove stale inflation artifacts
            if self._clear_local_client.service_is_ready():
                self._clear_local_client.call_async(ClearEntireCostmap.Request())
            if self._clear_global_client.service_is_ready():
                self._clear_global_client.call_async(ClearEntireCostmap.Request())

            # Inject an escape carrot ahead of current yaw into the plan
            with self._lock:
                rx, ry, ryaw = (self._raw.robot_x, self._raw.robot_y,
                                self._raw.robot_yaw)
            escape_x = rx + self._stuck_recovery_dist * math.cos(ryaw)
            escape_y = ry + self._stuck_recovery_dist * math.sin(ryaw)
            with self._plan_lock:
                self._plan = [(escape_x, escape_y)]
                self._plan_stamp = time.monotonic()

            # Reset history so recovery doesn't fire every tick
            with self._lock:
                self._position_history.clear()
        else:
            if self._stuck_warn_logged:
                self._pub_stuck.publish(Bool(data=False))
                self._stuck_warn_logged = False

    # ── Inference loop ───────────────────────────────────────────────────

    def _inference_tick(self) -> None:
        """Called at inference_rate Hz.  Runs the policy and publishes cmd_vel."""

        # ── Stuck detection ───────────────────────────────────────────────
        self._check_and_recover_stuck()

        # ── Plan staleness check ──────────────────────────────────────────
        with self._plan_lock:
            plan_exists = self._plan is not None
            plan_age = time.monotonic() - self._plan_stamp

        if plan_exists and plan_age > self._plan_max_age:
            if not self._plan_stale_logged:
                self.get_logger().error(
                    f'Global plan is stale ({plan_age:.1f} s > '
                    f'{self._plan_max_age:.1f} s) — planner may be down')
                self._plan_stale_logged = True
            self._publish_zero()
            return

        # ── 3-zone safety + reactive wall escape ─────────────────────────
        # Zone 3 (emergency, < safety_min_range):  pure escape, skip RL
        # Zone 2 (blend, safety..escape_blend_start): blend escape into RL
        # Zone 1 (normal, >= escape_blend_start):  pure RL (unchanged)
        with self._lock:
            scan_copy = (self._raw.scan_ranges.copy()
                         if self._raw.scan_ranges is not None else None)

        # LiDAR sensor dropout guard: >50% NaN means sensor failure
        if scan_copy is not None:
            nan_fraction = float(np.sum(np.isnan(scan_copy))) / len(scan_copy)
            if nan_fraction > 0.5:
                now = time.monotonic()
                if now - self._scan_nan_warn_time > 10.0:
                    self.get_logger().error(
                        f'LiDAR scan is {nan_fraction:.0%} NaN — sensor failure, '
                        'publishing zero velocity')
                    self._scan_nan_warn_time = now
                self._publish_zero()
                return

        min_range = float(np.nanmin(scan_copy)) if scan_copy is not None else 999.0

        if min_range < self._safety_range:
            # Emergency: compute escape direction from scan, publish directly
            evx, evy, evyaw = self._compute_escape_velocity(scan_copy)
            pvx, pvy, pvyaw = self._prev_cmd
            evx   = float(np.clip(evx,   pvx   - self._max_ax * self._dt,
                                         pvx   + self._max_ax * self._dt))
            evy   = float(np.clip(evy,   pvy   - self._max_ay * self._dt,
                                         pvy   + self._max_ay * self._dt))
            evyaw = float(np.clip(evyaw, pvyaw - self._max_ayaw * self._dt,
                                         pvyaw + self._max_ayaw * self._dt))
            self._prev_cmd = (evx, evy, evyaw)
            cmd = Twist()
            cmd.linear.x = evx
            cmd.linear.y = evy
            cmd.angular.z = evyaw
            self._pub_cmd.publish(cmd)
            return

        # Blend alpha: 0.0 = pure RL (at escape_blend_start), 1.0 = pure escape (at safety_range)
        if min_range < self._escape_blend_start:
            blend_range = self._escape_blend_start - self._safety_range
            alpha = float(np.clip(
                (self._escape_blend_start - min_range) / blend_range, 0.0, 1.0))
        else:
            alpha = 0.0

        # ── Extract local goal from global path ──────────────────────────
        carrot = self._extract_carrot()
        if carrot is None:
            self._publish_zero()
            return

        # ── Build observation + run policy ──────────────────────────────
        # Wrapped in try-except: any obs/ONNX error must not crash the node.
        # A single bad tick publishes zero and logs a warning instead.
        try:
            with self._lock:
                self._raw.goal_x = carrot[0]
                self._raw.goal_y = carrot[1]
                obs = build_observation(self._raw)

            action = self._policy.predict(obs)
            vx, vy, vyaw = scale_action(action)

            # ── Zone 2: blend escape vector into RL output ────────────────
            if alpha > 0.0:
                evx, evy, evyaw = self._compute_escape_velocity(scan_copy)
                vx    = (1.0 - alpha) * vx    + alpha * evx
                vy    = (1.0 - alpha) * vy    + alpha * evy
                vyaw  = (1.0 - alpha) * vyaw  + alpha * evyaw

            # ── Velocity clamping ─────────────────────────────────────────
            vx = np.clip(vx, -self._max_vx, self._max_vx)
            vy = np.clip(vy, -self._max_vy, self._max_vy)
            vyaw = np.clip(vyaw, -self._max_vyaw, self._max_vyaw)

            # ── Acceleration limiting ─────────────────────────────────────
            pvx, pvy, pvyaw = self._prev_cmd
            max_dvx = self._max_ax * self._dt
            max_dvy = self._max_ay * self._dt
            max_dvyaw = self._max_ayaw * self._dt

            vx = np.clip(vx, pvx - max_dvx, pvx + max_dvx)
            vy = np.clip(vy, pvy - max_dvy, pvy + max_dvy)
            vyaw = np.clip(vyaw, pvyaw - max_dvyaw, pvyaw + max_dvyaw)

            self._prev_cmd = (float(vx), float(vy), float(vyaw))

            # ── Publish ───────────────────────────────────────────────────
            cmd = Twist()
            cmd.linear.x = float(vx)
            cmd.linear.y = float(vy)
            cmd.angular.z = float(vyaw)
            self._pub_cmd.publish(cmd)

        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'Inference error (publishing zero): {exc}')
            self._publish_zero()

    def _publish_zero(self) -> None:
        self._pub_cmd.publish(Twist())
        self._prev_cmd = (0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = RLControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
