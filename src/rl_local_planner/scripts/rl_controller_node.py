#!/usr/bin/env python3
"""ROS2 RL controller node — drop-in replacement for DWB local planner.

Subscribes to the global path from Nav2's planner_server plus sensor
data, runs ONNX policy inference at 10 Hz, and publishes cmd_vel.

Safety layer:
  - Emergency stop when min LiDAR range < safety_min_range
  - Velocity clamping to configured limits
  - Acceleration limiting (smooth output between steps)

Graceful degradation:
  - If ONNX model is missing / fails to load → zero velocity (no crash)
  - If no /plan available → zero velocity
"""

from __future__ import annotations

import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan

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
        self.declare_parameter('max_vel_x', 0.5)
        self.declare_parameter('max_vel_y', 0.5)
        self.declare_parameter('max_vel_theta', 1.0)
        self.declare_parameter('safety_min_range', 0.18)
        self.declare_parameter('goal_tolerance', 0.4)
        self.declare_parameter('costmap_size', 84)
        self.declare_parameter('max_accel_x', 2.5)
        self.declare_parameter('max_accel_y', 2.5)
        self.declare_parameter('max_accel_theta', 3.2)

        model_path = self.get_parameter('model_path').value
        rate = self.get_parameter('inference_rate').value
        self._carrot_radius = self.get_parameter('carrot_radius').value
        self._max_vx = self.get_parameter('max_vel_x').value
        self._max_vy = self.get_parameter('max_vel_y').value
        self._max_vyaw = self.get_parameter('max_vel_theta').value
        self._safety_range = self.get_parameter('safety_min_range').value
        self._goal_tol = self.get_parameter('goal_tolerance').value
        self._max_ax = self.get_parameter('max_accel_x').value
        self._max_ay = self.get_parameter('max_accel_y').value
        self._max_ayaw = self.get_parameter('max_accel_theta').value
        self._dt = 1.0 / rate

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
        self._prev_cmd = (0.0, 0.0, 0.0)  # for acceleration limiting

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

        # ── Publisher ────────────────────────────────────────────────────
        self._pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

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

    def _plan_cb(self, msg: Path) -> None:
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._plan_lock:
            self._plan = points

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

        radius = self._carrot_radius
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

        # Fallback: closest path point within radius (no LOS check)
        min_dist = float('inf')
        closest = None
        for px, py in plan:
            dist = math.hypot(px - rx, py - ry)
            if dist < min_dist:
                min_dist = dist
                closest = (px, py)

        if closest is not None:
            return closest

        # Final fallback: end of path
        return plan[-1] if plan else None

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

    # ── Inference loop ───────────────────────────────────────────────────

    def _inference_tick(self) -> None:
        """Called at inference_rate Hz.  Runs the policy and publishes cmd_vel."""

        # ── Safety check ─────────────────────────────────────────────────
        with self._lock:
            min_range = float(np.nanmin(self._raw.scan_ranges)) \
                if self._raw.scan_ranges is not None else 999.0

        if min_range < self._safety_range:
            self._publish_zero()
            return

        # ── Extract local goal from global path ──────────────────────────
        carrot = self._extract_carrot()
        if carrot is None:
            self._publish_zero()
            return

        # ── Build observation ────────────────────────────────────────────
        with self._lock:
            self._raw.goal_x = carrot[0]
            self._raw.goal_y = carrot[1]
            obs = build_observation(self._raw)

        # ── Run policy ───────────────────────────────────────────────────
        action = self._policy.predict(obs)
        vx, vy, vyaw = scale_action(action)

        # ── Velocity clamping ────────────────────────────────────────────
        vx = np.clip(vx, -self._max_vx, self._max_vx)
        vy = np.clip(vy, -self._max_vy, self._max_vy)
        vyaw = np.clip(vyaw, -self._max_vyaw, self._max_vyaw)

        # ── Acceleration limiting ────────────────────────────────────────
        pvx, pvy, pvyaw = self._prev_cmd
        max_dvx = self._max_ax * self._dt
        max_dvy = self._max_ay * self._dt
        max_dvyaw = self._max_ayaw * self._dt

        vx = np.clip(vx, pvx - max_dvx, pvx + max_dvx)
        vy = np.clip(vy, pvy - max_dvy, pvy + max_dvy)
        vyaw = np.clip(vyaw, pvyaw - max_dvyaw, pvyaw + max_dvyaw)

        self._prev_cmd = (float(vx), float(vy), float(vyaw))

        # ── Publish ──────────────────────────────────────────────────────
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(vyaw)
        self._pub_cmd.publish(cmd)

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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
