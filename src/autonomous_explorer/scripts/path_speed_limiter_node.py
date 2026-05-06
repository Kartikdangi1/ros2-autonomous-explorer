#!/usr/bin/env python3
"""
path_speed_limiter_node.py — Curvature-based DWB speed limiter
==============================================================
Subscribes to the global path (/plan) and the robot's odometry, measures the
total heading change over a 1 m lookahead window ahead of the robot, and
publishes a nav2_msgs/SpeedLimit message.  Nav2's controller_server scales
DWB's output velocities by this limit automatically (speed_limit_topic param).

Result: full speed on straight corridors, automatic deceleration into turns.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry, Path
from nav2_msgs.msg import SpeedLimit

from autonomous_explorer.nbv_utils import _normalize_angle

LOOKAHEAD_M = 1.0    # arc-length window to measure curvature (metres)
PUBLISH_HZ  = 10.0   # speed-limit publish rate

# (curvature_threshold_rad_per_m, speed_limit_mps)
# speed_limit=0.0 means "no limit" to controller_server
SPEED_TABLE = [
    (0.3, 0.0),   # straight corridor — full speed (0.5 m/s max)
    (0.7, 0.35),  # gentle curve — ~70 % of max
    (9e9, 0.20),  # sharp turn  — ~40 % of max
]


class PathSpeedLimiterNode(Node):

    def __init__(self):
        super().__init__('path_speed_limiter')

        qos_best = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1)

        self._plan_poses = []   # list[PoseStamped]
        self._robot_x    = None
        self._robot_y    = None

        self.create_subscription(Path,    '/plan',               self._plan_cb, 1)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, qos_best)

        self._pub = self.create_publisher(SpeedLimit, '/speed_limit', 1)
        self.create_timer(1.0 / PUBLISH_HZ, self._timer_cb)

        self.get_logger().info('path_speed_limiter started')

    # ── callbacks ────────────────────────────────────────────────────────────

    def _plan_cb(self, msg: Path):
        self._plan_poses = msg.poses

    def _odom_cb(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    # ── main tick ────────────────────────────────────────────────────────────

    def _timer_cb(self):
        try:
            limit = self._compute_speed_limit()
        except Exception as exc:
            self.get_logger().warn(f'speed limit compute error: {exc}', throttle_duration_sec=5.0)
            limit = 0.0
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage    = False
        msg.speed_limit   = limit
        self._pub.publish(msg)

    def _compute_speed_limit(self) -> float:
        if not self._plan_poses or self._robot_x is None:
            return 0.0  # no plan yet — no limit

        poses = self._plan_poses
        rx, ry = self._robot_x, self._robot_y

        # Find path index closest to robot
        min_d2   = float('inf')
        start_idx = 0
        for i, ps in enumerate(poses):
            dx = ps.pose.position.x - rx
            dy = ps.pose.position.y - ry
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2    = d2
                start_idx = i

        # Walk forward LOOKAHEAD_M and accumulate absolute heading change
        total_dheading = 0.0
        dist_walked    = 0.0
        prev_yaw       = None

        for i in range(start_idx, len(poses) - 1):
            x0 = poses[i].pose.position.x
            y0 = poses[i].pose.position.y
            x1 = poses[i + 1].pose.position.x
            y1 = poses[i + 1].pose.position.y

            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-4:
                continue

            yaw = math.atan2(y1 - y0, x1 - x0)
            if prev_yaw is not None:
                total_dheading += abs(_normalize_angle(yaw - prev_yaw))
            prev_yaw = yaw

            dist_walked += seg
            if dist_walked >= LOOKAHEAD_M:
                break

        if dist_walked < 0.05:
            return 0.0  # too close to end of path — no limit

        curvature = total_dheading / dist_walked  # rad/m

        for threshold, speed_limit in SPEED_TABLE:
            if curvature < threshold:
                return speed_limit

        return SPEED_TABLE[-1][1]


def main():
    rclpy.init()
    node = PathSpeedLimiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
