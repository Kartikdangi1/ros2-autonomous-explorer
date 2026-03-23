"""
localization.py
===============
PoseProvider — looks up the robot pose and sensor-frame transform from TF2.

Separating this from the planner keeps TF2 queries in one place so they can
be tuned (timeout, fallback policy) without touching NBV planning code.

Usage
-----
from autonomous_explorer.localization import PoseProvider

provider = PoseProvider(node)
# In planning code:
pose = provider.get_robot_pose()            # np.ndarray [x, y, yaw] or None
sx, sy, syaw = provider.get_scan_pose(scan) # (float, float, float) or None
"""

import math

import numpy as np
import rclpy
import tf2_ros
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


class PoseProvider:
    """
    Thin wrapper around TF2 that returns robot and sensor poses as numpy values.

    Key behaviours:
    - Returns None (never raises) when a transform is temporarily unavailable.
    - Falls back to the robot (base_link) pose when the scan's own frame TF is
      momentarily unavailable — avoids dropping valid scans on TF hiccups.
    """

    def __init__(self, node, map_frame: str = 'map', base_frame: str = 'base_link'):
        self._node = node
        self._map_frame = map_frame
        self._base_frame = base_frame
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def tf_buffer(self) -> Buffer:
        """Direct access to the TF buffer (e.g. for passing to tf2_geometry_msgs)."""
        return self._tf_buffer

    def get_robot_pose(self) -> np.ndarray | None:
        """
        Return [x, y, yaw] of base_link in the map frame.

        Returns None when the TF is not yet available (e.g. at startup).
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            return self._tf_to_xyyaw(tf)
        except Exception as e:
            self._node.get_logger().warn(f'TF lookup failed: {e}')
            return None

    def get_scan_pose(self, scan: LaserScan) -> tuple[float, float, float] | None:
        """
        Return (x, y, yaw) of the scan's sensor frame in the map frame at the
        scan's timestamp.

        Falls back to the current robot pose when the exact-timestamp TF is
        unavailable (e.g. TF buffer too short or scan timestamp slightly stale).
        Returns None only when even the fallback lookup fails.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame,
                scan.header.frame_id,
                scan.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1))
            x, y, yaw = self._tf_to_xyyaw(tf)
            return x, y, yaw
        except Exception:
            pass

        # Fallback: use the robot pose (base_link → map, latest available)
        pose = self.get_robot_pose()
        if pose is not None:
            return float(pose[0]), float(pose[1]), float(pose[2])
        return None

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _tf_to_xyyaw(tf) -> np.ndarray:
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return np.array([x, y, yaw])
