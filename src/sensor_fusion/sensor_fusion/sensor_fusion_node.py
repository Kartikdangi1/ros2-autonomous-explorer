#!/usr/bin/env python3
"""
Sensor Fusion Preprocess Node
==============================
Layer 2 — Measurement fusion: fuses LiDAR + radar proxy + RGB-D depth into
          a single /fused_scan (sensor_msgs/LaserScan) topic.
Layer 3A — Priority binning: LiDAR > Depth > Radar per angular bin.
Optional — publishes /detected_obstacles_fusion (PoseArray) for near-field
           dynamic obstacles usable by the NBV-SPLAM controller.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSDurabilityPolicy, QoSHistoryPolicy)
from rclpy.duration import Duration
import rclpy.time

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose

import tf2_ros
import numpy as np


class SensorFusionPreprocessNode(Node):

    def __init__(self):
        super().__init__('sensor_fusion')
        self._declare_params()
        self._init_state()
        self._init_ros()
        self.get_logger().info('Sensor fusion preprocess node initialized')

    # ── Parameters ───────────────────────────────────────────────────────────

    def _declare_params(self):
        p = self.declare_parameter
        p('lidar_min',  0.08)
        p('lidar_max', 18.0)
        p('radar_min',  0.5)
        p('radar_max', 20.0)
        p('depth_min',  0.3)
        p('depth_max',  6.0)
        p('depth_height_min', -0.15)   # height filter in output frame (m)
        p('depth_height_max',  1.5)
        p('num_bins',     360)
        p('output_frame', 'lidar_link')
        p('depth_stride', 4)           # sub-sample depth image (speed)
        p('obstacle_publish_range', 5.0)

    def _gp(self, name):
        return self.get_parameter(name).value

    # ── State ────────────────────────────────────────────────────────────────

    def _init_state(self):
        self.latest_radar       = None
        self.latest_depth       = None
        self.latest_camera_info = None

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 4×4 homogeneous TF matrices (fixed joints — cached after first lookup)
        self._T_lidar_to_out  = None   # lidar_link          → output_frame
        self._T_radar_to_out  = None   # radar_link          → output_frame
        self._T_camopt_to_out = None   # camera_optical_link → output_frame

    # ── ROS plumbing ─────────────────────────────────────────────────────────

    def _init_ros(self):
        sq = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(LaserScan,  '/scan',             self._lidar_cb,  sq)
        self.create_subscription(LaserScan,  '/radar/scan',       self._radar_cb,  sq)
        self.create_subscription(Image,      '/rgbd/depth_image', self._depth_cb,  sq)
        self.create_subscription(CameraInfo, '/rgbd/camera_info', self._caminfo_cb, sq)

        self.fused_pub    = self.create_publisher(LaserScan, '/fused_scan',               10)
        self.obstacle_pub = self.create_publisher(PoseArray, '/detected_obstacles_fusion', 10)

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _radar_cb(self, msg):
        self.latest_radar = msg

    def _depth_cb(self, msg):
        self.latest_depth = msg

    def _caminfo_cb(self, msg):
        self.latest_camera_info = msg

    def _lidar_cb(self, msg):
        """Main trigger (LiDAR rate ~20 Hz): fuse all sensors → /fused_scan."""
        self._try_cache_transforms()

        out_frame = self._gp('output_frame')
        num_bins  = self._gp('num_bins')
        a_min     = -np.pi
        a_inc     = 2.0 * np.pi / num_bins

        # Per-sensor range bins (inf = no reading)
        lidar_bins = np.full(num_bins, np.inf, dtype=np.float32)
        depth_bins = np.full(num_bins, np.inf, dtype=np.float32)
        radar_bins = np.full(num_bins, np.inf, dtype=np.float32)

        # ── LiDAR ── (primary, 0.08-18 m, 360°)
        lidar_pts = self._scan_to_pts_2d(
            msg, self._gp('lidar_min'), self._gp('lidar_max'),
            self._T_lidar_to_out)
        self._bin_pts(lidar_pts, lidar_bins, num_bins, a_min, a_inc)

        # ── Radar proxy ── (forward 90°, 0.5-20 m)
        radar_pts = np.empty((0, 2), dtype=np.float32)
        if self.latest_radar is not None and self._T_radar_to_out is not None:
            radar_pts = self._scan_to_pts_2d(
                self.latest_radar,
                self._gp('radar_min'), self._gp('radar_max'),
                self._T_radar_to_out)
            self._bin_pts(radar_pts, radar_bins, num_bins, a_min, a_inc)

        # ── RGB-D depth ── (forward ~80°, 0.3-6 m, with height filter)
        depth_pts_2d = np.empty((0, 2), dtype=np.float32)
        if (self.latest_depth is not None
                and self.latest_camera_info is not None
                and self._T_camopt_to_out is not None):
            depth_pts_2d = self._depth_to_2d(
                self.latest_depth, self.latest_camera_info,
                self._gp('depth_min'), self._gp('depth_max'),
                self._gp('depth_height_min'), self._gp('depth_height_max'),
                self._gp('depth_stride'), self._T_camopt_to_out)
            self._bin_pts(depth_pts_2d, depth_bins, num_bins, a_min, a_inc)

        # ── Layer 3A priority merge: LiDAR > Depth > Radar ──
        fused = np.where(np.isfinite(lidar_bins), lidar_bins,
                np.where(np.isfinite(depth_bins), depth_bins,
                         radar_bins))

        # range_min/max reflect the actual fused sensor envelope, not just the LiDAR.
        fused_r_min = self._gp('lidar_min')
        fused_r_max = max(self._gp('lidar_max'), self._gp('radar_max'), self._gp('depth_max'))
        self.fused_pub.publish(
            self._build_laserscan(msg, fused, out_frame, num_bins, a_min, a_inc,
                                  fused_r_min, fused_r_max))

        # ── Optional: near-field dynamic obstacle publication ──
        self._maybe_publish_obstacles(
            radar_pts, depth_pts_2d, msg.header.stamp, out_frame)

    # ── TF helpers ────────────────────────────────────────────────────────────

    def _try_cache_transforms(self):
        """Look up fixed-joint transforms once; reuse on every subsequent call."""
        out = self._gp('output_frame')
        tmo = Duration(seconds=0.05)
        t0  = rclpy.time.Time()

        if self._T_lidar_to_out is None:
            try:
                ts = self.tf_buffer.lookup_transform(out, 'lidar_link', t0, timeout=tmo)
                self._T_lidar_to_out = self._ts_to_mat(ts)
                self.get_logger().info(
                    f'Cached TF: lidar_link → {out}', once=True)
            except Exception as e:
                self.get_logger().debug(f'TF lidar_link→{out} not ready: {e}')

        if self._T_radar_to_out is None:
            try:
                ts = self.tf_buffer.lookup_transform(out, 'radar_link', t0, timeout=tmo)
                self._T_radar_to_out = self._ts_to_mat(ts)
                self.get_logger().info(f'Cached TF: radar_link → {out}', once=True)
            except Exception as e:
                self.get_logger().debug(f'TF radar_link→{out} not ready: {e}')

        if self._T_camopt_to_out is None:
            try:
                ts = self.tf_buffer.lookup_transform(
                    out, 'camera_optical_link', t0, timeout=tmo)
                self._T_camopt_to_out = self._ts_to_mat(ts)
                self.get_logger().info(
                    f'Cached TF: camera_optical_link → {out}', once=True)
            except Exception as e:
                self.get_logger().debug(f'TF camera_optical_link→{out} not ready: {e}')

    @staticmethod
    def _ts_to_mat(ts):
        """TransformStamped → 4×4 homogeneous numpy matrix."""
        tr = ts.transform.translation
        ro = ts.transform.rotation
        x, y, z, w = ro.x, ro.y, ro.z, ro.w
        R = np.array([
            [1-2*(y*y+z*z),  2*(x*y-z*w),  2*(x*z+y*w)],
            [  2*(x*y+z*w),1-2*(x*x+z*z),  2*(y*z-x*w)],
            [  2*(x*z-y*w),  2*(y*z+x*w),1-2*(x*x+y*y)],
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tr.x, tr.y, tr.z]
        return T

    # ── Sensor → 2D point converters ─────────────────────────────────────────

    def _scan_to_pts_2d(self, msg, r_min, r_max, T):
        """LaserScan → Nx2 floor-plane points in output frame."""
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = (msg.angle_min
                  + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment)
        valid = (np.isfinite(ranges)
                 & (ranges >= max(float(r_min), msg.range_min))
                 & (ranges <= min(float(r_max), msg.range_max)))
        r = ranges[valid]; a = angles[valid]
        if len(r) == 0:
            return np.empty((0, 2), dtype=np.float32)

        # 2D points in sensor frame (z = 0 for flat LiDAR / radar)
        pts = np.column_stack([r * np.cos(a), r * np.sin(a)]).astype(np.float64)

        if T is None:
            return pts.astype(np.float32)

        # Apply 2D slice of the 4×4 transform (sensor is coplanar with floor)
        pts_h = np.hstack([pts, np.zeros((len(pts), 1)), np.ones((len(pts), 1))])
        out   = (T @ pts_h.T).T
        return out[:, :2].astype(np.float32)

    def _depth_to_2d(self, depth_msg, cam_info, d_min, d_max,
                     h_min, h_max, stride, T):
        """
        Depth image → Nx2 floor-plane points in output frame.
        Full 3D transform applied so height filter works in output frame.
        """
        enc = depth_msg.encoding
        if enc == '32FC1':
            raw   = np.frombuffer(bytes(depth_msg.data), dtype=np.float32)
            scale = 1.0
        elif enc in ('16UC1', 'mono16'):
            raw   = np.frombuffer(bytes(depth_msg.data), dtype=np.uint16)
            scale = 0.001
        else:
            return np.empty((0, 2), dtype=np.float32)

        H, W = depth_msg.height, depth_msg.width
        if raw.size != H * W:
            return np.empty((0, 2), dtype=np.float32)

        depth = raw.reshape(H, W).astype(np.float32) * scale

        fx = float(cam_info.k[0]); fy = float(cam_info.k[4])
        cx = float(cam_info.k[2]); cy = float(cam_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return np.empty((0, 2), dtype=np.float32)

        # Stride-based downsampling
        ds = depth[::stride, ::stride]
        rows = np.arange(0, H, stride, dtype=np.float32)
        cols = np.arange(0, W, stride, dtype=np.float32)
        cc, rr = np.meshgrid(cols, rows)

        z = ds.ravel(); u = cc.ravel(); v = rr.ravel()
        valid = np.isfinite(z) & (z >= d_min) & (z <= d_max)
        z = z[valid]; u = u[valid]; v = v[valid]
        if len(z) == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Unproject to 3D in camera_optical_link frame (Z-forward convention)
        xc = (u - cx) * z / fx
        yc = (v - cy) * z / fy
        pts_cam = np.column_stack([xc, yc, z, np.ones(len(z))]).astype(np.float64)

        # Transform to output frame (full 3D)
        out4 = (T @ pts_cam.T).T   # Nx4

        # Height filter in output frame
        z_out = out4[:, 2]
        keep  = (z_out >= h_min) & (z_out <= h_max)
        return out4[keep, :2].astype(np.float32)

    # ── Angular binning ───────────────────────────────────────────────────────

    @staticmethod
    def _bin_pts(pts, range_arr, num_bins, a_min, a_inc):
        """Map Nx2 floor points to angular bins; keep minimum range per bin."""
        if len(pts) == 0:
            return
        a = np.arctan2(pts[:, 1].astype(np.float64),
                       pts[:, 0].astype(np.float64))
        r = np.hypot(pts[:, 0], pts[:, 1]).astype(np.float32)
        idx  = np.floor((a - a_min) / a_inc).astype(np.int32)
        mask = (idx >= 0) & (idx < num_bins) & (r > 0.0)
        np.minimum.at(range_arr, idx[mask], r[mask])

    # ── Output LaserScan builder ──────────────────────────────────────────────

    @staticmethod
    def _build_laserscan(ref, ranges, frame_id, num_bins, a_min, a_inc,
                         r_min=0.08, r_max=20.0):
        out = LaserScan()
        out.header.stamp    = ref.header.stamp
        out.header.frame_id = frame_id
        out.angle_min       = float(a_min)
        out.angle_max       = float(a_min + a_inc * (num_bins - 1))
        out.angle_increment = float(a_inc)
        out.time_increment  = 0.0
        out.scan_time       = ref.scan_time
        out.range_min       = float(r_min)
        out.range_max       = float(r_max)
        # ROS convention: inf = no return (enables raytrace clearing in nav2)
        out.ranges      = np.where(np.isfinite(ranges), ranges, float('inf')).tolist()
        out.intensities = []
        return out

    # ── Dynamic obstacle helper ───────────────────────────────────────────────

    def _maybe_publish_obstacles(self, radar_pts, depth_pts, stamp, frame_id):
        obs_range = self._gp('obstacle_publish_range')
        near = []
        for pts in (radar_pts, depth_pts):
            if len(pts) > 0:
                d = np.hypot(pts[:, 0], pts[:, 1])
                near.append(pts[d < obs_range])
        non_empty = [p for p in near if len(p) > 0]
        if not non_empty:
            return
        all_near = np.vstack(non_empty)
        if len(all_near) == 0:
            return

        # Coarse voxel clustering (0.3 m grid)
        vox      = 0.3
        quant    = np.unique(np.floor(all_near / vox).astype(np.int32), axis=0)
        centroids = (quant.astype(np.float32) + 0.5) * vox

        pa = PoseArray()
        pa.header.stamp    = stamp
        pa.header.frame_id = frame_id
        for x, y in centroids:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.obstacle_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionPreprocessNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        # RuntimeError is thrown by the DDS layer on Ctrl-C when a subscription
        # callback is interrupted mid-deserialization (ROS 2 Humble race condition).
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
