#!/usr/bin/env python3
"""
NBV Goal Provider Node — Nav2 Mission Controller
=================================================
Responsibilities:
  1. Subscribe to /map (OccupancyGrid from SLAM Toolbox) and maintain an
     OccupancyMapper via OccupancyMapManager.
  2. Subscribe to /fused_scan (LaserScan) for frontier extraction.
  3. Use PoseProvider (TF2 wrapper) to obtain the robot pose in map frame.
  4. Compute the next-best-view (NBV) candidate.
  5. Send a NavigateToPose action goal to the Nav2 BT Navigator.
  6. On success, compute and send the next NBV goal (continuous exploration).
  7. On failure, blacklist the failed goal and try the next candidate.

Modular design:
  - OccupancyMapManager  (my_awesome_robot.mapping)    — map sync + expansion
  - PoseProvider         (my_awesome_robot.localization) — TF2 pose lookups
  - NBV algorithms       (my_awesome_robot.nbv_utils)   — OutlineExtractor etc.
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                       QoSProfile, QoSReliabilityPolicy)

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point as GPoint, PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from my_awesome_robot.mapping import OccupancyMapManager
from my_awesome_robot.localization import PoseProvider
from my_awesome_robot.nbv_utils import (
    OutlineExtractor,
    CandidateGenerator,
    NBVScorer,
)

# QoS profile for the SLAM Toolbox /map topic (transient-local durability)
MAP_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# How long a blacklisted goal stays blacklisted (seconds)
BLACKLIST_TTL = 30.0
# Minimum distance (m) between goals before the new one is considered distinct
CANDIDATE_MIN_DIST = 1.5
# Wait after an action failure before retrying (seconds)
RETRY_DELAY = 2.0
# Minimum visibility score required to publish a goal
MIN_VISIBILITY = 0.02


class NBVGoalProviderNode(Node):
    """Mission controller: computes NBV goals and sends NavigateToPose actions."""

    def __init__(self):
        super().__init__('nbv_goal_provider')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        # OutlineExtractor
        self.declare_parameter('num_sectors', 72)
        self.declare_parameter('jump_threshold', 1.0)
        self.declare_parameter('max_range', 19.0)
        # CandidateGenerator
        self.declare_parameter('candidate_offset', 0.5)
        self.declare_parameter('sample_spacing', 0.5)
        self.declare_parameter('exploration_radius', 15.0)
        # NBVScorer
        self.declare_parameter('weight_visibility', 3.0)
        self.declare_parameter('weight_distance', 1.0)
        self.declare_parameter('weight_orientation', 0.5)
        self.declare_parameter('num_rays', 72)
        # Goal
        self.declare_parameter('goal_tolerance', 0.8)
        self.declare_parameter('min_visibility_threshold', MIN_VISIBILITY)

        map_frame  = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value

        # ── Modules ───────────────────────────────────────────────────────────
        self._map_manager = OccupancyMapManager(self.get_logger())
        self._pose_provider = PoseProvider(self, map_frame=map_frame,
                                           base_frame=base_frame)

        self._outline_extractor = OutlineExtractor(
            num_sectors=self.get_parameter('num_sectors').value,
            jump_threshold=self.get_parameter('jump_threshold').value,
            max_range=self.get_parameter('max_range').value,
        )
        self._candidate_generator = CandidateGenerator(
            offset_distance=self.get_parameter('candidate_offset').value,
            sample_spacing=self.get_parameter('sample_spacing').value,
            exploration_radius=self.get_parameter('exploration_radius').value,
        )
        self._scorer = NBVScorer(
            num_rays=self.get_parameter('num_rays').value,
            weight_visibility=self.get_parameter('weight_visibility').value,
            weight_distance=self.get_parameter('weight_distance').value,
            weight_orientation=self.get_parameter('weight_orientation').value,
            exploration_radius=self.get_parameter('exploration_radius').value,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._latest_scan: LaserScan | None = None
        self._scan_lock = threading.Lock()
        self._blacklist: list[tuple[np.ndarray, float]] = []  # (pos, expiry)
        self._current_goal: np.ndarray | None = None
        self._nav_active = False
        self._nav_goal_handle = None
        # Latest computed outline (for continuous viz timer)
        self._latest_outline = None
        self._latest_outline_stamp = None
        self._outline_lock = threading.Lock()

        self._map_frame = map_frame

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, MAP_QOS)

        self._scan_sub = self.create_subscription(
            LaserScan, '/fused_scan', self._scan_callback, 10)

        # Publishers: visualisation only
        self._candidate_pub = self.create_publisher(MarkerArray, '/nbv_candidates', 5)
        self._outline_pub   = self.create_publisher(Marker,      '/outline_polygon', 5)
        self._jump_pub      = self.create_publisher(MarkerArray, '/jump_edges', 5)
        self._goal_pub      = self.create_publisher(PoseStamped, '/nbv_goal', 5)

        # Nav2 NavigateToPose action client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # NBV planning tick — only fires when Nav2 is idle
        self._nbv_timer = self.create_timer(5.0, self._nbv_tick)

        # Continuous visualisation timer — updates outline/jump_edges at 2 Hz
        # regardless of whether navigation is active, so RViz always shows fresh
        # frontier data (fixes the "outline polygon never changes" bug).
        self._viz_timer = self.create_timer(0.5, self._viz_tick)

        self.get_logger().info('NBVGoalProviderNode initialised — waiting for /map and Nav2')

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _map_callback(self, msg: OccupancyGrid):
        self._map_manager.update_from_slam(msg)

    def _scan_callback(self, msg: LaserScan):
        with self._scan_lock:
            self._latest_scan = msg

    # =========================================================================
    # Periodic timers
    # =========================================================================

    def _nbv_tick(self):
        """Send the next NBV goal if Nav2 is idle."""
        if self._nav_active:
            return
        if not self._nav_client.server_is_ready():
            self.get_logger().info('Waiting for navigate_to_pose action server …')
            return
        self._compute_and_send_nbv()

    def _viz_tick(self):
        """Recompute and publish outline + jump edges at 2 Hz (always-on)."""
        pose = self._pose_provider.get_robot_pose()
        if pose is None:
            return

        with self._scan_lock:
            scan = self._latest_scan
        if scan is None:
            return

        scan_pose = self._pose_provider.get_scan_pose(scan)
        if scan_pose is None:
            return
        scan_x, scan_y, scan_yaw = scan_pose

        points_2d = self._scan_to_points(scan, scan_x, scan_y, scan_yaw)
        outline = self._outline_extractor.extract(points_2d, pose)
        if outline is None:
            return

        stamp = self.get_clock().now().to_msg()
        with self._outline_lock:
            self._latest_outline       = outline
            self._latest_outline_stamp = stamp

        self._publish_outline(outline, stamp)

    # =========================================================================
    # Core NBV computation
    # =========================================================================

    def _compute_and_send_nbv(self):
        pose = self._pose_provider.get_robot_pose()
        if pose is None:
            return

        with self._scan_lock:
            scan = self._latest_scan
        if scan is None:
            self.get_logger().debug('No scan available yet')
            return

        mapper = self._map_manager.mapper
        if mapper is None:
            self.get_logger().debug('OccupancyMapper not initialised yet')
            return

        scan_pose = self._pose_provider.get_scan_pose(scan)
        if scan_pose is None:
            return
        scan_x, scan_y, scan_yaw = scan_pose

        points_2d = self._scan_to_points(scan, scan_x, scan_y, scan_yaw)

        # ── Frontier extraction ───────────────────────────────────────────────
        with self._outline_lock:
            outline = self._latest_outline  # reuse the viz-timer result if fresh
        if outline is None:
            outline = self._outline_extractor.extract(points_2d, pose)
        if outline is None or len(outline.jump_edges) == 0:
            self.get_logger().debug('No frontiers found')
            return

        # ── Prune expired blacklist entries ───────────────────────────────────
        now = time.monotonic()
        self._blacklist = [(p, t) for p, t in self._blacklist if t > now]
        blacklist_positions = [p for p, _ in self._blacklist]

        # ── Candidate generation + scoring ────────────────────────────────────
        candidates = self._candidate_generator.generate(outline, pose, mapper, blacklist_positions)
        if not candidates:
            self.get_logger().info('No valid candidates — exploration may be complete')
            return

        scored = self._scorer.score_candidates(candidates, pose, outline, mapper)
        if not scored:
            return

        # ── Pick the best candidate far enough from the current goal ──────────
        min_vis = self.get_parameter('min_visibility_threshold').value
        best = None
        for sc in scored:
            if sc.visibility_score < min_vis:
                continue
            pos = sc.candidate.position
            if self._current_goal is not None:
                if np.linalg.norm(pos - self._current_goal) < CANDIDATE_MIN_DIST:
                    continue
            best = sc
            break

        if best is None:
            self.get_logger().info(
                'All candidates below visibility threshold — exploration may be complete')
            return

        self._publish_candidates(scored, self.get_clock().now().to_msg())

        # ── Send NavigateToPose action goal ───────────────────────────────────
        target = best.candidate.position
        yaw    = best.candidate.orientation
        self._current_goal = target
        self._send_nav_goal(target[0], target[1], yaw)

    # =========================================================================
    # Scan conversion helper
    # =========================================================================

    def _scan_to_points(self, scan: LaserScan,
                        scan_x: float, scan_y: float, scan_yaw: float) -> np.ndarray:
        """Convert a LaserScan into a (N,2) array of map-frame points."""
        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        valid  = (np.isfinite(ranges)
                  & (ranges > scan.range_min)
                  & (ranges < scan.range_max))
        r, a   = ranges[valid], angles[valid]
        cos_yaw, sin_yaw = np.cos(scan_yaw), np.sin(scan_yaw)
        px = r * np.cos(a)
        py = r * np.sin(a)
        return np.column_stack([
            scan_x + cos_yaw * px - sin_yaw * py,
            scan_y + sin_yaw * px + cos_yaw * py,
        ])

    # =========================================================================
    # Nav2 action callbacks
    # =========================================================================

    def _send_nav_goal(self, x: float, y: float, yaw: float):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self._map_frame
        goal_msg.pose.header.stamp    = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._goal_pub.publish(goal_msg.pose)
        self.get_logger().info(
            f'Sending NBV goal: ({x:.2f}, {y:.2f}) yaw={math.degrees(yaw):.1f}°')

        self._nav_active = True
        send_future = self._nav_client.send_goal_async(
            goal_msg, feedback_callback=self._nav_feedback_cb)
        send_future.add_done_callback(self._nav_goal_response_cb)

    def _nav_goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal — will retry')
            self._nav_active = False
            if self._current_goal is not None:
                self._blacklist.append(
                    (self._current_goal, time.monotonic() + RETRY_DELAY * 3))
            return
        self._nav_goal_handle = handle
        self.get_logger().debug('Goal accepted by Nav2 BT Navigator')
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._nav_result_cb)

    def _nav_feedback_cb(self, feedback_msg):
        pass

    def _nav_result_cb(self, future):
        result = future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('NBV goal reached — computing next goal')
        else:
            self.get_logger().warn(
                f'Navigation failed (status={status}) — blacklisting goal and retrying')
            if self._current_goal is not None:
                self._blacklist.append(
                    (self._current_goal, time.monotonic() + BLACKLIST_TTL))
        self._nav_active   = False
        self._current_goal = None

    # =========================================================================
    # Visualisation helpers
    # =========================================================================

    def _publish_outline(self, outline, stamp):
        if len(outline.vertices) < 2:
            return
        m = Marker()
        m.header.frame_id = self._map_frame
        m.header.stamp    = stamp
        m.ns     = 'outline'
        m.id     = 0
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03
        m.color   = ColorRGBA(r=0.2, g=0.8, b=0.2, a=0.8)
        for v in outline.vertices:
            p = GPoint(); p.x, p.y = float(v[0]), float(v[1])
            m.points.append(p)
        self._outline_pub.publish(m)

        # Jump edges
        markers = MarkerArray()
        for i, je in enumerate(outline.jump_edges):
            jm = Marker()
            jm.header.frame_id = self._map_frame
            jm.header.stamp    = stamp
            jm.ns     = 'jump_edges'
            jm.id     = i
            jm.type   = Marker.ARROW
            jm.action = Marker.ADD
            jm.scale.x = 0.05
            jm.scale.y = 0.10
            jm.color   = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
            ps = GPoint(); ps.x, ps.y = float(je.start[0]), float(je.start[1])
            pe = GPoint(); pe.x, pe.y = float(je.end[0]),   float(je.end[1])
            jm.points = [ps, pe]
            markers.markers.append(jm)
        self._jump_pub.publish(markers)

    def _publish_candidates(self, scored, stamp):
        markers = MarkerArray()
        for i, sc in enumerate(scored[:20]):
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp    = stamp
            m.ns     = 'nbv_candidates'
            m.id     = i
            m.type   = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.4
            m.scale.y = 0.08
            m.scale.z = 0.08
            intensity = float(np.clip(sc.total_score / 5.0, 0.0, 1.0))
            m.color   = ColorRGBA(r=1.0 - intensity, g=intensity, b=0.0, a=0.8)
            m.pose.position.x = float(sc.candidate.position[0])
            m.pose.position.y = float(sc.candidate.position[1])
            yaw = sc.candidate.orientation
            m.pose.orientation.z = math.sin(yaw / 2.0)
            m.pose.orientation.w = math.cos(yaw / 2.0)
            markers.markers.append(m)
        self._candidate_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = NBVGoalProviderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
