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

New additions:
  A3: NBV tick timer reduced to 2 s for faster re-planning.
  A4: Look-ahead prefetch — next goal computed in background while navigating.
  B2: CoverageTracker — publishes /exploration_coverage, adaptive radius.
  B3: Uncertainty-aware scoring — subscribes to /odometry/filtered for EKF
      pose covariance and passes it to NBVScorer to scale weights dynamically.

Modular design:
  - OccupancyMapManager  (my_awesome_robot.mapping)    — map sync + expansion
  - PoseProvider         (my_awesome_robot.localization) — TF2 pose lookups
  - NBV algorithms       (my_awesome_robot.nbv_utils)   — OutlineExtractor etc.
"""

import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                       QoSProfile, QoSReliabilityPolicy)

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point as GPoint, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray

from my_awesome_robot.mapping import OccupancyMapManager
from my_awesome_robot.localization import PoseProvider
from my_awesome_robot.nbv_utils import (
    OutlineExtractor,
    CandidateGenerator,
    NBVScorer,
    GoalValidator,
    TopologicalMap,
    CoveragePlanner,
    CoverageTracker,
    ScoredCandidate,
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
        self.declare_parameter('use_entropy_scoring', False)  # Ext 2: feature flag
        self.declare_parameter('weight_frontier_size', 1.5)   # B1: frontier-size bonus
        # Goal
        self.declare_parameter('goal_tolerance', 0.8)
        self.declare_parameter('min_visibility_threshold', MIN_VISIBILITY)
        # Extension 1: Goal pre-validation
        self.declare_parameter('pre_validate_goals', True)
        self.declare_parameter('validator_robot_radius', 0.3)
        self.declare_parameter('validator_max_cells', 40000)
        # Extension 4: Topological map
        self.declare_parameter('enable_topo_map', True)
        self.declare_parameter('topo_node_spacing', 1.5)
        self.declare_parameter('topo_visited_radius', 2.0)
        self.declare_parameter('topo_penalty_weight', 0.8)
        # Extension 5: Coverage fallback
        self.declare_parameter('enable_coverage_fallback', True)
        self.declare_parameter('coverage_stagnation_ticks', 3)
        # B3: uncertainty-aware scoring
        self.declare_parameter('enable_uncertainty_scoring', True)

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
            use_entropy=self.get_parameter('use_entropy_scoring').value,
            weight_frontier_size=self.get_parameter('weight_frontier_size').value,
        )

        # Extension 1: goal pre-validator
        self._validator = GoalValidator(
            robot_radius=self.get_parameter('validator_robot_radius').value,
            max_search_cells=self.get_parameter('validator_max_cells').value,
        )

        # Extension 4: topological map (None if disabled)
        self._topo_map: Optional[TopologicalMap] = (
            TopologicalMap(
                node_spacing=self.get_parameter('topo_node_spacing').value,
                visited_radius=self.get_parameter('topo_visited_radius').value,
            ) if self.get_parameter('enable_topo_map').value else None
        )

        # Extension 5: coverage planner + stagnation counter
        self._coverage_planner = CoveragePlanner(row_spacing=1.0)
        self._zero_frontier_ticks: int = 0

        # B2: coverage tracker
        self._coverage_tracker = CoverageTracker()
        self._current_coverage: float = 0.0

        # B3: EKF pose uncertainty (covariance trace from /odometry/filtered)
        self._pose_uncertainty: float = 0.0
        self._uncertainty_lock = threading.Lock()

        # ── State ─────────────────────────────────────────────────────────────
        self._latest_scan: Optional[LaserScan] = None
        self._scan_lock = threading.Lock()
        self._blacklist: list = []  # list of (pos: np.ndarray, expiry: float)
        self._current_goal: Optional[np.ndarray] = None
        self._nav_active = False
        self._nav_goal_handle = None
        # Latest computed outline (for continuous viz timer)
        self._latest_outline = None
        self._latest_outline_stamp = None
        self._outline_lock = threading.Lock()

        self._map_frame = map_frame

        # A4: look-ahead prefetch state
        self._prefetched_goal: Optional[ScoredCandidate] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, MAP_QOS)

        self._scan_sub = self.create_subscription(
            LaserScan, '/fused_scan', self._scan_callback, 10)

        # B3: EKF odometry for pose uncertainty
        self._odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_callback, 10)

        # Publishers: visualisation only
        self._candidate_pub = self.create_publisher(MarkerArray, '/nbv_candidates', 5)
        self._outline_pub   = self.create_publisher(Marker,      '/outline_polygon', 5)
        self._jump_pub      = self.create_publisher(MarkerArray, '/jump_edges', 5)
        self._goal_pub      = self.create_publisher(PoseStamped, '/nbv_goal', 5)
        self._topo_pub      = self.create_publisher(MarkerArray, '/topo_graph', 5)  # Ext 4
        # B2: coverage metric
        self._coverage_pub  = self.create_publisher(Float32, '/exploration_coverage', 5)

        # Nav2 NavigateToPose action client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # A3: NBV planning tick reduced from 5 s to 2 s — fires when Nav2 is idle
        self._nbv_timer = self.create_timer(2.0, self._nbv_tick)

        # Continuous visualisation timer — updates outline/jump_edges at 2 Hz
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

    def _odom_callback(self, msg: Odometry):
        """B3: Extract position uncertainty from EKF covariance."""
        if not self.get_parameter('enable_uncertainty_scoring').value:
            return
        # pose.covariance is a flat 6×6 = 36-element array (row-major)
        # [0] = var(x), [7] = var(y)
        cov = msg.pose.covariance
        uncertainty = float(cov[0] + cov[7])  # position covariance trace (m²)
        with self._uncertainty_lock:
            self._pose_uncertainty = max(0.0, uncertainty)

    # =========================================================================
    # Periodic timers
    # =========================================================================

    def _nbv_tick(self):
        """A3: Send the next NBV goal if Nav2 is idle (fires every 2 s)."""
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

        mapper = self._map_manager.mapper
        points_2d = self._scan_to_points(scan, scan_x, scan_y, scan_yaw)
        outline = self._outline_extractor.extract(points_2d, pose, mapper=mapper)
        if outline is None:
            return

        stamp = self.get_clock().now().to_msg()
        with self._outline_lock:
            self._latest_outline       = outline
            self._latest_outline_stamp = stamp

        self._publish_outline(outline, stamp)
        if self._topo_map is not None:
            self._publish_topo_graph(stamp)

    # =========================================================================
    # Core NBV computation
    # =========================================================================

    def _compute_and_send_nbv(self):
        """Check for a prefetched goal first, otherwise compute synchronously."""
        # A4: use look-ahead result if available and still valid
        with self._prefetch_lock:
            prefetched = self._prefetched_goal
            self._prefetched_goal = None

        if prefetched is not None:
            self.get_logger().debug('Using prefetched NBV goal')
            self._send_scored_candidate(prefetched)
            return

        best = self._compute_best_candidate()
        if best is not None:
            self._send_scored_candidate(best)

    def _compute_best_candidate(self) -> Optional[ScoredCandidate]:
        """Core NBV planning: compute and return the best scored candidate."""
        pose = self._pose_provider.get_robot_pose()
        if pose is None:
            return None

        with self._scan_lock:
            scan = self._latest_scan
        if scan is None:
            self.get_logger().debug('No scan available yet')
            return None

        mapper = self._map_manager.mapper
        if mapper is None:
            self.get_logger().debug('OccupancyMapper not initialised yet')
            return None

        scan_pose = self._pose_provider.get_scan_pose(scan)
        if scan_pose is None:
            return None
        scan_x, scan_y, scan_yaw = scan_pose

        points_2d = self._scan_to_points(scan, scan_x, scan_y, scan_yaw)

        # Extension 4: update topological map with current robot position
        if self._topo_map is not None:
            self._topo_map.update(pose, mapper)

        # ── B2: coverage tracking + adaptive exploration radius ───────────────
        self._current_coverage = CoverageTracker.compute_coverage(mapper)
        cov_msg = Float32()
        cov_msg.data = float(self._current_coverage)
        self._coverage_pub.publish(cov_msg)

        adaptive_radius = CoverageTracker.adaptive_radius(
            self._current_coverage,
            base_radius=self.get_parameter('exploration_radius').value,
        )
        if adaptive_radius != self._scorer.exploration_radius:
            self._scorer.exploration_radius = adaptive_radius
            self._candidate_generator.exploration_radius = adaptive_radius
            self.get_logger().info(
                f'Coverage {self._current_coverage:.1%} → exploration_radius = {adaptive_radius:.1f} m')

        # ── Frontier extraction ───────────────────────────────────────────────
        with self._outline_lock:
            outline = self._latest_outline  # reuse the viz-timer result if fresh
        if outline is None:
            outline = self._outline_extractor.extract(points_2d, pose, mapper=mapper)

        # Extension 5: coverage fallback when stuck with no frontiers
        if outline is None or len(outline.jump_edges) == 0:
            self._zero_frontier_ticks += 1
            stagnation = self.get_parameter('coverage_stagnation_ticks').value
            if (self.get_parameter('enable_coverage_fallback').value
                    and self._zero_frontier_ticks >= stagnation):
                self._run_coverage_fallback(pose, mapper)
            else:
                self.get_logger().debug(
                    f'No frontiers found (tick {self._zero_frontier_ticks}/{stagnation})')
            return None
        self._zero_frontier_ticks = 0

        # ── Prune expired blacklist entries ───────────────────────────────────
        now = time.monotonic()
        self._blacklist = [(p, t) for p, t in self._blacklist if t > now]
        blacklist_positions = [p for p, _ in self._blacklist]

        # ── Candidate generation + scoring ────────────────────────────────────
        candidates = self._candidate_generator.generate(
            outline, pose, mapper, blacklist_positions)
        if not candidates:
            self.get_logger().info('No valid candidates — exploration may be complete')
            return None

        topo_penalty = self.get_parameter('topo_penalty_weight').value
        with self._uncertainty_lock:
            pose_uncertainty = self._pose_uncertainty

        scored = self._scorer.score_candidates(
            candidates, pose, outline, mapper,
            topo_map=self._topo_map,
            weight_topo_penalty=topo_penalty,
            pose_uncertainty=pose_uncertainty,
        )
        if not scored:
            return None

        # ── Pick the best candidate far enough from the current goal ──────────
        min_vis = self.get_parameter('min_visibility_threshold').value
        pre_validate = self.get_parameter('pre_validate_goals').value
        for sc in scored:
            if sc.visibility_score < min_vis:
                continue
            pos = sc.candidate.position
            if self._current_goal is not None:
                if np.linalg.norm(pos - self._current_goal) < CANDIDATE_MIN_DIST:
                    continue
            if pre_validate and not self._validator.is_reachable(pose, pos, mapper):
                self.get_logger().debug(
                    f'Goal ({pos[0]:.2f},{pos[1]:.2f}) pre-validation failed — skipping')
                self._blacklist.append((pos, time.monotonic() + 10.0))
                continue
            self._publish_candidates(scored, self.get_clock().now().to_msg())
            return sc

        self.get_logger().info(
            'All candidates below visibility threshold — exploration may be complete')
        return None

    def _send_scored_candidate(self, best: ScoredCandidate):
        """Send a scored candidate as a NavigateToPose goal and start prefetch."""
        target = best.candidate.position
        yaw    = best.candidate.orientation
        self._current_goal = target
        self._send_nav_goal(target[0], target[1], yaw)

        # A4: start computing the next goal in the background immediately
        self._start_prefetch()

    def _start_prefetch(self):
        """A4: Spawn a background thread to compute the next NBV candidate."""
        with self._prefetch_lock:
            self._prefetched_goal = None  # clear any stale result
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return  # already running
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True, name='nbv-prefetch')
        self._prefetch_thread.start()

    def _prefetch_worker(self):
        """A4: Background worker that pre-computes the next goal."""
        try:
            next_goal = self._compute_best_candidate()
            with self._prefetch_lock:
                self._prefetched_goal = next_goal
            if next_goal is not None:
                self.get_logger().debug('Prefetch complete — next goal ready')
        except Exception as exc:
            self.get_logger().debug(f'Prefetch worker error: {exc}')

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
            f'Sending NBV goal: ({x:.2f}, {y:.2f}) yaw={math.degrees(yaw):.1f}°  '
            f'coverage={self._current_coverage:.1%}')

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
            with self._prefetch_lock:
                self._prefetched_goal = None  # A4: discard prefetch on rejection
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
            # A4: stale prefetch is based on old map state, discard it
            with self._prefetch_lock:
                self._prefetched_goal = None
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

    # =========================================================================
    # Extension 5: Coverage fallback
    # =========================================================================

    def _run_coverage_fallback(self, pose: np.ndarray, mapper) -> None:
        """Generate a coverage sweep goal when no frontiers are available."""
        visited_r = self.get_parameter('topo_visited_radius').value
        candidates = self._coverage_planner.generate(
            pose, mapper, topo_map=self._topo_map, visited_radius=visited_r)
        if not candidates:
            self.get_logger().info('Coverage fallback: no unvisited free cells — exploration complete')
            return

        self.get_logger().info(
            f'Coverage fallback: {len(candidates)} sweep candidates generated')

        pre_validate = self.get_parameter('pre_validate_goals').value
        for c in candidates:
            pos = c.position
            if pre_validate and not self._validator.is_reachable(pose, pos, mapper):
                continue
            self._current_goal = pos
            self._send_nav_goal(pos[0], pos[1], c.orientation)
            return

        self.get_logger().info('Coverage fallback: all candidates unreachable')

    # =========================================================================
    # Extension 4: Topological graph visualisation
    # =========================================================================

    def _publish_topo_graph(self, stamp) -> None:
        """Publish topo nodes (spheres) and edges (lines) to /topo_graph."""
        if self._topo_map is None:
            return
        nodes = self._topo_map.get_all_nodes()
        if not nodes:
            return

        markers = MarkerArray()

        for i, node in enumerate(nodes):
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp = stamp
            m.ns = 'topo_nodes'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(node.position[0])
            m.pose.position.y = float(node.position[1])
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            if node.visited:
                m.color = ColorRGBA(r=0.0, g=0.9, b=0.2, a=0.7)
            else:
                m.color = ColorRGBA(r=0.9, g=0.9, b=0.0, a=0.5)
            markers.markers.append(m)

        edge_marker = Marker()
        edge_marker.header.frame_id = self._map_frame
        edge_marker.header.stamp = stamp
        edge_marker.ns = 'topo_edges'
        edge_marker.id = 0
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.02
        edge_marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.4)
        for node in nodes:
            for ni in node.neighbor_indices:
                if ni < len(nodes):
                    p1 = GPoint()
                    p1.x, p1.y = float(node.position[0]), float(node.position[1])
                    p2 = GPoint()
                    p2.x, p2.y = float(nodes[ni].position[0]), float(nodes[ni].position[1])
                    edge_marker.points.extend([p1, p2])
        markers.markers.append(edge_marker)

        self._topo_pub.publish(markers)

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
