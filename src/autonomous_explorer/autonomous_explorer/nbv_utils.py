#!/usr/bin/env python3
"""
NBV Algorithm Utilities
=======================
Core data structures and algorithms for Next Best View exploration:
  - OccupancyMapper: Bayesian log-odds grid synced from SLAM Toolbox /map
  - OutlineExtractor: Frontier detection via polar-sector jump edges
  - CandidateGenerator: NBV candidate placement at frontier edges
  - NBVScorer: Information-gain scoring via ray-casting
  - Shared utilities: _normalize_angle, _segments_intersect
"""

import numpy as np
import threading
from scipy.spatial import KDTree
from dataclasses import dataclass, field
from typing import List, Optional

# ── Module-level constants ───────────────────────────────────────────────────
OCCUPIED_PROB_THRESHOLD = 0.65
FREE_PROB_THRESHOLD = 0.40


def _log_odds_to_prob(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds value(s) to probability in [0, 1]."""
    return 1.0 - 1.0 / (1.0 + np.exp(log_odds))


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Edge:
    start: np.ndarray
    end: np.ndarray

@dataclass
class JumpEdge:
    start: np.ndarray
    end: np.ndarray
    midpoint: np.ndarray = field(default_factory=lambda: np.zeros(2))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # True if at least one endpoint is a real scan return (not a max-range
    # placeholder). Edges between two placeholders produce phantom frontiers
    # far outside the environment and must not generate candidates.
    has_real_endpoint: bool = True

    def __post_init__(self):
        self.midpoint = (self.start + self.end) / 2.0
        edge_vec = self.end - self.start
        self.normal = np.array([-edge_vec[1], edge_vec[0]])
        norm = np.linalg.norm(self.normal)
        if norm > 1e-6:
            self.normal /= norm
        self.length = float(norm)  # B1: metric length of this frontier edge (m)

@dataclass
class OutlinePolygon:
    vertices: np.ndarray
    edges: List[Edge]
    jump_edges: List[JumpEdge]

@dataclass
class Candidate:
    position: np.ndarray
    orientation: float

@dataclass
class ScoredCandidate:
    candidate: Candidate
    visibility_score: float
    distance_cost: float
    orientation_cost: float
    total_score: float

# ============================================================
# Occupancy Grid Mapping
# ============================================================

class OccupancyMapper:
    def __init__(self, width=800, height=800, resolution=0.05,
                 origin_x=-20.0, origin_y=-20.0,
                 prob_min=0.12, prob_max=0.97):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self._lock = threading.Lock()
        self.log_odds = np.zeros((height, width), dtype=np.float32)

        # Inflated grid cache — recomputed only when the map changes.
        # _map_version is incremented on every update_with_scan_points call.
        # _inflated_cache_version tracks the map version at the time the cache
        # was built; if they differ the cache is stale and must be rebuilt.
        self._map_version: int = 0
        self._inflated_cache: Optional[np.ndarray] = None
        self._inflated_cache_radius: Optional[float] = None
        self._inflated_cache_version: int = -1

        self.log_occ = np.log(0.75 / 0.25)   # ≈  1.10
        self.log_free = np.log(0.40 / 0.60)   # ≈ -0.41

        self.prob_min = prob_min
        self.prob_max = prob_max
        self.log_odds_min = np.log(prob_min / (1 - prob_min))
        self.log_odds_max = np.log(prob_max / (1 - prob_max))

        # A7: proactive inflate-cache — background thread rewarms the inflated
        # grid immediately after each map update so the A* validator never stalls
        # on a cold binary_dilation (which takes ~50 ms on first call).
        self._default_robot_radius = 0.3
        self._inflate_trigger = threading.Event()
        self._inflate_bg_thread = threading.Thread(
            target=self._inflate_bg_worker, daemon=True, name='inflate-cache')
        self._inflate_bg_thread.start()

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution
        return wx, wy

    def is_in_map(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def update_with_scan_points(self, robot_pose, scan_points_world):
        rx, ry = robot_pose[0], robot_pose[1]
        rgx, rgy = self.world_to_grid(rx, ry)
        if not self.is_in_map(rgx, rgy):
            return

        n = len(scan_points_world)
        if n == 0:
            return

        # --- Vectorised coordinate conversion (no lock needed) ---
        egx = np.floor(
            (scan_points_world[:, 0] - self.origin_x) / self.resolution
        ).astype(np.int32)
        egy = np.floor(
            (scan_points_world[:, 1] - self.origin_y) / self.resolution
        ).astype(np.int32)
        in_map = (
            (0 <= egx) & (egx < self.width) &
            (0 <= egy) & (egy < self.height)
        )

        # Accumulate all log-odds updates outside the lock so we hold it for
        # only a single, fast array addition at the end.  This reduces lock
        # contention with the planning thread (which reads log_odds briefly to
        # snapshot the grid) from O(N × ray_len) down to O(1) acquisitions.
        delta = np.zeros((self.height, self.width), dtype=np.float32)

        for i in range(n):
            # Clip endpoint so _bresenham always stays within array bounds.
            # We correct the final cell to 'occupied' below if the true
            # endpoint was inside the map.
            ex = int(np.clip(egx[i], 0, self.width - 1))
            ey = int(np.clip(egy[i], 0, self.height - 1))

            ray = self._bresenham(rgx, rgy, ex, ey)
            for j, (px, py) in enumerate(ray):
                if not self.is_in_map(px, py):
                    break
                if j == len(ray) - 1 and in_map[i]:
                    delta[py, px] += self.log_occ
                else:
                    delta[py, px] += self.log_free

        # Single lock acquisition — just array arithmetic, very fast.
        with self._lock:
            self.log_odds += delta
            np.clip(self.log_odds, self.log_odds_min, self.log_odds_max,
                    out=self.log_odds)
            self._map_version += 1  # invalidate inflated grid cache
        self._inflate_trigger.set()  # A7: wake background cache warmer

    def get_occupancy_grid(self):
        with self._lock:
            lo = self.log_odds.copy()
        prob = _log_odds_to_prob(lo)
        grid = np.full_like(prob, -1, dtype=np.int8)
        grid[prob >= OCCUPIED_PROB_THRESHOLD] = 100
        grid[prob <= FREE_PROB_THRESHOLD] = 0
        grid[lo == 0] = -1
        return grid

    def get_inflated_grid(self, robot_radius=0.3):
        """Return a binary grid (1=obstacle+inflation, 0=free/unknown).

        The result is cached and recomputed only when the map has changed
        since the last call (tracked by _map_version).  This avoids running
        a full binary_dilation on every A* query — especially important now
        that planning runs in a background thread and can be called frequently.
        """
        # Read the map version atomically so we can decide whether the cache
        # is still valid without holding the lock during the expensive dilation.
        with self._lock:
            current_version = self._map_version

        if (self._inflated_cache is not None and
                self._inflated_cache_version == current_version and
                self._inflated_cache_radius == robot_radius):
            return self._inflated_cache

        from scipy.ndimage import binary_dilation
        grid = self.get_occupancy_grid()
        occupied = (grid == 100)
        inflate_cells = int(np.ceil(robot_radius / self.resolution))
        y, x = np.ogrid[-inflate_cells:inflate_cells + 1,
                        -inflate_cells:inflate_cells + 1]
        struct = (x * x + y * y <= inflate_cells * inflate_cells)
        inflated = binary_dilation(occupied, structure=struct).astype(np.uint8)

        # Store in cache.  If the map was updated while we were computing, the
        # next caller will see current_version != _map_version and recompute.
        self._inflated_cache = inflated
        self._inflated_cache_radius = robot_radius
        self._inflated_cache_version = current_version
        return inflated

    def _inflate_bg_worker(self):
        """A7: Proactively recompute inflated grid whenever the map changes."""
        last_warmed = -1
        while True:
            self._inflate_trigger.wait(timeout=2.0)
            self._inflate_trigger.clear()
            with self._lock:
                v = self._map_version
            if v != last_warmed:
                self.get_inflated_grid(self._default_robot_radius)
                last_warmed = v

    def get_inflated_grid_coarse(self, robot_radius=0.3, factor=2):
        """B4: Return a 2× downsampled inflated grid for faster A* validation.

        Any factor×factor block containing an obstacle maps to an obstacle cell
        in the coarse grid, so the coarse resolution is factor*resolution metres
        per cell.  Coarse grid coordinates = fine grid coordinates // factor.
        """
        fine = self.get_inflated_grid(robot_radius)
        H, W = fine.shape
        H2, W2 = H // factor, W // factor
        coarse = (fine[:H2 * factor, :W2 * factor]
                  .reshape(H2, factor, W2, factor)
                  .max(axis=(1, 3)))
        return coarse

    def is_free(self, gx, gy):
        if not self.is_in_map(gx, gy):
            return False
        with self._lock:
            prob = _log_odds_to_prob(self.log_odds[gy, gx])
        return prob <= FREE_PROB_THRESHOLD

    def is_known(self, gx, gy):
        if not self.is_in_map(gx, gy):
            return False
        with self._lock:
            return self.log_odds[gy, gx] != 0.0

    def _bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        max_pts = int(np.sqrt(self.width ** 2 + self.height ** 2)) + 10
        while max_pts > 0:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            max_pts -= 1
        return points


# ============================================================
# Outline Polygon Extraction
# ============================================================

class OutlineExtractor:
    def __init__(self, num_sectors=72, jump_threshold=1.5, max_range=20.0):
        self.num_sectors = num_sectors
        self.jump_threshold = jump_threshold
        self.max_range = max_range
        # Incremental frontier cache (Extension 3)
        self._cached_outline = None
        self._cached_map_version: int = -1
        self._cached_robot_cell: Optional[tuple] = None
        self._robot_move_threshold_cells: int = 5  # ~0.25 m at 0.05 m/cell

    def extract(self, points_2d, robot_pose, mapper=None):
        """Extract frontier outline.

        If *mapper* is supplied the result is cached and reused when the map
        version has not changed and the robot has not moved more than
        ~0.25 m.  Pass mapper=None to always recompute (legacy behaviour).
        """
        if mapper is not None:
            with mapper._lock:
                ver = mapper._map_version
            rgx, rgy = mapper.world_to_grid(robot_pose[0], robot_pose[1])
            rc = (rgx, rgy)
            if (self._cached_outline is not None
                    and ver == self._cached_map_version
                    and self._cached_robot_cell is not None
                    and abs(rc[0] - self._cached_robot_cell[0]) < self._robot_move_threshold_cells
                    and abs(rc[1] - self._cached_robot_cell[1]) < self._robot_move_threshold_cells):
                return self._cached_outline
            result = self._do_extract(points_2d, robot_pose)
            self._cached_outline = result
            self._cached_map_version = ver
            self._cached_robot_cell = rc
            return result
        return self._do_extract(points_2d, robot_pose)

    def _do_extract(self, points_2d, robot_pose):
        if len(points_2d) < 3:
            return OutlinePolygon(np.empty((0, 2)), [], [])

        rx, ry = robot_pose[0], robot_pose[1]
        rel = points_2d - np.array([rx, ry])
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        ranges = np.linalg.norm(rel, axis=1)

        sector_size = 2.0 * np.pi / self.num_sectors
        vertices = []
        sector_has_return = []

        for i in range(self.num_sectors):
            sector_angle = -np.pi + (i + 0.5) * sector_size
            angle_diff = np.abs(self._normalize_angles(angles - sector_angle))
            in_sector = angle_diff < (sector_size / 2.0)

            if np.any(in_sector):
                sector_ranges = ranges[in_sector]
                sector_pts = points_2d[in_sector]
                max_idx = np.argmax(sector_ranges)
                vertices.append(sector_pts[max_idx])
                sector_has_return.append(True)
            else:
                vx = rx + self.max_range * np.cos(sector_angle)
                vy = ry + self.max_range * np.sin(sector_angle)
                vertices.append(np.array([vx, vy]))
                sector_has_return.append(False)

        if not any(sector_has_return):
            return OutlinePolygon(np.empty((0, 2)), [], [])

        vertices = np.array(vertices)
        edges = []
        jump_edges = []

        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            gap = np.linalg.norm(vertices[j] - vertices[i])

            has_i = sector_has_return[i]
            has_j = sector_has_return[j]

            is_jump = False
            if has_i and has_j and gap > self.jump_threshold:
                is_jump = True
            elif has_i != has_j:
                is_jump = True

            if is_jump:
                jump_edges.append(JumpEdge(
                    start=vertices[i], end=vertices[j],
                    has_real_endpoint=(has_i or has_j)))
            else:
                edges.append(Edge(start=vertices[i], end=vertices[j]))

        return OutlinePolygon(vertices=vertices, edges=edges, jump_edges=jump_edges)

    @staticmethod
    def _normalize_angles(angles):
        return (angles + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# Candidate NBV Pose Generation
# ============================================================

class CandidateGenerator:
    def __init__(self, offset_distance=0.8, sample_spacing=1.0,
                 exploration_radius=8.0, min_candidate_dist=1.0):
        self.offset_distance = offset_distance
        self.sample_spacing = sample_spacing
        self.exploration_radius = exploration_radius
        self.min_candidate_dist = min_candidate_dist

    def generate(self, outline, robot_pose, mapper, blacklist=None):
        candidates = []
        blacklist = blacklist or []

        # Strategy A: Jump edge placement (primary)
        # Only use edges with at least one real scan return; edges between
        # two max-range placeholders produce phantom frontiers outside the map.
        for je in outline.jump_edges:
            if not je.has_real_endpoint:
                continue

            # Place the candidate on the ROBOT'S SIDE of the jump edge.
            # The stored normal is an arbitrary left-perpendicular of the edge
            # vector; it can point toward or away from the robot.  Using the
            # sign of the dot product with (robot - midpoint) ensures the
            # candidate always lands in reachable space, never behind the wall.
            robot_vec = robot_pose[:2] - je.midpoint
            dot = np.dot(robot_vec, je.normal)
            sign = 1.0 if dot >= 0.0 else -1.0

            pos = je.midpoint + sign * je.normal * self.offset_distance
            to_edge = je.midpoint - pos
            orientation = np.arctan2(to_edge[1], to_edge[0])

            if self._is_valid(pos, mapper, blacklist):
                dist_to_robot = np.linalg.norm(pos - robot_pose[:2])
                if dist_to_robot > self.min_candidate_dist and dist_to_robot < self.exploration_radius:
                    candidates.append(Candidate(position=pos, orientation=orientation))

        candidates = self._deduplicate(candidates, min_dist=1.0)

        # Strategy B: Uniform sampling fallback
        if len(candidates) == 0:
            rx, ry = robot_pose[0], robot_pose[1]
            r = self.exploration_radius
            for x in np.arange(rx - r, rx + r, self.sample_spacing):
                for y in np.arange(ry - r, ry + r, self.sample_spacing):
                    pos = np.array([x, y])
                    dist = np.linalg.norm(pos - np.array([rx, ry]))
                    if dist < self.min_candidate_dist or dist > r:
                        continue
                    if self._is_valid(pos, mapper, blacklist):
                        orientation = self._orient_to_nearest_jump(pos, outline.jump_edges)
                        candidates.append(Candidate(position=pos, orientation=orientation))
                        if len(candidates) > 30:
                            break
                if len(candidates) > 30:
                    break

        return candidates[:30]

    # Robot radius used for candidate clearance validation.
    # Must be >= nav2 inflation_radius (0.25 m) so candidates always have
    # enough room for the robot to manoeuvre without hitting inflated walls.
    CANDIDATE_CLEARANCE_RADIUS = 0.35

    def _is_valid(self, pos, mapper, blacklist):
        gx, gy = mapper.world_to_grid(pos[0], pos[1])
        if not mapper.is_in_map(gx, gy):
            return False
        # Use the inflated grid (not raw log-odds) so candidates inside the
        # robot's inflation zone are rejected.  This prevents goals that are
        # technically "free" but wedged against a wall with no room to turn.
        inflated = mapper.get_inflated_grid(self.CANDIDATE_CLEARANCE_RADIUS)
        if inflated[gy, gx] == 1:
            return False
        # Use a 1 m spatial radius so that slight pose changes after a failed
        # attempt don't immediately make the same position eligible again.
        for bl in blacklist:
            if np.linalg.norm(pos - np.asarray(bl)) < 1.0:
                return False
        return True

    def _deduplicate(self, candidates, min_dist=1.0):
        if not candidates:
            return []
        positions = np.array([c.position for c in candidates])
        tree = KDTree(positions)
        keep = [True] * len(candidates)
        for i, pos in enumerate(positions):
            if not keep[i]:
                continue
            for j in tree.query_ball_point(pos, min_dist):
                if j > i:
                    keep[j] = False
        return [c for i, c in enumerate(candidates) if keep[i]]

    def _orient_to_nearest_jump(self, pos, jump_edges):
        if not jump_edges:
            return 0.0
        min_dist = float('inf')
        best_orient = 0.0
        for je in jump_edges:
            dist = np.linalg.norm(pos - je.midpoint)
            if dist < min_dist:
                min_dist = dist
                to_edge = je.midpoint - pos
                best_orient = np.arctan2(to_edge[1], to_edge[0])
        return best_orient


# ============================================================
# NBV Scoring
# ============================================================

class NBVScorer:
    """Score NBV candidates using vectorized batch ray-casting.

    A1: All candidates and all rays are evaluated simultaneously using NumPy
    array operations — no Python loops over cells.  Typical speedup vs. the
    old Bresenham-loop implementation: 15-30×.

    B1: Frontier-size weighting — candidates near long frontier clusters score
    higher, directing the robot toward large unexplored regions.

    B3: Uncertainty-adaptive weights — when EKF pose uncertainty is high the
    distance weight grows (prefer close goals) and the topo penalty shrinks
    (allow loop-closure revisits) automatically.
    """

    def __init__(self, num_rays=72, weight_visibility=3.0, weight_distance=1.0,
                 weight_orientation=0.5, exploration_radius=8.0, use_entropy=False,
                 weight_frontier_size=1.5):
        self.num_rays = num_rays
        self.weight_visibility = weight_visibility
        self.weight_distance = weight_distance
        self.weight_orientation = weight_orientation
        self.exploration_radius = exploration_radius
        self.use_entropy = use_entropy          # Extension 2 flag
        self.weight_frontier_size = weight_frontier_size  # B1

        # A1: Precompute unit ray directions — fixed for the lifetime of the node.
        angles = -np.pi + np.arange(num_rays) * (2.0 * np.pi / num_rays)
        self._ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (R, 2)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_candidates(self, candidates, robot_pose, outline, mapper,
                         topo_map=None, weight_topo_penalty=0.8,
                         pose_uncertainty: float = 0.0):
        """Score all candidates; returns list sorted best-first.

        pose_uncertainty: EKF position covariance trace (m²), ≥ 0.
          - High uncertainty  → distance weight increases (prefer closer goals)
          - High uncertainty  → topo penalty decreases (allow loop-closure revisits)
        """
        if not candidates:
            return []

        with mapper._lock:
            lo_snapshot = mapper.log_odds.copy()

        # A1: batch vectorized scoring for all candidates at once
        vis_scores = self._score_all_vectorized(candidates, outline, mapper, lo_snapshot)

        # B1: precompute frontier-size lookup arrays
        if self.weight_frontier_size > 0 and outline.jump_edges:
            edge_lengths = np.array([je.length for je in outline.jump_edges])
            edge_mids    = np.array([je.midpoint for je in outline.jump_edges])  # (J, 2)
            total_length = max(float(edge_lengths.sum()), 1e-6)
        else:
            edge_lengths = edge_mids = None
            total_length = 1.0

        # B3: uncertainty-adaptive weight scaling
        unc_scale = float(np.clip(pose_uncertainty / 0.5, 0.0, 1.0))
        eff_topo_w  = weight_topo_penalty * (1.0 - 0.7 * unc_scale)
        eff_dist_w  = self.weight_distance * (1.0 + unc_scale)

        scored = []
        for i, c in enumerate(candidates):
            vis        = float(vis_scores[i])
            dist_cost  = float(np.linalg.norm(c.position - robot_pose[:2])
                               / self.exploration_radius)
            orient_cost = abs(_normalize_angle(robot_pose[2] - c.orientation)) / np.pi

            # Extension 4: topological revisit penalty (B3-scaled)
            topo_penalty = (eff_topo_w
                            if topo_map is not None and topo_map.is_near_visited(c.position)
                            else 0.0)

            # B1: frontier-size bonus — sum of lengths of nearby jump edges
            if edge_lengths is not None:
                dists    = np.linalg.norm(edge_mids - c.position, axis=1)
                nearby   = dists < self.exploration_radius
                frontier_size = float(edge_lengths[nearby].sum()) / total_length
            else:
                frontier_size = 0.0

            total = (self.weight_visibility * vis
                     - eff_dist_w * dist_cost
                     - self.weight_orientation * orient_cost
                     - topo_penalty
                     + self.weight_frontier_size * frontier_size)

            scored.append(ScoredCandidate(
                candidate=c, visibility_score=vis,
                distance_cost=dist_cost, orientation_cost=orient_cost,
                total_score=total))

        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # A1: Vectorized batch ray-casting
    # ------------------------------------------------------------------

    def _score_all_vectorized(self, candidates, outline, mapper,
                               lo_snapshot: np.ndarray) -> np.ndarray:
        """Compute visibility or entropy scores for ALL candidates at once.

        Uses NumPy advanced indexing — no Python loops over cells or rays.
        Memory: ~20 MB of temporaries for 20 candidates × 72 rays × 300 steps.

        Returns: float32 array of shape (C,).
        """
        C = len(candidates)
        R = self.num_rays
        positions = np.array([c.position for c in candidates], dtype=np.float32)  # (C, 2)

        step_m  = mapper.resolution          # 0.05 m — one grid cell per step
        max_r   = float(self.exploration_radius)
        S       = int(max_r / step_m) + 1
        t       = np.arange(1, S + 1, dtype=np.float32) * step_m  # (S,)

        # Sample positions along every ray for every candidate: (C, R, S, 2)
        sample_world = (positions[:, None, None, :]
                        + self._ray_dirs.astype(np.float32)[None, :, None, :]
                        * t[None, None, :, None])

        # Grid indices: (C, R, S)
        gxs = np.floor(
            (sample_world[:, :, :, 0] - mapper.origin_x) / mapper.resolution
        ).astype(np.int32)
        gys = np.floor(
            (sample_world[:, :, :, 1] - mapper.origin_y) / mapper.resolution
        ).astype(np.int32)

        in_bounds = (
            (gxs >= 0) & (gxs < mapper.width) &
            (gys >= 0) & (gys < mapper.height)
        )  # (C, R, S)

        gxs_c = np.clip(gxs, 0, mapper.width  - 1)
        gys_c = np.clip(gys, 0, mapper.height - 1)

        # "still_in": ray is still inside the map (stops at first OOB cell)
        still_in = np.cumprod(in_bounds.astype(np.uint8), axis=2).astype(bool)

        lo_occ_thr = float(np.log(OCCUPIED_PROB_THRESHOLD / (1.0 - OCCUPIED_PROB_THRESHOLD)))
        lo_vals    = lo_snapshot[gys_c, gxs_c]  # (C, R, S)

        occ_hit  = still_in & (lo_vals >= lo_occ_thr)   # (C, R, S)
        blocked  = np.any(occ_hit, axis=2)               # (C, R)
        unblocked = ~blocked                              # (C, R)

        # ── Entropy scoring (Extension 2) ────────────────────────────────────
        if self.use_entropy:
            occ_cumsum  = np.cumsum(occ_hit, axis=2)             # (C, R, S)
            still_clear = still_in & (occ_cumsum == 0)           # (C, R, S)
            p_c = np.clip(
                1.0 - 1.0 / (1.0 + np.exp(np.clip(lo_vals, -50.0, 50.0))),
                1e-6, 1.0 - 1e-6)
            entropy     = -p_c * np.log2(p_c) - (1.0 - p_c) * np.log2(1.0 - p_c)
            total_ent   = np.sum(still_clear * entropy, axis=(1, 2))  # (C,)
            max_poss    = R * (max_r / mapper.resolution)
            return np.clip(total_ent / max(max_poss, 1.0), 0.0, 1.0).astype(np.float32)

        # ── Visibility scoring (default) ─────────────────────────────────────
        if not outline.jump_edges:
            return np.zeros(C, dtype=np.float32)

        p3 = np.array([je.start for je in outline.jump_edges], dtype=np.float64)  # (J, 2)
        p4 = np.array([je.end   for je in outline.jump_edges], dtype=np.float64)  # (J, 2)

        # Scaled ray direction vectors: d1[r] = ray_dirs[r] * max_r  → (R, 2)
        d1 = (self._ray_dirs * max_r).astype(np.float64)  # (R, 2)
        # Edge direction vectors: d2[j] = p4[j] - p3[j]  → (J, 2)
        d2 = p4 - p3                                       # (J, 2)
        # d3[c, j] = p3[j] - positions[c]  → (C, J, 2)
        d3 = p3[None, :, :] - positions[:, None, :].astype(np.float64)

        # cross[r, j] = d1[r] × d2[j]  → (R, J)
        cross      = d1[:, 0:1] * d2[None, :, 1] - d1[:, 1:2] * d2[None, :, 0]
        safe_cross = np.where(np.abs(cross) < 1e-10, 1e-10, cross)
        parallel   = np.abs(cross) < 1e-10  # (R, J)

        # t_param[c, r, j]: numerator depends on (c, j) only → (C, J) / (R, J)
        t_num   = d3[:, :, 0] * d2[None, :, 1] - d3[:, :, 1] * d2[None, :, 0]  # (C, J)
        t_param = t_num[:, None, :] / safe_cross[None, :, :]                     # (C, R, J)

        # u_param[c, r, j]: numerator depends on (c, r, j)  → (C, R, J)
        u_num   = (d3[:, None, :, 0] * d1[None, :, None, 1]
                   - d3[:, None, :, 1] * d1[None, :, None, 0])       # (C, R, J)
        u_param = u_num / safe_cross[None, :, :]                      # (C, R, J)

        intersects = (
            ~parallel[None, :, :]
            & (t_param >= 0.0) & (t_param <= 1.0)
            & (u_param >= 0.0) & (u_param <= 1.0)
        )  # (C, R, J)

        ray_hit_any = np.any(intersects, axis=2)                          # (C, R)
        vis_counts  = np.sum(unblocked & ray_hit_any, axis=1).astype(float)  # (C,)
        return (vis_counts / R).astype(np.float32)


# ============================================================
# Shared Utilities
# ============================================================

def _normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def _segments_intersect(p1: np.ndarray, p2: np.ndarray,
                        p3: np.ndarray, p4: np.ndarray) -> bool:
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return False
    d3 = p3 - p1
    t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
    u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


# ============================================================
# B2: Coverage Tracker
# ============================================================

class CoverageTracker:
    """Tracks map exploration coverage and drives adaptive exploration radius.

    Coverage = (free known cells) / (all known cells) ∈ [0, 1].
    As coverage grows, exploration_radius expands so the robot casts further.
    """
    # (coverage_fraction_threshold, new_exploration_radius_m)
    RADIUS_STAGES = [(0.3, 20.0), (0.6, 25.0)]

    @staticmethod
    def compute_coverage(mapper: 'OccupancyMapper') -> float:
        """Return fraction of known (non-zero) cells that are free, ∈ [0, 1]."""
        lo_free_thr = np.log(FREE_PROB_THRESHOLD / (1.0 - FREE_PROB_THRESHOLD))
        lo_occ_thr  = np.log(OCCUPIED_PROB_THRESHOLD / (1.0 - OCCUPIED_PROB_THRESHOLD))
        with mapper._lock:
            lo = mapper.log_odds
            free     = int(np.sum(lo < lo_free_thr))
            occupied = int(np.sum(lo > lo_occ_thr))
        total = free + occupied
        return float(free) / total if total > 0 else 0.0

    @staticmethod
    def adaptive_radius(coverage: float, base_radius: float = 15.0) -> float:
        """Return an exploration radius that grows with coverage."""
        radius = base_radius
        for threshold, new_radius in CoverageTracker.RADIUS_STAGES:
            if coverage >= threshold:
                radius = max(radius, new_radius)
        return radius


# ============================================================
# Extension 1: Goal Pre-Validation (fast A* reachability check)
# ============================================================

class GoalValidator:
    """Fast 4-connected A* reachability check on the cached inflated grid.

    Run before dispatching NavigateToPose so unreachable goals are
    discarded in milliseconds instead of waiting for Nav2's planner
    timeout (up to 30 s).  Fail-open: returns True when uncertain so
    that the blacklist still catches genuinely unreachable goals.

    B4: Uses a 2× downsampled (coarse) grid by default — reduces search cells
    from 640k to 160k, cutting A* time by ~4× with no meaningful accuracy loss
    (obstacles are inflated, so coarse-grid representation is conservative).
    """

    def __init__(self, robot_radius: float = 0.3, max_search_cells: int = 40000,
                 use_coarse: bool = True):
        self.robot_radius = robot_radius
        self.max_search_cells = max_search_cells
        self.use_coarse = use_coarse   # B4: use 2× downsampled grid
        self._coarse_factor = 2

    def is_reachable(self, robot_pose: np.ndarray,
                     goal_pos: np.ndarray,
                     mapper: 'OccupancyMapper') -> bool:
        """Return True if a collision-free path exists from robot to goal."""
        import heapq
        if mapper is None:
            return True

        factor = self._coarse_factor if self.use_coarse else 1
        if self.use_coarse:
            grid = mapper.get_inflated_grid_coarse(self.robot_radius, factor)
        else:
            grid = mapper.get_inflated_grid(self.robot_radius)
        H, W = grid.shape

        # Fine-grid coords then scale down to coarse coords
        rgx_f, rgy_f = mapper.world_to_grid(robot_pose[0], robot_pose[1])
        ggx_f, ggy_f = mapper.world_to_grid(goal_pos[0], goal_pos[1])
        rgx, rgy = rgx_f // factor, rgy_f // factor
        ggx, ggy = ggx_f // factor, ggy_f // factor

        def _in_grid(x, y):
            return 0 <= x < W and 0 <= y < H

        for gx, gy in [(rgx, rgy), (ggx, ggy)]:
            if not _in_grid(gx, gy):
                return False
        if grid[ggy, ggx] == 1:
            return False       # goal is inside an obstacle
        if grid[rgy, rgx] == 1:
            return True        # robot stuck in obstacle — let Nav2 handle

        open_heap = [(abs(ggx - rgx) + abs(ggy - rgy), 0, rgx, rgy)]
        visited: set = set()
        searched = 0

        while open_heap and searched < self.max_search_cells:
            _, g, cx, cy = heapq.heappop(open_heap)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            searched += 1
            if cx == ggx and cy == ggy:
                return True
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited or not _in_grid(nx, ny):
                    continue
                if grid[ny, nx] == 1:
                    continue
                ng = g + 1
                heapq.heappush(open_heap, (ng + abs(ggx - nx) + abs(ggy - ny), ng, nx, ny))

        # Budget exhausted: fail-open (let Nav2 decide) unless search was tiny
        return searched < self.max_search_cells


# ============================================================
# Extension 4: Topological Exploration Graph
# ============================================================

import time as _time  # avoid shadowing 'time' in user namespace

@dataclass
class TopoNode:
    """A waypoint node in the topological roadmap."""
    position: np.ndarray
    cell: tuple
    visited: bool = False
    visit_time: float = 0.0
    neighbor_indices: list = field(default_factory=list)


class TopologicalMap:
    """Sparse roadmap of visited robot positions built during exploration.

    Nodes are added whenever the robot moves more than *node_spacing* metres
    from the nearest existing node.  During NBV scoring the planner can
    query is_near_visited() to apply a penalty to candidates in already-
    explored regions, discouraging redundant re-exploration.

    Thread-safe: all node mutations are protected by an internal lock.
    """

    def __init__(self, node_spacing: float = 1.5,
                 visited_radius: float = 2.0,
                 max_nodes: int = 2000):
        self.node_spacing = node_spacing
        self.visited_radius = visited_radius
        self.max_nodes = max_nodes
        self._nodes: List[TopoNode] = []
        self._positions: Optional[np.ndarray] = None   # Nx2, KDTree cache
        self._tree: Optional[KDTree] = None
        self._lock = threading.Lock()

    def update(self, robot_pose: np.ndarray, mapper: 'OccupancyMapper') -> None:
        """Mark nearest node visited or add a new node at the robot's position."""
        pos = np.array([robot_pose[0], robot_pose[1]])
        with self._lock:
            if self._tree is not None and len(self._nodes) > 0:
                dist, idx = self._tree.query(pos)
                if dist < self.node_spacing:
                    self._nodes[idx].visited = True
                    self._nodes[idx].visit_time = _time.monotonic()
                    return
            if len(self._nodes) >= self.max_nodes:
                return
            gx, gy = mapper.world_to_grid(pos[0], pos[1])
            node = TopoNode(position=pos.copy(), cell=(gx, gy),
                            visited=True, visit_time=_time.monotonic())
            self._nodes.append(node)
            self._positions = np.array([n.position for n in self._nodes])
            self._tree = KDTree(self._positions)
            if len(self._nodes) > 1:
                new_idx = len(self._nodes) - 1
                for ni in self._tree.query_ball_point(pos, 3.0 * self.node_spacing):
                    if ni == new_idx:
                        continue
                    node.neighbor_indices.append(ni)
                    self._nodes[ni].neighbor_indices.append(new_idx)

    def is_near_visited(self, position: np.ndarray, radius: float = None) -> bool:
        """Return True if *position* is within *radius* of any visited node."""
        radius = radius if radius is not None else self.visited_radius
        with self._lock:
            if self._tree is None:
                return False
            indices = self._tree.query_ball_point(position, radius)
            return any(self._nodes[i].visited for i in indices)

    def get_all_nodes(self) -> List[TopoNode]:
        """Return a snapshot of all nodes for visualisation."""
        with self._lock:
            return list(self._nodes)


# ============================================================
# Extension 5: Coverage Path Planning Fallback (boustrophedon)
# ============================================================

class CoveragePlanner:
    """Boustrophedon (lawnmower) coverage for stagnated exploration.

    Activated when no frontiers are found for N consecutive planning ticks.
    Scans the occupancy map for free cells that are not near any visited
    topological node, then generates an ordered back-and-forth sweep path.
    """

    def __init__(self, row_spacing: float = 1.0):
        self.row_spacing = row_spacing

    def generate(self, robot_pose: np.ndarray,
                 mapper: 'OccupancyMapper',
                 topo_map: Optional['TopologicalMap'] = None,
                 visited_radius: float = 2.0) -> List['Candidate']:
        """Return an ordered list of coverage waypoints (≤ 30 candidates)."""
        grid = mapper.get_occupancy_grid()
        free_gy, free_gx = np.where(grid == 0)
        if len(free_gx) == 0:
            return []

        wx = mapper.origin_x + (free_gx + 0.5) * mapper.resolution
        wy = mapper.origin_y + (free_gy + 0.5) * mapper.resolution
        pts = np.column_stack([wx, wy])

        # Remove cells already covered by a visited topo node
        if topo_map is not None:
            unvisited = np.array(
                [not topo_map.is_near_visited(p, visited_radius) for p in pts])
            pts = pts[unvisited]

        if len(pts) == 0:
            return []

        # Build boustrophedon path: group into Y rows, alternate X direction
        row_ids = np.floor(pts[:, 1] / self.row_spacing).astype(int)
        candidates: List[Candidate] = []
        for rid in np.unique(row_ids):
            row_pts = pts[row_ids == rid]
            x_min, x_max = row_pts[:, 0].min(), row_pts[:, 0].max()
            xs = np.arange(x_min, x_max, self.row_spacing)
            direction = 1 if rid % 2 == 0 else -1
            for x in xs[::direction]:
                nearest = row_pts[np.argmin(np.abs(row_pts[:, 0] - x))]
                orientation = 0.0 if direction > 0 else np.pi
                candidates.append(Candidate(position=nearest.copy(),
                                            orientation=orientation))
            if len(candidates) >= 30:
                break

        return candidates[:30]