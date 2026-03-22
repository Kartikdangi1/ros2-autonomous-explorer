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

    def extract(self, points_2d, robot_pose):
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
            pos = je.midpoint + je.normal * self.offset_distance
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

    def _is_valid(self, pos, mapper, blacklist):
        gx, gy = mapper.world_to_grid(pos[0], pos[1])
        if not mapper.is_in_map(gx, gy):
            return False
        with mapper._lock:
            lo = mapper.log_odds[gy, gx]
        prob = _log_odds_to_prob(lo)
        if prob >= OCCUPIED_PROB_THRESHOLD:
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
    def __init__(self, num_rays=72, weight_visibility=3.0, weight_distance=1.0,
                 weight_orientation=0.5, exploration_radius=8.0):
        self.num_rays = num_rays
        self.weight_visibility = weight_visibility
        self.weight_distance = weight_distance
        self.weight_orientation = weight_orientation
        self.exploration_radius = exploration_radius

    def score_candidates(self, candidates, robot_pose, outline, mapper):
        with mapper._lock:
            lo_snapshot = mapper.log_odds.copy()
        scored = []
        for c in candidates:
            vis = self._visibility_score(c, outline, mapper, lo_snapshot)
            dist_cost = np.linalg.norm(c.position - robot_pose[:2]) / self.exploration_radius
            orient_cost = abs(_normalize_angle(robot_pose[2] - c.orientation)) / np.pi
            total = (self.weight_visibility * vis
                     - self.weight_distance * dist_cost
                     - self.weight_orientation * orient_cost)
            scored.append(ScoredCandidate(
                candidate=c, visibility_score=vis,
                distance_cost=dist_cost, orientation_cost=orient_cost,
                total_score=total))
        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored

    def _visibility_score(self, candidate, outline, mapper, lo_snapshot):
        if not outline.jump_edges:
            return 0.0
        px, py = candidate.position
        hits = 0
        gx0, gy0 = mapper.world_to_grid(px, py)
        for i in range(self.num_rays):
            angle = -np.pi + i * (2.0 * np.pi / self.num_rays)
            ray_end = np.array([
                px + self.exploration_radius * np.cos(angle),
                py + self.exploration_radius * np.sin(angle)])

            gx1, gy1 = mapper.world_to_grid(ray_end[0], ray_end[1])
            blocked = False
            ray_cells = mapper._bresenham(gx0, gy0, gx1, gy1)
            for cell_x, cell_y in ray_cells[1:]:
                if not mapper.is_in_map(cell_x, cell_y):
                    break
                prob = _log_odds_to_prob(lo_snapshot[cell_y, cell_x])
                if prob >= OCCUPIED_PROB_THRESHOLD:
                    blocked = True
                    break
            if blocked:
                continue

            ray_start = np.array([px, py])
            for je in outline.jump_edges:
                if _segments_intersect(ray_start, ray_end, je.start, je.end):
                    hits += 1
                    break

        return hits / self.num_rays if self.num_rays > 0 else 0.0


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