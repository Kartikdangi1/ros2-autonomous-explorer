"""Lightweight 2D point-mass Gymnasium environment for fast reward iteration.

Drop-in replacement for MuJoCoExplorerEnv — identical observation space,
action space, reward function, and curriculum integration.  Pure NumPy,
no physics engine required.

The robot is a point mass with (x, y, yaw) state and body-frame velocity
commands.  Walls are line segments matching the MuJoCo maze.  LiDAR is
simulated via vectorized 2D raycasting.

Expected FPS: 5,000–50,000+ (vs 50–200 in MuJoCo, ~5 in Gazebo).
"""

from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np

from rl_local_planner.obs_builder import (
    COSTMAP_OBS_SIZE,
    MAX_SCAN_RANGE,
    SCAN_DIM,
    RawSensorData,
    build_observation,
    scale_action,
)
from rl_local_planner.reward import RewardState, RewardWeights, compute_reward
from rl_local_planner.curriculum import CurriculumManager


from scipy.ndimage import maximum_filter


# ── Vectorized costmap (replaces Python-loop version in mujoco_sim.py) ──────

def _lidar_to_costmap_fast(
    scan_metres: np.ndarray,
    resolution: float = 0.12,
    size: int = 84,
    inflation_cells: int = 2,
) -> np.ndarray:
    """Vectorized lidar→costmap. Same output as mujoco_sim.lidar_to_costmap."""
    center = size // 2
    n_rays = len(scan_metres)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    x_m = scan_metres * np.cos(angles)
    y_m = scan_metres * np.sin(angles)
    cols = (center + x_m / resolution).astype(np.intp)
    rows = (center - y_m / resolution).astype(np.intp)
    valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[rows[valid], cols[valid]] = 254
    # Inflate using max filter
    inflated = maximum_filter(grid, size=2 * inflation_cells + 1)
    # inscribed=253 where inflated but not lethal
    result = np.where(grid == 254, 254, np.where(inflated > 0, 253, 0)).astype(np.uint8)
    return result.flatten()


# ── Constants (match mujoco_env.py) ─────────────────────────────────────────
COLLISION_THRESHOLD = 0.15    # metres — LiDAR-based collision
STUCK_WINDOW        = 25      # steps
STUCK_THRESHOLD     = 0.05    # metres
PHYSICS_TIMESTEP    = 0.05    # 20 Hz
ROBOT_RADIUS        = 0.25    # metres — collision body radius
SPAWN_MIN_CLEARANCE = 0.30    # metres
MAX_SPAWN_RETRIES   = 10
MAX_GOAL_RETRIES    = 20
GOAL_MIN_CLEARANCE  = 0.20    # metres


# ── Maze geometry (from mujoco_maze.xml) ────────────────────────────────────

def _box_to_segments(
    cx: float, cy: float, hx: float, hy: float, yaw: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Convert an oriented box to 4 line segments [(x1,y1,x2,y2), ...]."""
    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)
    corners = []
    for sx, sy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
        corners.append((cx + cos_a * sx - sin_a * sy,
                        cy + sin_a * sx + cos_a * sy))
    segs = []
    for i in range(4):
        j = (i + 1) % 4
        segs.append((corners[i][0], corners[i][1],
                      corners[j][0], corners[j][1]))
    return segs


def _build_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Build wall segments and circles arrays from maze definition."""
    # (cx, cy, hx, hy, yaw) — from mujoco_maze.xml
    boxes = [
        # Perimeter walls
        (0, 12.5, 12.8, 0.15, 0),
        (0, -12.5, 12.8, 0.15, 0),
        (12.5, 0, 0.15, 12.5, 0),
        (-12.5, 0, 0.15, 12.5, 0),
        # Inner horizontal
        (-4, 0, 4.0, 0.125, 0),
        (6.5, 0, 2.5, 0.125, 0),
        (-2, 6, 5.0, 0.125, 0),
        (8, 9, 2.0, 0.125, 0),
        (-3, -5, 3.5, 0.125, 0),
        (5, -8, 3.0, 0.125, 0),
        # Inner vertical
        (-8, 3, 0.125, 6.0, 0),
        (3, 4, 0.125, 4.0, 0),
        (9, -3, 0.125, 3.5, 0),
        (-5, -8, 0.125, 3.0, 0),
        (6, 9, 0.125, 2.5, 0),
        # Angled
        (-8, -9, 2.0, 0.1, 0.7854),
        (10, 5, 2.5, 0.1, -0.5236),
        # Crates (0.3 half-extent boxes)
        (1.0, 3.0, 0.3, 0.3, 0),
        (-3.0, -7.0, 0.3, 0.3, 0),
        (7.0, -5.0, 0.3, 0.3, 0),
        (-7.0, -2.0, 0.3, 0.3, 0),
        (5.0, 7.0, 0.3, 0.3, 0),
    ]
    segs = []
    for cx, cy, hx, hy, yaw in boxes:
        segs.extend(_box_to_segments(cx, cy, hx, hy, yaw))

    circles = [
        # Pillars
        (-4, 6, 0.25), (3, -5, 0.25), (8, 3, 0.30), (-9, -3, 0.25),
        # Barrels
        (-6.0, 2.0, 0.3), (-5.4, 2.0, 0.3), (-5.7, 2.55, 0.3),
        (11.0, 0.0, 0.3), (11.0, -1.0, 0.3),
    ]

    seg_arr = np.array(segs, dtype=np.float64)      # (N, 4)
    circ_arr = np.array(circles, dtype=np.float64)   # (M, 3)
    return seg_arr, circ_arr


# Pre-compute at module load
WALL_SEGMENTS, CIRCLES = _build_geometry()


# ── Vectorized 2D raycasting ────────────────────────────────────────────────

def raycast_2d(
    px: float, py: float, yaw: float,
    n_rays: int = SCAN_DIM,
    max_range: float = MAX_SCAN_RANGE,
) -> np.ndarray:
    """Cast 2D rays against maze geometry. Returns (n_rays,) float32 distances.

    Fully vectorized — no Python loops over rays or segments.
    """
    # Ray directions
    angles = yaw + np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dx = np.cos(angles)  # (R,)
    dy = np.sin(angles)  # (R,)

    scan = np.full(n_rays, max_range, dtype=np.float64)

    # ── Ray-segment intersection ────────────────────────────────────────
    # Segments: (N, 4) → A=(x1,y1), B=(x2,y2)
    if len(WALL_SEGMENTS) > 0:
        ax = WALL_SEGMENTS[:, 0]  # (N,)
        ay = WALL_SEGMENTS[:, 1]
        bx = WALL_SEGMENTS[:, 2]
        by = WALL_SEGMENTS[:, 3]
        sx = bx - ax  # segment direction (N,)
        sy = by - ay

        # cross(d, s) = dx*sy - dy*sx → (R, N)
        denom = dx[:, None] * sy[None, :] - dy[:, None] * sx[None, :]

        # Vector from ray origin to segment start
        qx = ax[None, :] - px  # (1, N) broadcast to (R, N)
        qy = ay[None, :] - py

        # t = cross(q, s) / denom
        t_num = qx * sy[None, :] - qy * sx[None, :]
        # u = cross(q, d) / denom
        u_num = qx * dy[:, None] - qy * dx[:, None]

        # Avoid division by zero
        safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        t = t_num / safe_denom
        u = u_num / safe_denom

        # Valid: t > 0, 0 <= u <= 1, denom != 0
        valid = (t > 0) & (u >= 0) & (u <= 1) & (np.abs(denom) >= 1e-12)
        t = np.where(valid, t, max_range)

        seg_min = np.min(t, axis=1)  # (R,)
        scan = np.minimum(scan, seg_min)

    # ── Ray-circle intersection ─────────────────────────────────────────
    if len(CIRCLES) > 0:
        cx = CIRCLES[:, 0]  # (M,)
        cy = CIRCLES[:, 1]
        cr = CIRCLES[:, 2]

        # f = origin - centre
        fx = px - cx[None, :]  # (1, M) → broadcast with (R, 1)
        fy = py - cy[None, :]

        # a = dx^2 + dy^2 = 1 (unit directions)
        # b = 2*(f·d)
        b = 2.0 * (fx * dx[:, None] + fy * dy[:, None])  # (R, M)
        # c = f·f - r^2
        c = fx * fx + fy * fy - (cr[None, :] ** 2)  # (R, M)

        disc = b * b - 4.0 * c  # a=1

        sqrt_disc = np.where(disc > 0, np.sqrt(np.maximum(disc, 0)), 0.0)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        # Nearest positive root
        t_circ = np.where(t1 > 0, t1, np.where(t2 > 0, t2, max_range))
        t_circ = np.where(disc > 0, t_circ, max_range)

        circ_min = np.min(t_circ, axis=1)  # (R,)
        scan = np.minimum(scan, circ_min)

    return np.clip(scan, 0.0, max_range).astype(np.float32)


def _point_collides(x: float, y: float, radius: float) -> bool:
    """Check if a circle at (x, y) with given radius overlaps any geometry."""
    # Segments: vectorized point-to-segment distance
    if len(WALL_SEGMENTS) > 0:
        ax = WALL_SEGMENTS[:, 0]
        ay = WALL_SEGMENTS[:, 1]
        bx = WALL_SEGMENTS[:, 2]
        by = WALL_SEGMENTS[:, 3]
        sx = bx - ax
        sy = by - ay
        len_sq = sx * sx + sy * sy
        len_sq = np.maximum(len_sq, 1e-12)
        t = np.clip(((x - ax) * sx + (y - ay) * sy) / len_sq, 0.0, 1.0)
        closest_x = ax + t * sx
        closest_y = ay + t * sy
        dist_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
        if np.any(dist_sq < radius * radius):
            return True

    # Circles
    if len(CIRCLES) > 0:
        dx = x - CIRCLES[:, 0]
        dy = y - CIRCLES[:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < (radius + CIRCLES[:, 2])):
            return True

    return False


def _ray_hits_before(
    sx: float, sy: float, gx: float, gy: float, clearance: float,
) -> bool:
    """Check line-of-sight from (sx,sy) to (gx,gy). Returns True if clear."""
    dx = gx - sx
    dy = gy - sy
    goal_dist = math.hypot(dx, dy)
    if goal_dist < 1e-6:
        return True
    # Cast single ray
    scan = raycast_2d(sx, sy, math.atan2(dy, dx), n_rays=1, max_range=goal_dist + 1.0)
    return float(scan[0]) >= (goal_dist - clearance)


# ── Environment ─────────────────────────────────────────────────────────────

class Point2DExplorerEnv(gym.Env):
    """Lightweight 2D env — same interface as MuJoCoExplorerEnv."""

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

        # Spaces (identical to MuJoCoExplorerEnv)
        self.observation_space = gym.spaces.Dict({
            'costmap':     gym.spaces.Box(0, 255, (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1), np.uint8),
            'scan':        gym.spaces.Box(0.0, 1.0, (SCAN_DIM,), np.float32),
            'goal_vector': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
            'velocity':    gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

        # Episode state
        self._x: float = 0.0
        self._y: float = 0.0
        self._yaw: float = 0.0
        self._vx_body: float = 0.0
        self._vy_body: float = 0.0
        self._vyaw: float = 0.0
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._step_count: int = 0
        self._reward_state = RewardState()
        self._position_history: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW)
        self._dynamics: dict = {}
        self._action_buffer: deque[np.ndarray] = deque()

    # ── Gymnasium interface ──────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample valid spawn
        spawn = self._curriculum.sample_spawn(self._rng)
        for _ in range(MAX_SPAWN_RETRIES):
            if not _point_collides(spawn[0], spawn[1], SPAWN_MIN_CLEARANCE):
                break
            spawn = self._curriculum.sample_spawn(self._rng)

        self._x, self._y = spawn[0], spawn[1]
        self._yaw = float(self._rng.uniform(0, 2 * math.pi))
        self._vx_body = 0.0
        self._vy_body = 0.0
        self._vyaw = 0.0

        # Sample goal with line-of-sight check
        goal = self._curriculum.sample_goal(spawn[0], spawn[1], self._rng)
        for _ in range(MAX_GOAL_RETRIES):
            if _ray_hits_before(spawn[0], spawn[1], goal[0], goal[1],
                                GOAL_MIN_CLEARANCE):
                break
            goal = self._curriculum.sample_goal(spawn[0], spawn[1], self._rng)
        self._goal_x, self._goal_y = goal

        init_dist = math.hypot(goal[0] - spawn[0], goal[1] - spawn[1])

        self._step_count = 0
        self._reward_state = RewardState(
            initial_goal_distance=max(init_dist, 0.1),
            prev_goal_distance=init_dist,
            prev_action=None,
        )
        self._position_history.clear()
        self._dynamics = self._curriculum.sample_dynamics(self._rng)
        self._action_buffer.clear()

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

        # ── Scale action → physical velocity (body frame) ────────────────
        vx_body, vy_body, vyaw = scale_action(action)
        vel_scale = self._dynamics.get('vel_scale', 1.0)
        vx_body *= vel_scale
        vy_body *= vel_scale
        vyaw *= vel_scale

        # ── Kinematic integration ────────────────────────────────────────
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        vx_world = cos_y * vx_body - sin_y * vy_body
        vy_world = sin_y * vx_body + cos_y * vy_body

        new_x = self._x + vx_world * PHYSICS_TIMESTEP
        new_y = self._y + vy_world * PHYSICS_TIMESTEP
        new_yaw = self._yaw + vyaw * PHYSICS_TIMESTEP
        new_yaw = (new_yaw + math.pi) % (2 * math.pi) - math.pi

        # Collision resolution: revert position if new pos collides
        collision = _point_collides(new_x, new_y, COLLISION_THRESHOLD)
        if not collision:
            self._x = new_x
            self._y = new_y
        self._yaw = new_yaw
        self._vx_body = vx_body if not collision else 0.0
        self._vy_body = vy_body if not collision else 0.0
        self._vyaw = vyaw

        # ── LiDAR ────────────────────────────────────────────────────────
        scan_metres = raycast_2d(self._x, self._y, self._yaw)

        # Sensor noise
        scan_noise = self._dynamics.get('scan_noise_sigma', 0.0)
        if scan_noise > 0:
            noise = self._rng.normal(0, scan_noise, size=scan_metres.shape).astype(np.float32)
            scan_metres = np.clip(scan_metres + noise, 0.0, MAX_SCAN_RANGE)

        # ── Observation ──────────────────────────────────────────────────
        obs = self._build_obs(scan_metres)

        # ── Goal distance ────────────────────────────────────────────────
        goal_dx = self._goal_x - self._x
        goal_dy = self._goal_y - self._y
        goal_dist = math.hypot(goal_dx, goal_dy)
        min_scan = float(np.nanmin(scan_metres))

        # ── Termination checks ───────────────────────────────────────────
        goal_reached = goal_dist < self._reward_weights.goal_tolerance
        # Also check LiDAR-based collision (matches mujoco_env)
        lidar_collision = min_scan < COLLISION_THRESHOLD
        collision = collision or lidar_collision

        self._position_history.append((self._x, self._y))
        stuck = False
        if len(self._position_history) >= STUCK_WINDOW:
            oldest = self._position_history[0]
            newest = self._position_history[-1]
            stuck = math.hypot(newest[0] - oldest[0],
                               newest[1] - oldest[1]) < STUCK_THRESHOLD

        timeout = self._step_count >= self._curriculum.config.max_steps
        terminated = goal_reached or collision
        truncated = timeout or stuck

        # ── Reward ───────────────────────────────────────────────────────
        reward, reward_info = compute_reward(
            curr_goal_distance=goal_dist,
            min_scan_range=min_scan,
            action=action,
            goal_reached=goal_reached,
            collision=collision,
            state=self._reward_state,
            weights=self._reward_weights,
            robot_yaw=self._yaw,
            goal_dx=goal_dx,
            goal_dy=goal_dy,
        )

        info = {
            'goal_reached':     goal_reached,
            'collision':        collision,
            'stuck':            stuck,
            'timeout':          timeout,
            'goal_distance':    goal_dist,
            'min_range':        min_scan,
            'step':             self._step_count,
            'curriculum_stage': self._curriculum.current_stage,
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_obs(self, scan_metres: np.ndarray) -> dict[str, np.ndarray]:
        raw = RawSensorData()
        raw.costmap = _lidar_to_costmap_fast(scan_metres)
        raw.costmap_width = COSTMAP_OBS_SIZE
        raw.costmap_height = COSTMAP_OBS_SIZE
        raw.costmap_resolution = 0.12
        raw.scan_ranges = scan_metres
        raw.robot_x = self._x
        raw.robot_y = self._y
        raw.robot_yaw = self._yaw
        raw.robot_vx = self._vx_body
        raw.robot_vy = self._vy_body
        raw.robot_vyaw = self._vyaw
        raw.goal_x = self._goal_x
        raw.goal_y = self._goal_y
        return build_observation(raw)

    def _get_obs(self) -> dict[str, np.ndarray]:
        scan = raycast_2d(self._x, self._y, self._yaw)
        return self._build_obs(scan)

    def set_curriculum_stage(self, stage: int) -> None:
        """Propagate curriculum stage from the main process to this worker."""
        self._curriculum.current_stage = stage
        self._curriculum._successes.clear()

    def close(self) -> None:
        pass
