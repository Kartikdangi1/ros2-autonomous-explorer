"""Curriculum learning manager for RL training.

CRITICAL FIX: Stage 0 goals reduced to 0.3-0.8m so that a random policy
can occasionally reach them by accident, bootstrapping the initial learning
signal. Previous 1-2m goals were unreachable by random walk.

Also: shorter episodes in early stages, and some near-wall spawns mixed
in so the robot actually encounters obstacles and learns to avoid them.

Progresses through 4 stages based on evaluation success rate:
  Stage 0 (bootstrap) — goals 0.3-0.8m, short episodes, mixed spawns
  Stage 1 (easy)      — goals 0.8-2.0m, medium episodes
  Stage 2 (medium)    — goals 2.0-4.0m, full episodes, all spawns
  Stage 3 (hard)      — goals 3.0-6.0m, dynamics randomization
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ── Predefined spawn points in the maze ──────────────────────────────────────
# "open" points have >= 2m clearance in all directions.
# "near_wall" points have 0.5-1.5m clearance — robot encounters obstacles.
# IMPORTANT: If you change the world file, update these coordinates.

OPEN_SPAWNS: list[tuple[float, float]] = [
    (-11.0, -11.0),
    (-9.0, -11.0),
    (-11.0, -9.0),
    (-6.0, -10.0),
    (-10.0, -6.0),
    (0.0, -3.0),
    (-3.0, 2.0),   # was (-3.0, 0.0) — sat on wall_h_center_left (y≈0)
    (5.0, 3.0),    # was (3.0, 3.0)  — sat on wall_v_center (x≈3)
    (-5.0, 5.0),
]

NEAR_WALL_SPAWNS: list[tuple[float, float]] = [
    (-4.0, -4.0),
    (-2.0, -6.0),
    (2.0, -8.0),
    (5.0, -5.0),
    (-7.0, 3.0),
    (7.0, 0.0),
    (-3.0, 7.0),
    (0.0, 5.0),
    (5.0, 5.0),
    (-8.0, -2.0),
]

# Stage 0 uses only open spawns — near-wall spawns risk placing goals inside
# walls (goal has no collision check), which makes episodes unwinnable and
# prevents curriculum advancement.
BOOTSTRAP_SPAWNS: list[tuple[float, float]] = OPEN_SPAWNS

ALL_SPAWNS: list[tuple[float, float]] = OPEN_SPAWNS + NEAR_WALL_SPAWNS

# Narrow corridor spawns — robot placed inside or just outside the 0.9 m-wide
# corridor at x≈0, y=-4 to -8 (added to mujoco_maze.xml). Only used in
# Stage 2+ when the policy already navigates open space reliably.
NARROW_CORRIDOR_SPAWNS: list[tuple[float, float]] = [
    (0.0, -5.0),   # inside corridor, north half
    (0.0, -6.0),   # inside corridor, center
    (0.0, -7.0),   # inside corridor, south half
    (0.0, -3.5),   # just outside north entrance — forces corridor entry
    (0.0, -8.5),   # just outside south entrance — forces corridor entry
]

# Maze half-extent in metres — goals sampled outside this are rejected
MAZE_BOUNDS = 15.0


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""

    goal_dist_min: float
    goal_dist_max: float
    spawn_points: list[tuple[float, float]]
    max_steps: int = 200               # episode length cap for this stage
    use_dynamics_randomization: bool = False
    # Dynamics randomization parameters
    friction_range: tuple[float, float] = (1.0, 1.0)
    vel_scale_range: tuple[float, float] = (1.0, 1.0)
    action_delay_max_steps: int = 0
    mass_scale_range: tuple[float, float] = (1.0, 1.0)   # robot body mass multiplier
    # Sensor noise
    scan_noise_sigma_max: float = 0.02
    odom_noise_sigma_max: float = 0.05
    lidar_dropout_max: float = 0.0    # fraction of beams randomly dropped


STAGES: list[StageConfig] = [
    # Stage 0: bootstrap — goals just outside goal_tolerance so a random policy
    # cannot reach them without directed movement; max 200 steps (MuJoCo needs
    # more time than point2d to accidentally reach goals and bootstrap reward)
    StageConfig(
        goal_dist_min=0.3,
        goal_dist_max=0.8,
        spawn_points=BOOTSTRAP_SPAWNS,
        max_steps=200,
        scan_noise_sigma_max=0.005,
        odom_noise_sigma_max=0.01,
    ),
    # Stage 1: easy — slightly longer goals, introduce near-wall spawns
    StageConfig(
        goal_dist_min=0.8,
        goal_dist_max=2.0,
        spawn_points=ALL_SPAWNS,
        max_steps=200,
        scan_noise_sigma_max=0.01,
        odom_noise_sigma_max=0.02,
    ),
    # Stage 2: medium — longer goals, all spawns + narrow corridor, full episodes
    StageConfig(
        goal_dist_min=2.0,
        goal_dist_max=4.0,
        spawn_points=ALL_SPAWNS + NARROW_CORRIDOR_SPAWNS,
        max_steps=350,
        scan_noise_sigma_max=0.02,
        odom_noise_sigma_max=0.05,
    ),
    # Stage 3: hard — full distance range + dynamics randomization + narrow corridor
    StageConfig(
        goal_dist_min=3.0,
        goal_dist_max=6.0,
        spawn_points=ALL_SPAWNS + NARROW_CORRIDOR_SPAWNS,
        max_steps=400,
        use_dynamics_randomization=True,
        friction_range=(0.7, 1.3),
        vel_scale_range=(0.8, 1.2),
        action_delay_max_steps=2,
        mass_scale_range=(0.85, 1.15),
        scan_noise_sigma_max=0.02,
        odom_noise_sigma_max=0.05,
        lidar_dropout_max=0.08,
    ),
]

# Success rate thresholds to advance
ADVANCE_THRESHOLDS = [0.40, 0.60, 0.50]  # 0→1, 1→2, 2→3


@dataclass
class CurriculumManager:
    """Tracks training progress and manages stage transitions."""

    current_stage: int = 0
    eval_window: int = 50
    _successes: deque = field(default_factory=lambda: deque(maxlen=50))
    _total_episodes: int = 0
    _total_advances: int = 0
    # Goal blacklist: positions that caused early collisions, temporarily avoided.
    # Mirrors JadNizam's frontier blacklist (up to 20 entries, TTL-based expiry).
    _goal_blacklist: list = field(default_factory=list)   # [(x, y, expiry_monotonic)]
    _blacklist_radius: float = 0.5    # metres — reject resamples within this distance
    _blacklist_ttl: float = 300.0     # seconds — entries auto-expire after 5 min

    @property
    def config(self) -> StageConfig:
        return STAGES[self.current_stage]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage >= len(STAGES) - 1

    def record_episode(self, success: bool) -> None:
        """Record an episode outcome for stage advancement tracking."""
        self._successes.append(1.0 if success else 0.0)
        self._total_episodes += 1

    @property
    def success_rate(self) -> float:
        if len(self._successes) == 0:
            return 0.0
        return float(np.mean(self._successes))

    def maybe_advance(self) -> bool:
        """Check whether to advance to the next stage. Returns True if advanced."""
        if self.is_final_stage:
            return False
        if len(self._successes) < self.eval_window:
            return False
        threshold = ADVANCE_THRESHOLDS[self.current_stage]
        if self.success_rate >= threshold:
            prev_stage = self.current_stage
            self.current_stage += 1
            self._total_advances += 1
            rate = self.success_rate
            self._successes.clear()
            self._goal_blacklist.clear()
            logger.info(
                'Curriculum: stage %d → %d (success_rate=%.2f >= %.2f, '
                'episodes=%d)',
                prev_stage, self.current_stage, rate, threshold,
                self._total_episodes,
            )
            return True
        return False

    def sample_spawn(self, rng: np.random.Generator) -> tuple[float, float]:
        idx = int(rng.integers(len(self.config.spawn_points)))
        return self.config.spawn_points[idx]

    def blacklist_goal(self, x: float, y: float) -> None:
        """Add (x, y) to the goal blacklist for _blacklist_ttl seconds.

        Called by the env when a goal causes an early collision (step < 20),
        indicating the position is likely behind a wall or in a tight corner
        that passes line-of-sight but is not practically navigable.
        Mirrors JadNizam's frontier blacklist (capped at 20 entries).
        """
        self._goal_blacklist.append((x, y, time.monotonic() + self._blacklist_ttl))
        if len(self._goal_blacklist) > 20:
            self._goal_blacklist.pop(0)
        logger.debug('Goal blacklisted at (%.2f, %.2f). Blacklist size: %d',
                     x, y, len(self._goal_blacklist))

    def sample_goal(
        self,
        spawn_x: float,
        spawn_y: float,
        rng: np.random.Generator,
        max_attempts: int = 50,
    ) -> tuple[float, float]:
        """Sample a random goal within the stage's distance range.

        Rejects goals outside MAZE_BOUNDS or within _blacklist_radius of any
        active blacklist entry.  Retries up to max_attempts times; falls back
        to a +X offset from spawn if all attempts fail.
        """
        cfg = self.config
        now = time.monotonic()
        self._goal_blacklist = [(bx, by, t) for bx, by, t in self._goal_blacklist if t > now]

        for _ in range(max_attempts):
            angle = float(rng.uniform(0.0, 2 * np.pi))
            dist = float(rng.uniform(cfg.goal_dist_min, cfg.goal_dist_max))
            gx = spawn_x + dist * np.cos(angle)
            gy = spawn_y + dist * np.sin(angle)
            if abs(gx) > MAZE_BOUNDS or abs(gy) > MAZE_BOUNDS:
                continue
            if any(math.hypot(gx - bx, gy - by) < self._blacklist_radius
                   for bx, by, _ in self._goal_blacklist):
                continue
            return (gx, gy)
        logger.warning(
            'sample_goal: all %d attempts placed goal outside maze bounds or '
            'near a blacklisted position from spawn (%.1f, %.1f). Using fallback.',
            max_attempts, spawn_x, spawn_y,
        )
        return (spawn_x + cfg.goal_dist_min, spawn_y)

    def sample_dynamics(self, rng: np.random.Generator) -> dict:
        """Sample domain-randomization parameters for the current stage."""
        cfg = self.config
        result = {
            'vel_scale': 1.0,
            'scan_noise_sigma': float(rng.uniform(0.0, cfg.scan_noise_sigma_max)),
            'odom_noise_sigma': float(rng.uniform(0.0, cfg.odom_noise_sigma_max)),
            'action_delay_steps': 0,
            'friction_scale': 1.0,
            'mass_scale': 1.0,
            'lidar_dropout_prob': float(rng.uniform(0.0, cfg.lidar_dropout_max)),
        }
        if cfg.use_dynamics_randomization:
            result['vel_scale'] = float(rng.uniform(*cfg.vel_scale_range))
            result['action_delay_steps'] = int(
                rng.integers(0, cfg.action_delay_max_steps + 1)
            )
            result['friction_scale'] = float(rng.uniform(*cfg.friction_range))
            result['mass_scale'] = float(rng.uniform(*cfg.mass_scale_range))
        return result

    def get_stats(self) -> dict:
        return {
            'curriculum_stage': self.current_stage,
            'curriculum_success_rate': self.success_rate,
            'curriculum_total_episodes': self._total_episodes,
            'curriculum_blacklist_size': len(self._goal_blacklist),
        }
