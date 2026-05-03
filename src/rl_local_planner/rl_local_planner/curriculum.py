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
    # Sensor noise
    scan_noise_sigma_max: float = 0.02
    odom_noise_sigma_max: float = 0.05


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
    # Stage 2: medium — longer goals, all spawns, full episodes
    StageConfig(
        goal_dist_min=2.0,
        goal_dist_max=4.0,
        spawn_points=ALL_SPAWNS,
        max_steps=300,
        scan_noise_sigma_max=0.02,
        odom_noise_sigma_max=0.05,
    ),
    # Stage 3: hard — full distance range + dynamics randomization
    StageConfig(
        goal_dist_min=3.0,
        goal_dist_max=6.0,
        spawn_points=ALL_SPAWNS,
        max_steps=400,
        use_dynamics_randomization=True,
        friction_range=(0.7, 1.3),
        vel_scale_range=(0.8, 1.2),
        action_delay_max_steps=2,
        scan_noise_sigma_max=0.02,
        odom_noise_sigma_max=0.05,
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

    def sample_goal(
        self,
        spawn_x: float,
        spawn_y: float,
        rng: np.random.Generator,
        max_attempts: int = 50,
    ) -> tuple[float, float]:
        """Sample a random goal within the stage's distance range.

        Rejects goals outside MAZE_BOUNDS and retries up to max_attempts times.
        Falls back to a +X offset from spawn if all attempts go out of bounds.
        """
        cfg = self.config
        for _ in range(max_attempts):
            angle = float(rng.uniform(0.0, 2 * np.pi))
            dist = float(rng.uniform(cfg.goal_dist_min, cfg.goal_dist_max))
            gx = spawn_x + dist * np.cos(angle)
            gy = spawn_y + dist * np.sin(angle)
            if abs(gx) <= MAZE_BOUNDS and abs(gy) <= MAZE_BOUNDS:
                return (gx, gy)
        logger.warning(
            'sample_goal: all %d attempts placed goal outside maze bounds '
            'from spawn (%.1f, %.1f). Using fallback.',
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
        }
        if cfg.use_dynamics_randomization:
            result['vel_scale'] = float(rng.uniform(*cfg.vel_scale_range))
            result['action_delay_steps'] = int(
                rng.integers(0, cfg.action_delay_max_steps + 1)
            )
        return result

    def get_stats(self) -> dict:
        return {
            'curriculum_stage': self.current_stage,
            'curriculum_success_rate': self.success_rate,
            'curriculum_total_episodes': self._total_episodes,
        }
