"""Reward function for the RL local planner.

Components:
  - Progress: rewards closing distance to goal (normalised by initial distance)
  - Heading: rewards facing the goal (bootstraps directional learning)
  - Near-goal: approach amplifier — rewards *closing distance* when near goal
    (approach-only, zero for hovering; replaces old position-based shaping
    that created a hovering optimum)
  - Goal bonus: one-time reward on reaching the goal
  - Collision penalty, proximity penalty, smoothness penalty, step cost
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    """Configurable reward weights. Loaded from training_config.yaml."""

    progress: float = 5.0         # × (fraction of initial distance closed)
    goal_reached: float = 25.0    # one-time bonus on success
    collision: float = -5.0       # one-time penalty on collision
    proximity: float = -0.5       # × max(0, threshold - min_range)
    smoothness: float = -0.1      # × ||action_t - action_{t-1}||
    step_cost: float = -0.03      # per-step penalty
    heading: float = 0.2          # × cos(angle_to_goal) — stronger directional bootstrap
    near_goal: float = 5.0        # × approach progress when inside near_goal_radius
    near_goal_radius: float = 0.8 # metres — approach-zone amplifier radius

    proximity_threshold: float = 0.4  # metres — soft warning zone
    goal_tolerance: float = 0.6       # metres — goal reached threshold


@dataclass
class RewardState:
    """Mutable state carried across steps within one episode."""

    initial_goal_distance: float = 1.0
    prev_goal_distance: float = 1.0
    prev_action: np.ndarray | None = None


def compute_reward(
    curr_goal_distance: float,
    min_scan_range: float,
    action: np.ndarray,
    goal_reached: bool,
    collision: bool,
    state: RewardState,
    weights: RewardWeights,
    robot_yaw: float = 0.0,
    goal_dx: float = 0.0,
    goal_dy: float = 0.0,
) -> tuple[float, dict]:
    """Compute the step reward and return (reward, info_dict).

    New parameters (optional, for heading reward):
        robot_yaw: robot's current yaw in world frame (radians)
        goal_dx: goal_x - robot_x (world frame)
        goal_dy: goal_y - robot_y (world frame)

    ``state`` is mutated in-place.
    """
    # ── Progress (normalised by initial distance) ────────────────────────
    raw_progress = state.prev_goal_distance - curr_goal_distance
    denom = max(state.initial_goal_distance, 0.1)
    progress_frac = raw_progress / denom
    r_progress = weights.progress * progress_frac

    # ── Heading alignment (NEW) ──────────────────────────────────────────
    # cos(angle_between_robot_heading_and_goal_direction)
    # Ranges from -1 (facing away) to +1 (facing toward goal)
    # Gives a consistent positive signal for facing the goal,
    # even when the robot hasn't moved yet.
    r_heading = 0.0
    if weights.heading > 0 and (abs(goal_dx) > 0.01 or abs(goal_dy) > 0.01):
        angle_to_goal = math.atan2(goal_dy, goal_dx)
        angle_diff = angle_to_goal - robot_yaw
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        r_heading = weights.heading * math.cos(angle_diff)

    # ── Near-goal shaping (approach-dominant) ──────────────────────────
    # Two components:
    #   1. Small position-based pull (provides gradient toward goal)
    #   2. Approach amplifier (rewards closing distance, zero for hovering)
    # The position component is sized so that hovering is net-negative
    # after step_cost, while approaching gets a large bonus.
    r_near_goal = 0.0
    if weights.near_goal > 0 and curr_goal_distance < weights.near_goal_radius:
        proximity_factor = 1.0 - curr_goal_distance / max(weights.near_goal_radius, 0.01)
        # Position pull: small constant gradient toward goal (0.02 × proximity)
        r_near_goal = 0.02 * proximity_factor
        # Approach amplifier: large bonus for closing distance
        approach = max(0.0, state.prev_goal_distance - curr_goal_distance)
        r_near_goal += weights.near_goal * approach * proximity_factor

    # ── Terminal bonuses / penalties ─────────────────────────────────────
    r_goal = weights.goal_reached if goal_reached else 0.0
    r_collision = weights.collision if collision else 0.0

    # ── Proximity penalty (soft, from LiDAR) ────────────────────────────
    clearance_deficit = max(0.0, weights.proximity_threshold - min_scan_range)
    r_proximity = weights.proximity * clearance_deficit

    # ── Smoothness penalty ───────────────────────────────────────────────
    if state.prev_action is not None:
        delta = float(np.linalg.norm(action - state.prev_action))
    else:
        delta = 0.0
    r_smooth = weights.smoothness * delta

    # ── Step cost ────────────────────────────────────────────────────────
    r_step = weights.step_cost

    # ── Total ────────────────────────────────────────────────────────────
    reward = (r_progress + r_heading + r_near_goal + r_goal + r_collision
              + r_proximity + r_smooth + r_step)

    # ── Update state ─────────────────────────────────────────────────────
    state.prev_goal_distance = curr_goal_distance
    state.prev_action = action.copy()

    info = {
        'r_progress': r_progress,
        'r_heading': r_heading,
        'r_near_goal': r_near_goal,
        'r_goal': r_goal,
        'r_collision': r_collision,
        'r_proximity': r_proximity,
        'r_smooth': r_smooth,
        'r_step': r_step,
    }
    return float(reward), info
