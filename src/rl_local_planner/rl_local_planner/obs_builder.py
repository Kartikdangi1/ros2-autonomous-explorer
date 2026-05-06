"""Shared observation construction for RL training and inference.

Builds the Dict observation consumed by the PPO policy from raw ROS2
sensor data.  Used identically in gym_env.py (training) and
rl_controller_node.py (inference) to guarantee observation parity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

_log = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────
COSTMAP_OBS_SIZE = 84        # downsampled costmap side length (pixels)
SCAN_DIM = 360               # number of LiDAR rays
MAX_SCAN_RANGE = 18.0        # metres (matches robot.urdf.xacro LiDAR config)
MAX_VEL_X = 0.5              # m/s (matches nav2_params.yaml)
MAX_VEL_Y = 0.5
MAX_VEL_THETA = 1.0          # rad/s
GOAL_NORM = 2.0              # normalisation distance for goal vector (metres)
POSE_HISTORY_LEN = 5         # number of past positions to include in observation


@dataclass
class RawSensorData:
    """Container for the latest raw sensor readings from ROS2 topics."""

    costmap: np.ndarray | None = None          # (H, W) uint8 occupancy grid
    costmap_resolution: float = 0.05           # m/cell
    costmap_origin_x: float = 0.0              # world‐frame origin of grid
    costmap_origin_y: float = 0.0
    costmap_width: int = 0
    costmap_height: int = 0

    scan_ranges: np.ndarray | None = None      # (360,) float32 metres

    robot_x: float = 0.0                       # world frame pose
    robot_y: float = 0.0
    robot_yaw: float = 0.0
    robot_vx: float = 0.0                      # body frame velocities
    robot_vy: float = 0.0
    robot_vyaw: float = 0.0

    goal_x: float = 0.0                        # world frame goal
    goal_y: float = 0.0

    # Optional: (POSE_HISTORY_LEN, 2) array of recent world-frame positions,
    # oldest first. None when history is unavailable (e.g. first few steps).
    pose_history: np.ndarray | None = None


def build_pose_history_obs(raw: RawSensorData) -> np.ndarray:
    """Build pose history observation: last N positions as body-frame displacements.

    Returns a (POSE_HISTORY_LEN * 2,) float32 array. Each pair is the
    (dx, dy) from the current position to a past position, rotated into
    body frame and normalised by GOAL_NORM. Gives the policy spatial memory
    so it can detect looping without relying solely on the reactive STUCK_WINDOW.
    """
    if raw.pose_history is None or len(raw.pose_history) == 0:
        return np.zeros(POSE_HISTORY_LEN * 2, dtype=np.float32)

    history = raw.pose_history  # (N, 2) float32, world frame
    cos_yaw = math.cos(-raw.robot_yaw)
    sin_yaw = math.sin(-raw.robot_yaw)

    result = np.zeros(POSE_HISTORY_LEN * 2, dtype=np.float32)
    # Pad from the front if we have fewer than POSE_HISTORY_LEN positions
    offset = POSE_HISTORY_LEN - len(history)
    for i, (px, py) in enumerate(history):
        dx_world = px - raw.robot_x
        dy_world = py - raw.robot_y
        dx_body = cos_yaw * dx_world - sin_yaw * dy_world
        dy_body = sin_yaw * dx_world + cos_yaw * dy_world
        idx = (offset + i) * 2
        result[idx]     = np.clip(dx_body / GOAL_NORM, -1.0, 1.0)
        result[idx + 1] = np.clip(dy_body / GOAL_NORM, -1.0, 1.0)
    return result


def build_costmap_obs(raw: RawSensorData) -> np.ndarray:
    """Downsample the local costmap to (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1).

    Nav2 costmap values:  0 = free, 253 = inscribed, 254 = lethal, 255 = unknown.
    We keep the raw uint8 values — the CNN learns to interpret them.
    """
    if raw.costmap is None:
        return np.zeros((COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1), dtype=np.uint8)

    grid = raw.costmap.reshape(raw.costmap_height, raw.costmap_width)
    resized = cv2.resize(
        grid,
        (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return resized.reshape(COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1)


def build_scan_obs(raw: RawSensorData) -> np.ndarray:
    """Normalise LiDAR ranges to [0, 1].  NaN/inf → 1.0 (max range)."""
    if raw.scan_ranges is None:
        return np.ones(SCAN_DIM, dtype=np.float32)

    scan = np.array(raw.scan_ranges, dtype=np.float32)
    scan = np.where(np.isfinite(scan), scan, MAX_SCAN_RANGE)
    scan = np.clip(scan / MAX_SCAN_RANGE, 0.0, 1.0)
    return scan


def build_goal_obs(raw: RawSensorData) -> np.ndarray:
    """Relative goal vector in robot body frame, normalised by GOAL_NORM."""
    if not (math.isfinite(raw.robot_x) and math.isfinite(raw.robot_y)
            and math.isfinite(raw.robot_yaw) and math.isfinite(raw.goal_x)
            and math.isfinite(raw.goal_y)):
        _log.warning('build_goal_obs: non-finite pose/goal values — returning zeros')
        return np.zeros(2, dtype=np.float32)

    dx_world = raw.goal_x - raw.robot_x
    dy_world = raw.goal_y - raw.robot_y

    cos_yaw = math.cos(-raw.robot_yaw)
    sin_yaw = math.sin(-raw.robot_yaw)
    dx_body = cos_yaw * dx_world - sin_yaw * dy_world
    dy_body = sin_yaw * dx_world + cos_yaw * dy_world

    goal = np.array([dx_body / GOAL_NORM, dy_body / GOAL_NORM], dtype=np.float32)
    return np.clip(goal, -1.0, 1.0)


def build_velocity_obs(raw: RawSensorData) -> np.ndarray:
    """Normalise body-frame velocity to [-1, 1]."""
    if not np.all(np.isfinite([raw.robot_vx, raw.robot_vy, raw.robot_vyaw])):
        _log.warning('build_velocity_obs: non-finite velocity values — returning zeros')
        return np.zeros(3, dtype=np.float32)

    vel = np.array([
        raw.robot_vx / MAX_VEL_X,
        raw.robot_vy / MAX_VEL_Y,
        raw.robot_vyaw / MAX_VEL_THETA,
    ], dtype=np.float32)
    return np.clip(vel, -1.0, 1.0)


def build_observation(raw: RawSensorData) -> dict[str, np.ndarray]:
    """Build the full Dict observation consumed by the PPO policy."""
    obs = {
        'costmap': build_costmap_obs(raw),
        'scan': build_scan_obs(raw),
        'goal_vector': build_goal_obs(raw),
        'velocity': build_velocity_obs(raw),
    }
    if raw.pose_history is not None:
        obs['pose_history'] = build_pose_history_obs(raw)
    return obs


def scale_action(action: np.ndarray) -> tuple[float, float, float]:
    """Scale normalised [-1, 1] policy action to physical velocity commands."""
    action = np.clip(action, -1.0, 1.0)
    vx = float(action[0]) * MAX_VEL_X
    vy = float(action[1]) * MAX_VEL_Y
    vyaw = float(action[2]) * MAX_VEL_THETA
    return vx, vy, vyaw
