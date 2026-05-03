"""Shared pytest fixtures for rl_local_planner unit tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rl_local_planner.curriculum import CurriculumManager
from rl_local_planner.obs_builder import COSTMAP_OBS_SIZE, SCAN_DIM, RawSensorData
from rl_local_planner.reward import RewardState, RewardWeights


@pytest.fixture
def raw_empty() -> RawSensorData:
    """RawSensorData with all sensor fields None (simulates cold start)."""
    return RawSensorData()


@pytest.fixture
def raw_populated() -> RawSensorData:
    """RawSensorData with realistic populated sensor data."""
    raw = RawSensorData()

    # 10×10 costmap grid, all free (0), 0.1 m/cell
    size = 10
    raw.costmap = np.zeros(size * size, dtype=np.uint8)
    raw.costmap_width = size
    raw.costmap_height = size
    raw.costmap_resolution = 0.1
    raw.costmap_origin_x = -0.5
    raw.costmap_origin_y = -0.5

    # Uniform scan at 5 m
    raw.scan_ranges = np.full(SCAN_DIM, 5.0, dtype=np.float32)

    # Robot at origin, facing forward
    raw.robot_x = 0.0
    raw.robot_y = 0.0
    raw.robot_yaw = 0.0
    raw.robot_vx = 0.25
    raw.robot_vy = 0.0
    raw.robot_vyaw = 0.5

    # Goal 5 m ahead
    raw.goal_x = 5.0
    raw.goal_y = 0.0

    return raw


@pytest.fixture
def default_weights() -> RewardWeights:
    return RewardWeights()


@pytest.fixture
def default_state() -> RewardState:
    return RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0)


@pytest.fixture
def curriculum() -> CurriculumManager:
    return CurriculumManager()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
