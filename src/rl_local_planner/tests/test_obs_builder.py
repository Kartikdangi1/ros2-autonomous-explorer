"""Unit tests for rl_local_planner.obs_builder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rl_local_planner.obs_builder import (
    COSTMAP_OBS_SIZE,
    GOAL_NORM,
    MAX_SCAN_RANGE,
    MAX_VEL_THETA,
    MAX_VEL_X,
    MAX_VEL_Y,
    SCAN_DIM,
    RawSensorData,
    build_costmap_obs,
    build_goal_obs,
    build_observation,
    build_scan_obs,
    build_velocity_obs,
    scale_action,
)


# ── build_costmap_obs ─────────────────────────────────────────────────────────

class TestBuildCostmapObs:
    def test_none_costmap_returns_zeros(self, raw_empty):
        out = build_costmap_obs(raw_empty)
        assert out.shape == (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1)
        assert out.dtype == np.uint8
        assert np.all(out == 0)

    def test_output_shape_and_dtype(self, raw_populated):
        out = build_costmap_obs(raw_populated)
        assert out.shape == (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1)
        assert out.dtype == np.uint8

    def test_free_costmap_stays_near_zero(self, raw_populated):
        out = build_costmap_obs(raw_populated)
        assert out.max() == 0  # all-free grid → all zeros after resize

    def test_lethal_cell_preserved(self):
        """A costmap with a lethal cell (254) should appear in the output."""
        size = 84
        raw = RawSensorData()
        raw.costmap_height = size
        raw.costmap_width = size
        raw.costmap_resolution = 0.05
        raw.costmap_origin_x = 0.0
        raw.costmap_origin_y = 0.0
        data = np.zeros(size * size, dtype=np.uint8)
        data[size * size // 2] = 254  # one lethal cell in the centre
        raw.costmap = data

        out = build_costmap_obs(raw)
        assert out.max() > 0  # lethal cell must survive resize


# ── build_scan_obs ────────────────────────────────────────────────────────────

class TestBuildScanObs:
    def test_none_scan_returns_ones(self, raw_empty):
        out = build_scan_obs(raw_empty)
        assert out.shape == (SCAN_DIM,)
        assert out.dtype == np.float32
        assert np.all(out == 1.0)

    def test_uniform_scan_normalised(self, raw_populated):
        # All ranges = 5.0 m → 5/18 ≈ 0.278
        out = build_scan_obs(raw_populated)
        expected = 5.0 / MAX_SCAN_RANGE
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_nan_becomes_one(self):
        raw = RawSensorData()
        raw.scan_ranges = np.full(SCAN_DIM, float('nan'), dtype=np.float32)
        out = build_scan_obs(raw)
        assert np.all(out == 1.0)

    def test_inf_becomes_one(self):
        raw = RawSensorData()
        raw.scan_ranges = np.full(SCAN_DIM, float('inf'), dtype=np.float32)
        out = build_scan_obs(raw)
        assert np.all(out == 1.0)

    def test_clipped_to_zero_one(self):
        raw = RawSensorData()
        # Ranges beyond MAX_SCAN_RANGE should be clipped to 1.0
        raw.scan_ranges = np.full(SCAN_DIM, MAX_SCAN_RANGE * 2, dtype=np.float32)
        out = build_scan_obs(raw)
        assert np.all(out == 1.0)

    def test_zero_range_clips_to_zero(self):
        raw = RawSensorData()
        raw.scan_ranges = np.zeros(SCAN_DIM, dtype=np.float32)
        out = build_scan_obs(raw)
        assert np.all(out == 0.0)

    def test_output_in_range(self, raw_populated):
        out = build_scan_obs(raw_populated)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ── build_goal_obs ────────────────────────────────────────────────────────────

class TestBuildGoalObs:
    def test_output_shape(self, raw_populated):
        out = build_goal_obs(raw_populated)
        assert out.shape == (2,)
        assert out.dtype == np.float32

    def test_goal_ahead_yaw_zero(self):
        """Robot at origin, yaw=0, goal 10 m ahead → body dx=10/10=1.0, dy=0."""
        raw = RawSensorData()
        raw.robot_x, raw.robot_y, raw.robot_yaw = 0.0, 0.0, 0.0
        raw.goal_x, raw.goal_y = 10.0, 0.0
        out = build_goal_obs(raw)
        np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-6)

    def test_goal_to_left_yaw_zero(self):
        """Goal 10 m to the left → body dx=0, dy=10/10=1.0."""
        raw = RawSensorData()
        raw.robot_x, raw.robot_y, raw.robot_yaw = 0.0, 0.0, 0.0
        raw.goal_x, raw.goal_y = 0.0, 10.0
        out = build_goal_obs(raw)
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)

    def test_goal_rotated_frame(self):
        """Robot yaw=π/2 (facing +Y), goal 1 m in +X → body frame: dy_body=-1."""
        raw = RawSensorData()
        raw.robot_x, raw.robot_y, raw.robot_yaw = 0.0, 0.0, math.pi / 2
        raw.goal_x, raw.goal_y = 1.0, 0.0
        out = build_goal_obs(raw)
        # dx_body = cos(-pi/2)*1 - sin(-pi/2)*0 = 0; dy_body = sin(-pi/2)*1 + cos(-pi/2)*0 = -1
        np.testing.assert_allclose(out, [0.0, -1.0 / GOAL_NORM], atol=1e-5)

    def test_output_clipped_to_minus_one_one(self):
        """Very far goal should be clipped to ±1."""
        raw = RawSensorData()
        raw.robot_x, raw.robot_y, raw.robot_yaw = 0.0, 0.0, 0.0
        raw.goal_x, raw.goal_y = 1000.0, 1000.0
        out = build_goal_obs(raw)
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)

    def test_goal_at_robot_position(self):
        raw = RawSensorData()
        out = build_goal_obs(raw)
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)


# ── build_velocity_obs ────────────────────────────────────────────────────────

class TestBuildVelocityObs:
    def test_output_shape_dtype(self, raw_populated):
        out = build_velocity_obs(raw_populated)
        assert out.shape == (3,)
        assert out.dtype == np.float32

    def test_max_velocity_normalises_to_one(self):
        raw = RawSensorData()
        raw.robot_vx = MAX_VEL_X
        raw.robot_vy = MAX_VEL_Y
        raw.robot_vyaw = MAX_VEL_THETA
        out = build_velocity_obs(raw)
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0], atol=1e-6)

    def test_zero_velocity_normalises_to_zero(self, raw_empty):
        out = build_velocity_obs(raw_empty)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-6)

    def test_negative_velocity(self):
        raw = RawSensorData()
        raw.robot_vx = -MAX_VEL_X
        out = build_velocity_obs(raw)
        assert out[0] == pytest.approx(-1.0)

    def test_clipped_beyond_max(self):
        raw = RawSensorData()
        raw.robot_vx = MAX_VEL_X * 10
        out = build_velocity_obs(raw)
        assert out[0] == pytest.approx(1.0)

    def test_half_velocity(self, raw_populated):
        """0.25 m/s vx with MAX_VEL_X=0.5 → 0.5 normalised."""
        out = build_velocity_obs(raw_populated)
        assert out[0] == pytest.approx(0.5)


# ── build_observation ─────────────────────────────────────────────────────────

class TestBuildObservation:
    def test_keys_present(self, raw_populated):
        obs = build_observation(raw_populated)
        assert set(obs.keys()) == {'costmap', 'scan', 'goal_vector', 'velocity'}

    def test_shapes(self, raw_populated):
        obs = build_observation(raw_populated)
        assert obs['costmap'].shape == (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1)
        assert obs['scan'].shape == (SCAN_DIM,)
        assert obs['goal_vector'].shape == (2,)
        assert obs['velocity'].shape == (3,)

    def test_empty_raw_has_correct_shapes(self, raw_empty):
        obs = build_observation(raw_empty)
        assert obs['costmap'].shape == (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1)
        assert obs['scan'].shape == (SCAN_DIM,)


# ── scale_action ──────────────────────────────────────────────────────────────

class TestScaleAction:
    def test_max_action(self):
        action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        vx, vy, vyaw = scale_action(action)
        assert vx == pytest.approx(MAX_VEL_X)
        assert vy == pytest.approx(MAX_VEL_Y)
        assert vyaw == pytest.approx(MAX_VEL_THETA)

    def test_zero_action(self):
        action = np.zeros(3, dtype=np.float32)
        assert scale_action(action) == (0.0, 0.0, 0.0)

    def test_negative_action(self):
        action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        vx, vy, vyaw = scale_action(action)
        assert vx == pytest.approx(-MAX_VEL_X)
        assert vy == pytest.approx(-MAX_VEL_Y)
        assert vyaw == pytest.approx(-MAX_VEL_THETA)

    def test_out_of_range_clipped(self):
        action = np.array([5.0, -5.0, 5.0], dtype=np.float32)
        vx, vy, vyaw = scale_action(action)
        assert vx == pytest.approx(MAX_VEL_X)
        assert vy == pytest.approx(-MAX_VEL_Y)
        assert vyaw == pytest.approx(MAX_VEL_THETA)
