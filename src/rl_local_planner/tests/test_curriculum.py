"""Unit tests for rl_local_planner.curriculum."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rl_local_planner.curriculum import (
    ADVANCE_THRESHOLDS,
    ALL_SPAWNS,
    BOOTSTRAP_SPAWNS,
    MAZE_BOUNDS,
    OPEN_SPAWNS,
    STAGES,
    CurriculumManager,
)


@pytest.fixture
def cm():
    return CurriculumManager()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── Initial state ─────────────────────────────────────────────────────────────

class TestInitialState:
    def test_starts_at_stage_zero(self, cm):
        assert cm.current_stage == 0

    def test_stage_zero_config(self, cm):
        assert cm.config.goal_dist_min == 0.3
        assert cm.config.goal_dist_max == 0.8
        assert cm.config.spawn_points is BOOTSTRAP_SPAWNS

    def test_success_rate_zero_with_no_episodes(self, cm):
        assert cm.success_rate == 0.0

    def test_not_final_stage_initially(self, cm):
        assert not cm.is_final_stage


# ── record_episode & success_rate ────────────────────────────────────────────

class TestSuccessTracking:
    def test_all_successes(self, cm):
        for _ in range(10):
            cm.record_episode(True)
        assert cm.success_rate == pytest.approx(1.0)

    def test_all_failures(self, cm):
        for _ in range(10):
            cm.record_episode(False)
        assert cm.success_rate == pytest.approx(0.0)

    def test_mixed_rate(self, cm):
        for i in range(10):
            cm.record_episode(i % 2 == 0)  # 5 successes out of 10
        assert cm.success_rate == pytest.approx(0.5)

    def test_window_size_capped_at_50(self, cm):
        """Only the last 50 episodes count."""
        for _ in range(60):
            cm.record_episode(False)  # 60 failures
        # Then 10 successes
        for _ in range(10):
            cm.record_episode(True)
        # Window holds last 50: 40 failures + 10 successes = 0.2
        assert cm.success_rate == pytest.approx(0.2)


# ── maybe_advance ─────────────────────────────────────────────────────────────

class TestMaybeAdvance:
    def test_does_not_advance_before_window_full(self, cm):
        for _ in range(49):  # one short of eval_window=50
            cm.record_episode(True)
        advanced = cm.maybe_advance()
        assert not advanced
        assert cm.current_stage == 0

    def test_advances_when_threshold_met(self, cm):
        threshold = ADVANCE_THRESHOLDS[0]  # 0.70
        n = cm.eval_window
        successes = math.ceil(n * threshold)
        for i in range(n):
            cm.record_episode(i < successes)
        advanced = cm.maybe_advance()
        assert advanced
        assert cm.current_stage == 1

    def test_does_not_advance_below_threshold(self, cm):
        n = cm.eval_window
        # Barely under stage-0 threshold (0.40)
        successes = math.ceil(n * 0.39) - 1
        for i in range(n):
            cm.record_episode(i < successes)
        advanced = cm.maybe_advance()
        assert not advanced
        assert cm.current_stage == 0

    def test_advances_through_all_stages(self, cm):
        for stage_idx in range(len(STAGES) - 1):
            threshold = ADVANCE_THRESHOLDS[stage_idx]
            n = cm.eval_window
            successes = math.ceil(n * threshold)
            for i in range(n):
                cm.record_episode(i < successes)
            cm.maybe_advance()
        assert cm.current_stage == len(STAGES) - 1

    def test_does_not_advance_past_final_stage(self, cm):
        cm.current_stage = len(STAGES) - 1
        for _ in range(cm.eval_window):
            cm.record_episode(True)
        advanced = cm.maybe_advance()
        assert not advanced
        assert cm.current_stage == len(STAGES) - 1

    def test_successes_cleared_after_advance(self, cm):
        n = cm.eval_window
        for _ in range(n):
            cm.record_episode(True)
        cm.maybe_advance()
        assert cm.success_rate == pytest.approx(0.0)

    def test_is_final_stage_at_last_stage(self, cm):
        cm.current_stage = len(STAGES) - 1
        assert cm.is_final_stage


# ── sample_spawn ──────────────────────────────────────────────────────────────

class TestSampleSpawn:
    def test_stage_zero_uses_bootstrap_spawns(self, cm, rng):
        for _ in range(20):
            spawn = cm.sample_spawn(rng)
            assert spawn in BOOTSTRAP_SPAWNS

    def test_stage_one_uses_all_spawns(self, cm, rng):
        cm.current_stage = 1
        spawns = {cm.sample_spawn(rng) for _ in range(100)}
        # Should eventually hit spawns outside OPEN_SPAWNS
        extra = set(ALL_SPAWNS) - set(OPEN_SPAWNS)
        assert spawns & extra  # at least one extra spawn seen

    def test_returns_tuple_of_two_floats(self, cm, rng):
        spawn = cm.sample_spawn(rng)
        assert len(spawn) == 2
        assert isinstance(spawn[0], float)
        assert isinstance(spawn[1], float)


# ── sample_goal ───────────────────────────────────────────────────────────────

class TestSampleGoal:
    def test_goal_distance_in_range_stage_zero(self, cm, rng):
        spawn = (0.0, 0.0)
        for _ in range(30):
            goal = cm.sample_goal(spawn[0], spawn[1], rng)
            dist = math.hypot(goal[0] - spawn[0], goal[1] - spawn[1])
            cfg = cm.config
            assert cfg.goal_dist_min <= dist <= cfg.goal_dist_max + 1e-6

    def test_goal_distance_in_range_stage_two(self, cm, rng):
        cm.current_stage = 2
        spawn = (0.0, 0.0)
        for _ in range(30):
            goal = cm.sample_goal(spawn[0], spawn[1], rng)
            dist = math.hypot(goal[0] - spawn[0], goal[1] - spawn[1])
            cfg = cm.config
            assert cfg.goal_dist_min <= dist <= cfg.goal_dist_max + 1e-6

    def test_goal_within_maze_bounds(self, cm, rng):
        """Goals must always be within MAZE_BOUNDS regardless of spawn position."""
        test_spawns = [(-11.0, -11.0), (7.0, 7.0), (0.0, 0.0)]
        for spawn in test_spawns:
            for _ in range(30):
                goal = cm.sample_goal(spawn[0], spawn[1], rng)
                assert abs(goal[0]) <= MAZE_BOUNDS
                assert abs(goal[1]) <= MAZE_BOUNDS

    def test_bad_spawn_removed(self):
        """(-8.0, -8.0) must not appear in any spawn list (it is inside a wall)."""
        assert (-8.0, -8.0) not in OPEN_SPAWNS
        assert (-8.0, -8.0) not in BOOTSTRAP_SPAWNS
        assert (-8.0, -8.0) not in ALL_SPAWNS


# ── sample_dynamics ──────────────────────────────────────────────────────────

class TestSampleDynamics:
    def test_stage_zero_no_randomisation(self, cm, rng):
        d = cm.sample_dynamics(rng)
        assert d['vel_scale'] == pytest.approx(1.0)
        assert d['action_delay_steps'] == 0
        assert 0.0 <= d['scan_noise_sigma'] <= STAGES[0].scan_noise_sigma_max
        assert 0.0 <= d['odom_noise_sigma'] <= STAGES[0].odom_noise_sigma_max

    def test_stage_three_has_randomisation(self, cm, rng):
        cm.current_stage = 3  # stage 3 is where dynamics randomization is enabled
        vel_scales = [cm.sample_dynamics(rng)['vel_scale'] for _ in range(50)]
        # Should see variation in vel_scale (not always 1.0)
        assert min(vel_scales) < 1.0 or max(vel_scales) > 1.0

    def test_stage_two_vel_scale_in_range(self, cm, rng):
        cm.current_stage = 2
        stage = STAGES[2]
        for _ in range(30):
            d = cm.sample_dynamics(rng)
            assert stage.vel_scale_range[0] <= d['vel_scale'] <= stage.vel_scale_range[1]

    def test_stage_two_delay_in_range(self, cm, rng):
        cm.current_stage = 2
        stage = STAGES[2]
        delays = {cm.sample_dynamics(rng)['action_delay_steps'] for _ in range(50)}
        assert delays <= set(range(stage.action_delay_max_steps + 1))

    def test_dynamics_dict_has_required_keys(self, cm, rng):
        d = cm.sample_dynamics(rng)
        assert {'vel_scale', 'scan_noise_sigma', 'odom_noise_sigma',
                'action_delay_steps'} <= set(d.keys())
