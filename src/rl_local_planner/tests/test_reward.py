"""Unit tests for rl_local_planner.reward."""

from __future__ import annotations

import numpy as np
import pytest

from rl_local_planner.reward import RewardState, RewardWeights, compute_reward


@pytest.fixture
def weights():
    return RewardWeights()


@pytest.fixture
def state():
    return RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0)


def _call(
    curr_dist: float = 4.5,
    min_scan: float = 2.0,
    action=None,
    goal_reached: bool = False,
    collision: bool = False,
    state: RewardState | None = None,
    weights: RewardWeights | None = None,
):
    if action is None:
        action = np.zeros(3, dtype=np.float32)
    if state is None:
        state = RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0)
    if weights is None:
        weights = RewardWeights()
    return compute_reward(curr_dist, min_scan, action, goal_reached, collision, state, weights)


class TestProgressReward:
    def test_positive_progress(self, weights, state):
        """Moving 0.5 m closer from 5 m → progress_frac = 0.5/5 = 0.1."""
        reward, info = compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        expected_progress = weights.progress * (0.5 / 5.0)
        assert info['r_progress'] == pytest.approx(expected_progress, rel=1e-5)

    def test_negative_progress(self, weights, state):
        """Moving 0.5 m further away → negative progress reward."""
        reward, info = compute_reward(5.5, 2.0, np.zeros(3), False, False, state, weights)
        expected_progress = weights.progress * (-0.5 / 5.0)
        assert info['r_progress'] == pytest.approx(expected_progress, rel=1e-5)

    def test_zero_progress(self, weights, state):
        reward, info = compute_reward(5.0, 2.0, np.zeros(3), False, False, state, weights)
        assert info['r_progress'] == pytest.approx(0.0, abs=1e-7)

    def test_normalised_by_initial_distance(self):
        """Same absolute progress but smaller initial_distance → larger fraction."""
        s1 = RewardState(initial_goal_distance=10.0, prev_goal_distance=10.0)
        s2 = RewardState(initial_goal_distance=2.0, prev_goal_distance=2.0)
        w = RewardWeights()
        _, info1 = compute_reward(9.0, 2.0, np.zeros(3), False, False, s1, w)
        _, info2 = compute_reward(1.0, 2.0, np.zeros(3), False, False, s2, w)
        assert abs(info2['r_progress']) > abs(info1['r_progress'])


class TestTerminalBonuses:
    def test_goal_reached_bonus(self, weights, state):
        _, info = compute_reward(0.3, 2.0, np.zeros(3), True, False, state, weights)
        assert info['r_goal'] == weights.goal_reached

    def test_no_goal_bonus_when_not_reached(self, weights, state):
        _, info = compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        assert info['r_goal'] == 0.0

    def test_collision_penalty(self, weights, state):
        _, info = compute_reward(4.5, 0.1, np.zeros(3), False, True, state, weights)
        assert info['r_collision'] == weights.collision

    def test_no_collision_penalty_when_clear(self, weights, state):
        _, info = compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        assert info['r_collision'] == 0.0


class TestProximityPenalty:
    def test_penalty_when_too_close(self, weights, state):
        """min_scan below threshold (0.4 m) → proximity penalty."""
        min_scan = 0.2  # 0.2 m below 0.4 threshold
        _, info = compute_reward(4.5, min_scan, np.zeros(3), False, False, state, weights)
        expected = weights.proximity * (weights.proximity_threshold - min_scan)
        assert info['r_proximity'] == pytest.approx(expected, rel=1e-5)

    def test_no_penalty_when_clear(self, weights, state):
        _, info = compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        assert info['r_proximity'] == pytest.approx(0.0, abs=1e-7)

    def test_exactly_at_threshold(self, weights, state):
        _, info = compute_reward(4.5, weights.proximity_threshold, np.zeros(3),
                                 False, False, state, weights)
        assert info['r_proximity'] == pytest.approx(0.0, abs=1e-7)


class TestSmoothnessPenalty:
    def test_no_penalty_on_first_step(self, weights):
        """prev_action=None → smoothness penalty is 0."""
        state = RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0,
                            prev_action=None)
        _, info = compute_reward(4.5, 2.0, np.array([1.0, 0.0, 0.0]), False, False,
                                 state, weights)
        assert info['r_smooth'] == pytest.approx(0.0, abs=1e-7)

    def test_penalty_on_action_change(self, weights):
        prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        curr = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state = RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0,
                            prev_action=prev)
        _, info = compute_reward(4.5, 2.0, curr, False, False, state, weights)
        delta = float(np.linalg.norm(curr - prev))
        expected = weights.smoothness * delta
        assert info['r_smooth'] == pytest.approx(expected, rel=1e-5)

    def test_zero_penalty_for_same_action(self, weights):
        action = np.array([0.5, 0.0, 0.5], dtype=np.float32)
        state = RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0,
                            prev_action=action.copy())
        _, info = compute_reward(4.5, 2.0, action.copy(), False, False, state, weights)
        assert info['r_smooth'] == pytest.approx(0.0, abs=1e-7)


class TestStepCost:
    def test_step_cost_always_applied(self, weights, state):
        _, info = compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        assert info['r_step'] == pytest.approx(weights.step_cost, rel=1e-5)


class TestStateMutation:
    def test_prev_goal_distance_updated(self, weights):
        state = RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0)
        compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        assert state.prev_goal_distance == pytest.approx(4.5)

    def test_prev_action_updated(self, weights):
        state = RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0)
        action = np.array([0.3, 0.1, 0.2], dtype=np.float32)
        compute_reward(4.5, 2.0, action, False, False, state, weights)
        np.testing.assert_array_equal(state.prev_action, action)

    def test_prev_action_is_copy(self, weights):
        """Mutating action after call should not affect stored prev_action."""
        state = RewardState(initial_goal_distance=5.0, prev_goal_distance=5.0)
        action = np.array([0.3, 0.1, 0.2], dtype=np.float32)
        compute_reward(4.5, 2.0, action, False, False, state, weights)
        action[:] = 0.0
        assert not np.all(state.prev_action == 0.0)


class TestReturnTypes:
    def test_reward_is_float(self, weights, state):
        reward, _ = compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        assert isinstance(reward, float)

    def test_info_keys(self, weights, state):
        _, info = compute_reward(4.5, 2.0, np.zeros(3), False, False, state, weights)
        assert set(info.keys()) == {
            'r_progress', 'r_heading', 'r_near_goal', 'r_goal', 'r_collision',
            'r_proximity', 'r_smooth', 'r_step'
        }
