"""Tests for EpsilonGreedyAgent.

Run with:
    pytest tests/test_epsilon_greedy.py -v
"""

import numpy as np
import pytest

from bandit_risk import EpsilonGreedyAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> EpsilonGreedyAgent:
    """A default 5-arm agent, seeded for reproducibility."""
    return EpsilonGreedyAgent(n_arms=5, epsilon=0.1, seed=42)


# ---------------------------------------------------------------------------
# Test 1 — select() returns a valid arm index
# ---------------------------------------------------------------------------


def test_select_returns_valid_arm(agent: EpsilonGreedyAgent) -> None:
    """select() must return an int in [0, n_arms) on every call."""
    for _ in range(200):
        arm = agent.select()
        assert isinstance(arm, int), f"Expected int, got {type(arm)}"
        assert 0 <= arm < agent.n_arms, f"Arm {arm} out of range [0, {agent.n_arms})"


# ---------------------------------------------------------------------------
# Test 2 — update() correctly increments n[arm] and modifies q[arm]
# ---------------------------------------------------------------------------


def test_update_increments_arm_count(agent: EpsilonGreedyAgent) -> None:
    """After one update on arm 2, n[2] must equal 1."""
    agent.update(arm=2, reward=1.0)
    assert agent.n[2] == 1, f"Expected n[2]==1, got {agent.n[2]}"


def test_update_changes_q_value(agent: EpsilonGreedyAgent) -> None:
    """After updating arm 0, q[0] should equal the reward (first pull)."""
    agent.update(arm=0, reward=0.75)
    assert agent.q[0] == pytest.approx(
        0.75
    ), f"Expected q[0]≈0.75 after first pull, got {agent.q[0]}"


def test_update_incremental_mean(agent: EpsilonGreedyAgent) -> None:
    """After two updates, q[arm] should equal the running mean of rewards."""
    agent.update(arm=1, reward=1.0)
    agent.update(arm=1, reward=0.0)
    assert agent.q[1] == pytest.approx(
        0.5
    ), f"Expected incremental mean 0.5, got {agent.q[1]}"


def test_update_does_not_affect_other_arms(agent: EpsilonGreedyAgent) -> None:
    """Updating arm 3 must leave all other arm Q-values at zero."""
    agent.update(arm=3, reward=1.0)
    for i in range(agent.n_arms):
        if i != 3:
            assert agent.q[i] == 0.0, f"q[{i}] should remain 0, got {agent.q[i]}"


# ---------------------------------------------------------------------------
# Test 3 — best_arm returns correct type and index
# ---------------------------------------------------------------------------


def test_best_arm_returns_int(agent: EpsilonGreedyAgent) -> None:
    """best_arm must return an int (not np.intp or similar)."""
    assert isinstance(agent.best_arm, int), f"Expected int, got {type(agent.best_arm)}"


def test_best_arm_reflects_updates(agent: EpsilonGreedyAgent) -> None:
    """best_arm should point to the arm with the highest Q after updates."""
    agent.update(arm=0, reward=0.3)
    agent.update(arm=4, reward=0.9)  # arm 4 should win
    assert agent.best_arm == 4, f"Expected best_arm=4, got {agent.best_arm}"


# ---------------------------------------------------------------------------
# Test 4 — epsilon decay
# ---------------------------------------------------------------------------


def test_epsilon_decays_after_update() -> None:
    """Epsilon must decrease after each update when decay < 1.0."""
    agent = EpsilonGreedyAgent(
        n_arms=3, epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.01
    )
    initial_eps = agent.epsilon
    agent.update(arm=0, reward=1.0)
    assert agent.epsilon < initial_eps, "Epsilon should decay after update"


def test_epsilon_floors_at_epsilon_min() -> None:
    """Epsilon must never go below epsilon_min."""
    agent = EpsilonGreedyAgent(
        n_arms=3, epsilon=0.5, epsilon_decay=0.5, epsilon_min=0.1
    )
    for _ in range(50):
        agent.update(arm=0, reward=1.0)
    assert (
        agent.epsilon >= agent.epsilon_min
    ), f"Epsilon {agent.epsilon} dropped below epsilon_min {agent.epsilon_min}"


# ---------------------------------------------------------------------------
# Test 5 — reset()
# ---------------------------------------------------------------------------


def test_reset_zeroes_state() -> None:
    """After reset(), q, n, t, and epsilon must return to initial values."""
    agent = EpsilonGreedyAgent(
        n_arms=4, epsilon=0.3, epsilon_decay=0.9, epsilon_min=0.01
    )
    for i in range(10):
        agent.update(arm=i % 4, reward=float(i))

    agent.reset()

    assert agent.t == 0
    assert agent.epsilon == pytest.approx(0.3)
    np.testing.assert_array_equal(agent.q, np.zeros(4))
    np.testing.assert_array_equal(agent.n, np.zeros(4))


# ---------------------------------------------------------------------------
# Test 6 — constructor validation
# ---------------------------------------------------------------------------


def test_invalid_epsilon_raises() -> None:
    with pytest.raises(ValueError):
        EpsilonGreedyAgent(n_arms=5, epsilon=1.5)


def test_invalid_epsilon_decay_raises() -> None:
    with pytest.raises(ValueError):
        EpsilonGreedyAgent(n_arms=5, epsilon_decay=0.0)
