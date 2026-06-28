"""Tests for InspectionBanditEnv.

Run with:
    python -m pytest tests/test_inspection_env.py -v
"""

import pytest

from bandit_risk import InspectionBanditEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> InspectionBanditEnv:
    """Seeded 5-arm inspection environment."""
    return InspectionBanditEnv(seed=42)


# ---------------------------------------------------------------------------
# Test 1 — step() returns a float in [0, 1]
# ---------------------------------------------------------------------------


def test_step_returns_float(env: InspectionBanditEnv) -> None:
    """step() must return a float for every arm."""
    for arm in range(env.n_arms):
        reward = env.step(arm)
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"


def test_step_reward_in_range(env: InspectionBanditEnv) -> None:
    """Rewards must be clipped to [0, 1]."""
    for _ in range(300):
        arm = int(env._rng.integers(0, env.n_arms))
        reward = env.step(arm)
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 2 — best_arm is Arm 0
# ---------------------------------------------------------------------------


def test_best_arm_is_zero(env: InspectionBanditEnv) -> None:
    """best_arm must equal 0 (Full structural, true mean=0.70)."""
    assert env.best_arm == 0, f"Expected best_arm=0, got {env.best_arm}"


def test_best_arm_returns_int(env: InspectionBanditEnv) -> None:
    """best_arm must be a Python int."""
    assert isinstance(env.best_arm, int)


# ---------------------------------------------------------------------------
# Test 3 — step() increments t
# ---------------------------------------------------------------------------


def test_step_increments_t(env: InspectionBanditEnv) -> None:
    """Each step() call must increment the step counter."""
    assert env.t == 0
    env.step(0)
    assert env.t == 1
    env.step(2)
    assert env.t == 2


# ---------------------------------------------------------------------------
# Test 4 — reset() zeroes t
# ---------------------------------------------------------------------------


def test_reset_zeroes_t(env: InspectionBanditEnv) -> None:
    """reset() must bring t back to 0."""
    for arm in range(env.n_arms):
        env.step(arm)
    assert env.t == env.n_arms
    env.reset()
    assert env.t == 0


# ---------------------------------------------------------------------------
# Test 5 — invalid arm raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_arm_raises(env: InspectionBanditEnv) -> None:
    """step() with out-of-range arm must raise ValueError."""
    with pytest.raises(ValueError):
        env.step(5)

    with pytest.raises(ValueError):
        env.step(-1)


# ---------------------------------------------------------------------------
# Test 6 — describe() structure
# ---------------------------------------------------------------------------


def test_describe_structure(env: InspectionBanditEnv) -> None:
    """describe() must return correct keys and arm count."""
    info = env.describe()
    assert info["n_arms"] == 5
    assert info["best_arm"] == 0
    assert len(info["arms"]) == 5
    first = info["arms"][0]
    assert first["arm"] == 0
    assert first["label"] == "Full structural inspection"
    assert first["true_mean"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Test 7 — n_arms is 5
# ---------------------------------------------------------------------------


def test_n_arms(env: InspectionBanditEnv) -> None:
    """InspectionBanditEnv must have exactly 5 arms."""
    assert env.n_arms == 5


# ---------------------------------------------------------------------------
# Integration — agent + environment interact correctly
# ---------------------------------------------------------------------------


def test_agent_env_loop() -> None:
    """EpsilonGreedyAgent must be able to run 100 steps in InspectionBanditEnv."""
    from bandit_risk import EpsilonGreedyAgent

    env = InspectionBanditEnv(seed=0)
    agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=0.1, seed=0)

    for _ in range(100):
        arm = agent.select()
        reward = env.step(arm)
        agent.update(arm, reward)

    # After 100 steps with epsilon=0.1, agent should lean toward best arm
    assert agent.best_arm in range(env.n_arms)
    assert env.t == 100
