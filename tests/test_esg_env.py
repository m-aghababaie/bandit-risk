"""Tests for ESGBanditEnv.

Run with:
    python -m pytest tests/test_esg_env.py -v
"""

import pytest

from bandit_risk import ESGBanditEnv


@pytest.fixture
def env() -> ESGBanditEnv:
    return ESGBanditEnv(seed=42)


def test_n_arms(env: ESGBanditEnv) -> None:
    """ESGBanditEnv must have exactly 6 arms."""
    assert env.n_arms == 6


def test_step_returns_float(env: ESGBanditEnv) -> None:
    for arm in range(env.n_arms):
        assert isinstance(env.step(arm), float)


def test_step_reward_in_range(env: ESGBanditEnv) -> None:
    for _ in range(300):
        arm = int(env._rng.integers(0, env.n_arms))
        r = env.step(arm)
        assert 0.0 <= r <= 1.0


def test_best_arm_is_heat_pump(env: ESGBanditEnv) -> None:
    """Heat pump (Arm 2) must be the true best arm — the counterintuitive result."""
    assert env.best_arm == 2


def test_solar_beats_nothing(env: ESGBanditEnv) -> None:
    """Solar PV (Arm 1) must be the worst arm — the post hook."""
    means = [env.true_mean(i) for i in range(env.n_arms)]
    assert means.index(min(means)) == 1


def test_optimal_mean(env: ESGBanditEnv) -> None:
    """optimal_mean must equal the heat pump's true mean (0.71)."""
    assert env.optimal_mean == pytest.approx(0.71)


def test_true_mean_matches_best_arm(env: ESGBanditEnv) -> None:
    assert env.true_mean(env.best_arm) == pytest.approx(env.optimal_mean)


def test_invalid_arm_raises(env: ESGBanditEnv) -> None:
    with pytest.raises(ValueError):
        env.step(6)
    with pytest.raises(ValueError):
        env.true_mean(-1)


def test_describe_structure(env: ESGBanditEnv) -> None:
    info = env.describe()
    assert info["n_arms"] == 6
    assert info["best_arm"] == 2
    assert len(info["arms"]) == 6
    assert info["arms"][2]["label"] == "Heat pump (ASHP)"


def test_reset_zeroes_t(env: ESGBanditEnv) -> None:
    for arm in range(env.n_arms):
        env.step(arm)
    env.reset()
    assert env.t == 0


def test_agent_converges_to_heat_pump() -> None:
    """Over many steps, ε-greedy should favour the heat pump arm."""
    from bandit_risk import EpsilonGreedyAgent

    env = ESGBanditEnv(seed=0)
    agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=0.1, seed=0)
    for _ in range(2000):
        arm = agent.select()
        agent.update(arm, env.step(arm))
    # Heat pump should have the highest estimated value
    assert agent.best_arm == 2
