"""Week 1 experiment — cumulative regret of ε-greedy across two environments.

Compares three epsilon values on InspectionBanditEnv and ESGBanditEnv,
averaged over multiple seeds. Produces TWO separate regret charts via the
shared plotting utility (one per environment).

Regret at step t = μ* − μ(aₜ), where μ* is the optimal arm's true mean and
μ(aₜ) is the true mean of the arm actually chosen — computed via each env's
public API (`optimal_mean`, `true_mean`), never private attributes.

Run:
    python -m experiments.run_week1
"""

from __future__ import annotations

import numpy as np

from bandit_risk import EpsilonGreedyAgent, ESGBanditEnv, InspectionBanditEnv
from bandit_risk.utils.plotting import plot_regret

N_STEPS = 1000
SEEDS = [0, 1, 2, 3, 4]
EPSILONS = [0.01, 0.10, 0.30]


def run_one(env_cls, epsilon: float, seed: int) -> np.ndarray:
    """Run a single agent/env episode and return its cumulative-regret curve."""
    env = env_cls(seed=seed)
    agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=epsilon, seed=seed)
    mu_star = env.optimal_mean

    step_regret = np.empty(N_STEPS, dtype=float)
    for t in range(N_STEPS):
        arm = agent.select()
        reward = env.step(arm)
        agent.update(arm, reward)
        step_regret[t] = mu_star - env.true_mean(arm)  # true-mean regret

    return np.cumsum(step_regret)


def mean_regret_by_epsilon(env_cls) -> dict[str, np.ndarray]:
    """Build a {label: mean_cumulative_regret} mapping for one environment."""
    results: dict[str, np.ndarray] = {}
    for eps in EPSILONS:
        curves = np.array([run_one(env_cls, eps, s) for s in SEEDS])
        results[f"epsilon = {eps}"] = curves.mean(axis=0)
    return results


def main() -> None:
    experiments = [
        (
            InspectionBanditEnv,
            "Inspection Strategy - cumulative regret",
            "plots/week1_inspection_regret.png",
        ),
        (
            ESGBanditEnv,
            "ESG Retrofit - cumulative regret",
            "plots/week1_esg_regret.png",
        ),
    ]

    for env_cls, title, out_path in experiments:
        results = mean_regret_by_epsilon(env_cls)
        plot_regret(results, title=title, save_path=out_path)


if __name__ == "__main__":
    main()
