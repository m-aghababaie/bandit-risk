"""Week 1 experiment — cumulative regret of ε-greedy across two environments.

Compares three epsilon values on InspectionBanditEnv and ESGBanditEnv,
averaged over multiple seeds. Produces a side-by-side regret chart.

Regret at step t = μ* − μ(aₜ), where μ* is the optimal arm's true mean and
μ(aₜ) is the true mean of the arm actually chosen. Computed via the env's
public API (`optimal_mean`, `true_mean`) — no private attributes touched.

Run:
    python experiments/run_week1.py
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

from bandit_risk import EpsilonGreedyAgent, InspectionBanditEnv, ESGBanditEnv

N_STEPS = 1000
SEEDS = [0, 1, 2, 3, 4]
EPSILONS = [0.01, 0.10, 0.30]


def run_one(env_cls, epsilon: int, seed: int) -> np.ndarray:
    """Run a single agent/env episode and return cumulative regret."""
    env = env_cls(seed=seed)
    agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=epsilon, seed=seed)
    mu_star = env.optimal_mean

    step_regret = np.empty(N_STEPS, dtype=float)
    for t in range(N_STEPS):
        arm = agent.select()
        reward = env.step(arm)
        agent.update(arm, reward)
        # Regret uses TRUE means, not observed rewards
        step_regret[t] = mu_star - env.true_mean(arm)

    return np.cumsum(step_regret)


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    envs = [
        (InspectionBanditEnv, "Inspection Strategy"),
        (ESGBanditEnv, "ESG Retrofit"),
    ]

    for ax, (env_cls, title) in zip(axes, envs):
        for eps in EPSILONS:
            curves = np.array([run_one(env_cls, eps, s) for s in SEEDS])
            mean_r = curves.mean(axis=0)
            std_r = curves.std(axis=0)
            ax.plot(mean_r, label=f"ε = {eps}")
            ax.fill_between(
                range(N_STEPS), mean_r - std_r, mean_r + std_r, alpha=0.15
            )
        ax.set_title(f"{title} — cumulative regret")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Cumulative regret")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "plots/week1_epsilon_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
