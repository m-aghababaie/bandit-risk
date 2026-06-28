# Changelog

All notable changes to **bandit-risk** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `EpsilonGreedyAgent` — ε-greedy policy with incremental-mean value updates and
  exponential epsilon decay (with floor). Full type hints, docstrings, input
  validation, and seeding for reproducibility.
- `InspectionBanditEnv` — 5-arm stationary bandit for insurance inspection-strategy
  selection. True best arm: full structural inspection (μ=0.70).
- `ESGBanditEnv` — 6-arm stationary bandit for ESG retrofit recommendation. True
  best arm: heat-pump ASHP (μ=0.71); worst: rooftop solar PV (μ=0.38).
- Public regret API on every environment: `true_mean(arm)` and `optimal_mean`.
- `plot_regret` — shared cumulative-regret plotting utility
  (`bandit_risk.utils.plotting`).
- `experiments/run_week1.py` — ε comparison across two environments and five seeds,
  producing per-environment regret charts.
- Documentation: mathematical foundations (`docs/mathematics/day5_mathematics.md`)
  and a genuine-bandit explainer (`docs/genuine_bandits.md`).
- Test suite: 37 tests covering agents, environments, and plotting.
- Tooling: Ruff (lint) and Black (format) configuration; CI lint job.

### Design notes
- Every environment is validated as a **genuine stationary bandit** (reward depends
  only on the action, never on history or elapsed time). A planned "MEES deadline"
  reward bonus was deliberately removed because it would have made the environment
  non-stationary; deadline-aware behaviour belongs in the agent's policy, not the
  environment's reward.

## [0.1.0] — planned (~Day 25)
- First public PyPI release: ε-greedy, UCB1, Thompson Sampling, and LinUCB agents
  across five validated environments.

[Unreleased]: https://github.com/m-aghababaie/bandit-risk/commits/main
