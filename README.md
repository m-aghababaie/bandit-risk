# bandit-risk

> Multi-armed bandit algorithms for insurance and property intelligence.

[![CI](https://github.com/m-aghababaie/bandit-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/m-aghababaie/bandit-risk/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is this?

Property insurers make hundreds of decisions daily — which inspection strategy to deploy,
which ESG retrofit to recommend, which outreach channel to invest in.

**bandit-risk** applies multi-armed bandit reinforcement learning to these decisions,
learning the optimal policy from experience rather than historical averages.

This is part of the *From Bandits to Buildings* open-source series by
[Mohammad at Hubbcast / Insurmatics](https://github.com/m-aghababaie).

---

## Quick Start

```bash
pip install bandit-risk
```

```python
from bandit_risk import EpsilonGreedyAgent

# 5 inspection strategies — which has the best loss-prevention ROI?
agent = EpsilonGreedyAgent(n_arms=5, epsilon=0.1, epsilon_decay=0.995)

for step in range(1000):
    arm = agent.select()           # choose an inspection strategy
    reward = env.step(arm)         # run inspection, observe reward
    agent.update(arm, reward)      # learn from the outcome

print(f"Best inspection strategy: Arm {agent.best_arm}")
```

---

## The Bandit Test

Every environment in bandit-risk passes the following test:

> **Reward depends only on the action taken, not on prior history.**
> Each decision is independent. There is no evolving state. If there were,
> the problem would be an MDP, not a bandit.

| Environment | Arms | Domain | True Best Arm |
|---|---|---|---|
| `InspectionBanditEnv` | 5 | Loss-prevention inspection strategy | Arm 0 — Full structural (μ=0.70) |
| `ESGBanditEnv` | 6 | MEES retrofit selection | Arm 2 — Heat pump ASHP (μ=0.71) |
| *(more coming in Weeks 2–4)* | | | |

---

## Agents

| Agent | Week | Key idea |
|---|---|---|
| `EpsilonGreedyAgent` | 1 | Explore randomly with prob ε; decay ε over time |
| `UCB1Agent` | 2 | Optimism under uncertainty — prefer untried arms |
| `ThompsonSamplingAgent` | 3 | Bayesian — sample from Beta posterior per arm |
| `LinUCBAgent` | 4 | Contextual — use IoT sensor features to personalise |

---

## Development

```bash
git clone https://github.com/hubbcast-rl/bandit-risk
cd bandit-risk
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Series

This library is built live as part of *From Bandits to Buildings* — a 90-day
open-source RL learning series connecting reinforcement learning to insurance
and property intelligence.

Follow along on [LinkedIn](https://linkedin.com) · [GitHub](https://github.com/m-aghababaie)
