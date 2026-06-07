"""bandit-risk — Multi-armed bandit algorithms for insurance and property intelligence.

A pip-installable open-source library applying reinforcement learning bandit
methods to real-world insurance and PropTech decision problems.

Agents
------
EpsilonGreedyAgent   : ε-greedy with incremental mean update and epsilon decay
UCB1Agent            : Upper Confidence Bound (Week 2)
ThompsonSamplingAgent: Beta-Bernoulli Thompson Sampling (Week 3)
LinUCBAgent          : Contextual LinUCB for IoT-enriched risk (Week 4)

Environments
------------
InspectionBanditEnv       : 5-arm inspection strategy optimisation (Week 1)
ESGBanditEnv              : 6-arm ESG retrofit selection with MEES urgency (Week 1)
OutreachBanditEnv         : 5-arm PropTech GTM channel optimisation (Week 2)
PolicyRecommendationEnv   : 4-arm insurance product selection (Week 3)
DispatchBanditEnv         : 5-arm emergency dispatch routing (Week 3)

Quick start
-----------
>>> from bandit_risk import EpsilonGreedyAgent, InspectionBanditEnv
>>> env   = InspectionBanditEnv(seed=0)
>>> agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=0.1, epsilon_decay=0.995)
>>> for _ in range(500):
...     arm    = agent.select()
...     reward = env.step(arm)
...     agent.update(arm, reward)
>>> print(f"Discovered best arm: {agent.best_arm}")   # should converge to 0
"""

from bandit_risk.agents.epsilon_greedy import EpsilonGreedyAgent
from bandit_risk.envs.inspection_env import InspectionBanditEnv

__version__ = "0.1.0"
__author__ = "Mohammad — Hubbcast / Insurmatics"

__all__ = [
    "EpsilonGreedyAgent",
    "InspectionBanditEnv",
]
