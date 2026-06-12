"""ESGBanditEnv — 6-arm bandit for ESG retrofit recommendation.

Domain context
--------------
A building owner facing MEES (Minimum Energy Efficiency Standards) compliance
must choose one retrofit measure per budget cycle. A recommendation engine wants
to learn which measure delivers the best outcome — modelled as a normalised score
combining energy-performance uplift, carbon reduction, and cost effectiveness.

Why this is a genuine bandit problem
--------------------------------------
Each recommendation is a *single-shot* decision for one building in one budget
cycle. The reward from recommending solar PV does not depend on what was
recommended in a previous cycle — there is no evolving state the agent must
track. This satisfies the multi-armed bandit criterion (reward depends on the
action alone, not on history). Rewards are stationary.

The counterintuitive result
----------------------------
Rooftop solar PV (Arm 1) is the *worst* arm, while the heat pump (Arm 2) is the
best. This reflects the reality that for many UK commercial buildings, heat-pump
retrofits deliver larger MEES-band uplift per pound than solar — a result the
agent discovers from data alone.

Arms
----
| Arm | Retrofit measure       | True mean | Std  |
|-----|------------------------|-----------|------|
|  0  | Cavity wall insulation | 0.52      | 0.12 |
|  1  | Rooftop solar PV       | 0.38      | 0.15 | ← counterintuitive loser
|  2  | Heat pump (ASHP)       | 0.71      | 0.18 | ← true best arm
|  3  | LED + controls         | 0.42      | 0.10 |
|  4  | BMS optimisation       | 0.68      | 0.14 |
|  5  | ASHP upgrade           | 0.55      | 0.16 |

Note on forced exploration
--------------------------
An earlier design considered a "MEES deadline" reward bonus that increased the
payoff of under-explored arms as a budget deadline approached. That was removed:
making reward depend on pull-counts or remaining steps would turn this into a
non-stationary / budgeted bandit (cf. Badanidiyuru et al., "Bandits with
Knapsacks", 2013) and break the stationary-bandit guarantee this library is
built on. Deadline-aware exploration belongs in the *agent's* policy, not in the
environment's reward — and is explored separately, not claimed as novel.

Reference: Sutton & Barto, *Reinforcement Learning* (2nd ed.), Chapter 2.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng


class ESGBanditEnv:
    """6-arm stationary bandit environment for ESG retrofit recommendation.

    Mirrors the interface of :class:`InspectionBanditEnv` so the two are
    interchangeable in experiments.

    Parameters
    ----------
    seed : int | None, optional
        Random seed for reproducibility. Default None.

    Attributes
    ----------
    n_arms : int
        Number of retrofit measures (6).
    t : int
        Number of steps taken since last reset.
    """

    ARM_LABELS: list[str] = [
        "Cavity wall insulation",   # 0
        "Rooftop solar PV",         # 1 — counterintuitive loser
        "Heat pump (ASHP)",         # 2 — true best
        "LED + controls",           # 3
        "BMS optimisation",         # 4
        "ASHP upgrade",             # 5
    ]

    _MEANS: list[float] = [0.52, 0.38, 0.71, 0.42, 0.68, 0.55]
    _STDS:  list[float] = [0.12, 0.15, 0.18, 0.10, 0.14, 0.16]

    def __init__(self, seed: int | None = None) -> None:
        self.n_arms: int = len(self.ARM_LABELS)
        self._means = np.array(self._MEANS, dtype=float)
        self._stds  = np.array(self._STDS,  dtype=float)
        self._rng   = default_rng(seed)
        self.t: int = 0

    # ---------------------------------------------------------------
    # Core bandit interface
    # ---------------------------------------------------------------

    def step(self, arm: int) -> float:
        """Pull *arm* and return a stochastic ESG-outcome reward.

        Parameters
        ----------
        arm : int
            Index of the retrofit measure, in ``[0, n_arms)``.

        Returns
        -------
        float
            Reward in ``[0, 1]``. Higher = better ESG outcome per pound.

        Raises
        ------
        ValueError
            If ``arm`` is outside ``[0, n_arms)``.
        """
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        self.t += 1
        reward = self._rng.normal(self._means[arm], self._stds[arm])
        return float(np.clip(reward, 0.0, 1.0))

    def reset(self) -> None:
        """Reset the step counter. Random state is not reset."""
        self.t = 0

    # ---------------------------------------------------------------
    # Introspection helpers
    # ---------------------------------------------------------------

    def true_mean(self, arm: int) -> float:
        """Return the true mean reward for *arm* (for regret computation).

        Exposed as a method so experiments can compute regret without
        touching private attributes.
        """
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        return float(self._means[arm])

    @property
    def optimal_mean(self) -> float:
        """The highest true mean across all arms (μ*) — used for regret."""
        return float(np.max(self._means))

    def describe(self) -> dict[str, object]:
        """Return a human-readable description of all arms."""
        arms = [
            {
                "arm":       i,
                "label":     self.ARM_LABELS[i],
                "true_mean": round(float(self._means[i]), 3),
                "true_std":  round(float(self._stds[i]),  3),
            }
            for i in range(self.n_arms)
        ]
        return {
            "n_arms":   self.n_arms,
            "arms":     arms,
            "best_arm": int(self.best_arm),
        }

    @property
    def best_arm(self) -> int:
        """Index of the arm with the highest true mean (Arm 2, heat pump)."""
        return int(np.argmax(self._means))

    def __repr__(self) -> str:
        return (
            f"ESGBanditEnv(n_arms={self.n_arms}, "
            f"best_arm={self.best_arm}, t={self.t})"
        )
