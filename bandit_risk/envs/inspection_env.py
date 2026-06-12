"""InspectionBanditEnv — 5-arm bandit for insurance inspection strategy selection.

Domain context
--------------
A commercial property insurer deploys surveyors across a portfolio of buildings.
Each building visit uses one of five inspection strategies. The insurer wants to
learn which strategy delivers the best loss-prevention ROI — measured as a
normalised score combining claim reduction potential and surveyor efficiency.

Why this is a genuine bandit problem
--------------------------------------
Each inspection decision is *stateless*: the reward from today's fire-safety
inspection does not alter the reward distribution of tomorrow's structural
inspection on a different building. There is no evolving state that the agent
must track across steps. This satisfies the defining criterion of a multi-armed
bandit vs an MDP.

Contrast with MDPs: a maintenance scheduling problem *is* an MDP because
choosing to defer roof maintenance today changes the condition of the roof
tomorrow — the state evolves. Inspection strategy selection does not.

Arms
----
| Arm | Strategy                  | True mean | Std  |
|-----|---------------------------|-----------|------|
|  0  | Full structural           | 0.70      | 0.20 | ← true best arm
|  1  | Fire safety only          | 0.55      | 0.15 |
|  2  | Water & leak focus        | 0.63      | 0.20 |
|  3  | ESG audit                 | 0.45      | 0.12 |
|  4  | Rapid drive-by            | 0.30      | 0.10 | ← worst arm

The agent does not know these means. It must discover Arm 0 through exploration.

Reference: Sutton & Barto, *Reinforcement Learning* (2nd ed.), Chapter 2.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng


class InspectionBanditEnv:
    """5-arm stateless bandit environment for inspection strategy selection.

    Each call to :meth:`step` simulates deploying one inspection strategy to a
    randomly drawn commercial property and returning a loss-prevention reward.
    Rewards are drawn from a Gaussian distribution whose mean and standard
    deviation are fixed per arm (unknown to the agent).

    Parameters
    ----------
    seed : int | None, optional
        Random seed for reproducibility. Default None.

    Attributes
    ----------
    n_arms : int
        Number of available inspection strategies (5).
    t : int
        Total number of steps taken since last reset.
    """

    # ---------------------------------------------------------------
    # Arm definitions — domain semantics live here, not in the agent
    # ---------------------------------------------------------------
    ARM_LABELS: list[str] = [
        "Full structural inspection",   # 0 — true best
        "Fire safety only",             # 1
        "Water & leak focus",           # 2
        "ESG audit",                    # 3
        "Rapid drive-by",               # 4 — worst
    ]

    _MEANS: list[float] = [0.70, 0.55, 0.63, 0.45, 0.30]
    _STDS:  list[float] = [0.20, 0.15, 0.20, 0.12, 0.10]

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
        """Pull *arm* and return a stochastic loss-prevention reward.

        The reward is drawn from a Gaussian distribution with the true mean
        and standard deviation for that arm. Rewards are clipped to [0, 1]
        to represent a normalised loss-prevention score.

        Parameters
        ----------
        arm : int
            Index of the inspection strategy to deploy, in ``[0, n_arms)``.

        Returns
        -------
        float
            Reward in ``[0, 1]``.  Higher = better loss-prevention ROI.

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
        """Reset the step counter.

        The random state is *not* reset so that sequential experiments remain
        independent.
        """
        self.t = 0

    # ---------------------------------------------------------------
    # Introspection helpers
    # ---------------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """Return a human-readable description of all arms.

        Useful for debugging, experiment logging, and social content.

        Returns
        -------
        dict
            Keys: ``'n_arms'``, ``'arms'`` (list of dicts with ``label``,
            ``true_mean``, ``true_std``), ``'best_arm'``.

        Examples
        --------
        >>> env = InspectionBanditEnv()
        >>> info = env.describe()
        >>> info['best_arm']
        0
        """
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

    @property
    def best_arm(self) -> int:
        """Index of the arm with the highest true mean reward.

        Returns
        -------
        int
            Always 0 (Full structural inspection, mean=0.70).
        """
        return int(np.argmax(self._means))

    # ---------------------------------------------------------------
    # Dunder helpers
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InspectionBanditEnv("
            f"n_arms={self.n_arms}, "
            f"best_arm={self.best_arm}, "
            f"t={self.t})"
        )
