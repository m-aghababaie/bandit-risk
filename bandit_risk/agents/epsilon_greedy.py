"""ε-Greedy agent with incremental mean update and epsilon decay.

This is the simplest exploration strategy for multi-armed bandit problems.
With probability ε the agent explores (random arm); with probability 1−ε it
exploits (greedy arm with highest estimated value).

Insurance application
---------------------
An underwriter running 5 inspection strategies across a property portfolio is a
classic bandit. Early in the portfolio cycle, high ε encourages trying all
strategies. As data accumulates, epsilon decays and the agent commits to the
strategy with the best observed loss-prevention ROI.

Reference: Sutton & Barto, *Reinforcement Learning* (2nd ed.), Section 2.4.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng


class EpsilonGreedyAgent:
    """ε-Greedy bandit agent with incremental mean update and epsilon decay.

    Parameters
    ----------
    n_arms : int
        Number of available actions (arms).
    epsilon : float, optional
        Initial exploration rate ∈ [0, 1]. Default 0.1.
    epsilon_decay : float, optional
        Multiplicative decay applied to epsilon after every update step.
        Set to 1.0 for no decay. Default 1.0.
    epsilon_min : float, optional
        Floor value; epsilon will never decay below this. Default 0.01.
    seed : int | None, optional
        Random seed for reproducibility. Default None.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}")
        if not 0.0 <= epsilon_min <= epsilon:
            raise ValueError(
                f"epsilon_min must be in [0, epsilon], got {epsilon_min}"
            )

        self.n_arms: int = n_arms
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self._initial_epsilon: float = epsilon  # stored for reset()

        self.q: np.ndarray = np.zeros(n_arms, dtype=float)   # estimated values
        self.n: np.ndarray = np.zeros(n_arms, dtype=float)   # arm pull counts
        self.t: int = 0                                       # total steps

        self._rng = default_rng(seed)

    # ------------------------------------------------------------------
    # Core bandit interface
    # ------------------------------------------------------------------

    def select(self) -> int:
        """Select an arm using the ε-greedy policy.

        With probability ε a random arm is chosen (exploration).
        Otherwise the arm with the highest estimated Q-value is chosen
        (exploitation). Ties in argmax are broken by numpy (lowest index).

        Returns
        -------
        int
            Index of the selected arm in ``[0, n_arms)``.
        """
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self.q))

    def update(self, arm: int, reward: float) -> None:
        """Update Q-estimate for *arm* using incremental mean, then decay ε.

        The incremental mean formula avoids storing all past rewards::

            Q(a) ← Q(a) + (R − Q(a)) / N(a)

        This is equivalent to a sample mean and is numerically stable.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Scalar reward observed after pulling ``arm``.
        """
        self.t += 1
        self.n[arm] += 1
        # Incremental mean update (Sutton & Barto eq. 2.3)
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
        # Epsilon decay — floor at epsilon_min
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        """Reset agent to its initial state.

        Zeroes all Q-estimates, arm counts, and the step counter.
        Restores epsilon to its initial value. Random state is *not* reset.
        """
        self.q = np.zeros(self.n_arms, dtype=float)
        self.n = np.zeros(self.n_arms, dtype=float)
        self.t = 0
        self.epsilon = self._initial_epsilon

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_arm(self) -> int:
        """Index of the arm with the highest current Q-estimate.

        Returns
        -------
        int
            ``np.argmax(self.q)``; ties broken by lowest index.
        """
        return int(np.argmax(self.q))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyAgent("
            f"n_arms={self.n_arms}, "
            f"epsilon={self.epsilon:.4f}, "
            f"epsilon_decay={self.epsilon_decay}, "
            f"epsilon_min={self.epsilon_min})"
        )
