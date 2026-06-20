"""bandit_risk.utils.plotting — Reusable plotting helpers for experiments.

Centralises chart code so experiment scripts don't duplicate matplotlib
boilerplate. All functions create their output directory automatically and
close figures to avoid memory leaks across repeated runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required (CI-safe)
import matplotlib.pyplot as plt
import numpy as np


def plot_regret(
    results: Mapping[str, np.ndarray],
    title: str,
    save_path: str | Path,
    xlabel: str = "Steps",
    ylabel: str = "Cumulative regret",
) -> None:
    """Plot cumulative-regret curves for multiple agents or epsilon values.

    Parameters
    ----------
    results : Mapping[str, np.ndarray]
        Mapping from curve label to a 1-D cumulative-regret array. Each array
        normally represents mean cumulative regret across seeds. All arrays
        should share the same length (number of steps).
    title : str
        Plot title.
    save_path : str | Path
        Destination PNG path. Parent directories are created if missing.
    xlabel : str, optional
        X-axis label. Default ``"Steps"``.
    ylabel : str, optional
        Y-axis label. Default ``"Cumulative regret"``.

    Returns
    -------
    None
        Writes a PNG to ``save_path`` and prints a confirmation line.

    Examples
    --------
    >>> import numpy as np
    >>> curves = {"eps=0.1": np.cumsum(np.full(100, 0.05))}
    >>> plot_regret(curves, "Demo", "plots/demo.png")   # doctest: +SKIP
    Saved plots/demo.png
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, regret in results.items():
        ax.plot(np.asarray(regret), label=label, linewidth=2)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)  # release figure memory — important across many runs

    print(f"Saved {save_path}")
