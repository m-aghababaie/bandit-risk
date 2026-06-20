"""Tests for the plot_regret plotting utility.

These tests verify file/directory side-effects without asserting on pixel
content. They use pytest's tmp_path fixture so no files leak into the repo.

Run with:
    python -m pytest tests/test_plotting.py -v
"""

import numpy as np
import pytest

from bandit_risk.utils import plot_regret


def test_plot_regret_creates_file(tmp_path) -> None:
    """plot_regret must write a PNG at the requested path."""
    out = tmp_path / "regret.png"
    curves = {"eps=0.1": np.cumsum(np.full(50, 0.05))}
    plot_regret(curves, title="Test", save_path=out)
    assert out.exists(), "plot_regret did not create the output file"
    assert out.stat().st_size > 0, "output PNG is empty"


def test_plot_regret_creates_parent_dirs(tmp_path) -> None:
    """plot_regret must create missing parent directories automatically."""
    out = tmp_path / "nested" / "deeper" / "regret.png"
    curves = {"eps=0.1": np.cumsum(np.full(20, 0.1))}
    plot_regret(curves, title="Test", save_path=out)
    assert out.exists(), "parent directories were not created"


def test_plot_regret_multiple_curves(tmp_path) -> None:
    """plot_regret must handle several labelled curves without error."""
    out = tmp_path / "multi.png"
    curves = {
        "eps=0.01": np.cumsum(np.full(30, 0.02)),
        "eps=0.10": np.cumsum(np.full(30, 0.05)),
        "eps=0.30": np.cumsum(np.full(30, 0.12)),
    }
    plot_regret(curves, title="Multi", save_path=out)
    assert out.exists()


def test_plot_regret_accepts_str_path(tmp_path) -> None:
    """save_path given as a plain string (not Path) must also work."""
    out = tmp_path / "as_string.png"
    curves = {"a": np.cumsum(np.full(10, 0.1))}
    plot_regret(curves, title="Str path", save_path=str(out))
    assert out.exists()
