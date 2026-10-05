"""Shared pytest fixtures for the vertax test suite."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every matplotlib figure after each test.

    Left-open figures accumulate across the suite; past matplotlib's default limit of 20 open
    figures, each new `plt.figure()`/`plt.subplots()` call raises a `RuntimeWarning`, which is
    fatal under this project's `filterwarnings = ["error"]`.
    """
    yield
    plt.close("all")
