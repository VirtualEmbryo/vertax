"""Shared pytest fixtures for the vertax test suite."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest

EXACT_REFERENCES_VARIABLE = "VERTAX_EXACT_REFERENCES"


@pytest.fixture()
def exact_references() -> bool:
    """Whether to compare with the stored references bit for bit (``VERTAX_EXACT_REFERENCES=1``).

    Some stored references (``tests/reference_result_test_{default_dtype,clamped,trace}.npz``) are
    floating-point results that depend on the machine: XLA's code generation and BLAS sum in an order
    and with fused multiply-adds that differ between CPUs, so another machine reproduces them only to
    the last bits. By default the tests that read them compare within a tolerance derived from the
    computation. Set ``VERTAX_EXACT_REFERENCES=1`` on the machine that produced the references to
    check them bit for bit, which is what those tests were written to establish there (for example,
    that the default float32 path is bit-identical to version 0.2.0).
    """
    return os.environ.get(EXACT_REFERENCES_VARIABLE) == "1"


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every matplotlib figure after each test.

    Left-open figures accumulate across the suite; past matplotlib's default limit of 20 open
    figures, each new `plt.figure()`/`plt.subplots()` call raises a `RuntimeWarning`, which is
    fatal under this project's `filterwarnings = ["error"]`.
    """
    yield
    plt.close("all")
