
"""Tests for ADA evaluation metrics."""

from math import sqrt
from types import SimpleNamespace

import pytest

from slam_for_ada.eval.metrics import (
    decision_accuracy,
    summarize_results,
)

def make_result(error, correct):
    """Create a small test result with a known error and decision."""
    return SimpleNamespace(
        error_in=error,
        decision_correct=correct,
    )

def test_measurement_errors():
    results = [
        make_result(2.0, True),
        make_result(-3.0, False),
        make_result(1.0, True),
    ]

    summary = summarize_results(results)

    assert summary.n == 3
    assert summary.mean_abs_error_in == pytest.approx(2.0)
    assert summary.median_abs_error_in == pytest.approx(2.0)
    assert summary.max_abs_error_in == pytest.approx(3.0)
    assert summary.rmse_in == pytest.approx(sqrt(14 / 3))

def test_ada_decision_accuracy():
    results = [
        make_result(1.0, True),
        make_result(-2.0, True),
        make_result(3.0, False),
        make_result(0.5, None),
    ]

    summary = summarize_results(results)

    assert summary.n_decisions == 3
    assert summary.n_correct == 2
    assert decision_accuracy(summary) == pytest.approx(2 / 3)

def test_no_ada_rule():
    results = [
        make_result(1.5, None),
        make_result(-0.5, None),
    ]

    summary = summarize_results(results)

    assert summary.n_decisions == 0
    assert summary.n_correct == 0
    assert decision_accuracy(summary) is None

def test_empty_results():
    with pytest.raises(ValueError):
        summarize_results([])
