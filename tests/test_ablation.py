
"""Unit tests for stereo-method ablation analysis."""

from math import sqrt
from types import SimpleNamespace

import pytest

from slam_for_ada.eval.ablation import (
    best_method,
    compare_feature_errors,
    compare_methods,
    group_by_method,
)

def make_result(method, feature, error, correct=True):
    """Create a mock measurement with a known error."""
    return SimpleNamespace(
        estimate=SimpleNamespace(
            method=SimpleNamespace(value=method),
            feature_id=feature,
        ),
        error_in=error,
        abs_error_in=abs(error),
        decision_correct=correct,
    )

def test_group_by_method():
    results = [
        make_result("sgbm", "door_1", 2.0),
        make_result("learned", "door_1", 1.0),
        make_result("sgbm", "door_2", -3.0),
        make_result("hybrid", "door_1", 0.5),
    ]

    groups = group_by_method(results)

    assert len(groups["sgbm"]) == 2
    assert len(groups["learned"]) == 1
    assert len(groups["hybrid"]) == 1

def test_compare_methods():
    results = {
        "sgbm": [
            make_result("sgbm", "door_1", 2.0),
            make_result("sgbm", "door_2", -4.0),
        ],
        "learned": [
            make_result("learned", "door_1", 1.0),
            make_result("learned", "door_2", -1.0),
        ],
        "hybrid": [
            make_result("hybrid", "door_1", 0.5),
            make_result("hybrid", "door_2", -0.5),
        ],
    }

    summaries = compare_methods(results)

    assert summaries["sgbm"].mean_abs_error_in == pytest.approx(3.0)
    assert summaries["sgbm"].rmse_in == pytest.approx(sqrt(10))
    assert summaries["learned"].mean_abs_error_in == pytest.approx(1.0)
    assert summaries["hybrid"].mean_abs_error_in == pytest.approx(0.5)

    assert best_method(summaries) == "hybrid"

def test_compare_feature_errors():
    results = {
        "sgbm": [
            make_result("sgbm", "door_1", -3.0),
            make_result("sgbm", "door_2", 2.0),
        ],
        "learned": [
            make_result("learned", "door_1", 1.0),
        ],
    }

    comparison = compare_feature_errors(results)

    assert comparison["sgbm"]["door_1"] == pytest.approx(3.0)
    assert comparison["sgbm"]["door_2"] == pytest.approx(2.0)
    assert comparison["learned"]["door_1"] == pytest.approx(1.0)

def test_empty_comparisons():
    assert compare_methods({}) == {}
    assert group_by_method([]) == {}
    assert best_method({}) is None

def test_decision_counts():
    results = {
        "sgbm": [
            make_result("sgbm", "door_1", 1.0, True),
            make_result("sgbm", "door_2", 2.0, False),
            make_result("sgbm", "outlet", 0.5, None),
        ],
    }

    summary = compare_methods(results)["sgbm"]

    assert summary.n == 3
    assert summary.n_decisions == 2
    assert summary.n_correct == 1
