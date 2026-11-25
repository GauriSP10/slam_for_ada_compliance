"""Evaluation metrics for ADA compliance measurements."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import mean, median

from slam_for_ada.datatypes import ComplianceResult, ErrorSummary

def summarize_results(
    results: Sequence[ComplianceResult],
    group: str = "all",
) -> ErrorSummary:
    """Summarize measurement errors and ADA decision accuracy."""

    if not results:
        raise ValueError("Cannot evaluate an empty set of results")

    errors = [r.error_in for r in results]
    absolute_errors = [abs(e) for e in errors]

    decisions = [
        r.decision_correct
        for r in results
        if r.decision_correct is not None
    ]

    return ErrorSummary(
        group=group,
        n=len(results),
        mean_abs_error_in=mean(absolute_errors),
        median_abs_error_in=median(absolute_errors),
        max_abs_error_in=max(absolute_errors),
        rmse_in=sqrt(mean(e**2 for e in errors)),
        n_decisions=len(decisions),
        n_correct=sum(decisions),
    )

def decision_accuracy(summary: ErrorSummary) -> float | None:
    """Return ADA decision accuracy, or None if no rules apply."""
    if summary.n_decisions == 0:
        return None

    return summary.n_correct / summary.n_decisions
