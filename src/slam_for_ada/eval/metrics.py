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


def write_error_report(
    results_csv,
    ground_truth_yaml,
    output_csv,
    matcher="sgbm",
):
    """Write error statistics grouped by matcher, method and feature kind."""
    import csv
    from collections import defaultdict
    from pathlib import Path

    import yaml

    with open(ground_truth_yaml, encoding="utf-8") as f:
        truth_data = yaml.safe_load(f)

    kinds = {
        feature["id"]: feature["kind"]
        for feature in truth_data["features"]
    }

    with open(results_csv, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    groups = defaultdict(list)

    for row in results:
        feature_kind = kinds[row["feature"]]
        categories = (
            ("matcher", matcher),
            ("method", row["method"]),
            ("feature_kind", feature_kind),
        )
        for category in categories:
            groups[category].append(row)

    report_rows = []

    for (group_type, group_name), rows in sorted(groups.items()):
        errors = [float(row["error_in"]) for row in rows]
        absolute_errors = [abs(error) for error in errors]
        decisions = [
            row["decision_correct"].strip().lower()
            for row in rows
            if row["decision_correct"].strip().lower()
            in ("yes", "no")
        ]

        n_correct = decisions.count("yes")

        report_rows.append({
            "group_type": group_type,
            "group": group_name,
            "n": len(rows),
            "mean_abs_error_in": mean(absolute_errors),
            "median_abs_error_in": median(absolute_errors),
            "max_abs_error_in": max(absolute_errors),
            "rmse_in": sqrt(mean(error ** 2 for error in errors)),
            "n_decisions": len(decisions),
            "n_correct": n_correct,
            "decision_accuracy": (
                n_correct / len(decisions) if decisions else ""
            ),
        })

    columns = [
        "group_type", "group", "n",
        "mean_abs_error_in", "median_abs_error_in",
        "max_abs_error_in", "rmse_in",
        "n_decisions", "n_correct", "decision_accuracy",
    ]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report_rows)

    return report_rows
