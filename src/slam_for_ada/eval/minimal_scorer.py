"""A minimal scorer so the vision side can be checked end to end on its own.

It matches each feature estimate to its tape measurement, computes the signed error, and gives the ADA verdict of
both the measurement and the tape value. It satisfies the ``Evaluator`` protocol, so a fuller scorer (breakdowns,
ablations, plots) can replace it without any change to the vision code.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from slam_for_ada.ada.evaluate import measured_verdict, true_verdict
from slam_for_ada.ada.rules import rule_for
from slam_for_ada.datatypes import AdaRule, ComplianceResult, FeatureEstimate, GroundTruthFeature


def score(
    estimates: Sequence[FeatureEstimate], truths: Mapping[str, GroundTruthFeature], rules: Sequence[AdaRule]
) -> list[ComplianceResult]:
    """One ComplianceResult per estimate that has a ground-truth entry."""
    out = []
    for est in estimates:
        gt = truths.get(est.feature_id)
        if gt is None:
            continue
        rule = rule_for(est.feature_id, rules)
        out.append(
            ComplianceResult(
                est,
                gt,
                rule,
                est.value_in - gt.value_in,
                measured_verdict(est.value_in, est.sigma_in, rule),
                true_verdict(gt.value_in, rule),
            )
        )
    return out


def _ok(r: ComplianceResult) -> str:
    c = r.decision_correct
    return "-" if c is None else ("yes" if c else "NO")


def results_rows(results: Sequence[ComplianceResult]) -> list[dict]:
    rows = []
    for r in sorted(results, key=lambda r: (r.estimate.feature_id, r.estimate.method.value)):
        e = r.estimate
        rows.append(
            {
                "feature": e.feature_id,
                "method": e.method.value,
                "n_obs": e.n_obs,
                "measured_in": round(e.value_in, 2),
                "sigma_in": round(e.sigma_in, 2),
                "tape_in": r.truth.value_in,
                "error_in": round(r.error_in, 2),
                "abs_error_in": round(r.abs_error_in, 2),
                "rule": r.rule.id if r.rule else "",
                "measured_verdict": r.measured_verdict.value,
                "tape_verdict": r.true_verdict.value,
                "decision_correct": _ok(r),
            }
        )
    return rows


def format_table(results: Sequence[ComplianceResult]) -> str:
    """Markdown table of every result."""
    rows = results_rows(results)
    cols = [
        "feature",
        "method",
        "n_obs",
        "measured_in",
        "sigma_in",
        "tape_in",
        "error_in",
        "measured_verdict",
        "tape_verdict",
        "decision_correct",
    ]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)


def write_csv(results: Sequence[ComplianceResult], path: Path) -> None:
    rows = results_rows(results)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)
