
"""Compare measurement accuracy across stereo-vision methods."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from slam_for_ada.datatypes import ComplianceResult, ErrorSummary
from slam_for_ada.eval.metrics import summarize_results

def compare_methods(
    results_by_method: Mapping[str, Sequence[ComplianceResult]],
) -> dict[str, ErrorSummary]:
    """Calculate accuracy metrics separately for each method."""
    summaries = {}

    for method, results in results_by_method.items():
        if not results:
            continue

        summaries[method] = summarize_results(
            results,
            group=method,
        )

    return summaries

def group_by_method(
    results: Sequence[ComplianceResult],
) -> dict[str, list[ComplianceResult]]:
    """Group measurements by the method that produced them."""
    groups = defaultdict(list)

    for result in results:
        method = result.estimate.method.value
        groups[method].append(result)

    return dict(groups)

def compare_feature_errors(
    results_by_method: Mapping[str, Sequence[ComplianceResult]],
) -> dict[str, dict[str, float]]:
    """Return absolute errors by feature for each method."""
    comparison = {}

    for method, results in results_by_method.items():
        comparison[method] = {
            result.estimate.feature_id: result.abs_error_in
            for result in results
        }

    return comparison

def best_method(
    summaries: Mapping[str, ErrorSummary],
) -> str | None:
    """Select the method with lowest MAE in the supplied results."""
    if not summaries:
        return None

    return min(
        summaries,
        key=lambda method: summaries[method].mean_abs_error_in,
    )
