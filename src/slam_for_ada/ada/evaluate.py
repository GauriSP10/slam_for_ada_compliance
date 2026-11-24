"""Turn a value and its uncertainty into PASS / FAIL / BORDERLINE against an ADA rule."""

from __future__ import annotations

from slam_for_ada.datatypes import AdaRule, Verdict


def margin_in(value_in: float, rule: AdaRule) -> float:
    """Signed distance to the nearest violated side: positive = inside the allowed range, negative = outside."""
    margins = []
    if rule.min_in is not None:
        margins.append(value_in - rule.min_in)
    if rule.max_in is not None:
        margins.append(rule.max_in - value_in)
    return min(margins)


def true_verdict(value_in: float, rule: AdaRule | None) -> Verdict:
    """Verdict of the tape-measure value. It has no modelled uncertainty, so it is never BORDERLINE."""
    if rule is None:
        return Verdict.NO_RULE
    return Verdict.PASS if margin_in(value_in, rule) >= 0 else Verdict.FAIL


def measured_verdict(value_in: float, sigma_in: float, rule: AdaRule | None) -> Verdict:
    """Verdict of a measurement. Within ``borderline_sigmas`` x sigma of a limit it is BORDERLINE, not a confident call."""
    if rule is None:
        return Verdict.NO_RULE
    m = margin_in(value_in, rule)
    band = rule.borderline_sigmas * sigma_in
    if m >= band:
        return Verdict.PASS
    if m <= -band:
        return Verdict.FAIL
    return Verdict.BORDERLINE
