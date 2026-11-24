"""ADA verdict logic, the rules file, and the minimal scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_for_ada.ada.evaluate import margin_in, measured_verdict, true_verdict
from slam_for_ada.ada.rules import load_rules, rule_for
from slam_for_ada.datatypes import (
    AdaRule,
    DepthMethod,
    FeatureEstimate,
    FeatureKind,
    FrameRef,
    Measurement,
    MeasureMethod,
    Verdict,
    Verification,
)
from slam_for_ada.eval.ground_truth import load_ground_truth
from slam_for_ada.eval.minimal_scorer import format_table, score

ROOT = Path(__file__).resolve().parents[1]
DOOR = AdaRule("door", FeatureKind.DOOR_WIDTH, min_in=32.0)
HAND = AdaRule("handrail", FeatureKind.HANDRAIL_HEIGHT, min_in=34.0, max_in=38.0)


def test_margin_and_true_verdict():
    assert margin_in(35.0, DOOR) == 3.0
    assert margin_in(36.0, HAND) == 2.0 and margin_in(33.0, HAND) == -1.0 and margin_in(40.0, HAND) == -2.0
    assert true_verdict(32.0, DOOR) is Verdict.PASS  # exactly on the limit passes
    assert true_verdict(31.9, DOOR) is Verdict.FAIL
    assert true_verdict(35.0, None) is Verdict.NO_RULE


def test_measured_verdict_has_a_borderline_band():
    # sigma 1 in, band = 2 sigma = 2 in
    assert measured_verdict(35.0, 1.0, DOOR) is Verdict.PASS  # 3 in margin > 2 in band
    assert measured_verdict(33.0, 1.0, DOOR) is Verdict.BORDERLINE  # 1 in margin inside the band
    assert measured_verdict(31.0, 1.0, DOOR) is Verdict.BORDERLINE  # 1 in below the limit, inside the band
    assert measured_verdict(29.0, 1.0, DOOR) is Verdict.FAIL
    assert measured_verdict(36.0, 0.1, HAND) is Verdict.PASS
    assert measured_verdict(33.0, 0.1, HAND) is Verdict.FAIL
    assert measured_verdict(5.0, 1.0, None) is Verdict.NO_RULE


def test_rules_file_and_ground_truth_agree():
    truths = load_ground_truth(ROOT / "data/ground_truth/ground_truth.yaml")
    rules = load_rules(ROOT / "config/ada_rules.yaml")
    assert len(truths) == 23 and len(rules) == 10
    assert all(r.verification is Verification.STANDARD_VERBATIM for r in rules)
    assert rule_for("thermostat_height", rules).id == "operable_part_reach"
    assert rule_for("dustbin_height", rules) is None
    assert true_verdict(truths["thermostat_height"].value_in, rule_for("thermostat_height", rules)) is Verdict.FAIL  # 51.25 > 48
    assert true_verdict(truths["door_interior_width"].value_in, rule_for("door_interior_width", rules)) is Verdict.PASS


def _estimate(fid, kind, value, sigma) -> FeatureEstimate:
    m = Measurement(FrameRef("b", 0, 0), kind, MeasureMethod.PLANE_OFFSET, value, sigma, 3.0, DepthMethod.FUSED, (), fid)
    return FeatureEstimate(fid, kind, MeasureMethod.PLANE_OFFSET, value, sigma, (m,))


def test_scorer_matches_truth_and_scores_decisions():
    truths = load_ground_truth(ROOT / "data/ground_truth/ground_truth.yaml")
    rules = load_rules(ROOT / "config/ada_rules.yaml")
    ests = [
        _estimate("thermostat_height", FeatureKind.OPERABLE_HEIGHT, 52.0, 0.6),  # tape 51.25 -> error +0.75, FAIL, correct
        _estimate("dustbin_height", FeatureKind.OBJECT_HEIGHT, 35.0, 0.6),  # no rule
        _estimate("not_in_ground_truth", FeatureKind.DOOR_WIDTH, 30.0, 0.6),
    ]  # skipped
    res = {r.estimate.feature_id: r for r in score(ests, truths, rules)}
    assert set(res) == {"thermostat_height", "dustbin_height"}
    t = res["thermostat_height"]
    assert abs(t.error_in - 0.75) < 1e-9 and t.measured_verdict is Verdict.FAIL and t.decision_correct is True
    assert res["dustbin_height"].decision_correct is None
    assert "thermostat_height" in format_table(list(res.values()))


def test_duplicate_rule_ids_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(
        "rules:\n"
        "- {id: a, kind: door_width, verification: standard_verbatim, min_in: 1}\n"
        "- {id: a, kind: door_width, verification: standard_verbatim, min_in: 1}\n"
    )
    with pytest.raises(ValueError):
        load_rules(p)
