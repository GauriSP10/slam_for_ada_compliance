"""Locks the datatypes.py contract, and checks the YAML data files stay consistent with it."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import yaml

from slam_for_ada.datatypes import (
    SCHEMA_VERSION,
    AdaRule,
    CameraPair,
    CameraSet,
    ComplianceResult,
    DepthMap,
    DepthMethod,
    FeatureEstimate,
    FeatureKind,
    FrameRef,
    GravityEstimate,
    GroundTruthFeature,
    Measurement,
    MeasureMethod,
    Plane3D,
    RectifiedPair,
    RunResult,
    Verdict,
    Verification,
)

ROOT = Path(__file__).resolve().parents[1]
REF = FrameRef("bag33", 10, 123)
PAIR = CameraPair(0, 2)


def _measurement(**over) -> Measurement:
    kw = dict(
        ref=REF,
        kind=FeatureKind.DOOR_WIDTH,
        method=MeasureMethod.STEREO_DEPTH,
        value_in=35.1,
        sigma_in=0.6,
        distance_m=4.0,
        depth_method=DepthMethod.FUSED,
        pairs=(PAIR,),
        feature_id="door_interior_width",
    )
    kw.update(over)
    return Measurement(**kw)


# --------------------------------------------------------------------------- validation
def test_camera_pair_requires_left_before_right():
    assert CameraPair(0, 4).spacing == 4
    for bad in [(2, 1), (1, 1), (-1, 2)]:
        with pytest.raises(ValueError):
            CameraPair(*bad)


def test_camera_set_checks_shapes_and_keys():
    ok = {i: np.zeros((4, 4, 3), np.uint8) for i in range(2)}
    cs = CameraSet(REF, ok, {0: 0, 1: 2_000_000})
    assert cs.max_spread_ns == 2_000_000
    with pytest.raises(ValueError):
        CameraSet(REF, {0: np.zeros((2, 2, 3), np.uint8), 1: np.zeros((3, 3, 3), np.uint8)}, {0: 0, 1: 0})
    with pytest.raises(ValueError):
        CameraSet(REF, ok, {0: 0})


def test_gravity_must_be_unit_and_stationarity_in_range():
    GravityEstimate(1, np.array([0.0, 1.0, 0.0]), 0.9)
    with pytest.raises(ValueError):
        GravityEstimate(1, np.array([0.0, 2.0, 0.0]), 0.9)
    with pytest.raises(ValueError):
        GravityEstimate(1, np.array([0.0, 1.0, 0.0]), 1.5)


def test_rectified_pair_checks():
    img = np.zeros((4, 4, 3), np.uint8)
    P, I3, I4 = np.zeros((3, 4)), np.eye(3), np.eye(4)
    RectifiedPair(REF, PAIR, img, img, I3, I3, P, P, I4, 0.33, 880.0)
    with pytest.raises(ValueError):
        RectifiedPair(REF, PAIR, img, np.zeros((5, 5, 3), np.uint8), I3, I3, P, P, I4, 0.33, 880.0)
    with pytest.raises(ValueError):
        RectifiedPair(REF, PAIR, img, img, I3, I3, P, P, I4, -0.1, 880.0)


def test_depth_map_checks_optional_layers():
    d = np.full((4, 6), np.nan, np.float32)
    DepthMap(REF, 0, np.eye(3), d, DepthMethod.FUSED, (PAIR,), confidence=np.ones((4, 6), np.float32))
    with pytest.raises(ValueError):
        DepthMap(REF, 0, np.eye(3), d, DepthMethod.FUSED, (PAIR,), confidence=np.ones((2, 2), np.float32))


def test_plane_must_be_unit_and_measures_height():
    floor = Plane3D(np.array([0.0, -1.0, 0.0]), 1.5, 100, 0.9, 0.01)  # camera frame: y down, up = -y
    h = floor.signed_distance(np.array([[0.0, 0.5, 2.0], [0.0, 1.5, 2.0]]))
    assert np.allclose(h, [1.0, 0.0])
    with pytest.raises(ValueError):
        Plane3D(np.array([0.0, -2.0, 0.0]), 0.0, 1, 1.0, 0.0)


def test_measurement_and_estimate_validate():
    m = _measurement()
    assert m.value_m == pytest.approx(35.1 * 0.0254)
    with pytest.raises(ValueError):
        _measurement(sigma_in=-1.0)
    with pytest.raises(ValueError):
        _measurement(value_in=float("nan"))
    est = FeatureEstimate("f", FeatureKind.DOOR_WIDTH, MeasureMethod.STEREO_DEPTH, 35.1, 0.6, (m,))
    assert est.n_obs == 1
    with pytest.raises(ValueError):
        FeatureEstimate("f", FeatureKind.DOOR_WIDTH, MeasureMethod.STEREO_DEPTH, 35.1, 0.6, ())


def test_ada_rule_needs_a_limit_and_sane_range():
    AdaRule("r", FeatureKind.DOOR_WIDTH, min_in=32.0)
    with pytest.raises(ValueError):
        AdaRule("r", FeatureKind.DOOR_WIDTH)
    with pytest.raises(ValueError):
        AdaRule("r", FeatureKind.HANDRAIL_HEIGHT, min_in=38.0, max_in=34.0)


def test_frozen_and_schema_version():
    with pytest.raises(FrozenInstanceError):
        REF.index = 5  # type: ignore[misc]
    rr = RunResult((), (), "hash", "2026-01-01T00:00:00Z")
    assert rr.schema_version == SCHEMA_VERSION


# --------------------------------------------------------------------------- decisions
def _result(measured: Verdict, true: Verdict, rule=True) -> ComplianceResult:
    m = _measurement()
    est = FeatureEstimate("door_interior_width", FeatureKind.DOOR_WIDTH, MeasureMethod.STEREO_DEPTH, 35.1, 0.6, (m,))
    gt = GroundTruthFeature("door_interior_width", FeatureKind.DOOR_WIDTH, 35.6, True, "IMG_3490")
    r = AdaRule("door_clear_width", FeatureKind.DOOR_WIDTH, min_in=32.0) if rule else None
    return ComplianceResult(est, gt, r, -0.5, measured, true)


def test_decision_correct_semantics():
    assert _result(Verdict.PASS, Verdict.PASS).decision_correct is True
    assert _result(Verdict.FAIL, Verdict.PASS).decision_correct is False
    assert _result(Verdict.BORDERLINE, Verdict.PASS).decision_correct is False  # borderline never counts
    assert _result(Verdict.NO_RULE, Verdict.NO_RULE, rule=False).decision_correct is None
    assert _result(Verdict.PASS, Verdict.PASS).abs_error_in == 0.5


# --------------------------------------------------------------------------- data files vs types
@pytest.fixture(scope="module")
def ground_truth() -> dict:
    doc = yaml.safe_load((ROOT / "data/ground_truth/ground_truth.yaml").read_text(encoding="utf-8"))
    return {f["id"]: f for f in doc["features"]}


def test_ground_truth_loads_into_types(ground_truth):
    assert len(ground_truth) == 23
    for f in ground_truth.values():
        GroundTruthFeature(
            id=f["id"],
            kind=FeatureKind(f["kind"]),
            value_in=float(f["value_in"]),
            confirmed=f["confirmed"],
            photo=f["photo"],
            bags=tuple(f["bags"]),
            note=f["note"],
        )


def test_rules_load_and_match_ground_truth(ground_truth):
    doc = yaml.safe_load((ROOT / "config/ada_rules.yaml").read_text(encoding="utf-8"))
    owner: dict[str, str] = {}
    for r in doc["rules"]:
        rule = AdaRule(
            id=r["id"],
            kind=FeatureKind(r["kind"]),
            applies_to=tuple(r["applies_to"]),
            min_in=r.get("min_in"),
            max_in=r.get("max_in"),
            borderline_sigmas=r["borderline_sigmas"],
            citation=r["citation"],
            verification=Verification(r["verification"]),
        )
        for fid in rule.applies_to:
            assert fid in ground_truth, f"{rule.id}: unknown ground-truth id {fid}"
            assert FeatureKind(ground_truth[fid]["kind"]) == rule.kind, f"{rule.id}: kind mismatch for {fid}"
            assert fid not in owner, f"{fid} covered by both {owner.get(fid)} and {rule.id}"
            owner[fid] = rule.id
