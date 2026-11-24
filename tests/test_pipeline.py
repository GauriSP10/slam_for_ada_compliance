"""Fusion of repeated observations, config loading, and the annotation file."""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_for_ada.config import Config, load_config
from slam_for_ada.dataio.annotations import load_annotations
from slam_for_ada.datatypes import CameraPair, DepthMethod, FeatureKind, FrameRef, Measurement, MeasureMethod
from slam_for_ada.eval.ground_truth import load_ground_truth
from slam_for_ada.fusion import fuse_observations

ROOT = Path(__file__).resolve().parents[1]


def _m(value, sigma, fid="f", method=MeasureMethod.PLANE_OFFSET) -> Measurement:
    return Measurement(
        FrameRef("b", 0, 0), FeatureKind.HANDRAIL_HEIGHT, method, value, sigma, 3.0, DepthMethod.FUSED, (CameraPair(0, 1),), fid
    )


def test_fusion_inverse_variance_mean():
    est = fuse_observations([_m(34.0, 0.5), _m(36.0, 1.0)])[0]
    w = (1 / 0.25, 1 / 1.0)
    assert abs(est.value_in - (34 * w[0] + 36 * w[1]) / sum(w)) < 1e-9
    assert est.n_obs == 2


def test_fusion_inflates_sigma_when_observations_disagree():
    agree = fuse_observations([_m(35.0, 0.5), _m(35.1, 0.5), _m(34.9, 0.5)])[0]
    clash = fuse_observations([_m(30.0, 0.5), _m(40.0, 0.5)])[0]
    assert clash.sigma_in > 3 * agree.sigma_in  # believe the disagreement over the error model


def test_fusion_drops_gross_outlier_with_three_or_more():
    est = fuse_observations([_m(35.0, 0.5), _m(35.2, 0.5), _m(34.8, 0.5), _m(12.0, 0.5)])[0]
    assert est.n_obs == 3 and abs(est.value_in - 35.0) < 0.2


def test_fusion_groups_by_feature_and_method_and_ignores_unlabelled():
    ests = fuse_observations([_m(1, 1, "a"), _m(2, 1, "a", MeasureMethod.HOMOGRAPHY), _m(3, 1, "b"), _m(4, 1, None)])
    assert sorted((e.feature_id, e.method.value) for e in ests) == [("a", "homography"), ("a", "plane_offset"), ("b", "plane_offset")]


def test_config_defaults_and_yaml(tmp_path):
    cfg = load_config()
    assert cfg.stereo.right_cams == (1, 2, 3, 4) and cfg.stereo.num_disparities == 256
    assert cfg.path("bags").name == "bags" and cfg.path("bags").is_absolute()
    p = tmp_path / "c.yaml"
    p.write_text("stereo:\n  num_disparities: 128\n  unknown_key: 1\nfloor: {ransac_iters: 50}\n")
    c2 = load_config(p)
    assert c2.stereo.num_disparities == 128 and c2.floor.ransac_iters == 50 and c2.cache.keyframe_window == 10
    assert isinstance(load_config(tmp_path / "missing.yaml"), Config)


def test_annotations_are_consistent_with_ground_truth_and_cache():
    cfg = load_config()
    anns = load_annotations(cfg.path("annotations"), cfg.path("bags")) if cfg.path("bags").exists() else pytest.skip("bags not present")
    truths = load_ground_truth(cfg.path("ground_truth"))
    assert anns
    for a in anns:
        assert a.feature_id in truths, f"{a.id}: unknown ground-truth id"
        assert truths[a.feature_id].kind == a.kind, f"{a.id}: kind differs from ground truth"
        g = a.geometry
        need = {"door_width": ("left_jamb", "right_jamb", "wall_rois"), "hallway_width": ("left_wall_roi", "right_wall_roi")}.get(
            a.kind.value
        )
        for k in need or (("point_px", "depth_roi") if a.kind.value.endswith("_height") else ()):
            assert k in g, f"{a.id}: missing {k}"
