"""Tier 2 pieces that can be tested without downloading models: pose tracking, disparity conversion, prompt mapping."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from slam_for_ada.config import StereoConfig, load_config
from slam_for_ada.datatypes import CameraPair, DepthMethod, FeatureKind, RectifiedPair
from slam_for_ada.vision.depth_stack import DepthStack
from slam_for_ada.vision.open_vocab import PROMPT_KINDS, RawDetection, to_object_detections
from slam_for_ada.vision.stereo_depth import disparity_to_ref_depth
from slam_for_ada.vision.temporal_fusion import propagate_point, relative_pose
from tests.helpers import REF, F, H, W, make_depth


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 200), dtype=np.uint8), (0, 0), 1.2)
    return cv2.addWeighted(g, 2.0, np.full_like(g, 0), 0, -120)  # extra contrast so ORB finds many corners


def test_relative_pose_recovers_a_sideways_move(K):
    """A fronto-parallel wall at 4 m; the second image is the first shifted 20 px, which is a camera move of 20*4/890 m."""
    tex = _texture()
    a, b = tex[:, 100 : 100 + W], tex[:, 120 : 120 + W]  # b(u) = a(u + 20): content moves left by 20 px
    depth = make_depth(K, np.full((H, W), 4.0))
    pose = relative_pose(a, depth, b)
    assert pose is not None and pose.n_inliers > 100 and pose.reproj_rms_px < 1.5
    assert abs(pose.T_b_a[0, 3] - (-20 * 4.0 / F)) < 0.01  # t_x = -s * Z / f
    assert np.allclose(pose.T_b_a[:3, :3], np.eye(3), atol=0.01)


def test_relative_pose_declines_without_overlap(K):
    a = _texture(1)[:, :W]
    b = _texture(2)[:, :W]  # unrelated image
    assert relative_pose(a, make_depth(K, np.full((H, W), 4.0)), b) is None


def test_propagate_point(K):

    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, -0.5]  # camera moves 0.5 m forward: points get 0.5 m nearer
    P = np.array([0.1, -0.2, 3.0])
    P_b, uv = propagate_point(P, T, K, (H, W))
    assert np.allclose(P_b, [0.1, -0.2, 2.5])
    assert propagate_point(np.array([0.1, 0.0, 0.5]), T, K, (H, W)) is None  # would end up behind the camera
    assert propagate_point(np.array([5.0, 0.0, 3.0]), np.eye(4), K, (H, W)) is None  # leaves the field of view


def test_disparity_to_ref_depth_identity_rectification(K):
    """With R1 = I and P1 = K, a constant disparity d maps to depth f*B/d everywhere on the reference grid."""
    img = np.zeros((H, W, 3), np.uint8)
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = P1.copy()
    P2[0, 3] = -F * 0.33
    pair = RectifiedPair(REF, CameraPair(0, 2), img, img, np.eye(3), np.eye(3), P1, P2, np.eye(4), 0.33, F)
    disp = np.full((H, W), 60.0, np.float32)
    disp[:100] = np.nan  # no match in the top rows
    d = disparity_to_ref_depth(disp, pair, K, StereoConfig(), DepthMethod.LEARNED, 0.3, 0.5)
    assert d.method is DepthMethod.LEARNED
    assert np.allclose(d.depth_m[200:400, 100:600], F * 0.33 / 60, rtol=1e-4)
    assert np.isnan(d.depth_m[:100]).all() and d.sigma_m is not None
    assert np.allclose(d.sigma_m[300, 300], (F * 0.33 / 60) ** 2 * 0.3 / (F * 0.33), rtol=1e-3)
    # too near for the pair's range -> dropped
    assert np.isnan(disparity_to_ref_depth(disp, pair, K, StereoConfig(), DepthMethod.LEARNED, 0.3, z_min_m=10.0).depth_m[300, 300])


def test_every_prompt_has_a_feature_kind_and_detections_convert():
    assert all(isinstance(k, FeatureKind) for k in PROMPT_KINDS.values())
    dets = [RawDetection("a wall thermostat", 0.4, (10, 10, 30, 40)), RawDetection("something unknown", 0.9, (0, 0, 5, 5))]
    out = to_object_detections(dets, REF)
    assert len(out) == 1 and out[0].kind is FeatureKind.OPERABLE_HEIGHT and out[0].bbox_px == (10, 10, 30, 40)


def test_hybrid_keeps_learned_where_it_agrees_and_falls_back_where_it_does_not(K):
    from slam_for_ada.vision.hybrid_depth import combine_matchers

    classical = np.full((H, W), 4.0)
    classical[:, :100] = np.nan  # block matching failed on the left (textureless)
    learned = np.full((H, W), 4.02)  # agrees within tolerance almost everywhere
    learned[200:260, 300:400] = 1.0  # but invents a wrong depth for a thin structure
    out = combine_matchers(make_depth(K, classical, sigma_m=0.05), make_depth(K, learned, sigma_m=0.03)).depth_m
    assert np.allclose(out[:, :100], 4.02)  # learned fills where classical is silent
    assert np.allclose(out[300:500, 150:250], 4.02)  # agreement -> learned value kept
    assert np.allclose(out[200:260, 300:400], 4.0)  # disagreement -> verified classical value wins
    other = make_depth(K, np.full((10, 10), 3.0))
    with pytest.raises(ValueError):
        combine_matchers(make_depth(K, classical), other)


def test_depth_stack_rejects_unknown_method():
    with pytest.raises(ValueError):
        DepthStack(load_config(), method="bogus")
