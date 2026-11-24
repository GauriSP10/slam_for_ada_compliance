"""Classical stereo on a synthetic pair with a known disparity, and fusion of several pairs."""

from __future__ import annotations

import cv2
import numpy as np

from slam_for_ada.config import StereoConfig
from slam_for_ada.datatypes import CameraPair, DepthMethod, RectifiedPair
from slam_for_ada.vision.multibaseline import fuse_depths
from slam_for_ada.vision.stereo_depth import SgbmDepth
from tests.helpers import REF, F, H, W, make_depth


def synthetic_pair(K, disparity_px: int, baseline_m: float = 0.33) -> RectifiedPair:
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 600), dtype=np.uint8), (0, 0), 1.5)
    left = tex[:, 200 : 200 + W]
    # a scene point at left column x appears at right column x - d, so right(x') = left(x' + d)
    right = tex[:, 200 + disparity_px : 200 + disparity_px + W]
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = P1.copy()
    P2[0, 3] = -F * baseline_m

    def bgr(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    return RectifiedPair(REF, CameraPair(0, 2), bgr(left), bgr(right), np.eye(3), np.eye(3), P1, P2, np.eye(4), baseline_m, F)


def test_sgbm_recovers_known_depth(K):
    d, B = 60, 0.33
    depth = SgbmDepth(StereoConfig(), K)(synthetic_pair(K, d, B))
    expected = F * B / d  # 4.895 m
    centre = depth.depth_m[150:400, 300:650]
    assert np.isfinite(centre).mean() > 0.8
    assert abs(float(np.nanmedian(centre)) - expected) / expected < 0.02
    assert depth.method is DepthMethod.SGBM and depth.sigma_m is not None


def test_fusion_weights_by_precision_and_rejects_outliers(K):
    z = np.full((H, W), 4.0)
    precise = make_depth(K, z + 0.00, sigma_m=0.01, pairs=(CameraPair(0, 4),))
    noisy = make_depth(K, z + 0.05, sigma_m=0.10, pairs=(CameraPair(0, 1),))
    fused = fuse_depths([precise, noisy])
    # weights 1/0.01^2 : 1/0.1^2 = 100 : 1  ->  4.0 + 0.05/101
    assert np.isclose(np.nanmedian(fused.depth_m), 4.0 + 0.05 / 101, atol=1e-4)
    assert fused.sigma_m is not None and np.nanmedian(fused.sigma_m) < 0.01  # better than either input
    assert fused.method is DepthMethod.FUSED and len(fused.pairs) == 2

    wrong = make_depth(K, z + 3.0, sigma_m=0.02, pairs=(CameraPair(0, 2),))  # a wrong match 3 m off
    fused3 = fuse_depths([precise, noisy, wrong])
    assert np.isclose(np.nanmedian(fused3.depth_m), 4.0, atol=0.01)  # outlier did not drag the result


def test_fusion_fills_gaps_and_keeps_nan_where_nobody_sees(K):
    a = np.full((H, W), np.nan)
    a[:, :300] = 5.0
    b = np.full((H, W), np.nan)
    b[:, 200:500] = 5.0
    fused = fuse_depths([make_depth(K, a), make_depth(K, b)]).depth_m
    assert np.isfinite(fused[:, :500]).all() and np.isnan(fused[:, 500:]).all()


def test_fusion_rejects_mismatched_grids(K):
    import pytest

    with pytest.raises(ValueError):
        fuse_depths([make_depth(K, np.full((H, W), 3.0)), make_depth(K, np.full((10, 10), 3.0))])
