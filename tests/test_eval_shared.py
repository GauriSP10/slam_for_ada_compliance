"""The shared evaluation helpers that replaced copy-pasted script code."""

from __future__ import annotations

import numpy as np

from slam_for_ada.eval.floor_scatter import floor_points, robust_sigma_cm
from tests.helpers import CAM_HEIGHT, H, W


def test_floor_points_on_a_perfect_floor(floor_depth_map, floor_plane, K):
    z, h = floor_points(floor_depth_map.depth_m, K, floor_plane, v_min_frac=0.5)
    assert z.size > 1000 and np.abs(h).max() < 0.01  # every point is on the plane
    assert z.min() > 4.0  # nothing nearer than the camera can see the floor


def test_floor_points_mask_restricts_pixels(floor_depth_map, floor_plane, K):
    full, _ = floor_points(floor_depth_map.depth_m, K, floor_plane, 0.5)
    left = np.zeros((H, W), bool)
    left[:, :200] = True
    part, _ = floor_points(floor_depth_map.depth_m, K, floor_plane, 0.5, left)
    assert 0 < part.size < full.size


def test_floor_points_drops_points_far_from_the_floor(floor_depth_map, floor_plane, K):
    d = floor_depth_map.depth_m.copy()
    d[400:, :] += 1.0  # a patch 1 m too far: not on the floor
    z, h = floor_points(d, K, floor_plane, 0.5)
    assert np.abs(h).max() < 0.15


def test_robust_sigma_ignores_outliers_and_needs_enough_points():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 0.02, 5000)  # 2 cm
    dirty = np.concatenate([clean, np.full(200, 5.0)])
    assert abs(robust_sigma_cm(clean) - 2.0) < 0.2
    assert abs(robust_sigma_cm(dirty) - robust_sigma_cm(clean)) < 0.3  # 4 percent gross outliers barely move it
    assert np.isnan(robust_sigma_cm(clean[:10]))
    assert CAM_HEIGHT > 0
