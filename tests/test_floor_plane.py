"""Floor fitting and pooling."""

from __future__ import annotations

import numpy as np

from slam_for_ada.config import FloorConfig
from slam_for_ada.datatypes import GravityEstimate, Plane3D
from slam_for_ada.vision.floor_plane import combine_floors, fit_floor, floor_spread, reject_floor_outliers
from tests.helpers import CAM_HEIGHT, H, W, make_depth


def test_fit_floor_recovers_height_and_normal(floor_depth_map):
    fl = fit_floor(floor_depth_map, FloorConfig())
    assert fl is not None
    assert np.allclose(fl.normal, [0, -1, 0], atol=0.01)
    assert abs(fl.d - CAM_HEIGHT) < 0.02  # d is the camera height above the floor
    assert abs(fl.signed_distance(np.array([[0.0, CAM_HEIGHT, 5.0]]))[0]) < 0.02  # a point on the floor has height 0


def test_wall_is_not_mistaken_for_floor(K):
    """With the upright prior, a lower image half that is a wall must not come back as a floor."""
    d = np.full((H, W), 4.0)  # everything is a wall 4 m away
    fl = fit_floor(make_depth(K, d), FloorConfig())
    assert fl is None


def test_gravity_sets_up_direction(floor_depth_map):
    g = GravityEstimate(0, np.array([0.0, 1.0, 0.0]), 1.0)  # down is +y in the camera frame
    fl = fit_floor(floor_depth_map, FloorConfig(), g)
    assert fl is not None and fl.normal[1] < -0.99


def test_pooling_rejects_outlier_frames():
    good = [Plane3D(np.array([0.0, -1.0, 0.0]), 1.27 + e, 100, 0.3, 0.01) for e in (-0.01, 0.0, 0.01, 0.005, -0.005)]
    bad_height = Plane3D(np.array([0.0, -1.0, 0.0]), 0.29, 100, 0.3, 0.01)  # RANSAC locked onto another surface
    n = np.array([0.0, -0.99, 0.14])
    n /= np.linalg.norm(n)
    bad_tilt = Plane3D(n, 1.27, 100, 0.3, 0.01)
    planes = good + [bad_height, bad_tilt]
    assert len(reject_floor_outliers(planes)) == len(good)
    pooled = combine_floors(planes)
    assert abs(pooled.d - 1.27) < 0.01 and pooled.normal[1] < -0.999
    tilt, hstd = floor_spread(reject_floor_outliers(planes))
    assert tilt < 0.5 and hstd < 0.02
