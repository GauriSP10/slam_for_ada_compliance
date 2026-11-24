"""Rig model and rectification, checked against the real calibration file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slam_for_ada.calib.rectify import build_rectifiers
from slam_for_ada.calib.rig import baseline_m, load_rig, relative_transform
from slam_for_ada.calib.undistort import RefUndistorter

CALIB = Path(__file__).resolve().parents[1] / "data/calibration/cams_calib.yaml"
pytestmark = pytest.mark.skipif(not CALIB.exists(), reason="calibration file not present")


def test_rig_loads_five_cameras():
    rig = load_rig(CALIB)
    assert sorted(rig.cameras) == [0, 1, 2, 3, 4] and sorted(rig.T_cn_cnm1) == [1, 2, 3, 4]
    c0 = rig.cameras[0]
    assert c0.size == (720, 540) and 880 < c0.fx < 900 and np.allclose(c0.K[2], [0, 0, 1])


def test_baselines_are_multiples_of_the_neighbour_spacing():
    rig = load_rig(CALIB)
    b = {k: baseline_m(rig, 0, k) for k in (1, 2, 3, 4)}
    assert all(0.15 < b[1] < 0.18 for _ in [0])
    for k in (2, 3, 4):
        assert abs(b[k] / b[1] - k) < 0.05  # evenly spaced row of cameras
    with pytest.raises(ValueError):
        relative_transform(rig, 2, 1)


def test_relative_transform_chains():
    rig = load_rig(CALIB)
    T02 = relative_transform(rig, 0, 2)
    assert np.allclose(T02, rig.T_cn_cnm1[2] @ rig.T_cn_cnm1[1])
    assert np.allclose(T02[3], [0, 0, 0, 1])


def test_rectifier_geometry_matches_baseline():
    rig = load_rig(CALIB)
    for k, r in build_rectifiers(rig, (1, 2, 4)).items():
        assert abs(r.baseline_m - baseline_m(rig, 0, k)) < 0.005
        assert 850 < r.focal_px < 920
        assert r.P1[1, 1] == r.P2[1, 1] and abs(r.P1[1, 2] - r.P2[1, 2]) < 1e-6  # same rows in both: epipolar lines horizontal
        assert np.allclose(r.R1 @ r.R1.T, np.eye(3), atol=1e-9)


def test_undistorter_keeps_the_centre_pixel():
    rig = load_rig(CALIB)
    u = RefUndistorter(rig)
    img = np.zeros((540, 720, 3), np.uint8)
    cx, cy = int(rig.cameras[0].cx), int(rig.cameras[0].cy)
    img[cy - 1 : cy + 2, cx - 1 : cx + 2] = 255
    out = u(img)
    ys, xs = np.nonzero(out[..., 0] > 100)
    assert abs(xs.mean() - cx) < 2 and abs(ys.mean() - cy) < 2  # the principal point is a fixed point of undistortion
