"""Measurements on synthetic scenes with known geometry: heights, door and hallway widths, depth-patch gate."""

from __future__ import annotations

import numpy as np

from slam_for_ada.config import MeasureConfig
from slam_for_ada.datatypes import MeasureMethod
from slam_for_ada.vision.geometry import project
from slam_for_ada.vision.height import height_of_point
from slam_for_ada.vision.measure import roi_depth
from slam_for_ada.vision.protrusion import protrusion_from_wall
from slam_for_ada.vision.width import (
    door_width_homography,
    door_width_stereo,
    edge_energy,
    hallway_width_floor,
    snap_jamb,
)
from tests.helpers import CAM_HEIGHT, CX, CY, F, H, W, make_depth


def test_height_of_point_on_wall(depth_map, floor_plane):
    # wall at z=3 m: the pixel row y is at height 1.3 - (y - cy)/f * 3
    y = 200.0
    m = height_of_point(depth_map, floor_plane, (400.0, y), (380, 190, 420, 210))
    expected = CAM_HEIGHT - (y - CY) / F * 3.0
    assert m is not None and abs(m.value_m - expected) < 0.01
    assert m.method is MeasureMethod.PLANE_OFFSET and m.sigma_m > 0


def test_roi_gate_rejects_edges_gaps_and_disagreement(K):
    z = np.full((H, W), 3.0)
    z[:, 400:] = 12.0  # a depth edge at x = 400
    d = make_depth(K, z)
    assert roi_depth(d, (350, 100, 380, 140)).reliable  # entirely on the near surface
    straddle = roi_depth(d, (380, 100, 420, 140))
    assert not straddle.reliable and "scatter" in straddle.why_not  # patch straddles the edge
    empty = np.full((H, W), np.nan)
    empty[0:5, 0:5] = 3.0
    assert not roi_depth(make_depth(K, empty), (100, 100, 140, 140)).reliable
    assert not roi_depth(make_depth(K, np.full((H, W), 3.0), conf=0.3), (100, 100, 140, 140)).reliable  # pairs disagree
    assert (
        height_of_point(
            d,
            __import__("slam_for_ada.datatypes", fromlist=["Plane3D"]).Plane3D(np.array([0, -1.0, 0]), 1.3, 1, 1, 0),
            (400.0, 120.0),
            (380, 100, 420, 140),
        )
        is None
    )


def test_door_width_both_methods(K, floor_plane, depth_map):
    """Jambs at x=300 and x=500 on the wall at z=3 m: width = 200 * 3 / 890 = 0.674 m."""
    left, right = ((300.0, 100.0), (300.0, 260.0)), ((500.0, 100.0), (500.0, 260.0))
    expected = 200 * 3.0 / F
    cfg = MeasureConfig()
    s = door_width_stereo(depth_map, floor_plane, left, right, cfg)
    h = door_width_homography(depth_map, floor_plane, left, right, [(310, 80, 490, 250)], cfg)
    assert s is not None and abs(s.value_m - expected) < 0.01
    assert h is not None and abs(h.value_m - expected) < 0.01
    assert s.method is MeasureMethod.STEREO_DEPTH and h.method is MeasureMethod.HOMOGRAPHY


def test_homography_declines_grazing_incidence_and_implausible_widths(K, floor_plane, depth_map):
    cfg = MeasureConfig()
    # a leaf almost parallel to the viewing direction: ray meets the plane at ~17 degrees -> ill-conditioned
    n = np.array([-0.99, 0.0, -0.05])
    n /= np.linalg.norm(n)
    u = np.arange(W)[None, :] * np.ones((H, 1))
    ray_x = (u - CX) / F
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -1.2 / (n[0] * ray_x + n[2])
    z[(z <= 0) | (z > 15)] = np.nan
    left, right = ((585.0, 100.0), (585.0, 260.0)), ((640.0, 100.0), (640.0, 260.0))
    assert door_width_homography(make_depth(K, z), floor_plane, left, right, [(590, 60, 636, 300)], cfg) is None
    assert door_width_stereo(make_depth(K, z), floor_plane, left, right, cfg, [(590, 60, 636, 300)]) is None  # same rule for stereo
    # jambs 600 px apart on a wall 3 m away are 2.0 m = 79 in: not a door, so neither method reports it
    far_l, far_r = ((60.0, 100.0), (60.0, 260.0)), ((660.0, 100.0), (660.0, 260.0))
    assert door_width_homography(depth_map, floor_plane, far_l, far_r, [(80, 80, 640, 250)], cfg) is None
    assert door_width_stereo(depth_map, floor_plane, far_l, far_r, cfg) is None


def test_snap_jamb_finds_the_edge():
    img = np.zeros((300, 400), np.uint8)
    img[:, 210:] = 200  # a vertical edge at x = 210
    snapped, strength = snap_jamb(edge_energy(img), ((200.0, 20.0), (200.0, 280.0)), 18)
    assert abs(snapped[0][0] - 209.5) < 1.5 and strength > 10


def test_hallway_width_from_floor_contacts(K, floor_plane):
    """Walls meet the floor at lateral x = -0.9 and +0.9 m; project known floor points to pixels and measure back."""

    def contact(x, zs):
        pts = np.array([[x, CAM_HEIGHT, z] for z in zs])
        return [tuple(p) for p in project(pts, K)]

    r = hallway_width_floor(floor_plane, K, contact(-0.9, [3.0, 4.0, 5.0]), contact(0.9, [3.5, 4.5, 5.5]))
    assert r is not None and abs(r.value_m - 1.8) < 0.02 and r.method is MeasureMethod.HOMOGRAPHY
    assert hallway_width_floor(floor_plane, K, [(0.0, 0.0)], [(1.0, 1.0)]) is None


def test_protrusion_from_wall(K, floor_plane):
    z = np.full((H, W), 3.0)
    z[200:260, 400:460] = 2.82  # an object 18 cm proud of the wall
    r = protrusion_from_wall(make_depth(K, z), floor_plane, (100, 100, 300, 300), (405, 205, 455, 255))
    assert r is not None and abs(r.value_m - 0.18) < 0.01
