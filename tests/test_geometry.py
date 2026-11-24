"""Geometry helpers: back-projection, plane fitting, ray-plane intersection, homography."""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import Plane3D
from slam_for_ada.vision.geometry import (
    backproject_map,
    backproject_points,
    fit_plane_svd,
    intersect_rays_with_plane,
    orient_normal,
    pixel_rays,
    project,
    ransac_plane,
)
from slam_for_ada.vision.homography import plane_frame


def test_backproject_project_round_trip(K):
    uv = np.array([[100.0, 50.0], [369.0, 275.0], [650.0, 500.0]])
    z = np.array([2.0, 5.0, 3.0])
    assert np.allclose(project(backproject_points(uv, z, K), K), uv)
    depth = np.full((10, 10), 4.0)
    pts = backproject_map(depth, K)
    assert pts.shape == (10, 10, 3) and np.allclose(pts[..., 2], 4.0)


def test_fit_plane_svd_recovers_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (500, 2))
    pts = np.column_stack([xy, 0.3 * xy[:, 0] + 2.0 + rng.normal(0, 0.001, 500)])  # z = 0.3 x + 2
    n, d, rms = fit_plane_svd(pts)
    assert rms < 0.002
    expected = np.array([-0.3, 0, 1]) / np.linalg.norm([-0.3, 0, 1])
    assert abs(float(n @ expected)) > 0.9999


def test_ransac_ignores_outliers_and_respects_prior():
    rng = np.random.default_rng(1)
    floor = np.column_stack([rng.uniform(-2, 2, 800), np.full(800, 1.3) + rng.normal(0, 0.005, 800), rng.uniform(1, 6, 800)])
    wall = np.column_stack([rng.uniform(-2, 2, 1200), rng.uniform(-1, 1, 1200), np.full(1200, 3.0)])  # bigger set: a wall
    pts = np.vstack([floor, wall])
    # with an upright prior the floor must win even though the wall has more points
    n, d, inl = ransac_plane(pts, 300, 0.02, np.random.default_rng(0), prior_normal=np.array([0, -1.0, 0]), max_angle_deg=10)
    assert abs(n[1]) > 0.99 and abs(abs(d) - 1.3) < 0.02
    assert inl[:800].mean() > 0.9 and inl[800:].mean() < 0.1


def test_intersect_rays_with_plane(K):
    wall = Plane3D(np.array([0.0, 0.0, -1.0]), 3.0, 1, 1, 0)  # plane z = 3, normal toward the camera
    uv = np.array([[369.0, 275.0], [369.0 + 89.0, 275.0]])
    p = intersect_rays_with_plane(pixel_rays(uv, K), wall.normal, wall.d)
    assert np.allclose(p[:, 2], 3.0) and np.isclose(p[1, 0], 0.3)


def test_orient_normal_flips():
    n, d = orient_normal(np.array([0.0, 1.0, 0.0]), 2.0, np.array([0.0, -1.0, 0.0]))
    assert n[1] == -1.0 and d == -2.0


def test_plane_frame_gives_metric_lengths(K):
    """Two pixels 100 px apart on a fronto-parallel wall at 3 m are 100*3/890 m apart."""
    wall = Plane3D(np.array([0.0, 0.0, -1.0]), 3.0, 1, 1, 0)
    frame = plane_frame(K, wall, np.array([0.0, -1.0, 0.0]))
    p = frame.to_plane(np.array([[300.0, 275.0], [400.0, 275.0]]))
    assert np.isclose(abs(p[1, 0] - p[0, 0]), 100 * 3.0 / 890.0, rtol=1e-6)
    # and the homography matrix gives the same answer as the ray intersection
    h = frame.H @ np.array([400.0, 275.0, 1.0])
    assert np.allclose(h[:2] / h[2], p[1])
