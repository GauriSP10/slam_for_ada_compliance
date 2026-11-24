"""Metric rectification of a plane (a wall or the floor): image pixels <-> metric coordinates on the plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam_for_ada.datatypes import Mat3, Plane3D, Points3D, Vec3
from slam_for_ada.vision.geometry import intersect_rays_with_plane, pixel_rays


@dataclass(frozen=True, slots=True, eq=False)
class PlaneFrame:
    """A metric 2D coordinate system lying in a plane, defined relative to a camera.

    ``e_right`` and ``e_up`` are orthonormal in-plane directions (in metres, camera frame); the plane origin is
    the foot of the camera's perpendicular. ``H`` is the 3x3 homography from image pixels to plane coordinates.
    """

    K: Mat3
    normal: Vec3
    d: float
    e_right: Vec3
    e_up: Vec3
    H: Mat3

    def to_plane(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) metric plane coordinates (right, up), NaN where the ray misses the plane."""
        pts = intersect_rays_with_plane(pixel_rays(uv, self.K), self.normal, self.d)
        return np.column_stack([pts @ self.e_right, pts @ self.e_up])

    def to_3d(self, uv: np.ndarray) -> Points3D:
        """(N, 2) pixels -> (N, 3) camera-frame points where the viewing rays hit the plane."""
        return intersect_rays_with_plane(pixel_rays(uv, self.K), self.normal, self.d)


def plane_frame(K: Mat3, plane: Plane3D, up_hint: Vec3) -> PlaneFrame:
    """Build the metric frame of ``plane``. ``up_hint`` (e.g. the floor normal) fixes which in-plane axis is 'up'."""
    n = plane.normal
    e_up = up_hint - float(up_hint @ n) * n
    if np.linalg.norm(e_up) < 1e-6:  # plane parallel to the hint (e.g. the floor): pick any in-plane axis
        e_up = np.array([0.0, 0.0, 1.0]) - n[2] * n
    e_up = e_up / np.linalg.norm(e_up)
    e_right = np.cross(e_up, n)
    e_right /= np.linalg.norm(e_right)
    B = np.vstack([e_right, e_up])  # (2, 3)
    Kinv = np.linalg.inv(K)
    # P = lam * Kinv @ uv_h with lam = -d / (n . Kinv uv_h);  plane coords = B P  ->  homogeneous [B Kinv uv_h ; (n . Kinv uv_h) / -d]
    H = np.vstack([B @ Kinv, -(n @ Kinv)[None, :] / plane.d])
    return PlaneFrame(K, n, plane.d, e_right, e_up, H)
