"""Small, pure 3D geometry helpers shared by the vision stages (camera frame: x right, y down, z forward)."""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import Mat3, Points3D, Vec3


def backproject_map(depth_m: np.ndarray, K: Mat3) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) camera-frame points; NaN where depth is NaN."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth_m.astype(np.float64)
    return np.dstack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])


def backproject_points(uv: np.ndarray, z: np.ndarray, K: Mat3) -> Points3D:
    """(N, 2) pixels and (N,) depths -> (N, 3) points."""
    uv = np.asarray(uv, dtype=np.float64)
    return np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z])


def pixel_rays(uv: np.ndarray, K: Mat3) -> Points3D:
    """(N, 2) pixels -> (N, 3) viewing rays with z = 1 (not unit length)."""
    uv = np.asarray(uv, dtype=np.float64)
    return np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))])


def project(points: Points3D, K: Mat3) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixels."""
    return np.column_stack([K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2], K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]])


def fit_plane_svd(points: Points3D) -> tuple[Vec3, float, float]:
    """Least-squares plane ``n . x + d = 0``. Returns (unit normal, d, rms residual in metres). Normal sign is arbitrary."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    d = -float(n @ c)
    return n, d, float(np.sqrt(np.mean((points @ n + d) ** 2)))


def ransac_plane(
    points: Points3D,
    iters: int,
    thresh_m: float,
    rng: np.random.Generator,
    prior_normal: Vec3 | None = None,
    max_angle_deg: float | None = None,
) -> tuple[Vec3 | None, float, np.ndarray]:
    """Robust plane fit. Returns (unit normal, d, inlier mask), or (None, 0, empty mask) if nothing fits.

    With ``prior_normal`` and ``max_angle_deg``, candidate planes whose normal is further from the prior
    (either sign) are skipped, so a wall can never win over a floor.
    """
    n_pts = len(points)
    best_count, best = 0, (None, 0.0, np.zeros(n_pts, bool))
    if n_pts < 3:
        return best
    cos_max = np.cos(np.radians(max_angle_deg)) if (prior_normal is not None and max_angle_deg is not None) else None
    for _ in range(iters):
        a, b, c = points[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if cos_max is not None and abs(float(n @ prior_normal)) < cos_max:
            continue
        d = -float(n @ a)
        inl = np.abs(points @ n + d) < thresh_m
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count, best = cnt, (n, d, inl)
    if best[0] is None:
        return best
    n, d, rms = fit_plane_svd(points[best[2]])  # refine on the inliers
    inl = np.abs(points @ n + d) < thresh_m
    return n, d, inl


def orient_normal(n: Vec3, d: float, toward: Vec3) -> tuple[Vec3, float]:
    """Flip (n, d) so that n points toward ``toward`` (same plane)."""
    return (n, d) if float(n @ toward) >= 0 else (-n, -d)


def intersect_rays_with_plane(rays: Points3D, normal: Vec3, d: float) -> Points3D:
    """Points where rays from the camera origin hit the plane ``n . x + d = 0``. NaN if a ray is parallel to it."""
    denom = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -d / denom
    t = np.where(t > 0, t, np.nan)
    return rays * t[:, None]
