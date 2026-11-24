"""Tier 2: follow a feature through neighbouring keyframes and measure it again in each.

A height above the floor does not depend on where the camera is, so repeated heights of the SAME object can simply be
combined, which averages down the matching noise. The hard part is knowing it is the same object. We estimate the camera
motion between consecutive keyframes (ORB matches + depth -> PnP), carry the object's 3D point into the next frame, and
take a fresh, independent depth reading there at the predicted pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from slam_for_ada.datatypes import DepthMap, Mat3, Mat4
from slam_for_ada.vision.geometry import backproject_points, project

MIN_INLIERS = 60


@dataclass(frozen=True)
class RelativePose:
    T_b_a: Mat4  # maps points in frame a's camera coordinates into frame b's
    n_inliers: int
    reproj_rms_px: float


def relative_pose(
    gray_a: np.ndarray, depth_a: DepthMap, gray_b: np.ndarray, ratio: float = 0.75, reproj_px: float = 3.0
) -> RelativePose | None:
    """Camera motion from frame a to frame b. Both images are cam0-undistorted, so ``depth_a.K`` applies to both.

    ORB features matched across the frames give 2D-2D correspondences; frame a's depth lifts each to 3D, and RANSAC-PnP
    finds the motion that reprojects those points onto frame b. People and other movers are rejected as outliers.
    Returns None when too few matches survive (blur, or too little overlap).
    """
    K = depth_a.K
    orb = cv2.ORB_create(2500)  # pyright: ignore[reportAttributeAccessIssue]  # missing from the OpenCV 5.1 stubs
    ka, da = orb.detectAndCompute(gray_a, None)
    kb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    pts3, pts2 = [], []
    for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2):
        if len(m) < 2 or m[0].distance >= ratio * m[1].distance:
            continue
        u, v = ka[m[0].queryIdx].pt
        z = (
            depth_a.depth_m[int(round(v)), int(round(u))]
            if 0 <= int(round(v)) < depth_a.shape[0] and 0 <= int(round(u)) < depth_a.shape[1]
            else np.nan
        )
        if np.isfinite(z) and 0.5 < z < 15.0:
            pts3.append(backproject_points(np.array([[u, v]]), np.array([z]), K)[0])
            pts2.append(kb[m[0].trainIdx].pt)
    if len(pts3) < MIN_INLIERS:
        return None
    P, p = np.array(pts3, np.float64), np.array(pts2, np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(P, p, K, None, iterationsCount=300, reprojectionError=reproj_px, confidence=0.995)
    if not ok or inl is None or len(inl) < MIN_INLIERS:
        return None
    inl = inl[:, 0].astype(int)
    rvec, tvec = cv2.solvePnPRefineLM(P[inl], p[inl], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec[:, 0]
    err = np.linalg.norm(project((R @ P[inl].T).T + tvec[:, 0], K) - p[inl], axis=1)
    return RelativePose(T, int(len(inl)), float(np.sqrt(np.mean(err**2))))


def propagate_point(P_a: np.ndarray, T_b_a: Mat4, K: Mat3, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Carry a 3D point from frame a into frame b. Returns (its 3D position in b, its pixel in b), or None if it left the view."""
    P_b = T_b_a[:3, :3] @ P_a + T_b_a[:3, 3]
    if P_b[2] < 0.4:
        return None
    uv = project(P_b[None], K)[0]
    h, w = shape
    if not (10 <= uv[0] < w - 10 and 10 <= uv[1] < h - 10):
        return None
    return P_b, uv
