"""Depth precision measured on the floor: it is flat, so the scatter of floor points about the floor plane needs no ground truth."""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import Mat3, Plane3D
from slam_for_ada.vision.geometry import backproject_map


def floor_points(
    depth_m: np.ndarray, K: Mat3, floor: Plane3D, v_min_frac: float, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(distance, height above the floor) of lower-image pixels within 15 cm of the floor plane.

    ``mask`` restricts the pixels, so two matchers can be compared on exactly the pixels both produce.
    """
    d = depth_m if mask is None else np.where(mask, depth_m, np.nan)
    pts = backproject_map(d, K)
    p = pts[int(v_min_frac * pts.shape[0]) :, ::2].reshape(-1, 3)
    p = p[np.isfinite(p[:, 2])]
    h = floor.signed_distance(p)
    keep = np.abs(h) < 0.15
    return p[keep, 2], h[keep]


def robust_sigma_cm(heights_m: np.ndarray, min_points: int = 50) -> float:
    """Outlier-resistant standard deviation (1.4826 x MAD) in centimetres; NaN if there are too few points."""
    if heights_m.size < min_points:
        return float("nan")
    return float(1.4826 * np.median(np.abs(heights_m - np.median(heights_m))) * 100)
