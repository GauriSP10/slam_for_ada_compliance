"""Quantify rectification quality: after rectifying, matching points must lie on the same image row."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from slam_for_ada.calib.rectify import PairRectifier
from slam_for_ada.datatypes import CameraSet


@dataclass(frozen=True)
class EpipolarReport:
    """Vertical misalignment of matched features for one camera pair, over many frames."""

    pair: tuple[int, int]
    baseline_m: float
    n_frames: int
    n_matches: int
    median_abs_dy_px: float
    p90_abs_dy_px: float
    mean_dy_px: float  # signed: a non-zero mean means a systematic vertical offset between the two cameras


def match_dy(
    rect_left: np.ndarray, rect_right: np.ndarray, max_features: int = 3000, ratio: float = 0.75, max_abs_dy: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical (and horizontal) offsets of ORB feature matches between two rectified images.

    Matches pass Lowe's ratio test and a loose |dy| gate that only rejects gross mismatches.
    Returns (dy, dx) arrays, where dy = y_left - y_right and dx = x_left - x_right (the disparity).
    """
    orb = cv2.ORB_create(max_features)  # pyright: ignore[reportAttributeAccessIssue]  # missing from the OpenCV 5.1 stubs
    gl, gr = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    kl, dl = orb.detectAndCompute(gl, None)
    kr, dr = orb.detectAndCompute(gr, None)
    if dl is None or dr is None or len(kl) < 8 or len(kr) < 8:
        return np.zeros(0), np.zeros(0)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dl, dr, k=2)
    dy, dx = [], []
    for p in pairs:
        if len(p) < 2 or p[0].distance >= ratio * p[1].distance:
            continue
        a, b = kl[p[0].queryIdx].pt, kr[p[0].trainIdx].pt
        if abs(a[1] - b[1]) <= max_abs_dy and a[0] - b[0] > 0:  # positive disparity only
            dy.append(a[1] - b[1])
            dx.append(a[0] - b[0])
    return np.array(dy), np.array(dx)


def check_pair(rectifier: PairRectifier, cam_sets: Iterable[CameraSet]) -> EpipolarReport:
    """Rectify many frames with one pair and summarise the vertical error of matched features."""
    all_dy, n_frames = [], 0
    for cs in cam_sets:
        dy, _ = match_dy(*(lambda r: (r.left, r.right))(rectifier.rectify(cs)))
        all_dy.append(dy)
        n_frames += 1
    dy = np.concatenate(all_dy) if all_dy else np.zeros(0)
    if dy.size == 0:
        return EpipolarReport((rectifier.pair.left, rectifier.pair.right), rectifier.baseline_m, n_frames, 0, np.nan, np.nan, np.nan)
    a = np.abs(dy)
    return EpipolarReport(
        (rectifier.pair.left, rectifier.pair.right),
        rectifier.baseline_m,
        n_frames,
        int(dy.size),
        float(np.median(a)),
        float(np.percentile(a, 90)),
        float(dy.mean()),
    )
