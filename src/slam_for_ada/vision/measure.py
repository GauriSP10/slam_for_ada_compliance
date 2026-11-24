"""Shared pieces of the measurement stages: the raw result type, ROI depth sampling, and wall-plane fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from slam_for_ada.datatypes import (
    M_PER_INCH,
    DepthMap,
    FeatureKind,
    FrameRef,
    Measurement,
    MeasureMethod,
    Plane3D,
    Vec3,
)
from slam_for_ada.vision.geometry import fit_plane_svd, orient_normal, ransac_plane

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in cam0 undistorted pixels


@dataclass(frozen=True)
class RawMeasurement:
    """A measurement in metres before it is attached to a frame and feature (see ``to_measurement``)."""

    value_m: float
    sigma_m: float
    distance_m: float  # camera-to-feature range
    method: MeasureMethod
    note: str = ""


def to_measurement(raw: RawMeasurement, ref: FrameRef, kind: FeatureKind, feature_id: str | None, depth: DepthMap) -> Measurement:
    """Attach frame, feature and the depth provenance, and convert to inches."""
    return Measurement(
        ref,
        kind,
        raw.method,
        raw.value_m / M_PER_INCH,
        raw.sigma_m / M_PER_INCH,
        raw.distance_m,
        depth.method,
        depth.pairs,
        feature_id,
        raw.note,
    )


def _clip_box(box: Box, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = box
    return max(int(round(x0)), 0), max(int(round(y0)), 0), min(int(round(x1)), w), min(int(round(y1)), h)


def roi_points(depth: DepthMap, box: Box) -> np.ndarray:
    """Valid camera-frame 3D points of every pixel in ``box``. Shape (N, 3)."""
    x0, y0, x1, y1 = _clip_box(box, depth.shape)
    z = depth.depth_m[y0:y1, x0:x1].astype(np.float64)
    u, v = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
    K = depth.K
    pts = np.dstack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]).reshape(-1, 3)
    return pts[np.isfinite(pts[:, 2])]


@dataclass(frozen=True)
class RoiDepth:
    """Depth statistics of a pixel patch, with a verdict on whether the patch can be trusted."""

    z: float  # median depth, metres (NaN if no valid pixel)
    sigma: float  # 1-sigma uncertainty of that median, metres
    n: int  # valid pixels
    fill: float  # share of the patch that has a depth value
    spread: float  # robust spread (1.4826 * MAD) of depth inside the patch, metres
    agreement: float  # mean share of camera pairs that agreed at each pixel (1 if unknown)
    reliable: bool
    why_not: str = ""


def roi_depth(depth: DepthMap, box: Box, min_fill: float = 0.25, min_agreement: float = 0.6) -> RoiDepth:
    """Median depth inside ``box`` plus a reliability verdict.

    A patch is rejected when it is mostly empty, when its depths scatter far more than a flat surface should
    (it straddles an edge or matched background through glass), or when the camera pairs disagree. Rejecting is
    deliberate: one such patch silently averaged into a result is worse than a missing observation.
    The uncertainty is the median modelled sigma, tightened by the number of roughly independent samples but
    never below 1 cm, because neighbouring pixels share their matching errors.
    """
    x0, y0, x1, y1 = _clip_box(box, depth.shape)
    area = max((x1 - x0) * (y1 - y0), 1)
    z = depth.depth_m[y0:y1, x0:x1]
    ok = np.isfinite(z)
    n = int(ok.sum())
    if n == 0:
        return RoiDepth(float("nan"), float("nan"), 0, 0.0, float("nan"), 0.0, False, "no valid depth")
    zv = z[ok].astype(np.float64)
    med = float(np.median(zv))
    spread = float(1.4826 * np.median(np.abs(zv - med)))
    fill = n / area
    agree = float(np.nanmean(depth.confidence[y0:y1, x0:x1][ok])) if depth.confidence is not None else 1.0
    sig = float(np.nanmedian(depth.sigma_m[y0:y1, x0:x1][ok])) if depth.sigma_m is not None else 0.03
    sigma = max(sig / np.sqrt(max(n / 25.0, 1.0)), 0.01)
    why = ""
    if fill < min_fill:
        why = f"only {fill:.0%} of the patch has depth"
    elif spread > max(0.08, 0.04 * med):
        why = f"depth scatters by {spread * 100:.0f} cm inside the patch (edge or mismatch)"
    elif agree < min_agreement:
        why = f"camera pairs agree on only {agree:.0%} of pixels"
    return RoiDepth(med, sigma, n, fill, spread, agree, not why, why)


def fit_wall_plane(
    depth: DepthMap, boxes: Sequence[Box], up: Vec3, thresh_m: float = 0.02, iters: int = 400, max_tilt_deg: float = 15.0, seed: int = 0
) -> tuple[Plane3D, np.ndarray] | None:
    """RANSAC plane through the depth points inside ``boxes``, restricted to near-vertical planes.

    Returns (plane with its normal pointing toward the camera, the inlier points) or None.
    ``up`` is the floor normal; a wall's normal must be within ``max_tilt_deg`` of perpendicular to it.
    """
    pts = np.vstack([roi_points(depth, b) for b in boxes]) if boxes else np.zeros((0, 3))
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    # a wall normal is horizontal, i.e. perpendicular to "up": sample freely, then reject tilted candidates
    n, d, inl = ransac_plane(pts, iters, thresh_m, rng)
    if n is None or inl.sum() < 50:
        return None
    n, d, rms = fit_plane_svd(pts[inl])
    if abs(float(n @ up)) > np.sin(np.radians(max_tilt_deg)):
        return None
    n, d = orient_normal(n, d, -pts[inl].mean(axis=0))  # point toward the camera (origin)
    return Plane3D(n, d, int(inl.sum()), float(inl.mean()), rms, "wall"), pts[inl]


def horizontal(v: np.ndarray, up: Vec3) -> np.ndarray:
    """Remove the component of ``v`` along ``up`` (project onto the horizontal plane)."""
    return v - np.outer(v @ up, up) if v.ndim == 2 else v - float(v @ up) * up
