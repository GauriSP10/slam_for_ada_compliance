"""Find the floor: the plane that defines height zero for every height measurement."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from slam_for_ada.config import FloorConfig
from slam_for_ada.datatypes import DepthMap, GravityEstimate, Plane3D
from slam_for_ada.vision.geometry import backproject_map, fit_plane_svd, orient_normal, ransac_plane

_DEFAULT_UP = np.array([0.0, -1.0, 0.0])  # camera frame has y pointing down, so "up" is -y when the rig is roughly upright


def fit_floor(depth: DepthMap, cfg: FloorConfig, gravity: GravityEstimate | None = None) -> Plane3D | None:
    """RANSAC plane fit on the lower part of the image. Satisfies the ``FloorFitter`` protocol.

    Gravity is optional. With it, "up" is the measured direction; without it we assume the rig is roughly
    upright and only accept planes within ``max_tilt_deg`` of the camera's vertical, which is enough to stop
    a wall from being mistaken for the floor. Returns None when too few usable points exist.
    """
    up = -gravity.down_cam if gravity is not None else _DEFAULT_UP
    pts = backproject_map(depth.depth_m, depth.K)
    h = pts.shape[0]
    rows = slice(int(cfg.v_min_frac * h), h, 3)
    cand = pts[rows, ::3].reshape(-1, 3)
    z = cand[:, 2]
    cand = cand[np.isfinite(z) & (z >= cfg.depth_min_m) & (z <= cfg.depth_max_m)]
    if len(cand) < 200:
        return None
    rng = np.random.default_rng(cfg.seed)
    n, d, inl = ransac_plane(cand, cfg.ransac_iters, cfg.inlier_thresh_m, rng, up, cfg.max_tilt_deg)
    if n is None or inl.sum() < 100:
        return None
    n, d, rms = fit_plane_svd(cand[inl])
    n, d = orient_normal(n, d, up)
    return Plane3D(n, d, int(inl.sum()), float(inl.mean()), rms, "ransac")


def reject_floor_outliers(planes: Sequence[Plane3D], max_dev_m: float = 0.10, max_tilt_deg: float = 3.0) -> list[Plane3D]:
    """Drop per-frame floors that disagree with the consensus.

    A few frames lock RANSAC onto some other near-horizontal surface (glass, a table top, a far wall) and
    give a camera height of tens of centimetres with a visibly tilted normal. The consensus is the median
    normal and height; frames further than ``max_dev_m`` in height or ``max_tilt_deg`` in tilt are removed.
    """
    if not planes:
        return []
    n0 = np.median(np.stack([p.normal for p in planes]), axis=0)
    n0 /= np.linalg.norm(n0)
    d0 = float(np.median([p.d for p in planes]))
    keep = []
    for p in planes:
        tilt = np.degrees(np.arccos(np.clip(float(p.normal @ n0), -1.0, 1.0)))
        if abs(p.d - d0) <= max_dev_m and tilt <= max_tilt_deg:
            keep.append(p)
    return keep


def combine_floors(planes: Sequence[Plane3D], max_dev_m: float = 0.10, max_tilt_deg: float = 3.0) -> Plane3D:
    """One robust floor from many per-frame floors: outliers rejected, then the median normal and offset.

    The rig is a rigid mount on a wheeled platform, so camera height and tilt above the floor are close to
    constant across the sequence (per-frame heights agree to about 1.5 cm here). Pooling many frames averages
    out per-frame noise, which matters because a 0.5 degree tilt error is already 3.5 cm of height error at 4 m.
    """
    if not planes:
        raise ValueError("no planes to combine")
    kept = reject_floor_outliers(planes, max_dev_m, max_tilt_deg) or list(planes)
    n = np.median(np.stack([p.normal for p in kept]), axis=0)
    n /= np.linalg.norm(n)
    d = float(np.median([p.d for p in kept]))
    return Plane3D(
        n,
        d,
        int(sum(p.n_inliers for p in kept)),
        float(np.mean([p.inlier_ratio for p in kept])),
        float(np.median([p.rmse_m for p in kept])),
        f"median of {len(kept)}/{len(planes)} frames",
    )


def floor_spread(planes: Sequence[Plane3D]) -> tuple[float, float]:
    """(std of the normal's tilt in degrees, std of camera height in metres) across per-frame floors."""
    n = np.stack([p.normal for p in planes])
    ref = np.median(n, axis=0)
    ref /= np.linalg.norm(ref)
    ang = np.degrees(np.arccos(np.clip(n @ ref, -1, 1)))
    return float(np.sqrt(np.mean(ang**2))), float(np.std([p.d for p in planes]))
