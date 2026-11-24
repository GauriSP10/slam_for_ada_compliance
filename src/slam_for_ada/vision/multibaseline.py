"""Fuse the depth maps of several camera pairs into one, weighting each by its modelled precision."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from slam_for_ada.datatypes import DepthMap, DepthMethod


def fuse_depths(depths: Sequence[DepthMap], outlier_sigmas: float = 3.0) -> DepthMap:
    """Per-pixel inverse-variance fusion across pairs. Satisfies the ``DepthFuser`` protocol.

    Depth error grows as z^2 / (f*B), so a wide baseline is far more precise at range and a narrow one is
    the only one that sees close. At each pixel:
      1. take the pairs that produced a value,
      2. reject any whose value disagrees with the median of those pairs by more than
         ``outlier_sigmas`` of its own uncertainty (plus a small floor), which removes wrong stereo matches,
      3. average the survivors weighted by 1/sigma^2.
    The output ``sigma_m`` is the fused standard deviation and ``confidence`` the share of valid pairs that agreed.
    """
    if not depths:
        raise ValueError("need at least one depth map")
    ref = depths[0]
    for d in depths:
        if d.shape != ref.shape or d.ref_cam != ref.ref_cam:
            raise ValueError("all depth maps must share one grid (same reference camera and size)")
        if d.sigma_m is None:
            raise ValueError("every depth map needs sigma_m for fusion")

    Z = np.stack([d.depth_m for d in depths]).astype(np.float64)  # (P, H, W)
    S = np.stack([d.sigma_m for d in depths if d.sigma_m is not None]).astype(np.float64)
    valid = np.isfinite(Z) & np.isfinite(S) & (S > 0)
    Zv = np.where(valid, Z, np.nan)
    with warnings.catch_warnings():  # pixels that no pair can see are all-NaN on purpose; numpy warns about them
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Zv, axis=0)  # (H, W); NaN where no pair is valid
    gate = outlier_sigmas * S + 0.01 * np.nan_to_num(med, nan=0.0)  # 1 percent of depth as a floor
    keep = valid & (np.abs(Z - med[None]) <= gate)
    W = np.where(keep, 1.0 / np.where(S > 0, S, 1.0) ** 2, 0.0)
    wsum = W.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fused = (W * np.where(keep, Z, 0.0)).sum(axis=0) / wsum
        sigma = 1.0 / np.sqrt(wsum)
        conf = keep.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    fused[wsum == 0] = np.nan
    sigma[wsum == 0] = np.nan
    pairs = tuple(p for d in depths for p in d.pairs)
    return DepthMap(
        ref.ref, ref.ref_cam, ref.K, fused.astype(np.float32), DepthMethod.FUSED, pairs, conf.astype(np.float32), sigma.astype(np.float32)
    )
