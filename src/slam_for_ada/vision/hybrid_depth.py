"""Hybrid matcher: learned coverage, classical verification.

A dense network answers at almost every pixel, including thin or transparent structures (a handrail in front of glass)
where it can return a smooth, plausible and wrong depth. Block matching answers at fewer pixels but its failures are
geometric, so where both have a value they can be cross-checked. This keeps the learned depth where it agrees with the
classical one (or where the classical one is silent, e.g. textureless walls) and falls back to the classical value where
the two disagree beyond their combined error bars.
"""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import DepthMap, DepthMethod

NAN = np.float32("nan")


def combine_matchers(classical: DepthMap, learned: DepthMap, n_sigmas: float = 3.0, rel_tol: float = 0.05) -> DepthMap:
    """Per-pixel combination of two depth maps of the same pair, on the same grid.

    Where both are valid: if they agree within ``n_sigmas`` combined sigma (and at least ``rel_tol`` of the depth) the
    learned value is kept; otherwise the classical value wins. Where only one is valid, that one is used.
    The result is labelled FUSED because it mixes two matchers, and lists the pair once.
    """
    if classical.shape != learned.shape or classical.ref_cam != learned.ref_cam:
        raise ValueError("both depth maps must share one grid")
    if classical.sigma_m is None or learned.sigma_m is None:
        raise ValueError("both depth maps need sigma_m")
    zc, zl = classical.depth_m.astype(np.float64), learned.depth_m.astype(np.float64)
    sc, sl = classical.sigma_m.astype(np.float64), learned.sigma_m.astype(np.float64)
    vc, vl = np.isfinite(zc), np.isfinite(zl)
    both = vc & vl
    tol = np.maximum(n_sigmas * np.hypot(np.nan_to_num(sc), np.nan_to_num(sl)), rel_tol * np.nan_to_num(zc))
    agree = both & (np.abs(zc - zl) <= tol)
    use_learned = agree | (vl & ~vc)
    use_classical = vc & ~use_learned
    z = np.where(use_learned, zl, np.where(use_classical, zc, np.nan))
    s = np.where(use_learned, sl, np.where(use_classical, sc, np.nan))
    return DepthMap(
        classical.ref, classical.ref_cam, classical.K, z.astype(np.float32), DepthMethod.FUSED, classical.pairs, None, s.astype(np.float32)
    )
