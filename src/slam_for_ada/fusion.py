"""Combine repeated observations of one feature into a single estimate."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from slam_for_ada.datatypes import FeatureEstimate, Measurement


def fuse_observations(measurements: Sequence[Measurement]) -> list[FeatureEstimate]:
    """One estimate per (feature, method): the inverse-variance weighted mean of its per-frame observations.

    With three or more observations, those further than three sigma (or 5 percent) from the median are dropped as gross
    outliers. The reported sigma is never smaller than the disagreement among the observations, so a feature measured
    several times with inconsistent results is not given false confidence.
    """
    groups: dict[tuple[str, str], list[Measurement]] = defaultdict(list)
    for m in measurements:
        if m.feature_id is not None:
            groups[(m.feature_id, m.method.value)].append(m)
    out = []
    for (fid, _), obs in groups.items():
        v = np.array([o.value_in for o in obs])
        s = np.array([max(o.sigma_in, 1e-3) for o in obs])
        if len(obs) >= 3:  # with fewer than 3 we cannot tell which one is wrong
            med = float(np.median(v))
            keep = np.abs(v - med) <= np.maximum(3 * s, 0.05 * abs(med))
            if keep.sum() >= 2:
                obs, v, s = [o for o, k in zip(obs, keep) if k], v[keep], s[keep]
        w = 1.0 / s**2
        mean = float((w * v).sum() / w.sum())
        model_sigma = float(1.0 / np.sqrt(w.sum()))
        scatter = float(np.sqrt((w * (v - mean) ** 2).sum() / w.sum()) / np.sqrt(len(obs))) if len(obs) > 1 else 0.0
        out.append(FeatureEstimate(fid, obs[0].kind, obs[0].method, mean, max(model_sigma, scatter), tuple(obs)))
    return out
