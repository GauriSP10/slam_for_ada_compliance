"""CLI: how precise is each camera pair, and does fusing them help?  (task A4 chart)

The floor is flat, so after fitting one pooled floor plane, the scatter of floor points about that plane,
binned by distance, measures each pair's depth precision with no ground truth needed.
Usage: python scripts/evaluate_depth.py [frames]
"""

from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slam_for_ada.config import load_config
from slam_for_ada.dataio.frame_cache import load_camera_set
from slam_for_ada.eval.floor_scatter import floor_points, robust_sigma_cm
from slam_for_ada.eval.sampling import sample_keyframes
from slam_for_ada.vision.depth_stack import DepthStack
from slam_for_ada.vision.floor_plane import combine_floors, fit_floor, floor_spread, reject_floor_outliers

BINS = [(1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 8.0)]


def main(n_frames: int = 60) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    stack = DepthStack(cfg)
    picks = sample_keyframes(cfg, n_frames)

    frames, planes = [], []
    for bag, k in picks:
        fused, maps = stack.depth(load_camera_set(cfg.path("cache"), bag, k))
        fl = fit_floor(fused, cfg.floor)
        if fl is not None:
            planes.append(fl)
        frames.append((fused, maps))
    pooled = combine_floors(planes)
    kept = reject_floor_outliers(planes)
    tilt, hstd = floor_spread(kept)
    print(
        f"{len(planes)}/{len(picks)} frames gave a floor; {len(kept)} agree. Pooled camera height {pooled.d:.3f} m; "
        f"spread of the agreeing frames: tilt {tilt:.2f} deg, height {hstd * 100:.1f} cm"
    )

    names = [f"0-{k} ({stack.rectifiers[k].baseline_m:.2f} m)" for k in stack.rectifiers] + ["fused"]
    acc = {n: [[] for _ in BINS] for n in names}
    for fused, maps in frames:
        # compare like with like: only pixels where every pair and the fused map have a value
        common = np.all([np.isfinite(m.depth_m) for m in maps + [fused]], axis=0)
        for name, m in zip(names, maps + [fused]):
            z, r = floor_points(m.depth_m, stack.K_ref, pooled, cfg.floor.v_min_frac, common)
            for bi, (lo, hi) in enumerate(BINS):
                sel = (z >= lo) & (z < hi)
                acc[name][bi].append(r[sel])

    sigma = {n: [] for n in names}
    print("\nrobust height scatter on the floor (cm), by distance; [n points]")
    print(f"{'pair':16s}" + "".join(f"{lo:>4.1f}-{hi:<4.1f} m   " for lo, hi in BINS))
    for n in names:
        row = []
        for bi in range(len(BINS)):
            r = np.concatenate(acc[n][bi]) if acc[n][bi] else np.zeros(0)
            s = robust_sigma_cm(r)
            sigma[n].append(s)
            row.append(f"{s:5.1f} [{r.size // 1000:3d}k] ")
        print(f"{n:16s}" + "  ".join(row))

    out = cfg.path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mid = [(lo + hi) / 2 for lo, hi in BINS]
    for n in names:
        ax.plot(mid, sigma[n], marker="o", lw=3 if n == "fused" else 1.5, label=n, color="black" if n == "fused" else None)
    ax.set_xlabel("distance to floor point (m)")
    ax.set_ylabel("floor height scatter, robust sigma (cm)")
    ax.set_title("Depth precision by camera pair and distance (lower is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "depth_precision_by_baseline.png", dpi=150)
    print(f"\nsaved {out / 'depth_precision_by_baseline.png'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
