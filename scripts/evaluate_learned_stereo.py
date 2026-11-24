"""CLI: classical SGBM vs learned Fast-FoundationStereo, on the same pixels (task A11).

Two numbers per camera pair and distance band:
  coverage  share of the floor region that has a depth value
  scatter   robust sigma (cm) of floor-point height about the pooled floor plane, on pixels BOTH methods produce
Usage: python scripts/evaluate_learned_stereo.py [frames] [pairs, e.g. 2 4]
The learned disparity of each frame is cached, so a repeat run is fast.
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
from slam_for_ada.pipeline import VisionPipeline
from slam_for_ada.vision.depth_stack import DepthStack

BINS = [(3.5, 4.5), (4.5, 6.0), (6.0, 8.0)]


def main(n_frames: int = 16, pairs: tuple[int, ...] = (2, 4)) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    floor = VisionPipeline(cfg).pooled_floor()
    sg, ln = DepthStack(cfg, method="sgbm"), DepthStack(cfg, method="learned")
    print(f"learned model on: {ln.estimator.provider}", flush=True)  # type: ignore[attr-defined]
    picks = sample_keyframes(cfg, n_frames)

    cover = {p: {"sgbm": [], "learned": []} for p in pairs}
    res = {p: {"sgbm": [[] for _ in BINS], "learned": [[] for _ in BINS]} for p in pairs}
    for i, (bag, key) in enumerate(picks):
        cs = load_camera_set(cfg.path("cache"), bag, key)
        for p in pairs:
            rect = sg.rectifiers[p]
            rp = rect.rectify(cs)
            ds, dl = sg.estimator(rp), ln.estimator(rp)
            lower = slice(int(cfg.floor.v_min_frac * ds.shape[0]), None)
            cover[p]["sgbm"].append(np.isfinite(ds.depth_m[lower]).mean())
            cover[p]["learned"].append(np.isfinite(dl.depth_m[lower]).mean())
            both = np.isfinite(ds.depth_m) & np.isfinite(dl.depth_m)
            for name, m in (("sgbm", ds), ("learned", dl)):
                z, h = floor_points(m.depth_m, sg.K_ref, floor, cfg.floor.v_min_frac, both)
                for bi, (lo, hi) in enumerate(BINS):
                    res[p][name][bi].append(h[(z >= lo) & (z < hi)])
        print(f"  frame {i + 1}/{len(picks)} done", flush=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mid = [(lo + hi) / 2 for lo, hi in BINS]
    print("\nfloor coverage (share of lower image with a depth value) and scatter (cm), common pixels")
    for p in pairs:
        B = sg.rectifiers[p].baseline_m
        print(f" pair 0-{p} (B={B:.2f} m): coverage sgbm {np.mean(cover[p]['sgbm']):.0%}  learned {np.mean(cover[p]['learned']):.0%}")
        for name in ("sgbm", "learned"):
            row = []
            for bi in range(len(BINS)):
                r = np.concatenate(res[p][name][bi]) if res[p][name][bi] else np.zeros(0)
                row.append(robust_sigma_cm(r))
            print(f"   {name:8s}" + "".join(f"  {lo:.1f}-{hi:.1f} m: {s_:5.2f}" for (lo, hi), s_ in zip(BINS, row)))
            ax.plot(mid, row, marker="o", ls="-" if name == "learned" else "--", label=f"{name} 0-{p}")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("floor height scatter, robust sigma (cm)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Learned vs classical stereo on the same floor pixels (lower is better)")
    out = cfg.path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "learned_vs_sgbm_floor.png", dpi=150)
    print(f"\nsaved {out / 'learned_vs_sgbm_floor.png'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    pr = tuple(int(a) for a in sys.argv[2:]) if len(sys.argv) > 2 else (2, 4)
    main(n, pr)
