"""CLI: does fusing a feature over neighbouring keyframes beat a single frame?  (task A13)

For each annotated height feature, track it through up to N keyframes on each side (visual odometry + a fresh depth
reading in each), then compare the single-frame error, the fused error, and the spread between frames.
Usage: python scripts/evaluate_temporal_fusion.py [neighbours_each_side]
"""

from __future__ import annotations

import csv
import sys
import warnings

import numpy as np

from slam_for_ada.config import load_config
from slam_for_ada.dataio.annotations import load_annotations
from slam_for_ada.eval.ground_truth import load_ground_truth
from slam_for_ada.features import HEIGHT_KINDS
from slam_for_ada.fusion import fuse_observations
from slam_for_ada.pipeline import VisionPipeline


def main(n: int = 3) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pipe = VisionPipeline(cfg, "sgbm")
    truths = load_ground_truth(cfg.path("ground_truth"))
    rows = []
    print(
        f"{'annotation':34s} {'frames':>6s} {'single':>8s} {'fused':>8s} {'tape':>6s}   "
        f"{'single err':>10s} {'fused err':>9s}   frame-to-frame std"
    )
    for a in load_annotations(cfg.path("annotations"), cfg.path("bags")):
        if a.kind not in HEIGHT_KINDS:
            continue
        obs = pipe.temporal_observations(a, n)
        if len(obs) < 2:
            print(f"{a.id:34s} {len(obs):6d}   (could not track into neighbouring frames)")
            continue
        tape = truths[a.feature_id].value_in
        center = next((o for o in obs if o.ref.index == a.key), obs[0])
        fused = fuse_observations(obs)[0]
        vals = np.array([o.value_in for o in obs])
        print(
            f"{a.id:34s} {len(obs):6d} {center.value_in:8.2f} {fused.value_in:8.2f} {tape:6.2f}   "
            f"{center.value_in - tape:+10.2f} {fused.value_in - tape:+9.2f}   {vals.std():.2f} in"
        )
        rows.append(
            {
                "annotation": a.id,
                "frames": len(obs),
                "single_in": round(center.value_in, 2),
                "fused_in": round(fused.value_in, 2),
                "tape_in": tape,
                "single_err": round(center.value_in - tape, 2),
                "fused_err": round(fused.value_in - tape, 2),
                "frame_std_in": round(float(vals.std()), 2),
                "values": " ".join(f"{v:.1f}" for v in vals),
            }
        )
    if rows:
        s = np.mean([abs(r["single_err"]) for r in rows])
        f = np.mean([abs(r["fused_err"]) for r in rows])
        print(f"\nmean |error| over {len(rows)} features: single frame {s:.2f} in, fused over neighbours {f:.2f} in")
        out = cfg.path("outputs") / "temporal_fusion.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
