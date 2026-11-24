"""CLI: does the open-vocabulary detector find the features we annotated, and can it measure them unaided?  (task A12)

For every hand-annotated height feature, a detection is a HIT when its mask contains the annotated point and its prompt
matches the feature kind. For each hit, the height is measured straight from the mask and compared with the height from
the hand-placed point and with the tape. Detections are cached, so a repeat run does not need the GPU models again.
Writes overlays to outputs/open_vocab/ and a table to outputs/open_vocab_results.csv.
Usage: python scripts/evaluate_open_vocab.py [threshold]
"""

from __future__ import annotations

import csv
import sys
import warnings

import cv2
import numpy as np

from slam_for_ada.config import load_config
from slam_for_ada.dataio.annotations import load_annotations
from slam_for_ada.dataio.frame_cache import load_camera_set
from slam_for_ada.datatypes import FeatureKind
from slam_for_ada.eval.ground_truth import load_ground_truth
from slam_for_ada.features import HEIGHT_KINDS
from slam_for_ada.pipeline import VisionPipeline
from slam_for_ada.vision.auto_measure import handrail_height, height_from_detection
from slam_for_ada.vision.measure import to_measurement
from slam_for_ada.vision.open_vocab import PROMPT_KINDS, OpenVocabDetector, to_object_detections
from slam_for_ada.viz.overlays import put_label


def _fmt(m) -> str:
    return f"{m.value_in:7.2f}" if m is not None else "     --"


def _err(m, tape: float) -> str:
    return f"{m.value_in - tape:+8.2f}" if m is not None else "      --"


def main(threshold: float = 0.15) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pipe = VisionPipeline(cfg, "sgbm")
    floor = pipe.pooled_floor()
    truths = load_ground_truth(cfg.path("ground_truth"))
    det = OpenVocabDetector()
    prompts = list(PROMPT_KINDS)
    out = cfg.path("outputs") / "open_vocab"
    out.mkdir(parents=True, exist_ok=True)
    anns = [a for a in load_annotations(cfg.path("annotations"), cfg.path("bags")) if a.kind in HEIGHT_KINDS]
    by_frame: dict = {}
    for a in anns:
        by_frame.setdefault((a.bag, a.key), []).append(a)
    rows, hits = [], 0
    print(f"{'annotation':34s} {'found':>5s} {'auto':>7s} {'by hand':>8s} {'tape':>6s}  {'auto err':>8s} {'hand err':>8s}")
    for (bag, key), group in by_frame.items():
        cs = load_camera_set(cfg.path("cache"), bag, key)
        img = pipe.undistort(cs.images[0])
        depth, _ = pipe.stack.depth(cs)
        raw = det.detect(img, prompts, threshold)
        objs = to_object_detections(raw, cs.ref)
        vis = img.copy()
        for d in raw:
            if d.mask is not None:
                vis[d.mask] = (0.6 * vis[d.mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
            x0, y0, x1, y1 = [int(v) for v in d.box]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
            put_label(vis, f"{d.prompt.split(' ', 1)[1]} {d.score:.2f}", (x0, max(y0 - 4, 12)), scale=0.4)
        for a in group:
            px, py = [int(round(v)) for v in a.geometry["point_px"]]
            cands = [o for o in objs if o.kind is a.kind and o.mask is not None and o.mask[py, px]]
            best = max(cands, key=lambda o: o.score) if cands else None
            tape = truths[a.feature_id].value_in
            hand = next(iter(pipe.measure(a, cs, depth, floor)), None)
            auto = None
            if best is not None:
                r = (
                    handrail_height(depth, floor, best)
                    if a.kind is FeatureKind.HANDRAIL_HEIGHT
                    else height_from_detection(depth, floor, best)
                )
                auto = to_measurement(r, cs.ref, a.kind, a.feature_id, depth) if r is not None else None
            hits += best is not None
            cv2.circle(vis, (px, py), 6, (255, 0, 0) if best is not None else (0, 0, 255), -1)
            print(
                f"{a.id:34s} {'HIT' if best is not None else 'miss':>5s} {_fmt(auto)} {_fmt(hand):>8s} {tape:6.2f}  "
                f"{_err(auto, tape)} {_err(hand, tape)}",
                flush=True,
            )
            rows.append(
                {
                    "annotation": a.id,
                    "found": best is not None,
                    "auto_in": round(auto.value_in, 2) if auto else "",
                    "hand_in": round(hand.value_in, 2) if hand else "",
                    "tape_in": tape,
                }
            )
        cv2.imwrite(str(out / f"{bag[-4:]}_{key}.jpg"), vis)
    both = [r for r in rows if r["auto_in"] != "" and r["hand_in"] != ""]
    print(f"\n{hits}/{len(rows)} annotated features found by an open-vocabulary detection of the right kind")
    if both:
        ae = np.mean([abs(r["auto_in"] - r["tape_in"]) for r in both])
        he = np.mean([abs(r["hand_in"] - r["tape_in"]) for r in both])
        print(f"on the {len(both)} with both: mean |error| vs tape: automatic {ae:.2f} in, hand-placed {he:.2f} in")
    path = cfg.path("outputs") / "open_vocab_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.15)
