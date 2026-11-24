"""CLI: automatic survey. Detect and measure features in EVERY usable keyframe, with no hand annotation (tasks A12 + A13).

Prompts: levers, thermostats, card readers, outlets, handrails. Each detection with a mask and a reliable depth patch is
measured; results go to outputs/survey_open_vocab.csv and a per-class summary is printed next to the tape value.
Many objects of one class exist (every door has a lever), so the summary is a population check, not a per-object match.
Usage: python scripts/survey_open_vocab.py [score_threshold]
"""

from __future__ import annotations

import csv
import sys
import warnings

import numpy as np

from slam_for_ada.config import load_config
from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.dataio.frame_cache import load_camera_set, load_index
from slam_for_ada.datatypes import M_PER_INCH, FeatureKind
from slam_for_ada.pipeline import VisionPipeline
from slam_for_ada.vision.auto_measure import handrail_height, height_from_detection
from slam_for_ada.vision.open_vocab import PROMPT_KINDS, OpenVocabDetector, to_object_detections

SURVEY = {  # prompt -> tape value (inches) it is compared with in the summary
    "a door lever lock": ("door_hardware_wooden", 39.5),
    "a wall thermostat": ("thermostat_height", 51.25),
    "a card reader": ("card_reader_height", 47.0),
    "an electrical outlet": ("plug_point_height", 19.3),
    "a handrail": ("handrail_height", 35.5),
}


def main(threshold: float = 0.25) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pipe = VisionPipeline(cfg, "sgbm")
    floor = pipe.pooled_floor()
    det = OpenVocabDetector()
    rows = []
    frames = [(b.stem, r["key"]) for b in list_bags(cfg.path("bags")) for r in load_index(cfg.path("cache"), b.stem) if r["usable"]]
    for i, (bag, key) in enumerate(frames):
        cs = load_camera_set(cfg.path("cache"), bag, key)
        depth, _ = pipe.stack.depth(cs)
        raw = det.detect(pipe.undistort(cs.images[0]), list(SURVEY), threshold)
        kept = [d for d in raw if d.prompt in PROMPT_KINDS]  # to_object_detections drops prompts without a kind
        for obj, d in zip(to_object_detections(raw, cs.ref), kept):
            r = handrail_height(depth, floor, obj) if obj.kind is FeatureKind.HANDRAIL_HEIGHT else height_from_detection(depth, floor, obj)
            if r is None:
                continue
            rows.append(
                {
                    "bag": bag[-4:],
                    "key": key,
                    "prompt": d.prompt,
                    "score": round(d.score, 3),
                    "height_in": round(r.value_m / M_PER_INCH, 2),
                    "sigma_in": round(r.sigma_m / M_PER_INCH, 2),
                    "distance_m": round(r.distance_m, 2),
                }
            )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(frames)} keyframes, {len(rows)} measurements so far", flush=True)
    out = cfg.path("outputs") / "survey_open_vocab.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} automatic measurements from {len(frames)} keyframes -> {out}\n")
    print(f"{'class':22s} {'n':>4s} {'median':>7s} {'IQR':>13s} {'tape':>6s}  {'median err':>10s}")
    for prompt, (_fid, tape) in SURVEY.items():
        v = np.array([r["height_in"] for r in rows if r["prompt"] == prompt])
        if v.size == 0:
            print(f"{prompt:22s} {0:4d}   (none)")
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        print(f"{prompt:22s} {v.size:4d} {med:7.1f} {q1:6.1f}-{q3:5.1f} {tape:6.2f}  {med - tape:+10.2f}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.25)
