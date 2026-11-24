"""CLI: annotated frames for every measurement, plus a demo video of the hallway run (task A10).

  outputs/overlays/<annotation id>.jpg   the measurement drawn on its frame, with measured vs tape value
  outputs/demo_hallway.mp4               undistorted frame with the ADA reach band tinted | fused depth | detected doors
Usage: python scripts/make_overlays.py [bag-suffix ...]     (default video bags: 32 33 34)
"""

from __future__ import annotations

import sys
import warnings

import cv2
import numpy as np

from slam_for_ada.config import load_config
from slam_for_ada.dataio.annotations import load_annotations
from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.dataio.frame_cache import load_camera_set, load_index
from slam_for_ada.datatypes import FeatureKind
from slam_for_ada.eval.ground_truth import load_ground_truth
from slam_for_ada.features import HEIGHT_KINDS
from slam_for_ada.pipeline import VisionPipeline
from slam_for_ada.vision.door_detector import detect_doors
from slam_for_ada.vision.width import as_jamb, edge_energy, snap_jamb
from slam_for_ada.viz.overlays import depth_colormap, draw_box, draw_jamb, draw_point, put_label, reach_band_overlay


def annotation_overlays(pipe: VisionPipeline, cfg) -> None:
    out = cfg.path("outputs") / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    truths = load_ground_truth(cfg.path("ground_truth"))
    floor = pipe.pooled_floor()
    for a in load_annotations(cfg.path("annotations"), cfg.path("bags")):
        cs = load_camera_set(cfg.path("cache"), a.bag, a.key)
        depth, _ = pipe.stack.depth(cs)
        img = reach_band_overlay(pipe.undistort(cs.images[0]), depth, floor)
        g, k = a.geometry, a.kind
        if k is FeatureKind.DOOR_WIDTH:
            energy = edge_energy(cv2.cvtColor(pipe.undistort(cs.images[0]), cv2.COLOR_BGR2GRAY))
            for key in ("left_jamb", "right_jamb"):
                draw_jamb(img, snap_jamb(energy, as_jamb(g[key]), cfg.measure.jamb_snap_px)[0])
            for b in g["wall_rois"]:
                draw_box(img, b, (255, 200, 0))
        elif k in HEIGHT_KINDS:
            draw_box(img, g["depth_roi"], (255, 200, 0))
            draw_point(img, g["point_px"], (0, 0, 255))
        elif k is FeatureKind.HALLWAY_WIDTH:
            draw_box(img, g["left_wall_roi"], (255, 200, 0))
            draw_box(img, g["right_wall_roi"], (255, 200, 0))
            for side in ("left", "right"):
                for p in g.get("floor_contacts", {}).get(side, []):
                    draw_point(img, p, (0, 0, 255))
        ms = pipe.measure(a, cs, depth, floor)
        tape = truths[a.feature_id].value_in
        put_label(img, f"{a.feature_id}   tape {tape:g} in", (10, 24), scale=0.6)
        for i, m in enumerate(ms):
            put_label(
                img,
                f"{m.method.value}: {m.value_in:.1f} in  (err {m.value_in - tape:+.1f}, +-{m.sigma_in:.1f})",
                (10, 48 + 22 * i),
                (0, 255, 255),
                0.55,
            )
        if not ms:
            put_label(img, "no reliable measurement (gate declined)", (10, 48), (0, 165, 255), 0.55)
        cv2.imwrite(str(out / f"{a.id}.jpg"), img)
    print(f"overlays -> {out}")


def demo_video(pipe: VisionPipeline, cfg, suffixes: list[str]) -> None:
    floor = pipe.pooled_floor()
    path = cfg.path("outputs") / "demo_hallway.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (1440, 540))  # pyright: ignore[reportAttributeAccessIssue]  # missing from the OpenCV 5.1 stubs
    n = 0
    for bag in [b for b in list_bags(cfg.path("bags")) if any(b.stem.endswith("_" + s) for s in suffixes)]:
        for rec in load_index(cfg.path("cache"), bag.stem):
            if not rec["usable"]:
                continue
            cs = load_camera_set(cfg.path("cache"), bag.stem, rec["key"])
            depth, _ = pipe.stack.depth(cs)
            img = pipe.undistort(cs.images[0])
            left = reach_band_overlay(img, depth, floor)
            for c in detect_doors(img, depth, cs.ref)[:3]:
                cv2.polylines(left, [c.corners_px.astype(np.int32)], True, (0, 165, 255) if c.is_glass else (255, 120, 0), 2, cv2.LINE_AA)
            put_label(left, f"{bag.stem[-4:]}  key {rec['key']}   green = 15-48 in above floor", (8, 24), scale=0.55)
            right = depth_colormap(depth.depth_m)
            put_label(right, "fused multi-baseline depth (0-8 m)", (8, 24), scale=0.55)
            vw.write(np.hstack([left, right]))
            n += 1
    vw.release()
    print(f"video -> {path} ({n} frames)")


def main(argv: list[str]) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pipe = VisionPipeline(cfg)
    annotation_overlays(pipe, cfg)
    demo_video(pipe, cfg, argv or ["32", "33", "34"])


if __name__ == "__main__":
    main(sys.argv[1:])
