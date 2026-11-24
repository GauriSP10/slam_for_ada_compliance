"""CLI: does the Harris+edge door detector find the doors we annotated by hand?  (task A6 check)

A detection counts as a hit when both of its jambs lie within 15 px of the annotated jambs.
Also writes an overlay image per frame to outputs/door_detection/.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from slam_for_ada.calib.undistort import RefUndistorter
from slam_for_ada.config import load_config
from slam_for_ada.dataio.annotations import load_annotations
from slam_for_ada.dataio.frame_cache import load_camera_set
from slam_for_ada.datatypes import FeatureKind
from slam_for_ada.vision.depth_stack import DepthStack
from slam_for_ada.vision.door_detector import detect_doors
from slam_for_ada.viz.overlays import put_label


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    stack = DepthStack(cfg)
    undist = RefUndistorter(stack.rig)
    out = cfg.path("outputs") / "door_detection"
    out.mkdir(parents=True, exist_ok=True)
    doors = [a for a in load_annotations(cfg.path("annotations"), cfg.path("bags")) if a.kind is FeatureKind.DOOR_WIDTH]
    hits = 0
    for a in doors:
        cs = load_camera_set(cfg.path("cache"), a.bag, a.key)
        depth, _ = stack.depth(cs)
        img = undist(cs.images[0])
        cands = detect_doors(img, depth, cs.ref)
        want = (a.geometry["left_jamb"][0][0], a.geometry["right_jamb"][0][0])
        hit = None
        for i, c in enumerate(cands):
            if abs(c.corners_px[0, 0] - want[0]) <= 15 and abs(c.corners_px[1, 0] - want[1]) <= 15:
                hit = i
                break
        hits += hit is not None
        print(f"{a.id:32s} {len(cands)} candidates; annotated door " + (f"found as #{hit + 1}" if hit is not None else "NOT found"))
        vis = img.copy()
        for i, c in enumerate(cands):
            col = (0, 255, 0) if i == hit else ((0, 165, 255) if c.is_glass else (255, 120, 0))
            cv2.polylines(vis, [c.corners_px.astype(np.int32)], True, col, 2, cv2.LINE_AA)
            put_label(vis, f"#{i + 1} {c.score:.2f}", (int(c.corners_px[0, 0]) + 4, int(c.corners_px[0, 1]) + 22 + 16 * i), scale=0.5)
        cv2.rectangle(vis, (int(want[0]) - 1, 0), (int(want[1]) + 1, 539), (0, 0, 255), 1)
        cv2.imwrite(str(out / f"{a.id}.jpg"), vis)
    print(f"\n{hits}/{len(doors)} annotated doors recovered; overlays in {out}")


if __name__ == "__main__":
    main()
