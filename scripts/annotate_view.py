"""CLI: render an UNDISTORTED cam0 keyframe with a pixel grid, for reading annotation coordinates off by eye.

Usage: python scripts/annotate_view.py <bag-suffix> <key> <out.jpg> [--depth]
The grid labels are cam0-undistorted pixel coordinates: the same grid every depth map and measurement uses.
"""

from __future__ import annotations

import sys

import cv2

from slam_for_ada.calib.rig import load_rig
from slam_for_ada.calib.undistort import RefUndistorter
from slam_for_ada.config import load_config
from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.dataio.frame_cache import load_camera_set
from slam_for_ada.viz.overlays import draw_grid


def main(argv: list[str]) -> None:
    bag_suffix, key, out = argv[0], int(argv[1]), argv[2]
    cfg = load_config()
    bag = next(b.stem for b in list_bags(cfg.path("bags")) if b.stem.endswith(bag_suffix))
    rig = load_rig(cfg.path("calibration") / "cams_calib.yaml")
    img = RefUndistorter(rig)(load_camera_set(cfg.path("cache"), bag, key).images[0])
    cv2.imwrite(out, draw_grid(img))
    print(f"wrote {out} ({bag}, key {key})")


if __name__ == "__main__":
    main(sys.argv[1:])
