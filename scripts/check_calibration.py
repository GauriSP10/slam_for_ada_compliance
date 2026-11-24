"""CLI: rectification quality over many cached keyframes.  Usage: python scripts/check_calibration.py [frames_per_bag]"""

from __future__ import annotations

import sys

from slam_for_ada.calib.calib_check import check_pair
from slam_for_ada.calib.rectify import build_rectifiers
from slam_for_ada.calib.rig import load_rig
from slam_for_ada.config import load_config
from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.dataio.frame_cache import load_camera_set, load_index


def main(per_bag: int = 12) -> None:
    cfg = load_config()
    rig = load_rig(cfg.path("calibration") / "cams_calib.yaml")
    rects = build_rectifiers(rig, cfg.stereo.right_cams)
    sets = []
    for bag in list_bags(cfg.path("bags")):
        idx = load_index(cfg.path("cache"), bag.stem)
        step = max(1, len(idx) // per_bag)
        sets += [load_camera_set(cfg.path("cache"), bag.stem, r["key"]) for r in idx[::step] if r["usable"]]
    print(f"{len(sets)} keyframes\n pair   baseline  matches  median|dy|  p90|dy|  mean dy (px)")
    for k, rect in rects.items():
        r = check_pair(rect, sets)
        print(
            f" 0-{k}    {r.baseline_m:.3f} m  {r.n_matches:7d}   {r.median_abs_dy_px:6.2f}     "
            f"{r.p90_abs_dy_px:6.2f}   {r.mean_dy_px:+6.2f}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
