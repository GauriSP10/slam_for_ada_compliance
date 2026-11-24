"""One floor plane for the whole sequence, pooled over many frames and cached."""

from __future__ import annotations

import json

import numpy as np

from slam_for_ada.config import Config
from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.dataio.frame_cache import load_camera_set, load_index
from slam_for_ada.datatypes import Plane3D
from slam_for_ada.vision.depth_stack import DepthStack
from slam_for_ada.vision.floor_plane import combine_floors, fit_floor


def pooled_floor(cfg: Config, stack: DepthStack, stride: int = 4, use_cache: bool = True) -> Plane3D:
    """The floor plane shared by every frame (the rig is rigidly mounted, so camera height and tilt are constant).

    Per-frame fits (every ``stride``-th usable keyframe of every bag) are pooled with outlier rejection. The result is
    cached in ``data/cache/floor.json``. ``stack`` should be the fast classical matcher, since ~50 frames are processed.
    """
    cache = cfg.path("cache") / "floor.json"
    if use_cache and cache.exists():
        j = json.loads(cache.read_text())
        return Plane3D(np.array(j["normal"]), j["d"], j["n_inliers"], j["inlier_ratio"], j["rmse_m"], j["source"])
    planes = []
    for bag in list_bags(cfg.path("bags")):
        for k in [r["key"] for r in load_index(cfg.path("cache"), bag.stem) if r["usable"]][::stride]:
            fused, _ = stack.depth(load_camera_set(cfg.path("cache"), bag.stem, k))
            p = fit_floor(fused, cfg.floor)
            if p is not None:
                planes.append(p)
    floor = combine_floors(planes)
    cache.write_text(
        json.dumps(
            {
                "normal": floor.normal.tolist(),
                "d": floor.d,
                "n_inliers": floor.n_inliers,
                "inlier_ratio": floor.inlier_ratio,
                "rmse_m": floor.rmse_m,
                "source": floor.source,
            }
        )
    )
    return floor
