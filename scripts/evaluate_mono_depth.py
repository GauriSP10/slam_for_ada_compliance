"""CLI: independent scale check of the stereo depth against single-image metric depth (task A14).

Per keyframe: (1) the ratio of stereo to monocular depth on shared pixels (1.5-8 m), and
(2) the camera height obtained by fitting a floor to the MONOCULAR depth, to compare with the stereo floor (1.27 m).
Usage: python scripts/evaluate_mono_depth.py [frames]
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

from slam_for_ada.config import load_config
from slam_for_ada.dataio.frame_cache import load_camera_set
from slam_for_ada.datatypes import DepthMap, DepthMethod
from slam_for_ada.eval.sampling import sample_keyframes
from slam_for_ada.pipeline import VisionPipeline
from slam_for_ada.vision.floor_plane import fit_floor
from slam_for_ada.vision.mono_depth import MetricMonoDepth


def main(n_frames: int = 24) -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pipe = VisionPipeline(cfg, "sgbm")
    mono = MetricMonoDepth()
    floor = pipe.pooled_floor()
    ratios, heights, rel = [], [], []
    for bag, key in sample_keyframes(cfg, n_frames):
        cs = load_camera_set(cfg.path("cache"), bag, key)
        stereo, _ = pipe.stack.depth(cs)
        zm = mono(pipe.undistort(cs.images[0]))
        zs = stereo.depth_m
        ok = np.isfinite(zs) & (zs > 1.5) & (zs < 8.0) & (zm > 0.1)
        if ok.sum() < 5000:
            continue
        ratios.append(float(np.median(zs[ok] / zm[ok])))
        rel.append(float(np.median(np.abs(zs[ok] - zm[ok]) / zs[ok])))
        mdm = DepthMap(cs.ref, 0, stereo.K, zm.astype(np.float32), DepthMethod.MONOCULAR, (), None, None)
        f = fit_floor(mdm, cfg.floor)
        if f is not None:
            heights.append(f.d)
    r, h, rl = np.array(ratios), np.array(heights), np.array(rel)
    print(f"{len(r)} frames")
    print(
        f"stereo / mono depth ratio per frame: median {np.median(r):.3f}, "
        f"spread (IQR) {np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f}"
    )
    print(f"median relative difference |stereo - mono| / stereo: {np.median(rl):.1%}")
    print(
        f"camera height from a floor fitted to MONOCULAR depth: median {np.median(h):.3f} m over {len(h)} frames "
        f"(stereo floor: {floor.d:.3f} m, difference {100 * (np.median(h) - floor.d):+.1f} cm)"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
