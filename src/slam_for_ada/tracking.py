"""Follow a height feature through neighbouring keyframes and measure it again in each (Tier 2, A13)."""

from __future__ import annotations

import cv2
import numpy as np

from slam_for_ada.config import Config
from slam_for_ada.dataio.annotations import Annotation
from slam_for_ada.dataio.frame_cache import load_camera_set, load_index
from slam_for_ada.datatypes import Measurement, Plane3D
from slam_for_ada.features import HEIGHT_KINDS, Undistorter
from slam_for_ada.vision.depth_stack import DepthStack
from slam_for_ada.vision.geometry import pixel_rays
from slam_for_ada.vision.height import height_of_point
from slam_for_ada.vision.measure import RawMeasurement, roi_depth, to_measurement
from slam_for_ada.vision.temporal_fusion import propagate_point, relative_pose


def track_height_feature(
    ann: Annotation, cfg: Config, stack: DepthStack, floor: Plane3D, undistort: Undistorter, n_neighbors: int = 3
) -> list[Measurement]:
    """Measure one annotated height feature in the keyframes around its own, tracking it with visual odometry.

    The feature's 3D point is carried into each neighbouring frame using the camera motion between consecutive
    keyframes; there a fresh, independent depth reading is taken at the predicted pixel. Heights above the floor do not
    depend on camera position, so the observations can be fused. A chain ends in a direction when the motion cannot be
    estimated, a frame is unusable, or the point leaves the view.
    """
    if ann.kind not in HEIGHT_KINDS:
        return []
    g, K = ann.geometry, stack.K_ref
    cache = cfg.path("cache")
    usable = {r["key"] for r in load_index(cache, ann.bag) if r["usable"]}

    def gray(cs):
        return cv2.cvtColor(undistort(cs.images[0]), cv2.COLOR_BGR2GRAY)

    cs0 = load_camera_set(cache, ann.bag, ann.key)
    d0, _ = stack.depth(cs0)
    rd = roi_depth(d0, tuple(g["depth_roi"]))
    if not rd.reliable:
        return []
    P0 = pixel_rays(np.array([g["point_px"]], float), K)[0] * rd.z
    half0 = 0.5 * float(np.mean([g["depth_roi"][2] - g["depth_roi"][0], g["depth_roi"][3] - g["depth_roi"][1]]))
    out: list[Measurement] = []
    first = height_of_point(d0, floor, tuple(g["point_px"]), tuple(g["depth_roi"]))
    if first is not None:
        out.append(to_measurement(first, cs0.ref, ann.kind, ann.feature_id, d0))
    for direction in (+1, -1):
        T, prev_depth, prev_gray = np.eye(4), d0, gray(cs0)
        for step in range(1, n_neighbors + 1):
            key = ann.key + direction * step
            if key not in usable:
                break
            cs = load_camera_set(cache, ann.bag, key)
            depth, _ = stack.depth(cs)
            cur_gray = gray(cs)
            pose = relative_pose(prev_gray, prev_depth, cur_gray)
            if pose is None:
                break
            T = pose.T_b_a @ T
            moved = propagate_point(P0, T, K, depth.shape)
            if moved is None:
                break
            P_b, uv = moved
            half = max(half0 * rd.z / P_b[2], 4.0)  # an object looks smaller the further away it is
            box = (uv[0] - half, uv[1] - half, uv[0] + half, uv[1] + half)
            raw = height_of_point(depth, floor, (float(uv[0]), float(uv[1])), box)
            if raw is not None:
                raw = RawMeasurement(
                    raw.value_m, raw.sigma_m, raw.distance_m, raw.method, raw.note + f"; tracked {direction * step:+d} keyframes"
                )
                out.append(to_measurement(raw, cs.ref, ann.kind, ann.feature_id, depth))
            prev_depth, prev_gray = depth, cur_gray
    return out
