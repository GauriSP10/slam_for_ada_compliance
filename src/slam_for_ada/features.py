"""Measure one annotated feature: choose the right measurement stage for its kind."""

from __future__ import annotations

from collections.abc import Callable

import cv2

from slam_for_ada.config import MeasureConfig
from slam_for_ada.dataio.annotations import Annotation
from slam_for_ada.datatypes import CameraSet, DepthMap, FeatureKind, ImageBGR, Measurement, Plane3D
from slam_for_ada.vision.height import height_of_point
from slam_for_ada.vision.measure import RawMeasurement, to_measurement
from slam_for_ada.vision.protrusion import protrusion_from_wall
from slam_for_ada.vision.width import (
    as_jamb,
    door_width_homography,
    door_width_stereo,
    edge_energy,
    hallway_width_floor,
    hallway_width_planes,
    snap_jamb,
)

HEIGHT_KINDS = frozenset(
    {
        FeatureKind.HARDWARE_HEIGHT,
        FeatureKind.OPERABLE_HEIGHT,
        FeatureKind.HANDRAIL_HEIGHT,
        FeatureKind.SIGNAGE_HEIGHT,
        FeatureKind.FOUNTAIN_HEIGHT,
        FeatureKind.SURFACE_HEIGHT,
        FeatureKind.OUTLET_HEIGHT,
        FeatureKind.OBJECT_HEIGHT,
    }
)

Undistorter = Callable[[ImageBGR], ImageBGR]


def measure_annotation(
    ann: Annotation, cam_set: CameraSet, depth: DepthMap, floor: Plane3D, undistort: Undistorter, cfg: MeasureConfig
) -> list[Measurement]:
    """All measurements an annotation supports: a door gets two methods, a hallway two, a height one."""
    g, ref = ann.geometry, cam_set.ref
    raws: list[RawMeasurement | None] = []
    if ann.kind is FeatureKind.DOOR_WIDTH:
        energy = edge_energy(cv2.cvtColor(undistort(cam_set.images[0]), cv2.COLOR_BGR2GRAY))
        left, _ = snap_jamb(energy, as_jamb(g["left_jamb"]), cfg.jamb_snap_px)
        right, _ = snap_jamb(energy, as_jamb(g["right_jamb"]), cfg.jamb_snap_px)
        rois = [tuple(b) for b in g["wall_rois"]]
        raws += [door_width_stereo(depth, floor, left, right, cfg, rois), door_width_homography(depth, floor, left, right, rois, cfg)]
    elif ann.kind in HEIGHT_KINDS:
        raws.append(height_of_point(depth, floor, tuple(g["point_px"]), tuple(g["depth_roi"])))
    elif ann.kind is FeatureKind.HALLWAY_WIDTH:
        raws.append(hallway_width_planes(depth, floor, tuple(g["left_wall_roi"]), tuple(g["right_wall_roi"])))
        if "floor_contacts" in g:
            raws.append(hallway_width_floor(floor, depth.K, g["floor_contacts"]["left"], g["floor_contacts"]["right"]))
    elif ann.kind is FeatureKind.PROTRUSION:
        raws.append(protrusion_from_wall(depth, floor, tuple(g["wall_roi"]), tuple(g["object_roi"])))
    return [to_measurement(r, ref, ann.kind, ann.feature_id, depth) for r in raws if r is not None]
