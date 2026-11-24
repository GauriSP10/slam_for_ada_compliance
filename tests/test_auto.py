"""Choosing and measuring from the detection that contains an annotated point."""

from __future__ import annotations

import numpy as np

from slam_for_ada.auto import AutoStats, matching_detection, measure_from_detections
from slam_for_ada.dataio.annotations import Annotation
from slam_for_ada.datatypes import CameraSet, DetectionSource, FeatureKind, ObjectDetection
from tests.helpers import CAM_HEIGHT, CY, REF, F, H, W


def _obj(kind, y0, y1, x0, x1, score=0.5) -> ObjectDetection:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return ObjectDetection(REF, kind, (float(x0), float(y0), float(x1), float(y1)), score, DetectionSource.OPEN_VOCAB, m)


def _ann(kind=FeatureKind.OPERABLE_HEIGHT, point=(410, 190)) -> Annotation:
    return Annotation("a1", "thermostat_height", kind, "bag", 0, {"point_px": list(point), "depth_roi": [400, 180, 420, 200]})


def _cam_set() -> CameraSet:
    return CameraSet(REF, {0: np.zeros((H, W, 3), np.uint8)}, {0: 0})


def test_matching_detection_needs_the_right_kind_and_the_point_inside_the_mask():
    near = _obj(FeatureKind.OPERABLE_HEIGHT, 170, 211, 400, 421)
    wrong_kind = _obj(FeatureKind.HANDRAIL_HEIGHT, 170, 211, 400, 421, score=0.99)
    elsewhere = _obj(FeatureKind.OPERABLE_HEIGHT, 300, 340, 100, 140, score=0.9)
    assert matching_detection((410, 190), FeatureKind.OPERABLE_HEIGHT, [near, wrong_kind, elsewhere]) is near
    assert matching_detection((410, 190), FeatureKind.OPERABLE_HEIGHT, [wrong_kind, elsewhere]) is None
    assert matching_detection((9999, 9999), FeatureKind.OPERABLE_HEIGHT, [near]) is None  # point outside the image


def test_matching_detection_prefers_the_higher_score_when_masks_overlap():
    a = _obj(FeatureKind.OPERABLE_HEIGHT, 170, 211, 400, 421, score=0.3)
    b = _obj(FeatureKind.OPERABLE_HEIGHT, 160, 220, 390, 430, score=0.8)
    assert matching_detection((410, 190), FeatureKind.OPERABLE_HEIGHT, [a, b]) is b


def test_measure_from_detections_matches_geometry(depth_map, floor_plane):
    objs = [_obj(FeatureKind.OPERABLE_HEIGHT, 170, 211, 400, 421)]
    m = measure_from_detections(_ann(), _cam_set(), depth_map, floor_plane, objs)
    assert m is not None and "from detection mask" in m.note
    assert abs(m.value_in * 0.0254 - (CAM_HEIGHT - (190 - CY) / F * 3.0)) < 0.01  # wall at 3 m


def test_measure_from_detections_declines_without_a_match(depth_map, floor_plane):
    assert measure_from_detections(_ann(), _cam_set(), depth_map, floor_plane, []) is None
    far = [_obj(FeatureKind.OPERABLE_HEIGHT, 400, 440, 50, 90)]
    assert measure_from_detections(_ann(), _cam_set(), depth_map, floor_plane, far) is None


def test_auto_stats_total():
    s = AutoStats(from_detection=7, from_fallback=2, unmeasured=1)
    assert s.total == 10
