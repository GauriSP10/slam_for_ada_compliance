"""Measuring straight from detection masks."""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import DetectionSource, FeatureKind, ObjectDetection
from slam_for_ada.vision.auto_measure import handrail_height, height_from_detection, mask_anchor
from tests.helpers import CAM_HEIGHT, CY, REF, F, H, W


def _det(mask, kind=FeatureKind.OPERABLE_HEIGHT) -> ObjectDetection:
    ys, xs = np.nonzero(mask)
    return ObjectDetection(
        REF, kind, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), 0.9, DetectionSource.OPEN_VOCAB, mask
    )


def _rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_anchor_centroid_top_and_inner_patch():
    m = _rect(100, 141, 300, 321)
    (cx, cy), box = mask_anchor(m, top=False)
    assert abs(cx - 310) < 1 and abs(cy - 120) < 1
    (tx, ty), _ = mask_anchor(m, top=True)
    assert ty == 100 and abs(tx - 310) < 1
    assert box[0] > 300 and box[2] < 320 and box[1] > 100 and box[3] < 140  # the patch stays inside the mask


def test_height_from_detection_on_a_wall(depth_map, floor_plane):
    # wall at z = 3 m: a point at image row y is 1.3 - (y - cy)/f * 3 above the floor
    r = height_from_detection(depth_map, floor_plane, _det(_rect(170, 211, 400, 421)))
    assert r is not None and abs(r.value_m - (CAM_HEIGHT - (190 - CY) / F * 3.0)) < 0.01


def test_outlet_is_anchored_to_the_top_edge(depth_map, floor_plane):
    top = height_from_detection(depth_map, floor_plane, _det(_rect(300, 341, 400, 421), FeatureKind.OUTLET_HEIGHT))
    mid = height_from_detection(depth_map, floor_plane, _det(_rect(300, 341, 400, 421), FeatureKind.OPERABLE_HEIGHT))
    assert top is not None and mid is not None
    assert top.value_m > mid.value_m  # the top edge is higher than the centre
    assert abs(top.value_m - (CAM_HEIGHT - (300 - CY) / F * 3.0)) < 0.01


def test_no_measurement_without_a_usable_mask(depth_map, floor_plane):
    assert (
        height_from_detection(
            depth_map, floor_plane, ObjectDetection(REF, FeatureKind.OPERABLE_HEIGHT, (0, 0, 5, 5), 0.9, DetectionSource.OPEN_VOCAB)
        )
        is None
    )
    assert height_from_detection(depth_map, floor_plane, _det(_rect(100, 103, 100, 103))) is None  # 9 pixels: too small


def test_handrail_height_uses_the_top_of_the_rail(depth_map, floor_plane):
    rail = _rect(250, 257, 150, 600)  # a level rail, 7 px thick
    r = handrail_height(depth_map, floor_plane, _det(rail, FeatureKind.HANDRAIL_HEIGHT))
    assert r is not None and abs(r.value_m - (CAM_HEIGHT - (250 - CY) / F * 3.0)) < 0.01
    assert (
        handrail_height(depth_map, floor_plane, _det(_rect(250, 252, 150, 160), FeatureKind.HANDRAIL_HEIGHT)) is None
    )  # too short to trust
