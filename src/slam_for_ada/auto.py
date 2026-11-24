"""Automatic measurement of height features from open-vocabulary detections (Tier 2).

A tape-measure note says "thermostat 51.25 in" but not WHERE the thermostat is, so something must say which object a
ground-truth entry refers to. What the detector removes is everything else: given one point on the object, it finds the
object's mask, and the point to measure and the depth patch both come from that mask. If no detection of the right kind
contains the point, the caller falls back to the hand-drawn depth box.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from slam_for_ada.dataio.annotations import Annotation
from slam_for_ada.datatypes import CameraSet, DepthMap, FeatureKind, Measurement, ObjectDetection, Plane3D
from slam_for_ada.vision.auto_measure import handrail_height, height_from_detection
from slam_for_ada.vision.measure import RawMeasurement, to_measurement


@dataclass
class AutoStats:
    """How many height features were measured from a detection, from the hand-drawn fallback, or not at all."""

    from_detection: int = 0
    from_fallback: int = 0
    unmeasured: int = 0

    @property
    def total(self) -> int:
        return self.from_detection + self.from_fallback + self.unmeasured


def matching_detection(point_px: tuple[float, float], kind: FeatureKind, objects: Sequence[ObjectDetection]) -> ObjectDetection | None:
    """The highest-scoring detection of ``kind`` whose mask contains the pixel, or None."""
    px, py = int(round(point_px[0])), int(round(point_px[1]))
    best: ObjectDetection | None = None
    for o in objects:
        if o.kind is not kind or o.mask is None:
            continue
        h, w = o.mask.shape
        if 0 <= py < h and 0 <= px < w and o.mask[py, px] and (best is None or o.score > best.score):
            best = o
    return best


def measure_from_detections(
    ann: Annotation, cam_set: CameraSet, depth: DepthMap, floor: Plane3D, objects: Sequence[ObjectDetection]
) -> Measurement | None:
    """Measure a height annotation from the detection mask under its point. None if there is no usable detection."""
    det = matching_detection(tuple(ann.geometry["point_px"]), ann.kind, objects)
    if det is None:
        return None
    raw = handrail_height(depth, floor, det) if ann.kind is FeatureKind.HANDRAIL_HEIGHT else height_from_detection(depth, floor, det)
    if raw is None:
        return None
    raw = RawMeasurement(raw.value_m, raw.sigma_m, raw.distance_m, raw.method, f"{raw.note}; from detection mask (score {det.score:.2f})")
    return to_measurement(raw, cam_set.ref, ann.kind, ann.feature_id, depth)
