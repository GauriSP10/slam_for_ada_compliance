"""Measure a feature straight from a detection mask, with no hand annotation (Tier 2).

The mask supplies both the pixel whose height is measured and the patch the range is read from, so the quality gates of
``roi_depth`` apply exactly as they do for hand-placed points.
"""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import DepthMap, FeatureKind, MeasureMethod, ObjectDetection, Plane3D
from slam_for_ada.vision.height import height_of_point
from slam_for_ada.vision.measure import RawMeasurement

# which point of the object the tape measure refers to (see ground_truth.yaml: the outlet is to the TOP of the plate)
TOP_ANCHORED = {FeatureKind.OUTLET_HEIGHT}


def mask_anchor(mask: np.ndarray, top: bool) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    """(point to measure, depth patch box) for a mask: its centroid, or the middle of its top edge.

    The patch is the central part of the mask's bounding box, so it stays on the object and away from its ragged boundary.
    """
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = float(np.median(xs)), float(np.median(ys))
    point = (cx, float(y0)) if top else (cx, cy)
    mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    return point, (x0 + mx, y0 + my, x1 - mx, y1 - my)


def height_from_detection(depth: DepthMap, floor: Plane3D, det: ObjectDetection) -> RawMeasurement | None:
    """Height of a detected object above the floor. None if there is no mask or the depth patch is unreliable."""
    if det.mask is None or det.mask.sum() < 20:
        return None
    point, box = mask_anchor(det.mask, det.kind in TOP_ANCHORED)
    if det.kind is FeatureKind.OUTLET_HEIGHT:  # a plate is flat on the wall: read range just below its top edge
        box = (box[0], point[1] + 2, box[2], max(box[3], point[1] + 6))
    return height_of_point(depth, floor, point, box)


def handrail_height(depth: DepthMap, floor: Plane3D, det: ObjectDetection, n_columns: int = 9) -> RawMeasurement | None:
    """Height of the TOP of a handrail from a long thin mask.

    A rail slopes across the image, so no single point represents it. At several columns along the mask, take its top
    pixel, measure the height there (range from the rail's own pixels in that column), and report the median.
    """
    if det.mask is None:
        return None
    ys, xs = np.nonzero(det.mask)
    if xs.size < 50:
        return None
    cols = np.linspace(np.percentile(xs, 15), np.percentile(xs, 85), n_columns).astype(int)
    vals = []
    for c in cols:
        col = np.nonzero(det.mask[:, c])[0]
        if col.size < 2:
            continue
        top, bot = float(col.min()), float(col.max())
        r = height_of_point(depth, floor, (float(c), top), (c - 3, top, c + 4, bot))
        if r is not None:
            vals.append(r)
    if len(vals) < 3:
        return None
    v = np.array([r.value_m for r in vals])
    med = float(np.median(v))
    sigma = float(max(np.median([r.sigma_m for r in vals]), 1.4826 * np.median(np.abs(v - med)) / np.sqrt(len(v))))
    return RawMeasurement(
        med,
        sigma,
        float(np.median([r.distance_m for r in vals])),
        MeasureMethod.PLANE_OFFSET,
        f"median over {len(vals)} columns of the rail mask",
    )
