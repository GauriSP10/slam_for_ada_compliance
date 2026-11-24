"""How far an object sticks out from the wall it is mounted on."""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import DepthMap, MeasureMethod, Plane3D
from slam_for_ada.vision.measure import Box, RawMeasurement, fit_wall_plane, roi_points


def protrusion_from_wall(depth: DepthMap, floor: Plane3D, wall_roi: Box, object_roi: Box) -> RawMeasurement | None:
    """Distance from the wall plane (fitted beside the object) to the front of the object.

    The front is the median signed distance of the object's depth points from the wall plane, positive toward the camera.
    """
    fit = fit_wall_plane(depth, [wall_roi], floor.normal)
    if fit is None:
        return None
    wall, _ = fit
    pts = roi_points(depth, object_roi)
    if len(pts) < 20:
        return None
    off = wall.signed_distance(pts)  # wall normal points toward the camera
    value = float(np.median(off))
    sigma = float(np.hypot(np.std(off) / np.sqrt(max(len(off) / 25.0, 1.0)), 0.01))
    dist = float(np.linalg.norm(pts.mean(axis=0)))
    return RawMeasurement(value, sigma, dist, MeasureMethod.PLANE_OFFSET, f"wall plane rms {wall.rmse_m * 100:.1f} cm")
