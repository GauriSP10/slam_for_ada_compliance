"""Height above the floor of a marked point on a wall-mounted or free-standing object."""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import DepthMap, MeasureMethod, Plane3D
from slam_for_ada.vision.geometry import pixel_rays
from slam_for_ada.vision.measure import Box, RawMeasurement, roi_depth

FLOOR_SIGMA_M = 0.015  # spread of the pooled floor height across frames (see scripts/evaluate_depth.py)


def height_of_point(
    depth: DepthMap, floor: Plane3D, point_px: tuple[float, float], depth_roi: Box, pixel_sigma: float = 2.0
) -> RawMeasurement | None:
    """Height of the 3D point seen at ``point_px``, taking its range from the depth patch ``depth_roi``.

    The point is often an edge (the top of a bin, a handrail), where depth is unreliable, so the range comes from a patch
    on the same surface nearby and the pixel's viewing ray places the point. This assumes the patch and the point are at
    the same distance, which holds for the nearly fronto-parallel surfaces we measure.
    """
    rd = roi_depth(depth, depth_roi)
    if not rd.reliable or rd.n < 10:
        return None
    z, sig_z, n = rd.z, rd.sigma, rd.n
    ray = pixel_rays(np.array([point_px], float), depth.K)[0]
    P = ray * z  # ray has z = 1, so P has depth z
    height = float(floor.signed_distance(P[None, :])[0])
    # error budget: range error along the ray, floor error, and the point's pixel position (vertical pixels move it in height)
    dh_dz = float(ray @ floor.normal)
    sigma = float(np.sqrt((dh_dz * sig_z) ** 2 + FLOOR_SIGMA_M**2 + (pixel_sigma * z / depth.K[1, 1]) ** 2))
    return RawMeasurement(height, sigma, float(np.linalg.norm(P)), MeasureMethod.PLANE_OFFSET, f"depth patch {n} px, z={z:.2f} m")
