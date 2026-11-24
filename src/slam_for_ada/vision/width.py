"""Door and hallway widths, each by two independent methods so they can be compared against the tape measure.

Door width
    stereo_depth : 3D points on each jamb read straight from the fused depth map.
    homography   : fit the wall plane, intersect the jamb viewing rays with it, measure in the rectified wall.
Hallway width
    stereo_depth : fit both side walls from depth and measure the distance between the two planes.
    homography   : rectify the FLOOR plane to a top-down view and measure between the wall/floor boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from slam_for_ada.config import MeasureConfig
from slam_for_ada.datatypes import M_PER_INCH, DepthMap, MeasureMethod, Plane3D
from slam_for_ada.vision.geometry import fit_plane_svd, orient_normal, pixel_rays
from slam_for_ada.vision.homography import plane_frame
from slam_for_ada.vision.measure import Box, RawMeasurement, fit_wall_plane, horizontal, roi_depth

Jamb = tuple[tuple[float, float], tuple[float, float]]  # ((x_top, y_top), (x_bottom, y_bottom)) of a vertical edge


def as_jamb(raw) -> Jamb:
    """A jamb from YAML-style nested numbers: ((x_top, y_top), (x_bottom, y_bottom)) as floats."""
    (xt, yt), (xb, yb) = raw
    return ((float(xt), float(yt)), (float(xb), float(yb)))


# --------------------------------------------------------------------------- edge snapping
def edge_energy(gray: np.ndarray) -> np.ndarray:
    """Absolute horizontal gradient: strong along vertical edges such as door jambs."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))


def snap_jamb(energy: np.ndarray, jamb: Jamb, search_px: int) -> tuple[Jamb, float]:
    """Slide a hand-placed jamb sideways onto the strongest vertical edge within ``search_px``.

    The whole line moves by one sub-pixel offset (a parabola through the best score and its neighbours), so a
    rough annotation becomes an accurate edge. Returns the snapped jamb and the mean edge strength along it.
    """
    (xt, yt), (xb, yb) = jamb
    ys = np.arange(int(min(yt, yb)), int(max(yt, yb)) + 1)
    if len(ys) < 5:
        return jamb, 0.0
    xs = np.interp(ys, [yt, yb] if yt < yb else [yb, yt], [xt, xb] if yt < yb else [xb, xt])
    offs = np.arange(-search_px, search_px + 1)
    scores = np.array([np.mean(energy[ys, np.clip(np.round(xs + o).astype(int), 0, energy.shape[1] - 1)]) for o in offs])
    i = int(np.argmax(scores))
    off = float(offs[i])
    if 0 < i < len(offs) - 1:
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            off += 0.5 * (a - c) / den
    return ((xt + off, yt), (xb + off, yb)), float(scores[i])


def _line_x_at(jamb: Jamb, y: np.ndarray) -> np.ndarray:
    (xt, yt), (xb, yb) = jamb
    return xt + (xb - xt) * (y - yt) / (yb - yt)


def _mid_rows(left: Jamb, right: Jamb, n: int) -> np.ndarray:
    """Rows in the middle 40 percent of the shorter jamb (away from the floor, header and any trim)."""
    top = max(left[0][1], right[0][1])
    bot = min(left[1][1], right[1][1])
    return np.linspace(top + 0.3 * (bot - top), top + 0.7 * (bot - top), n)


# --------------------------------------------------------------------------- door width
def door_width_stereo(
    depth: DepthMap, floor: Plane3D, left: Jamb, right: Jamb, cfg: MeasureConfig, wall_rois: Sequence[Box] | None = None
) -> RawMeasurement | None:
    """Door width from the 3D position of each jamb, with range taken from the leaf just inside it.

    If ``wall_rois`` is given, the leaf plane is fitted and the measurement is refused when the door is seen at a grazing
    angle (the same conditioning rule as the homography method): an oblique, close door has a pixel gap that is mostly
    foreshortening, and the two jamb ranges then differ so much that a small error becomes a very large length error.

    A jamb is a depth discontinuity (leaf in front, frame or background beside it), so a depth read exactly on the
    edge mixes surfaces. Instead, at each sampled row the range comes from a small patch on the leaf a few pixels
    inside the jamb, and the jamb's own pixel supplies the viewing direction. Each jamb uses its own patch, so the
    method still works when the door is seen obliquely and the range changes across the leaf.
    """
    rows = _mid_rows(left, right, cfg.sample_rows)
    K = depth.K
    if wall_rois:
        fit = fit_wall_plane(depth, wall_rois, floor.normal)
        if fit is None:
            return None
        xl_, xr_ = _line_x_at(left, rows), _line_x_at(right, rows)
        rays = pixel_rays(np.array([[xl_.mean(), rows.mean()], [xr_.mean(), rows.mean()]]), K)
        if float(np.min(np.abs(rays @ fit[0].normal) / np.linalg.norm(rays, axis=1))) < cfg.min_cos_incidence:
            return None
    P, S = [], []
    for jamb, inward in ((left, +1), (right, -1)):
        pts, sig = [], []
        for y, x in zip(rows, _line_x_at(jamb, rows)):
            xa, xb = sorted((x + inward * 8, x + inward * 26))
            rd = roi_depth(depth, (xa, y - 2, xb, y + 3), min_fill=0.3)
            if rd.reliable:
                pts.append(pixel_rays(np.array([[x, y]]), K)[0] * rd.z)
                sig.append(rd.sigma)
        if len(pts) < max(5, cfg.sample_rows // 4):
            return None
        P.append(np.median(np.array(pts), axis=0))
        S.append(float(np.median(sig)))
    pl, pr = P
    width = float(np.linalg.norm(horizontal(pr - pl, floor.normal)))
    if not cfg.min_door_width_in <= width / M_PER_INCH <= cfg.max_door_width_in:  # not door-sized: the ranges are wrong
        return None
    dist = float(np.linalg.norm((pl + pr) / 2))
    statistical = float(np.hypot(*S) + 0.5 * dist / K[0, 0] * np.sqrt(2))  # depth noise plus 0.5 px of edge position
    sigma = float(np.hypot(statistical, cfg.stereo_edge_bias_frac * width))  # plus the systematic bias of reading depth at an edge
    return RawMeasurement(width, sigma, dist, MeasureMethod.STEREO_DEPTH, "jamb ray x leaf range just inside each jamb")


def door_width_homography(
    depth: DepthMap, floor: Plane3D, left: Jamb, right: Jamb, wall_rois: Sequence[Box], cfg: MeasureConfig
) -> RawMeasurement | None:
    """Door width by rectifying the wall plane: only the plane (a robust fit over many pixels) comes from depth.

    The jamb edges are located in the image, not in depth, so the depth discontinuity at the door does not matter.
    A pixel on the plane maps to a metric position through a homography, and the width is the horizontal gap.
    """
    fit = fit_wall_plane(depth, wall_rois, floor.normal)
    if fit is None:
        return None
    wall, pts = fit
    frame = plane_frame(depth.K, wall, floor.normal)
    rows = _mid_rows(left, right, cfg.sample_rows)
    xl, xr = _line_x_at(left, rows), _line_x_at(right, rows)
    # Conditioning: where a viewing ray meets the plane at a grazing angle, a tiny pixel error becomes a huge length
    # error (an oblique, close door can read 70 in wide). cos(incidence) = |n . unit ray|; refuse below the threshold.
    jamb_rays = pixel_rays(np.array([[xl.mean(), rows.mean()], [xr.mean(), rows.mean()]]), depth.K)
    cos_inc = float(np.min(np.abs(jamb_rays @ wall.normal) / np.linalg.norm(jamb_rays, axis=1)))
    if cos_inc < cfg.min_cos_incidence:
        return None
    pl = frame.to_plane(np.column_stack([xl, rows]))
    pr = frame.to_plane(np.column_stack([xr, rows]))
    gap = np.abs(pr[:, 0] - pl[:, 0])
    gap = gap[np.isfinite(gap)]
    if gap.size < 5:
        return None
    width = float(np.median(gap))
    if not cfg.min_door_width_in <= width / M_PER_INCH <= cfg.max_door_width_in:  # not door-sized: the fit or the edges are wrong
        return None
    dist = float(abs(wall.d))  # camera-to-wall distance
    # a plane offset error of dd scales the whole rectified wall by dd / distance; 0.5 px of edge error is magnified by 1/cos(incidence)
    sigma = float(np.hypot(width * 0.015 / max(dist, 0.5), 0.5 * dist / depth.K[0, 0] * np.sqrt(2) / cos_inc))
    return RawMeasurement(
        width,
        sigma,
        dist,
        MeasureMethod.HOMOGRAPHY,
        f"wall plane {wall.n_inliers} pts, rms {wall.rmse_m * 100:.1f} cm, cos incidence {cos_inc:.2f}",
    )


# --------------------------------------------------------------------------- hallway width
def hallway_width_planes(
    depth: DepthMap, floor: Plane3D, left_roi: Box, right_roi: Box, n_boot: int = 30, seed: int = 0
) -> RawMeasurement | None:
    """Hallway width as the gap between planes fitted to the left and right walls from depth."""
    up = floor.normal
    fl = fit_wall_plane(depth, [left_roi], up)
    fr = fit_wall_plane(depth, [right_roi], up)
    if fl is None or fr is None:
        return None

    def width_of(pl: np.ndarray, pr: np.ndarray, nl: np.ndarray, nr: np.ndarray) -> float:
        axis = horizontal(nl - nr, up)
        axis /= np.linalg.norm(axis)
        return float((pr.mean(axis=0) - pl.mean(axis=0)) @ axis)

    (wl, ptl), (wr, ptr) = fl, fr
    value = width_of(ptl, ptr, wl.normal, wr.normal)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):  # statistical spread from resampling the wall points
        a = ptl[rng.integers(0, len(ptl), len(ptl))]
        b = ptr[rng.integers(0, len(ptr), len(ptr))]
        na, da, _ = fit_plane_svd(a)
        na, _ = orient_normal(na, da, wl.normal)
        nb, db, _ = fit_plane_svd(b)
        nb, _ = orient_normal(nb, db, wr.normal)
        boots.append(width_of(a, b, na, nb))
    parallel = float(
        np.degrees(
            np.arccos(
                np.clip(
                    abs(horizontal(wl.normal, up) @ horizontal(wr.normal, up))
                    / (np.linalg.norm(horizontal(wl.normal, up)) * np.linalg.norm(horizontal(wr.normal, up))),
                    -1,
                    1,
                )
            )
        )
    )
    sigma = float(np.hypot(np.std(boots), 0.02))  # plus 2 cm for depth bias common to both walls
    dist = float(np.linalg.norm((ptl.mean(axis=0) + ptr.mean(axis=0)) / 2))
    return RawMeasurement(abs(value), sigma, dist, MeasureMethod.STEREO_DEPTH, f"walls differ by {parallel:.1f} deg")


def hallway_width_floor(
    floor: Plane3D,
    K: np.ndarray,
    left_pts: Sequence[tuple[float, float]],
    right_pts: Sequence[tuple[float, float]],
    pixel_sigma: float = 2.0,
    seed: int = 0,
) -> RawMeasurement | None:
    """Hallway width on the rectified floor: where each wall meets the floor, the perpendicular distance between them.

    The floor is a plane known to about a centimetre, so every wall/floor contact pixel maps to a floor position
    without needing any depth at the wall itself. At least two contact points per wall are needed to give its direction.
    """
    if len(left_pts) < 2 or len(right_pts) < 2:
        return None
    frame = plane_frame(K, floor, np.array([0.0, 0.0, 1.0]))  # 'up' in the floor plane = forward

    def one(lp, rp) -> float:
        L, R = frame.to_plane(np.array(lp, float)), frame.to_plane(np.array(rp, float))
        if not (np.isfinite(L).all() and np.isfinite(R).all()):
            return float("nan")

        def perp(line_pts, pts):
            c = line_pts.mean(axis=0)
            _, _, vt = np.linalg.svd(line_pts - c)
            n = np.array([-vt[0][1], vt[0][0]])
            return float(np.abs((pts - c) @ n).mean())

        return 0.5 * (perp(L, R) + perp(R, L))

    value = one(left_pts, right_pts)
    if not np.isfinite(value):
        return None
    rng = np.random.default_rng(seed)
    boots = [
        one(
            np.array(left_pts) + rng.normal(0, pixel_sigma, (len(left_pts), 2)),
            np.array(right_pts) + rng.normal(0, pixel_sigma, (len(right_pts), 2)),
        )
        for _ in range(40)
    ]
    sigma = float(np.hypot(np.nanstd(boots), 0.01))
    dist = float(np.linalg.norm(frame.to_3d(np.array(left_pts[:1], float))[0]))
    return RawMeasurement(value, sigma, dist, MeasureMethod.HOMOGRAPHY, "floor-plane rectification")
