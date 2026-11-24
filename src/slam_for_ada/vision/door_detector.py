"""Tier 1 door detection: Harris corners + long vertical edges, filtered by metric size from depth.

A door frame shows up as two long vertical edges (the jambs) a door-width apart, joined at the top by a horizontal
head that makes a corner. Edges alone produce many pairs (every pillar, panel seam and window mullion), so each pair
is checked in metres: depth turns the pixel gap into a width, and only door-sized pairs survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from slam_for_ada.datatypes import M_PER_INCH, DepthMap, DetectionSource, DoorCandidate, FrameRef, ImageBGR


@dataclass(frozen=True)
class DoorDetectorParams:
    canny_low: int = 40
    canny_high: int = 120
    min_height_px: int = 200  # a jamb must be at least this long to count as a door edge
    min_overlap: float = 0.7  # vertical overlap of the two jambs, as a share of the shorter one
    min_width_in: float = 20.0  # door-sized in metres; low enough to allow doors seen obliquely
    max_width_in: float = 48.0
    corner_radius_px: int = 22  # search radius around each jamb's top end for a Harris corner
    harris_rel_thresh: float = 0.02  # corner response relative to the strongest in the image
    glass_fill: float = 0.35  # below this share of valid depth between the jambs we call it glass
    max_candidates: int = 6


def vertical_segments(gray: np.ndarray, p: DoorDetectorParams) -> list[tuple[float, float, float]]:
    """Long, near-vertical edge segments as (x, y_top, y_bottom)."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), p.canny_low, p.canny_high)
    edges = cv2.dilate(edges, np.ones((1, 5), np.uint8))  # tolerate a slightly slanted edge
    keep = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((p.min_height_px // 2, 1), np.uint8))  # keep only tall runs
    n, _, stats, cent = cv2.connectedComponentsWithStats(keep, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h >= p.min_height_px and w <= max(12, 0.08 * h):  # tall and thin
            segs.append((float(cent[i][0]), float(y), float(y + h)))
    return sorted(segs)


def harris_response(gray: np.ndarray) -> np.ndarray:
    """Harris corner strength, normalised to the strongest corner in the image."""
    r = cv2.cornerHarris(gray.astype(np.float32), 3, 3, 0.04)
    return r / max(float(r.max()), 1e-9)


def detect_doors(image: ImageBGR, depth: DepthMap, ref: FrameRef, params: DoorDetectorParams = DoorDetectorParams()) -> list[DoorCandidate]:
    """Door-frame candidates in a cam0-undistorted image, best first."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segs = vertical_segments(gray, params)
    harris = harris_response(gray)
    f = depth.K[0, 0]
    out: list[tuple[float, DoorCandidate]] = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (xl, tl, bl), (xr, tr, br) = segs[i], segs[j]
            dx = xr - xl
            top, bot = max(tl, tr), min(bl, br)
            if dx < 25 or bot - top < params.min_overlap * min(bl - tl, br - tr):
                continue
            ya, yb = int(top + 0.25 * (bot - top)), int(top + 0.75 * (bot - top))
            between = depth.depth_m[ya:yb, int(xl) + 6 : int(xr) - 5]
            ok = np.isfinite(between)
            fill = float(ok.mean()) if ok.size else 0.0
            # Metric width = 3D distance between the two jambs. Each jamb's range comes from the surface just inside it,
            # so an obliquely seen door (whose pixel gap is foreshortened) is still measured at its true size.
            pts = []
            for x_jamb, inward in ((xl, +1), (xr, -1)):
                xa, xb = sorted((int(x_jamb + inward * 8), int(x_jamb + inward * 28)))
                patch = depth.depth_m[ya:yb, max(xa, 0) : xb]
                z = patch[np.isfinite(patch)]
                if z.size < 30:
                    break
                pts.append(
                    np.array([(x_jamb - depth.K[0, 2]) / f, (0.5 * (ya + yb) - depth.K[1, 2]) / depth.K[1, 1], 1.0]) * float(np.median(z))
                )
            if len(pts) < 2:
                continue
            width_in = float(np.linalg.norm(pts[1] - pts[0])) / M_PER_INCH
            if not params.min_width_in <= width_in <= params.max_width_in:
                continue

            # corner evidence at the top of each jamb (the door head)
            def corner_near(x, y):
                r = params.corner_radius_px
                y0, y1, x0, x1 = max(int(y) - r, 0), int(y) + r, max(int(x) - r, 0), int(x) + r
                return float(harris[y0:y1, x0:x1].max() >= params.harris_rel_thresh)

            corner = 0.5 * (corner_near(xl, top) + corner_near(xr, top))
            size_fit = 1.0 - min(abs(width_in - 34.0) / 34.0, 1.0)
            score = 0.5 * size_fit + 0.3 * corner + 0.2 * min((bot - top) / 400.0, 1.0)
            cand = DoorCandidate(
                ref,
                np.array([[xl, top], [xr, top], [xr, bot], [xl, bot]], dtype=np.float64),
                float(score),
                DetectionSource.HARRIS_EDGES,
                is_glass=fill < params.glass_fill,
            )
            out.append((score, cand))
    out.sort(key=lambda t: -t[0])
    picked: list[DoorCandidate] = []
    for _, c in out:  # non-maximum suppression on the jamb positions
        xs = c.corners_px[[0, 1], 0]
        if all(np.abs(xs - p.corners_px[[0, 1], 0]).max() > 20 for p in picked):
            picked.append(c)
        if len(picked) == params.max_candidates:
            break
    return picked
