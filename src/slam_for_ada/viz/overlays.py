"""Drawing helpers: coordinate grids for annotating, depth colour maps, and measurement overlays."""

from __future__ import annotations

import cv2
import numpy as np

from slam_for_ada.datatypes import ImageBGR


def depth_colormap(depth_m: np.ndarray, vmax: float = 8.0) -> ImageBGR:
    """Turbo colour map of a depth map; invalid (NaN) pixels are black."""
    a = np.clip(np.nan_to_num(depth_m, nan=0.0) / vmax, 0, 1)
    c = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~np.isfinite(depth_m)] = 0
    return np.asarray(c, dtype=np.uint8)


def draw_grid(image: ImageBGR, step: int = 50, scale: int = 2, label_every: int = 2) -> ImageBGR:
    """Upscale ``image`` and overlay a labelled pixel grid, so coordinates can be read off by eye.

    Labels are in ORIGINAL pixel units, whatever the scale. Lines every ``step`` px; a label every ``label_every`` lines.
    """
    big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    for x in range(0, w, step):
        cv2.line(big, (x * scale, 0), (x * scale, h * scale), (0, 255, 255) if x % (step * label_every) == 0 else (0, 160, 160), 1)
        if x % (step * label_every) == 0:
            cv2.putText(big, str(x), (x * scale + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(big, (0, y * scale), (w * scale, y * scale), (0, 255, 255) if y % (step * label_every) == 0 else (0, 160, 160), 1)
        if y % (step * label_every) == 0:
            cv2.putText(big, str(y), (3, y * scale - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return np.asarray(big, dtype=np.uint8)


def reach_band_overlay(image: ImageBGR, depth, floor, lo_in: float = 15.0, hi_in: float = 48.0, alpha: float = 0.35) -> ImageBGR:
    """Tint every pixel whose 3D point lies between ``lo_in`` and ``hi_in`` inches above the floor.

    The 15-48 in band is the ADA unobstructed reach range, so operable parts (switches, readers, handles)
    should sit inside the tint. Needs the depth map and the floor plane, which is the point of the whole pipeline.
    """
    from slam_for_ada.datatypes import M_PER_INCH
    from slam_for_ada.vision.geometry import backproject_map

    pts = backproject_map(depth.depth_m, depth.K)
    h = pts @ floor.normal + floor.d  # height above the floor, metres
    band = np.isfinite(h) & (h >= lo_in * M_PER_INCH) & (h <= hi_in * M_PER_INCH)
    out = image.copy()
    tint = np.zeros_like(image)
    tint[..., 1] = 255
    out[band] = cv2.addWeighted(image, 1 - alpha, tint, alpha, 0)[band]
    return out


def draw_jamb(image: ImageBGR, jamb, colour=(0, 255, 0), scale: int = 1) -> None:
    (xt, yt), (xb, yb) = jamb
    cv2.line(image, (int(xt * scale), int(yt * scale)), (int(xb * scale), int(yb * scale)), colour, 2, cv2.LINE_AA)


def draw_box(image: ImageBGR, box, colour=(255, 200, 0), scale: int = 1, label: str | None = None) -> None:
    x0, y0, x1, y1 = [int(v * scale) for v in box]
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, 1, cv2.LINE_AA)
    if label:
        cv2.putText(image, label, (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def draw_point(image: ImageBGR, pt, colour=(0, 0, 255), scale: int = 1, label: str | None = None) -> None:
    x, y = int(pt[0] * scale), int(pt[1] * scale)
    cv2.circle(image, (x, y), 5, colour, -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(image, label, (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, label, (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def put_label(image: ImageBGR, text: str, org=(10, 28), colour=(255, 255, 255), scale: float = 0.8) -> None:
    """Text with a dark outline so it reads on any background."""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
