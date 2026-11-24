"""Door detector on a synthetic frame: a dark door in a bright wall at a known distance."""

from __future__ import annotations

import numpy as np

from slam_for_ada.vision.door_detector import DoorDetectorParams, detect_doors, harris_response, vertical_segments
from tests.helpers import REF, H, W, make_depth


def _scene(width_px: int, x0: int = 250):
    """Bright wall with a dark door whose jambs are at x0 and x0+width_px, 3 m from the camera."""
    img = np.full((H, W, 3), 200, np.uint8)
    img[40:500, x0 : x0 + width_px] = 40
    img[40:50, x0 : x0 + width_px] = 20  # a head line, so the top corners carry a Harris response
    return img


def test_vertical_segments_find_the_two_jambs():
    img = _scene(250)
    import cv2

    segs = vertical_segments(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), DoorDetectorParams())
    xs = sorted(round(s[0]) for s in segs)
    assert len(xs) >= 2 and abs(xs[0] - 250) < 8 and abs(xs[-1] - 500) < 8


def test_detects_a_door_sized_pair_and_rejects_a_pillar_pair(K):
    depth = make_depth(K, np.full((H, W), 3.0))
    # 3 m away, 250 px = 0.84 m = 33 in: a door
    door = detect_doors(_scene(250), depth, REF)
    assert door, "a door-sized pair should be detected"
    xs = door[0].corners_px[:2, 0]
    assert abs(xs[0] - 250) < 10 and abs(xs[1] - 500) < 10 and not door[0].is_glass
    # 3 m away, 600 px = 2.0 m = 79 in: far too wide to be a door
    assert detect_doors(_scene(600, x0=60), depth, REF) == []


def test_missing_depth_between_jambs_flags_glass(K):
    z = np.full((H, W), 3.0)
    z[:, 270:480] = np.nan  # stereo found nothing inside the opening
    door = detect_doors(_scene(250), make_depth(K, z), REF)
    assert door and door[0].is_glass


def test_harris_response_is_normalised():
    import cv2

    r = harris_response(cv2.cvtColor(_scene(250), cv2.COLOR_BGR2GRAY))
    assert r.max() == 1.0 or abs(r.max() - 1.0) < 1e-6
