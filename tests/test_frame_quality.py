"""Blur scoring and the usable/unusable decision."""

from __future__ import annotations

import cv2
import numpy as np

from slam_for_ada.config import CacheConfig
from slam_for_ada.dataio.frame_cache import reassess
from slam_for_ada.vision.frame_quality import assess, laplacian_sharpness
from tests.helpers import REF


def _textured() -> np.ndarray:
    rng = np.random.default_rng(0)
    return cv2.cvtColor(rng.integers(0, 255, (540, 720), dtype=np.uint8), cv2.COLOR_GRAY2BGR)


def test_blur_lowers_sharpness():
    sharp = _textured()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    assert laplacian_sharpness(blurred) < 0.1 * laplacian_sharpness(sharp)


def test_assess_flags_blur_relative_to_the_bag_and_people():
    cfg = CacheConfig()
    ok = assess(REF, {0: 600.0, 1: 580.0}, 0.0, cfg, ref_sharpness=600.0)
    assert ok.usable and not ok.reasons
    blurry = assess(REF, {0: 600.0, 1: 200.0}, 0.0, cfg, ref_sharpness=600.0)  # 33% of the bag median
    assert not blurry.usable and "blur" in blurry.reasons[0]
    crowd = assess(REF, {0: 600.0}, 0.3, cfg, ref_sharpness=600.0)
    assert not crowd.usable and "people" in crowd.reasons[0]
    both = assess(REF, {0: 100.0}, 0.3, cfg, ref_sharpness=600.0)
    assert len(both.reasons) == 2


def test_reassess_uses_the_bag_median():
    recs = [{"sharpness": {"0": s}, "person_fraction": 0.0, "usable": True, "reasons": []} for s in (500, 520, 480, 150)]
    out = reassess(recs, CacheConfig())
    assert [r["usable"] for r in out] == [True, True, True, False]
