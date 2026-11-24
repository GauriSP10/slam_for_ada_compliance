"""Constants and builders shared by the tests (importable, unlike conftest.py)."""

from __future__ import annotations

import numpy as np

from slam_for_ada.datatypes import CameraPair, DepthMap, DepthMethod, FrameRef

F, CX, CY, W, H = 890.0, 369.0, 275.0, 720, 540
CAM_HEIGHT = 1.3  # metres above the floor
REF = FrameRef("synthetic", 0, 0)


DEFAULT_PAIRS = (CameraPair(0, 1),)


def make_depth(K, depth_m, sigma_m=0.02, pairs=DEFAULT_PAIRS, conf=1.0) -> DepthMap:
    """A DepthMap on cam0's grid with uniform sigma and confidence."""
    return DepthMap(
        REF,
        0,
        K,
        depth_m.astype(np.float32),
        DepthMethod.FUSED,
        pairs,
        np.full(depth_m.shape, conf, np.float32),
        np.full(depth_m.shape, sigma_m, np.float32),
    )
