"""Undistort cam0 onto the pinhole grid that every depth map and every annotated pixel refers to."""

from __future__ import annotations

import cv2
import numpy as np

from slam_for_ada.datatypes import ImageBGR, RigCalibration


class RefUndistorter:
    """Undistorts camera images with their own distortion model but keeps the original intrinsics K.

    Because K is unchanged, a pixel in the output has the same viewing ray as the same pixel in cam0's depth map.
    """

    def __init__(self, rig: RigCalibration, cam: int = 0):
        c = rig.cameras[cam]
        self.K = c.K
        self._maps = cv2.initUndistortRectifyMap(c.K, np.array(c.dist), None, c.K, c.size, cv2.CV_32FC1)

    def __call__(self, image: ImageBGR) -> ImageBGR:
        return np.asarray(cv2.remap(image, *self._maps, cv2.INTER_LINEAR), dtype=np.uint8)
