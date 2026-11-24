"""Rectify a camera pair so epipolar lines are horizontal, and relate rectified pixels back to camera 0."""

from __future__ import annotations

import cv2
import numpy as np

from slam_for_ada.calib.rig import relative_transform
from slam_for_ada.datatypes import CameraPair, CameraSet, RectifiedPair, RigCalibration


class PairRectifier:
    """Rectification for one pair. Builds the remap tables once, then rectifies any number of camera sets.

    The left camera of every pair is the reference camera (cam0), so a depth map computed in the
    rectified left frame can be warped back onto cam0's undistorted grid (see ``vision.stereo_depth.disparity_to_ref_depth``).
    """

    def __init__(self, rig: RigCalibration, pair: CameraPair, alpha: float = 0.0):
        self.rig, self.pair = rig, pair
        L, R = rig.cameras[pair.left], rig.cameras[pair.right]
        if L.size != R.size:
            raise ValueError("cameras of a pair must share an image size")
        T = relative_transform(rig, pair.left, pair.right)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            L.K,
            np.array(L.dist),
            R.K,
            np.array(R.dist),
            L.size,
            T[:3, :3],
            T[:3, 3].reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha,
        )
        self.R1, self.R2, self.P1, self.P2, self.Q = (np.asarray(m, dtype=np.float64) for m in (R1, R2, P1, P2, Q))
        self.size = L.size
        self.focal_px = float(P1[0, 0])
        self.baseline_m = float(abs(P2[0, 3] / P2[0, 0]))
        self._map_l = cv2.initUndistortRectifyMap(L.K, np.array(L.dist), R1, P1, L.size, cv2.CV_32FC1)
        self._map_r = cv2.initUndistortRectifyMap(R.K, np.array(R.dist), R2, P2, R.size, cv2.CV_32FC1)

    def rectify(self, cam_set: CameraSet) -> RectifiedPair:
        """Warp both images of this pair onto the common rectified grid."""
        left = cv2.remap(cam_set.images[self.pair.left], *self._map_l, cv2.INTER_LINEAR)
        right = cv2.remap(cam_set.images[self.pair.right], *self._map_r, cv2.INTER_LINEAR)
        return RectifiedPair(
            cam_set.ref,
            self.pair,
            np.asarray(left, dtype=np.uint8),
            np.asarray(right, dtype=np.uint8),
            self.R1,
            self.R2,
            self.P1,
            self.P2,
            self.Q,
            self.baseline_m,
            self.focal_px,
        )


def build_rectifiers(rig: RigCalibration, right_cams, alpha: float = 0.0) -> dict[int, PairRectifier]:
    """One rectifier per (0, k) pair, keyed by k."""
    return {k: PairRectifier(rig, CameraPair(0, k), alpha) for k in right_cams}
