"""Metric depth from one rectified pair. Tier 1: classical semi-global block matching. Tier 2 lives in learned_stereo.py."""

from __future__ import annotations

import cv2
import numpy as np

from slam_for_ada.config import StereoConfig
from slam_for_ada.datatypes import DepthMap, DepthMethod, Mat3, RectifiedPair

NAN = np.float32("nan")


def disparity_to_ref_depth(
    disp: np.ndarray, pair: RectifiedPair, K_ref: Mat3, cfg: StereoConfig, method: DepthMethod, sigma_disparity_px: float, z_min_m: float
) -> DepthMap:
    """Turn a rectified-left disparity map into a metric DepthMap on cam0's undistorted grid.

    Shared by every matcher: depth = f*B/d in the rectified frame, a per-pixel uncertainty z^2 * sigma_d / (f*B),
    then a homography resample onto the reference grid so maps from different pairs are pixel-aligned.
    ``disp`` is NaN where the matcher failed; depths nearer than ``z_min_m`` or beyond ``cfg.max_depth_m`` are dropped.
    """
    K_ref = np.asarray(K_ref, dtype=np.float64)
    f, B = pair.focal_px, pair.baseline_m
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (f * B / disp).astype(np.float32)
    z[(z < max(cfg.min_depth_m, z_min_m)) | (z > cfg.max_depth_m)] = NAN
    sigma = (z.astype(np.float64) ** 2 * sigma_disparity_px / (f * B)).astype(np.float32)

    h, w = z.shape
    H = pair.P1[:, :3] @ pair.R1 @ np.linalg.inv(K_ref)  # ref pixel -> rectified pixel

    def warp(a: np.ndarray) -> np.ndarray:
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        return cv2.warpPerspective(a, H, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))

    z_ref, s_ref = warp(z), warp(sigma)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ray = np.dstack([(u - K_ref[0, 2]) / K_ref[0, 0], (v - K_ref[1, 2]) / K_ref[1, 1], np.ones((h, w))])
    rz = np.einsum("ij,hwj->hwi", pair.R1, ray)[..., 2]  # z of the viewing ray in the rectified frame
    scale = (1.0 / rz).astype(np.float32)  # z_ref = z_rect / rz
    return DepthMap(pair.ref, 0, K_ref, (z_ref * scale).astype(np.float32), method, (pair.pair,), None, (s_ref * scale).astype(np.float32))


class SgbmDepth:
    """Satisfies the ``DepthEstimator`` protocol. Every result is resampled onto cam0's undistorted grid.

    A pair with spacing k has baseline about 0.164*k m. Its largest measurable disparity is
    ``num_disparities``, so it cannot see closer than ``f*B/num_disparities``: the widest pair is blind
    below about 2.3 m but is the most precise far away, and the narrow pairs cover the near field.
    """

    def __init__(self, cfg: StereoConfig, K_ref: Mat3):
        self.cfg, self.K_ref = cfg, np.asarray(K_ref, dtype=np.float64)
        bs = cfg.block_size
        self._sgbm = cv2.StereoSGBM_create(  # pyright: ignore[reportAttributeAccessIssue]  # missing from the OpenCV 5.1 stubs
            minDisparity=0,
            numDisparities=cfg.num_disparities,
            blockSize=bs,
            P1=8 * bs * bs,
            P2=32 * bs * bs,
            disp12MaxDiff=cfg.disp12_max_diff,
            preFilterCap=63,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=cfg.speckle_window,
            speckleRange=cfg.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def disparity(self, pair: RectifiedPair) -> np.ndarray:
        """Sub-pixel disparity of the rectified left image, NaN where matching failed."""
        gl = cv2.cvtColor(pair.left, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(pair.right, cv2.COLOR_BGR2GRAY)
        # SGBM cannot evaluate the leftmost num_disparities columns (its search window would run off the
        # image), which would discard a third of a 720 px image. Pad both images with black on the left,
        # match, then crop the padding off again.
        pad = self.cfg.num_disparities
        gl = cv2.copyMakeBorder(gl, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
        gr = cv2.copyMakeBorder(gr, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
        d = self._sgbm.compute(gl, gr)[:, pad:].astype(np.float32) / 16.0
        d[d < 0.5] = NAN  # SGBM marks invalid pixels with a negative value
        return d

    def __call__(self, pair: RectifiedPair) -> DepthMap:
        z_min = pair.focal_px * pair.baseline_m / (self.cfg.num_disparities - 1)  # closer than this wraps past the search range
        return disparity_to_ref_depth(
            self.disparity(pair), pair, self.K_ref, self.cfg, DepthMethod.SGBM, self.cfg.sigma_disparity_px, z_min
        )
