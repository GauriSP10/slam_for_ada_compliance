"""Tier 2 stereo: Fast-FoundationStereo (NVIDIA, zero-shot) run from its ONNX export with onnxruntime.

The ONNX file is a computation graph (data), not executable pickle, so loading it runs no third-party code.
It is slower than the PyTorch release (about 14 s per 736x544 pair on an RTX 4050, because the export replaces the
fused correlation kernel with generic ops), so every disparity map is cached on disk and a frame is only ever
computed once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from slam_for_ada.config import StereoConfig
from slam_for_ada.datatypes import DepthMap, DepthMethod, Mat3, RectifiedPair
from slam_for_ada.vision.stereo_depth import NAN, disparity_to_ref_depth

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None] * 255.0
_STD = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None] * 255.0
MODEL_MAX_DISPARITY = 416  # from the model's cfg.yaml; sets the nearest range it can see
DIVISIBLE_BY = 32


class OnnxStereoDepth:
    """Satisfies the ``DepthEstimator`` protocol. Results land on cam0's undistorted grid, like ``SgbmDepth``."""

    def __init__(
        self, cfg: StereoConfig, K_ref: Mat3, model_path: Path | None = None, cache_dir: Path | None = None, session: Any | None = None
    ):
        """``session`` (anything with ``run(None, feed)``) replaces the ONNX model; used by the tests."""
        self.cfg, self.K_ref, self.cache_dir = cfg, np.asarray(K_ref, dtype=np.float64), cache_dir
        if session is not None:
            self._sess, self.provider = session, "injected"
        else:
            if model_path is None:
                raise ValueError("need a model_path or an injected session")
            import onnxruntime as ort
            import torch  # noqa: F401  (importing torch first puts the CUDA runtime DLLs where onnxruntime can find them)

            so = ort.SessionOptions()
            so.log_severity_level = 3
            self._sess = ort.InferenceSession(str(model_path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.provider = self._sess.get_providers()[0]
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _cache_file(self, pair: RectifiedPair) -> Path | None:
        if not self.cache_dir:
            return None
        r = pair.ref
        return Path(self.cache_dir) / f"{r.bag}_{r.index:05d}_{pair.pair.left}{pair.pair.right}.npy"

    def disparity(self, pair: RectifiedPair) -> np.ndarray:
        """Dense sub-pixel disparity of the rectified left image, NaN where the model reports nothing usable."""
        cache = self._cache_file(pair)
        if cache is not None and cache.exists():
            return np.load(cache)
        h, w = pair.left.shape[:2]
        ph, pw = (-h) % DIVISIBLE_BY, (-w) % DIVISIBLE_BY

        def prep(img: np.ndarray) -> np.ndarray:
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            x = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            return (np.ascontiguousarray(x.transpose(2, 0, 1)[None]) - _MEAN) / _STD  # ImageNet normalisation, as the export expects

        out = self._sess.run(None, {"left_image": prep(pair.left), "right_image": prep(pair.right)})[0]
        d = np.squeeze(np.asarray(out))[:h, :w].astype(np.float32)
        d[d < 0.5] = NAN
        if cache is not None:
            np.save(cache, d)
        return d

    def __call__(self, pair: RectifiedPair) -> DepthMap:
        z_min = pair.focal_px * pair.baseline_m / (MODEL_MAX_DISPARITY - 1)  # nearest range the model can represent
        return disparity_to_ref_depth(
            self.disparity(pair), pair, self.K_ref, self.cfg, DepthMethod.LEARNED, self.cfg.learned_sigma_disparity_px, z_min
        )
