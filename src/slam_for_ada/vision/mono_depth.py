"""Tier 3: metric monocular depth (Depth Anything V2, indoor), used only as an independent check on the stereo scale.

A single image carries no baseline, so this model predicts metric depth from learned priors. It is far less precise than
calibrated stereo, but its errors are independent of ours: if it also puts the camera about 1.27 m above the floor, the
stereo scale is corroborated. Loaded from ``safetensors`` only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from slam_for_ada.datatypes import ImageBGR

MONO_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

LowResDepth = Callable[[ImageBGR], np.ndarray]


class HfDepthBackend:
    """Depth Anything V2 from Hugging Face; returns the model's native-resolution depth in metres."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = AutoImageProcessor.from_pretrained(MONO_MODEL)
        self._model = AutoModelForDepthEstimation.from_pretrained(MONO_MODEL, use_safetensors=True).to(self.device).eval()

    def __call__(self, image: ImageBGR) -> np.ndarray:
        import torch
        from PIL import Image

        self._load()
        inputs = self._proc(images=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self._model(**inputs).predicted_depth  # (1, h', w')
        return pred[0].float().cpu().numpy()


class MetricMonoDepth:
    """Single-image metric depth on the input image's own grid (metres). Pass ``backend`` to substitute the model."""

    def __init__(self, backend: LowResDepth | None = None, device: str | None = None):
        self.backend: LowResDepth = backend or HfDepthBackend(device)

    def __call__(self, image: ImageBGR) -> np.ndarray:
        """(H, W) float32 depth in metres for a BGR image: the backend's output, resampled to the image size."""
        h, w = image.shape[:2]
        low = np.asarray(self.backend(image), dtype=np.float32)
        if low.ndim != 2:
            raise ValueError(f"backend must return a 2D depth map, got shape {low.shape}")
        return cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
