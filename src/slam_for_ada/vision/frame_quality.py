"""Decide whether a camera set is good enough to measure on: motion blur and people in the way."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from slam_for_ada.config import CacheConfig
from slam_for_ada.datatypes import CameraId, CameraSet, FrameQuality, ImageBGR

_PERSON_CLASS = 0  # COCO class id of "person"


def laplacian_sharpness(image: ImageBGR, scale: float = 0.5) -> float:
    """Variance of the Laplacian of the grayscale image: high = sharp, low = blurred.

    Downscaling first stops sensor noise from counting as detail.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def set_sharpness(cam_set: CameraSet, scale: float = 0.5) -> dict[CameraId, float]:
    """Sharpness of every camera in a set."""
    return {c: laplacian_sharpness(im, scale) for c, im in cam_set.images.items()}


class PersonDetector:
    """YOLOv8-nano person detector (CPU is fast enough for keyframes). Loaded lazily."""

    def __init__(self, model_dir: Path, weights: str = "yolov8n.pt", conf: float = 0.35):
        self.model_dir, self.weights, self.conf = Path(model_dir), weights, conf
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self.model_dir.mkdir(parents=True, exist_ok=True)
            path = self.model_dir / self.weights
            self._model = YOLO(str(path) if path.exists() else self.weights)
            if not path.exists():  # ultralytics downloaded it to the working dir; keep it with the models
                src = Path(self.weights)
                if src.exists():
                    src.replace(path)
        return self._model

    def boxes(self, image: ImageBGR) -> np.ndarray:
        """(N, 4) person boxes x0, y0, x1, y1 in pixels."""
        results: Any = self._load().predict(image, classes=[_PERSON_CLASS], conf=self.conf, verbose=False)
        res = results[0]
        return res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))

    def mask(self, image: ImageBGR) -> np.ndarray:
        """Boolean (H, W) mask, True inside any detected person box."""
        h, w = image.shape[:2]
        m = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in self.boxes(image):
            m[max(int(y0), 0) : int(y1) + 1, max(int(x0), 0) : int(x1) + 1] = True
        return m


def assess(ref, sharpness: Mapping[CameraId, float], person_frac: float, cfg: CacheConfig, ref_sharpness: float) -> FrameQuality:
    """Combine blur and person coverage into a usable / not-usable decision, with reasons.

    ``ref_sharpness`` is the bag's median of per-keyframe minimum sharpness. Absolute Laplacian
    variance depends on what the camera is looking at, so blur is judged relative to the bag.
    """
    reasons = []
    worst = min(sharpness.values())
    if worst < cfg.min_sharpness_ratio * ref_sharpness:
        reasons.append(f"blur (sharpness {worst:.0f} is {worst / ref_sharpness:.0%} of the bag median)")
    if person_frac > cfg.max_person_fraction:
        reasons.append(f"people cover {person_frac:.0%} of the image")
    return FrameQuality(ref, dict(sharpness), float(person_frac), not reasons, tuple(reasons))
