"""Tier 2 detection: text-prompted boxes (OWLv2) refined into masks (SAM), no training data needed.

Both models are loaded from ``safetensors`` only, so no pickled code from the download is ever executed.
A prompt such as "a wall thermostat" is turned into candidate boxes; each box is segmented; the masks are what the
measurement stages use to place a point on the object and to take range from the object's own pixels.

The logic (ranking, per-prompt limits, clipping, mask attachment) lives in ``OpenVocabDetector`` and is independent of the
models; the models sit behind the small ``DetectionBackend`` interface so they can be replaced by a fake in tests.
"""

# This module is a thin adapter to Hugging Face transformers, whose type stubs are incomplete for these models.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from slam_for_ada.datatypes import DetectionSource, FeatureKind, FrameRef, ImageBGR, ObjectDetection

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
SAM_MODEL = "facebook/sam-vit-base"

# prompt text -> the kind of feature it measures. The prompt wording matters to the detector, the kind to the pipeline.
PROMPT_KINDS: dict[str, FeatureKind] = {
    "a door handle": FeatureKind.HARDWARE_HEIGHT,
    "a door lever lock": FeatureKind.HARDWARE_HEIGHT,
    "a wall thermostat": FeatureKind.OPERABLE_HEIGHT,
    "a card reader": FeatureKind.OPERABLE_HEIGHT,
    "a handrail": FeatureKind.HANDRAIL_HEIGHT,
    "an electrical outlet": FeatureKind.OUTLET_HEIGHT,
    "a wall sign": FeatureKind.SIGNAGE_HEIGHT,
    "a drinking fountain": FeatureKind.FOUNTAIN_HEIGHT,
    "a trash can": FeatureKind.OBJECT_HEIGHT,
}

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RawDetection:
    prompt: str
    score: float
    box: Box
    mask: np.ndarray | None = None


class DetectionBackend(Protocol):
    """What the detector needs from the models."""

    def boxes(self, image: ImageBGR, prompts: Sequence[str], threshold: float) -> list[tuple[str, float, Box]]:
        """Candidate (prompt, score, box) triples in the image's own pixel coordinates."""
        ...

    def masks(self, image: ImageBGR, boxes: Sequence[Box]) -> list[np.ndarray]:
        """One boolean (H, W) mask per box."""
        ...


class HfOwlSamBackend:
    """OWLv2 (boxes) + SAM (masks) from Hugging Face. Models load on first use and stay on the GPU."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._owl: Any = None

    def _load(self) -> None:
        if self._owl is not None:
            return
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._owl_proc = Owlv2Processor.from_pretrained(OWL_MODEL)
        self._owl = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL, use_safetensors=True).to(self.device).eval()
        self._sam_proc = SamProcessor.from_pretrained(SAM_MODEL)
        self._sam = SamModel.from_pretrained(SAM_MODEL, use_safetensors=True).to(self.device).eval()

    @staticmethod
    def _pil(image: ImageBGR) -> Any:
        from PIL import Image

        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def boxes(self, image: ImageBGR, prompts: Sequence[str], threshold: float) -> list[tuple[str, float, Box]]:
        import torch

        self._load()
        h, w = image.shape[:2]
        inputs = self._owl_proc(text=[list(prompts)], images=self._pil(image), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._owl(**inputs)
        side = max(h, w)  # OWLv2 pads to a square, so boxes are relative to it
        res = self._owl_proc.post_process_grounded_object_detection(
            out, threshold=threshold, target_sizes=torch.tensor([[side, side]]), text_labels=[list(prompts)]
        )[0]
        return [
            (res["text_labels"][i], float(res["scores"][i]), tuple(float(v) for v in res["boxes"][i].cpu().numpy()))  # type: ignore[misc]
            for i in range(len(res["scores"]))
        ]

    def masks(self, image: ImageBGR, boxes: Sequence[Box]) -> list[np.ndarray]:
        import torch

        self._load()
        inputs = self._sam_proc(self._pil(image), input_boxes=[[list(b) for b in boxes]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._sam(**inputs, multimask_output=False)
        masks = self._sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]
        return [m[0].numpy().astype(bool) for m in masks]


class OpenVocabDetector:
    """Ranks and filters the backend's candidates and attaches masks. Pass ``backend`` to substitute the models."""

    def __init__(self, backend: DetectionBackend | None = None, device: str | None = None):
        self.backend: DetectionBackend = backend or HfOwlSamBackend(device)

    def detect(
        self, image: ImageBGR, prompts: Sequence[str], threshold: float = 0.15, with_masks: bool = True, max_per_prompt: int = 6
    ) -> list[RawDetection]:
        """Boxes (and masks) for each text prompt, best first, at most ``max_per_prompt`` per prompt.

        Boxes are clipped to the image; boxes thinner than 3 px are dropped.
        """
        h, w = image.shape[:2]
        dets: list[RawDetection] = []
        per_prompt: dict[str, int] = {}
        for prompt, score, (x0, y0, x1, y1) in sorted(self.backend.boxes(image, prompts, threshold), key=lambda t: -t[1]):
            if score < threshold or per_prompt.get(prompt, 0) >= max_per_prompt:
                continue
            box = (max(x0, 0.0), max(y0, 0.0), min(x1, w - 1.0), min(y1, h - 1.0))
            if box[2] - box[0] < 3 or box[3] - box[1] < 3:
                continue
            per_prompt[prompt] = per_prompt.get(prompt, 0) + 1
            dets.append(RawDetection(prompt, float(score), box))
        if with_masks and dets:
            masks = self.backend.masks(image, [d.box for d in dets])
            dets = [RawDetection(d.prompt, d.score, d.box, m) for d, m in zip(dets, masks)]
        return dets


def to_object_detections(dets: Sequence[RawDetection], ref: FrameRef) -> list[ObjectDetection]:
    """RawDetection -> the shared ``ObjectDetection`` contract, dropping prompts that have no feature kind."""
    return [
        ObjectDetection(ref, PROMPT_KINDS[d.prompt], d.box, d.score, DetectionSource.OPEN_VOCAB, d.mask)
        for d in dets
        if d.prompt in PROMPT_KINDS
    ]
