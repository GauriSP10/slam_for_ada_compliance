"""The vision pipeline end to end: cached keyframes -> depth -> floor -> measurements -> one estimate per feature.

This module only orchestrates. The work lives in: ``features`` (per-kind measurement), ``floor_pool`` (the shared floor),
``tracking`` (the same feature in neighbouring frames) and ``fusion`` (combining repeated observations).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from slam_for_ada.auto import AutoStats, measure_from_detections
from slam_for_ada.calib.undistort import RefUndistorter
from slam_for_ada.config import Config
from slam_for_ada.dataio.annotations import Annotation, load_annotations
from slam_for_ada.dataio.frame_cache import load_camera_set
from slam_for_ada.datatypes import CameraSet, DepthMap, FeatureEstimate, Measurement, ObjectDetection, Plane3D
from slam_for_ada.features import HEIGHT_KINDS, measure_annotation
from slam_for_ada.floor_pool import pooled_floor
from slam_for_ada.fusion import fuse_observations
from slam_for_ada.tracking import track_height_feature
from slam_for_ada.vision.depth_stack import DepthStack
from slam_for_ada.vision.open_vocab import PROMPT_KINDS, OpenVocabDetector, to_object_detections


class VisionPipeline:
    """Owns the depth stack and the pooled floor, and measures annotated features frame by frame."""

    def __init__(self, cfg: Config, method: str | None = None):
        self.cfg = cfg
        self.stack = DepthStack(cfg, method=method)  # the matcher used for measurements
        # The floor is pooled over ~50 frames, so it always uses the fast classical matcher even when measurements
        # use the slower learned one (the rig is rigid, so the same floor applies to every frame either way).
        self._floor_stack = self.stack if self.stack.method == "sgbm" else DepthStack(cfg, rig=self.stack.rig, method="sgbm")
        self.undistort = RefUndistorter(self.stack.rig)
        self._floor: Plane3D | None = None

    def pooled_floor(self, stride: int = 4, use_cache: bool = True) -> Plane3D:
        """The sequence-wide floor plane (computed once, then cached in memory and on disk)."""
        if self._floor is None:
            self._floor = pooled_floor(self.cfg, self._floor_stack, stride, use_cache)
        return self._floor

    def measure(self, ann: Annotation, cam_set: CameraSet, depth: DepthMap, floor: Plane3D) -> list[Measurement]:
        """All measurements one annotation supports."""
        return measure_annotation(ann, cam_set, depth, floor, self.undistort, self.cfg.measure)

    def temporal_observations(self, ann: Annotation, n_neighbors: int = 3) -> list[Measurement]:
        """The same height feature measured in the keyframes around its own."""
        return track_height_feature(ann, self.cfg, self.stack, self.pooled_floor(), self.undistort, n_neighbors)

    def run(
        self, annotations: Sequence[Annotation], detector: OpenVocabDetector | None = None, threshold: float = 0.12
    ) -> tuple[list[Measurement], AutoStats]:
        """Measure every annotation. Depth is computed once per keyframe even if it holds several features.

        Without a ``detector`` every feature is measured from its hand-drawn geometry. With one, each height feature is
        measured from the detection mask under its annotated point, falling back to the hand-drawn depth box when no
        detection of the right kind is found. Doors and hallways always use their hand-drawn geometry.
        Returns the measurements and how each height feature was handled.
        """
        floor = self.pooled_floor()
        by_frame: dict[tuple[str, int], list[Annotation]] = defaultdict(list)
        for a in annotations:
            by_frame[(a.bag, a.key)].append(a)
        out: list[Measurement] = []
        stats = AutoStats()
        for (bag, key), anns in by_frame.items():
            cam_set = load_camera_set(self.cfg.path("cache"), bag, key)
            depth, _ = self.stack.depth(cam_set)
            objects: list[ObjectDetection] = []
            if detector is not None and any(a.kind in HEIGHT_KINDS for a in anns):
                objects = to_object_detections(
                    detector.detect(self.undistort(cam_set.images[0]), list(PROMPT_KINDS), threshold), cam_set.ref
                )
            for a in anns:
                if detector is not None and a.kind in HEIGHT_KINDS:
                    auto = measure_from_detections(a, cam_set, depth, floor, objects)
                    if auto is not None:
                        out.append(auto)
                        stats.from_detection += 1
                        continue
                    fallback = self.measure(a, cam_set, depth, floor)
                    out += fallback
                    stats.from_fallback += bool(fallback)
                    stats.unmeasured += not fallback
                else:
                    out += self.measure(a, cam_set, depth, floor)
        return out, stats


def run_vision(
    cfg: Config, method: str | None = None, auto: bool = False, detector: OpenVocabDetector | None = None
) -> tuple[list[Measurement], list[FeatureEstimate], Plane3D, AutoStats]:
    """Load the annotations, measure everything, fuse repeated observations. The single entry point of the vision side.

    ``method`` picks the stereo matcher for the measurements ("sgbm", "learned" or "hybrid"); the default is
    ``cfg.stereo.method``. With ``auto=True`` height features are measured from open-vocabulary detections (a detector is
    created if none is passed). The last return value says how many were measured from a detection vs the fallback.
    """
    pipe = VisionPipeline(cfg, method)
    if auto and detector is None:
        detector = OpenVocabDetector()
    ms, stats = pipe.run(load_annotations(cfg.path("annotations"), cfg.path("bags")), detector if auto else None)
    return ms, fuse_observations(ms), pipe.pooled_floor(), stats
