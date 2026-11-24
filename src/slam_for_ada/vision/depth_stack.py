"""One object that turns a 5-camera set into per-pair and fused depth maps (rectify -> match -> fuse)."""

from __future__ import annotations

from slam_for_ada.calib.rectify import PairRectifier, build_rectifiers
from slam_for_ada.calib.rig import load_rig
from slam_for_ada.config import PROJECT_ROOT, Config
from slam_for_ada.datatypes import CameraSet, DepthMap, RigCalibration
from slam_for_ada.vision.hybrid_depth import combine_matchers
from slam_for_ada.vision.multibaseline import fuse_depths
from slam_for_ada.vision.stereo_depth import SgbmDepth


class DepthStack:
    """Holds the rig, the rectification tables and the matcher, so each frame costs only the matching.

    ``method`` selects the matcher: ``"sgbm"`` (Tier 1, classical), ``"learned"`` (Tier 2, Fast-FoundationStereo) or
    ``"hybrid"`` (learned coverage, cross-checked against the classical matcher). It defaults to ``cfg.stereo.method``.
    """

    def __init__(self, cfg: Config, rig: RigCalibration | None = None, method: str | None = None):
        self.cfg = cfg
        self.method = method or cfg.stereo.method
        self.rig = rig or load_rig(cfg.path("calibration") / "cams_calib.yaml")
        self.rectifiers: dict[int, PairRectifier] = build_rectifiers(self.rig, cfg.stereo.right_cams)
        if self.method == "sgbm":
            self.estimator = SgbmDepth(cfg.stereo, self.rig.cameras[0].K)
        elif self.method in ("learned", "hybrid"):
            from slam_for_ada.vision.learned_stereo import OnnxStereoDepth

            learned = OnnxStereoDepth(
                cfg.stereo, self.rig.cameras[0].K, PROJECT_ROOT / cfg.stereo.learned_model, cfg.path("cache") / "learned_disparity"
            )
            if self.method == "learned":
                self.estimator = learned
            else:
                classical = SgbmDepth(cfg.stereo, self.rig.cameras[0].K)
                self.estimator = lambda pair: combine_matchers(classical(pair), learned(pair))  # DepthEstimator protocol
        else:
            raise ValueError(f"unknown stereo method {self.method!r}")

    @property
    def K_ref(self):
        """Intrinsics of cam0's undistorted grid; every depth map and every pixel coordinate downstream uses it."""
        return self.rig.cameras[0].K

    def per_pair(self, cam_set: CameraSet) -> list[DepthMap]:
        """One depth map per camera pair, narrowest baseline first, all on cam0's grid."""
        return [self.estimator(r.rectify(cam_set)) for r in self.rectifiers.values()]

    def depth(self, cam_set: CameraSet) -> tuple[DepthMap, list[DepthMap]]:
        """(fused depth map, the per-pair maps it was built from)."""
        maps = self.per_pair(cam_set)
        return fuse_depths(maps, self.cfg.stereo.fuse_outlier_sigmas), maps
