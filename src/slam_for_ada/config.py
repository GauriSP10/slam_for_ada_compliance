"""Typed settings loaded from config/default.yaml. No tunable number lives anywhere else."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class PathsConfig:
    bags: str = "data/raw/bags"
    calibration: str = "data/calibration"
    ground_truth: str = "data/ground_truth/ground_truth.yaml"
    annotations: str = "data/annotations/rois.yaml"
    rules: str = "config/ada_rules.yaml"
    cache: str = "data/cache"
    models: str = "data/models"
    outputs: str = "outputs"


@dataclass(frozen=True)
class CacheConfig:
    keyframe_window: int = 10  # keep the sharpest camera set out of every N consecutive sets (20 Hz -> 2 Hz)
    max_sync_spread_ms: float = 5.0  # cameras further apart than this are not one set
    png_compress: int = 1  # 0..9, low = fast
    sharpness_scale: float = 0.5  # downscale before the Laplacian so noise does not dominate
    min_sharpness_ratio: float = 0.6  # blurred if the keyframe is below this fraction of the bag's median sharpness
    # (absolute values depend on scene content, so the rule is relative)
    max_person_fraction: float = 0.10  # share of image covered by detected people
    person_conf: float = 0.35
    person_model: str = "yolov8n.pt"


@dataclass(frozen=True)
class StereoConfig:
    right_cams: tuple[int, ...] = (1, 2, 3, 4)  # pairs are (0, k); every depth lands on cam0's grid
    num_disparities: int = 256  # multiple of 16
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    sigma_disparity_px: float = 0.3  # 1-sigma matching error used to model depth uncertainty
    min_depth_m: float = 0.4
    max_depth_m: float = 20.0
    fuse_outlier_sigmas: float = 3.0
    method: str = "sgbm"  # "sgbm" (Tier 1) or "learned" (Tier 2, Fast-FoundationStereo ONNX)
    learned_model: str = "data/models/ffs/ffs_dynhw.onnx"
    learned_sigma_disparity_px: float = 0.3  # calibrated against floor scatter in scripts/evaluate_learned_stereo.py


@dataclass(frozen=True)
class FloorConfig:
    ransac_iters: int = 500
    inlier_thresh_m: float = 0.03
    v_min_frac: float = 0.55  # floor candidates come from the lower part of the image
    depth_min_m: float = 1.0
    depth_max_m: float = 7.0
    max_tilt_deg: float = 12.0  # reject planes whose normal is further than this from straight up
    seed: int = 0


@dataclass(frozen=True)
class MeasureConfig:
    jamb_snap_px: int = 18  # how far a hand-given jamb position may be snapped to an edge
    min_door_width_in: float = 20.0
    max_door_width_in: float = 60.0
    min_cos_incidence: float = 0.3  # reject a ray-plane intersection when the ray meets the plane within ~17 degrees (ill-conditioned)
    stereo_edge_bias_frac: float = 0.10  # systematic error of a stereo read at a depth edge, as a share of the width.
    # Empirical (door errors were 3-6 in on ~35 in) and calibrated on only 2 doors: a rough
    # model, not a validated one
    sample_rows: int = 40  # rows of a jamb used for the width estimate


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    floor: FloorConfig = field(default_factory=FloorConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)

    def path(self, name: str) -> Path:
        """Absolute path of one of the entries in ``paths``."""
        return PROJECT_ROOT / getattr(self.paths, name)


def _build(cls: type, values: Mapping[str, Any] | None):
    """Instantiate a section dataclass from a YAML mapping, ignoring unknown keys but coercing lists to tuples."""
    values = dict(values or {})
    known = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in values.items():
        if k not in known:
            continue
        kwargs[k] = tuple(v) if isinstance(v, list) else v
    return cls(**kwargs)


def load_config(path: Path | None = None) -> Config:
    """Load settings from YAML; any missing section or key falls back to the dataclass default."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    return Config(
        paths=_build(PathsConfig, raw.get("paths")),
        cache=_build(CacheConfig, raw.get("cache")),
        stereo=_build(StereoConfig, raw.get("stereo")),
        floor=_build(FloorConfig, raw.get("floor")),
        measure=_build(MeasureConfig, raw.get("measure")),
    )
