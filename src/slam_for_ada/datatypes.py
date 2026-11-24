"""Shared data contracts for the Level 2 pipeline.

This file is the shared contract between the vision stages and the rules / scoring stages.
Change it only by agreement, because both sides code against it.

Data flow
---------
    bag -> FrameRef/CameraSet -> FrameQuality -> RectifiedPair -> DepthMap (per pair)
        -> DepthMap (fused) -> Plane3D (floor) -> DoorCandidate / ObjectDetection
        -> Measurement (many per feature) -> FeatureEstimate (one per feature)
        -> ComplianceResult (vs GroundTruthFeature and AdaRule) -> ErrorSummary

Conventions (all code must follow these)
----------------------------------------
Units       Geometry is in METRES everywhere inside the vision code. Anything that is
            compared against the tape measure or an ADA rule is in INCHES
            (Measurement, GroundTruthFeature, AdaRule, ComplianceResult).
            Convert once, at the edge: see M_PER_INCH.
Time        Integer nanoseconds (``t_ns``), the ROS header stamp.
Camera      OpenCV camera frame: x right, y down, z forward. Cameras are cam0 (leftmost)
            to cam4 (rightmost); ``CameraPair(left, right)`` always has left < right.
Images      ``ImageBGR`` is uint8 (H, W, 3), BGR channel order (OpenCV).
Depth       float32 metres, ``NaN`` where invalid. Never use 0 for "unknown".
Transforms  4x4 homogeneous. ``T_cn_cnm1`` maps points from camera n-1 into camera n
            (Kalibr convention, as in cams_calib.yaml).
Plane       ``n . x + d = 0`` with ``n`` a unit vector. For the floor, ``n`` points UP
            (away from the floor), so ``signed_distance(x)`` is height above the floor.
Pixels      (x, y) = (column, row), origin top-left.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------- aliases
ImageBGR = NDArray[np.uint8]  # (H, W, 3)
ImageGray = NDArray[np.uint8]  # (H, W)
Mask = NDArray[np.bool_]  # (H, W)
FloatMap = NDArray[np.float32]  # (H, W)
Vec3 = NDArray[np.float64]  # (3,)
Mat3 = NDArray[np.float64]  # (3, 3)
Mat4 = NDArray[np.float64]  # (4, 4)
Points3D = NDArray[np.float64]  # (N, 3), metres

TimeNs = int
CameraId = int  # 0..4 for the front array

M_PER_INCH = 0.0254
INCH_PER_M = 1.0 / M_PER_INCH

# Bump when a field is added, removed or changes meaning. RunResult records it so results
# produced under an older contract are recognisable.
SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- enums
class FeatureKind(str, Enum):
    """What is being measured. Values are the strings used in ground_truth.yaml / ada_rules.yaml."""

    DOOR_WIDTH = "door_width"
    HALLWAY_WIDTH = "hallway_width"
    HARDWARE_HEIGHT = "hardware_height"  # door handles
    OPERABLE_HEIGHT = "operable_height"  # thermostat, card reader
    HANDRAIL_HEIGHT = "handrail_height"
    SIGNAGE_HEIGHT = "signage_height"
    FOUNTAIN_HEIGHT = "fountain_height"
    PROTRUSION = "protrusion"  # depth an object sticks out from a wall
    SURFACE_HEIGHT = "surface_height"  # tables, counters, microwave
    OUTLET_HEIGHT = "outlet_height"
    OBJECT_HEIGHT = "object_height"  # e.g. dustbin; no ADA rule
    HALLWAY_LENGTH = "hallway_length"  # no ADA rule


class DepthMethod(str, Enum):
    """How a depth map was produced."""

    SGBM = "sgbm"  # Tier 1: classical block matching
    LEARNED = "learned"  # Tier 2: zero-shot learned stereo
    FUSED = "fused"  # multi-baseline fusion of several maps
    MONOCULAR = "monocular"  # Tier 3: metric monocular cross-check


class MeasureMethod(str, Enum):
    """How a Measurement was obtained. Used to break the error table down."""

    STEREO_DEPTH = "stereo_depth"  # back-project edges / points to 3D, take distance
    HOMOGRAPHY = "homography"  # rectify the wall plane, measure in the plane
    MASK_PLANE_3D = "mask_plane_3d"  # Tier 2: segmentation mask lifted to 3D with plane fit
    PLANE_OFFSET = "plane_offset"  # height above floor / offset from wall plane


class DetectionSource(str, Enum):
    HARRIS_EDGES = "harris_edges"  # Tier 1
    OPEN_VOCAB = "open_vocab"  # Tier 2


class Verification(str, Enum):
    """How well an ADA threshold has been checked against the source text."""

    STANDARD_VERBATIM = "standard_verbatim"
    GUIDE_VERBATIM = "guide_verbatim"
    SECONDARY_SOURCES = "secondary_sources"
    SINGLE_SOURCE_UNCHECKED = "single_source_unchecked"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BORDERLINE = "borderline"  # inside the uncertainty band around the limit
    NO_RULE = "no_rule"  # feature has no ADA rule; accuracy-only


# --------------------------------------------------------------------------- identifiers
@dataclass(frozen=True, slots=True)
class FrameRef:
    """Names one moment in one bag. Used as the key for cached data."""

    bag: str  # bag file stem, e.g. "2023-03-14-12-14-02_33"
    index: int  # running index of the 5-camera set within that bag
    t_ns: TimeNs  # timestamp of the reference camera (cam0)


@dataclass(frozen=True, slots=True)
class CameraPair:
    """Two cameras used as a stereo pair. ``left < right``."""

    left: CameraId
    right: CameraId

    def __post_init__(self) -> None:
        if not 0 <= self.left < self.right:
            raise ValueError(f"need 0 <= left < right, got {self.left}, {self.right}")

    @property
    def spacing(self) -> int:
        """Number of camera slots apart (1..4). Baseline is about 0.164 m * spacing."""
        return self.right - self.left


# --------------------------------------------------------------------------- calibration
@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole + radial-tangential model for one camera (from cams_calib.yaml)."""

    camera: CameraId
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, float, float, float]  # k1, k2, p1, p2

    @property
    def K(self) -> Mat3:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def size(self) -> tuple[int, int]:
        """(width, height), the order OpenCV expects."""
        return (self.width, self.height)


@dataclass(frozen=True, slots=True, eq=False)
class RigCalibration:
    """The whole 5-camera array. Built by calib/rig.py; read-only afterwards."""

    cameras: Mapping[CameraId, CameraIntrinsics]
    T_cn_cnm1: Mapping[CameraId, Mat4]  # keys 1..4; maps cam n-1 coords -> cam n coords
    source_file: str = ""


# --------------------------------------------------------------------------- raw sensor data
@dataclass(frozen=True, slots=True, eq=False)
class CameraSet:
    """Images from all five cameras taken at (nearly) the same instant."""

    ref: FrameRef
    images: Mapping[CameraId, ImageBGR]
    stamps_ns: Mapping[CameraId, TimeNs]

    def __post_init__(self) -> None:
        shapes = {im.shape for im in self.images.values()}
        if len(shapes) != 1:
            raise ValueError(f"all images must share one shape, got {shapes}")
        if set(self.images) != set(self.stamps_ns):
            raise ValueError("images and stamps_ns must cover the same cameras")

    @property
    def max_spread_ns(self) -> int:
        """Largest timestamp difference between any two cameras (about 1.6 ms in this dataset)."""
        vals = list(self.stamps_ns.values())
        return max(vals) - min(vals)


@dataclass(frozen=True, slots=True)
class ImuSample:
    """One IMU reading (Vectornav, 200 Hz)."""

    t_ns: TimeNs
    accel: tuple[float, float, float]  # m/s^2, IMU frame, includes gravity
    gyro: tuple[float, float, float]  # rad/s, IMU frame


@dataclass(frozen=True, slots=True, eq=False)
class GravityEstimate:
    """Which way is down, expressed in the reference camera frame."""

    t_ns: TimeNs
    down_cam: Vec3  # unit vector, direction gravity pulls, camera frame
    stationarity: float  # 0..1, how well the sensor was still (low = shaky, less trustworthy)
    source: str = "accel"  # e.g. "accel" (stand-in) or "complementary"

    def __post_init__(self) -> None:
        if self.down_cam.shape != (3,) or not np.isclose(np.linalg.norm(self.down_cam), 1.0, atol=1e-6):
            raise ValueError("down_cam must be a unit 3-vector")
        if not 0.0 <= self.stationarity <= 1.0:
            raise ValueError("stationarity must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FrameQuality:
    """Decides whether a CameraSet is good enough to measure on."""

    ref: FrameRef
    sharpness: Mapping[CameraId, float]  # e.g. variance of Laplacian per camera
    person_fraction: float  # share of the image covered by people, 0..1
    usable: bool
    reasons: tuple[str, ...] = ()  # why it was rejected, if it was


# --------------------------------------------------------------------------- stereo
@dataclass(frozen=True, slots=True, eq=False)
class RectifiedPair:
    """Two cameras warped so epipolar lines are horizontal. Output of calib/rectify.py."""

    ref: FrameRef
    pair: CameraPair
    left: ImageBGR
    right: ImageBGR
    R1: Mat3
    R2: Mat3
    P1: NDArray[np.float64]  # (3, 4) rectified projection of the left camera
    P2: NDArray[np.float64]  # (3, 4)
    Q: Mat4  # disparity-to-depth matrix
    baseline_m: float
    focal_px: float

    def __post_init__(self) -> None:
        if self.left.shape != self.right.shape:
            raise ValueError("left and right images must share one shape")
        if self.P1.shape != (3, 4) or self.P2.shape != (3, 4):
            raise ValueError("P1 and P2 must be (3, 4)")
        if not (self.baseline_m > 0 and self.focal_px > 0):
            raise ValueError("baseline_m and focal_px must be positive")

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.left.shape[:2]
        return (w, h)


@dataclass(frozen=True, slots=True, eq=False)
class DepthMap:
    """Metric depth on a pixel grid.

    Every DepthMap, including each per-pair one, is resampled onto the UNDISTORTED
    pinhole image grid of ``ref_cam`` (intrinsics ``K``). That makes maps from different
    pairs pixel-aligned, so multi-baseline fusion is a per-pixel operation.
    """

    ref: FrameRef
    ref_cam: CameraId
    K: Mat3
    depth_m: FloatMap  # NaN where invalid
    method: DepthMethod
    pairs: tuple[CameraPair, ...]  # which pairs contributed (several if fused)
    confidence: FloatMap | None = None  # 0..1 per pixel
    sigma_m: FloatMap | None = None  # per-pixel 1-sigma depth uncertainty, if modelled

    def __post_init__(self) -> None:
        if self.depth_m.ndim != 2:
            raise ValueError("depth_m must be (H, W)")
        for name in ("confidence", "sigma_m"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != self.depth_m.shape:
                raise ValueError(f"{name} shape must match depth_m")

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth_m.shape


@dataclass(frozen=True, slots=True, eq=False)
class Plane3D:
    """A plane ``n . x + d = 0`` in the ref-camera frame, with fit statistics.

    Used for the floor (n points up) and for wall planes.
    """

    normal: Vec3
    d: float
    n_inliers: int
    inlier_ratio: float
    rmse_m: float
    source: str = "ransac"

    def __post_init__(self) -> None:
        if not np.isclose(np.linalg.norm(self.normal), 1.0, atol=1e-6):
            raise ValueError("normal must be a unit vector")

    def signed_distance(self, points: Points3D) -> NDArray[np.float64]:
        """Signed distance in metres of each point from the plane. Floor: height above floor."""
        return points @ self.normal + self.d


# --------------------------------------------------------------------------- detection
@dataclass(frozen=True, slots=True, eq=False)
class DoorCandidate:
    """A possible door frame in the ref camera's undistorted image."""

    ref: FrameRef
    corners_px: NDArray[np.float64]  # (4, 2), order: top-left, top-right, bottom-right, bottom-left
    score: float  # 0..1
    source: DetectionSource
    is_glass: bool = False  # glass doors break stereo; flagged and normally rejected
    mask: Mask | None = None

    def __post_init__(self) -> None:
        if self.corners_px.shape != (4, 2):
            raise ValueError("corners_px must be (4, 2)")


@dataclass(frozen=True, slots=True, eq=False)
class ObjectDetection:
    """A non-door object to measure a height or protrusion for (card reader, sign, ...)."""

    ref: FrameRef
    kind: FeatureKind
    bbox_px: tuple[float, float, float, float]  # x0, y0, x1, y1
    score: float
    source: DetectionSource
    mask: Mask | None = None


# --------------------------------------------------------------------------- measurements
@dataclass(frozen=True, slots=True)
class Measurement:
    """One measurement of one feature in one frame. The core currency between vision and scoring."""

    ref: FrameRef
    kind: FeatureKind
    method: MeasureMethod
    value_in: float  # inches
    sigma_in: float  # 1-sigma uncertainty, inches (must be modelled, not guessed)
    distance_m: float  # camera-to-feature range; error grows with this
    depth_method: DepthMethod
    pairs: tuple[CameraPair, ...]  # camera pairs the depth came from
    feature_id: str | None = None  # ground-truth id once matched, e.g. "door_interior_width"
    note: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.value_in):
            raise ValueError("value_in must be finite")
        if not (np.isfinite(self.sigma_in) and self.sigma_in >= 0):
            raise ValueError("sigma_in must be finite and >= 0")

    @property
    def value_m(self) -> float:
        return self.value_in * M_PER_INCH


@dataclass(frozen=True, slots=True)
class FeatureEstimate:
    """One physical feature after combining its per-frame Measurements (temporal fusion)."""

    feature_id: str
    kind: FeatureKind
    method: MeasureMethod
    value_in: float
    sigma_in: float
    observations: tuple[Measurement, ...]

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("a FeatureEstimate needs at least one observation")
        if not (np.isfinite(self.value_in) and np.isfinite(self.sigma_in) and self.sigma_in >= 0):
            raise ValueError("value_in must be finite and sigma_in finite and >= 0")

    @property
    def n_obs(self) -> int:
        return len(self.observations)


# --------------------------------------------------------------------------- truth and rules
@dataclass(frozen=True, slots=True)
class GroundTruthFeature:
    """One tape-measured value. Loaded from data/ground_truth/ground_truth.yaml."""

    id: str
    kind: FeatureKind
    value_in: float
    confirmed: bool  # False = handwriting reading still uncertain
    photo: str
    description: str = ""
    bags: tuple[str, ...] = ()  # filled by the frame index; empty = not yet checked
    note: str = ""


@dataclass(frozen=True, slots=True)
class AdaRule:
    """One ADA limit. Loaded from config/ada_rules.yaml; values must be verified against the standard."""

    id: str
    kind: FeatureKind
    applies_to: tuple[str, ...] = ()  # ground-truth ids this rule is scored against; one kind can
    # fall under different rules (e.g. sink vs table), so match by id
    min_in: float | None = None  # value must be >= this
    max_in: float | None = None  # value must be <= this
    borderline_sigmas: float = 2.0  # within this many sigma of a limit => BORDERLINE
    citation: str = ""  # ADA standard section the limit comes from
    verification: Verification = Verification.SINGLE_SOURCE_UNCHECKED
    note: str = ""

    def __post_init__(self) -> None:
        if self.min_in is None and self.max_in is None:
            raise ValueError("a rule needs min_in, max_in, or both")
        if None not in (self.min_in, self.max_in) and self.min_in > self.max_in:  # type: ignore[operator]
            raise ValueError("min_in must not exceed max_in")


@dataclass(frozen=True, slots=True)
class ComplianceResult:
    """The final row: estimate vs truth, and verdict vs the true verdict."""

    estimate: FeatureEstimate
    truth: GroundTruthFeature
    rule: AdaRule | None
    error_in: float  # estimate - truth (signed)
    measured_verdict: Verdict
    true_verdict: Verdict  # the rule applied to the tape-measure value

    @property
    def abs_error_in(self) -> float:
        return abs(self.error_in)

    @property
    def decision_correct(self) -> bool | None:
        """None when there is no rule; BORDERLINE never counts as correct."""
        if self.measured_verdict is Verdict.NO_RULE:
            return None
        return self.measured_verdict is self.true_verdict


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    """Aggregate error for one group of results, e.g. group='method=homography'."""

    group: str
    n: int
    mean_abs_error_in: float
    median_abs_error_in: float
    max_abs_error_in: float
    rmse_in: float
    n_decisions: int = 0  # results that had a rule
    n_correct: int = 0  # of those, how many verdicts were right


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything one pipeline run produces; enough to regenerate every table and plot."""

    results: tuple[ComplianceResult, ...]
    summaries: tuple[ErrorSummary, ...]
    config_hash: str  # hash of the config used, for reproducibility
    created_utc: str  # ISO 8601
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------- stage interfaces
# Typing-only contracts: the signature each stage must satisfy. Implementations live in the
# modules named in each docstring. They let the vision and scoring sides code against each
# other before either is finished (swap in a fake that satisfies the Protocol).


class FrameSource(Protocol):
    """dataio/bag_reader.py + dataio/frame_cache.py."""

    def frames(self, bag: str) -> Iterator[CameraSet]: ...
    def imu(self, bag: str) -> Iterator[ImuSample]: ...


class GravityEstimator(Protocol):
    """imu/gravity.py. Must return a unit ``down_cam`` in the reference camera frame."""

    def __call__(self, samples: Sequence[ImuSample], t_ns: TimeNs) -> GravityEstimate: ...


class DepthEstimator(Protocol):
    """vision/stereo_depth.py. Tier 1 SGBM, Tier 2 learned; same signature."""

    def __call__(self, pair: RectifiedPair) -> DepthMap: ...


class DepthFuser(Protocol):
    """vision/multibaseline.py. Combines per-pair maps into one ``DepthMethod.FUSED`` map."""

    def __call__(self, depths: Sequence[DepthMap]) -> DepthMap: ...


class FloorFitter(Protocol):
    """vision/floor_plane.py. ``gravity`` is optional; the fit must work without it."""

    def __call__(self, depth: DepthMap, gravity: GravityEstimate | None = None) -> Plane3D: ...


class FeatureMeasurer(Protocol):
    """vision/width.py, height.py, protrusion.py. One call yields measurements for one frame."""

    def __call__(
        self, depth: DepthMap, floor: Plane3D, doors: Sequence[DoorCandidate], objects: Sequence[ObjectDetection]
    ) -> Sequence[Measurement]: ...


class Evaluator(Protocol):
    """ada/evaluate.py + eval/metrics.py. Estimates + truth + rules -> scored results."""

    def __call__(
        self, estimates: Sequence[FeatureEstimate], truths: Mapping[str, GroundTruthFeature], rules: Sequence[AdaRule]
    ) -> Sequence[ComplianceResult]: ...
