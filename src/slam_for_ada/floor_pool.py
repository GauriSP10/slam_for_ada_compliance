
"""One floor plane for a sequence, pooled across frames and cached.

Optional IMU-assisted floor fitting requires independently verified
camera-to-IMU calibration. The original stereo-only behavior remains
the default.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np

from slam_for_ada.config import Config
from slam_for_ada.dataio.bag_reader import iter_imu, list_bags
from slam_for_ada.dataio.frame_cache import (
    load_camera_set,
    load_index,
)
from slam_for_ada.datatypes import Plane3D
from slam_for_ada.imu.calibration import load_imu_to_cam0_rotation
from slam_for_ada.imu.gravity import ImuGravityEstimator
from slam_for_ada.vision.depth_stack import DepthStack
from slam_for_ada.vision.floor_plane import (
    combine_floors,
    fit_floor,
)

def _read_cached_floor(cache):
    """Restore the original cached Plane3D representation."""
    data = json.loads(cache.read_text(encoding="utf-8"))

    return Plane3D(
        np.array(data["normal"]),
        data["d"],
        data["n_inliers"],
        data["inlier_ratio"],
        data["rmse_m"],
        data["source"],
    )

def _write_cached_floor(cache, floor):
    """Save a fitted plane using the existing cache format."""
    cache.parent.mkdir(parents=True, exist_ok=True)

    cache.write_text(
        json.dumps(
            {
                "normal": floor.normal.tolist(),
                "d": floor.d,
                "n_inliers": floor.n_inliers,
                "inlier_ratio": floor.inlier_ratio,
                "rmse_m": floor.rmse_m,
                "source": floor.source,
            }
        ),
        encoding="utf-8",
    )

def _fit_bag_floors(
    cfg,
    stack,
    bag,
    stride,
    estimator=None,
    min_stationarity=0.8,
):
    """Fit selected frames, optionally using nearby IMU measurements."""
    records = [
        record
        for record in load_index(cfg.path("cache"), bag.stem)
        if record["usable"]
    ][::stride]

    planes = []

    # IMU readings are processed in timestamp order. This avoids
    # storing an entire large ROS bag's IMU stream in memory.
    if estimator is not None:
        imu_stream = iter(iter_imu(bag))
        next_sample = next(imu_stream, None)
        window = deque()
        half_window = estimator.window_ns // 2
    else:
        imu_stream = None
        next_sample = None
        window = None
        half_window = 0

    for record in records:
        gravity = None

        if estimator is not None:
            timestamp = int(record["t_ns"])

            # Retain samples within the estimator's time window.
            upper = timestamp + half_window
            lower = timestamp - half_window

            while (
                next_sample is not None
                and next_sample.t_ns <= upper
            ):
                window.append(next_sample)
                next_sample = next(imu_stream, None)

            while window and window[0].t_ns < lower:
                window.popleft()

            if len(window) >= 3:
                try:
                    candidate = estimator(
                        list(window),
                        timestamp,
                    )

                    if candidate.stationarity >= min_stationarity:
                        gravity = candidate
                except ValueError:
                    # Invalid or insufficient IMU data:
                    # fall back to the original visual floor fitting.
                    gravity = None

        camera_set = load_camera_set(
            cfg.path("cache"),
            bag.stem,
            record["key"],
        )

        fused, _ = stack.depth(camera_set)

        plane = fit_floor(
            fused,
            cfg.floor,
            gravity=gravity,
        )

        if plane is not None:
            planes.append(plane)

    return planes

def pooled_floor(
    cfg: Config,
    stack: DepthStack,
    stride: int = 4,
    use_cache: bool = True,
    *,
    use_imu: bool = False,
    calibration_verified: bool = False,
    min_stationarity: float = 0.8,
) -> Plane3D:
    """Compute a robust pooled floor from usable keyframes.

    Default:
        Stereo-only fitting, preserving the original behavior.

    Optional:
        Use IMU-derived gravity during stationary intervals.
        The physical calibration must first be verified.

    Stereo and IMU-assisted results have separate cache files
    to prevent accidental reuse across modes.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")

    if not 0.0 <= min_stationarity <= 1.0:
        raise ValueError(
            "min_stationarity must be between 0 and 1"
        )

    if use_imu and not calibration_verified:
        raise ValueError(
            "IMU assistance requires physically verified "
            "camera-to-IMU calibration. Set "
            "calibration_verified=True only after validation."
        )

    cache_root = cfg.path("cache")
    cache_name = "floor_imu.json" if use_imu else "floor.json"
    cache = cache_root / cache_name

    if use_cache and cache.exists():
        return _read_cached_floor(cache)

    estimator = None

    if use_imu:
        calibration_dir = cfg.path("calibration")

        rotation = load_imu_to_cam0_rotation(
            calibration_dir / "cams_calib.yaml",
            calibration_dir / "cam2_imu_calib.yaml",
        )

        estimator = ImuGravityEstimator(rotation)

    planes = []

    for bag in list_bags(cfg.path("bags")):
        planes.extend(
            _fit_bag_floors(
                cfg,
                stack,
                bag,
                stride,
                estimator=estimator,
                min_stationarity=min_stationarity,
            )
        )

    if not planes:
        raise ValueError(
            "No valid floor planes could be estimated."
        )

    floor = combine_floors(planes)
    _write_cached_floor(cache, floor)

    return floor
