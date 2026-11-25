
"""Estimate gravity direction from VectorNav IMU measurements."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from slam_for_ada.datatypes import (
    GravityEstimate,
    ImuSample,
)

class ImuGravityEstimator:
    """Estimate camera-frame down from near-stationary IMU samples.

    R_cam_imu rotates vectors from the IMU frame to the
    reference camera frame. It must come from calibration.
    """

    def __init__(
        self,
        R_cam_imu: np.ndarray,
        window_ns: int = 100_000_000,
    ) -> None:
        rotation = np.asarray(R_cam_imu, dtype=float)

        if rotation.shape != (3, 3):
            raise ValueError("Rotation must be 3x3")

        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-3
        ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
            raise ValueError("Invalid rotation matrix")

        if window_ns <= 0:
            raise ValueError("window_ns must be positive")

        self.R_cam_imu = rotation
        self.window_ns = window_ns

    def __call__(
        self,
        samples: Sequence[ImuSample],
        t_ns: int,
    ) -> GravityEstimate:
        """Estimate gravity near the requested camera timestamp."""

        nearby = [
            s for s in samples
            if abs(s.t_ns - t_ns) <= self.window_ns // 2
        ]

        if not nearby:
            raise ValueError("No IMU samples near requested time")

        accel = np.asarray([s.accel for s in nearby], dtype=float)
        gyro = np.asarray([s.gyro for s in nearby], dtype=float)

        if not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro)):
            raise ValueError("Non-finite IMU measurements")

        mean_accel = np.mean(accel, axis=0)
        magnitude = np.linalg.norm(mean_accel)

        if magnitude < 1e-6:
            raise ValueError("Acceleration vector is too small")

        # A stationary accelerometer measures specific force
        # opposite to gravity. Negate to obtain the down direction.
        down_imu = -mean_accel / magnitude

        down_cam = self.R_cam_imu @ down_imu
        down_cam /= np.linalg.norm(down_cam)

        # Simple confidence proxy; not a calibrated probability.
        norm_error = abs(magnitude - 9.80665) / 9.80665
        gyro_rms = float(
            np.sqrt(np.mean(np.sum(gyro**2, axis=1)))
        )
        accel_spread = float(
            np.sqrt(np.mean(np.sum(
                (accel - mean_accel)**2, axis=1
            )))
        )

        stationarity = float(np.clip(
            1.0
            - norm_error
            - gyro_rms / 0.5
            - accel_spread / 3.0,
            0.0,
            1.0,
        ))

        return GravityEstimate(
            t_ns=t_ns,
            down_cam=down_cam,
            stationarity=stationarity,
            source="accel",
        )
