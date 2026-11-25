
"""Tests for VectorNav IMU to camera 0 calibration."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from slam_for_ada.calib.rig import load_rig, relative_transform
from slam_for_ada.datatypes import ImuSample
from slam_for_ada.imu.calibration import load_imu_to_cam0_rotation
from slam_for_ada.imu.gravity import ImuGravityEstimator

CAMERAS = Path("data/calibration/cams_calib.yaml")
IMU_CALIB = Path("data/calibration/cam2_imu_calib.yaml")

def test_rotation_matrix_is_valid():
    """The calibrated rotation must be orthogonal and proper."""
    rotation = load_imu_to_cam0_rotation(CAMERAS, IMU_CALIB)

    assert rotation.shape == (3, 3)

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-3,
    )

    assert np.linalg.det(rotation) == pytest.approx(
        1.0, abs=1e-3
    )

def test_camera_transform_composition():
    """Check the cam2-to-cam0 rotation composition."""
    rotation = load_imu_to_cam0_rotation(CAMERAS, IMU_CALIB)

    rig = load_rig(CAMERAS)
    T_cam2_cam0 = relative_transform(rig, 0, 2)

    with IMU_CALIB.open(encoding="utf-8") as file:
        calibration = yaml.safe_load(file)

    T_cam2_imu = np.asarray(
        calibration["cam0"]["T_cam_imu"],
        dtype=float,
    )

    # cam0 -> cam2, then IMU -> cam2
    expected = (
        T_cam2_cam0[:3, :3].T
        @ T_cam2_imu[:3, :3]
    )

    np.testing.assert_allclose(
        rotation, expected, atol=1e-6
    )

def test_gravity_estimator_accepts_calibration():
    """The gravity estimator accepts the real calibrated matrix."""
    rotation = load_imu_to_cam0_rotation(CAMERAS, IMU_CALIB)

    estimator = ImuGravityEstimator(rotation)

    sample = ImuSample(
        t_ns=1_000_000_000,
        accel=(0.0, 0.0, 9.80665),
        gyro=(0.0, 0.0, 0.0),
    )

    result = estimator([sample], sample.t_ns)

    assert result.down_cam.shape == (3,)
    assert np.linalg.norm(result.down_cam) == pytest.approx(1.0)
    assert result.stationarity == pytest.approx(1.0)

def test_missing_calibration_file():
    """Missing calibration data must raise an error."""
    with pytest.raises(FileNotFoundError):
        load_imu_to_cam0_rotation(
            CAMERAS,
            Path("data/calibration/nonexistent.yaml"),
        )
