
"""Unit tests for VectorNav IMU gravity estimation."""

import numpy as np
import pytest

from slam_for_ada.datatypes import ImuSample
from slam_for_ada.imu.gravity import ImuGravityEstimator

def make_sample(t_ns, accel, gyro=(0.0, 0.0, 0.0)):
    """Create a synthetic IMU sample."""
    return ImuSample(
        t_ns=t_ns,
        accel=accel,
        gyro=gyro,
    )

def test_stationary_gravity():
    """A stationary accelerometer points opposite to gravity."""
    estimator = ImuGravityEstimator(np.eye(3))

    samples = [
        make_sample(0, (0.0, 9.80665, 0.0)),
        make_sample(10_000_000, (0.0, 9.80665, 0.0)),
        make_sample(20_000_000, (0.0, 9.80665, 0.0)),
    ]

    result = estimator(samples, 10_000_000)

    np.testing.assert_allclose(
        result.down_cam,
        [0.0, -1.0, 0.0],
        atol=1e-6,
    )

    assert result.stationarity == pytest.approx(1.0)
    assert result.source == "accel"

def test_camera_frame_rotation():
    """Gravity must be rotated from the IMU into camera coordinates."""
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])

    estimator = ImuGravityEstimator(rotation)

    samples = [
        make_sample(0, (-9.80665, 0.0, 0.0)),
    ]

    result = estimator(samples, 0)

    np.testing.assert_allclose(
        result.down_cam,
        [0.0, 1.0, 0.0],
        atol=1e-6,
    )

def test_no_nearby_samples():
    estimator = ImuGravityEstimator(np.eye(3))

    samples = [
        make_sample(0, (0.0, 9.80665, 0.0)),
    ]

    with pytest.raises(ValueError, match="No IMU samples"):
        estimator(samples, 1_000_000_000)

def test_invalid_acceleration():
    estimator = ImuGravityEstimator(np.eye(3))

    samples = [
        make_sample(0, (0.0, 0.0, 0.0)),
    ]

    with pytest.raises(ValueError, match="too small"):
        estimator(samples, 0)

def test_invalid_rotation():
    with pytest.raises(ValueError, match="Rotation must be 3x3"):
        ImuGravityEstimator(np.zeros((2, 2)))

    with pytest.raises(ValueError, match="Invalid rotation"):
        ImuGravityEstimator(np.zeros((3, 3)))

def test_motion_reduces_stationarity():
    estimator = ImuGravityEstimator(np.eye(3))

    stationary = [
        make_sample(0, (0.0, 9.80665, 0.0)),
    ]

    moving = [
        make_sample(
            0,
            (0.0, 9.80665, 0.0),
            gyro=(0.5, 0.0, 0.0),
        ),
    ]

    still_result = estimator(stationary, 0)
    moving_result = estimator(moving, 0)

    assert moving_result.stationarity < still_result.stationarity
    assert 0.0 <= moving_result.stationarity <= 1.0
