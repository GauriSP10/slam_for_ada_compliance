
"""Tests for optional IMU-assisted floor pooling."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slam_for_ada.datatypes import ImuSample
from slam_for_ada.floor_pool import (
    _fit_bag_floors,
    pooled_floor,
)
from slam_for_ada.imu.gravity import ImuGravityEstimator

def make_config(tmp_path):
    """Minimal configuration for isolated floor-pooling tests."""
    return SimpleNamespace(
        path=lambda name: tmp_path / name,
        floor=SimpleNamespace(),
    )

def make_samples(gyro=(0.0, 0.0, 0.0)):
    """Stationary IMU readings around a camera timestamp."""
    return [
        ImuSample(
            t_ns=t,
            accel=(0.0, 9.80665, 0.0),
            gyro=gyro,
        )
        for t in (
            970_000_000,
            990_000_000,
            1_010_000_000,
        )
    ]

def setup_bag_mocks(monkeypatch, samples):
    """Replace bag and depth operations with lightweight test data."""
    import slam_for_ada.floor_pool as module

    monkeypatch.setattr(
        module,
        "load_index",
        lambda cache, bag: [
            {
                "key": 0,
                "t_ns": 1_000_000_000,
                "usable": True,
            }
        ],
    )

    monkeypatch.setattr(
        module,
        "load_camera_set",
        lambda cache, bag, key: "fake_camera_set",
    )

    monkeypatch.setattr(
        module,
        "iter_imu",
        lambda bag: iter(samples),
    )

    stack = SimpleNamespace(
        depth=lambda camera_set: ("fake_depth", None)
    )

    cfg = SimpleNamespace(
        path=lambda name: Path("/unused") / name,
        floor="fake_floor_config",
    )

    return cfg, stack

def test_stationary_imu_points_down():
    """A stationary sensor produces a unit downward direction."""
    estimator = ImuGravityEstimator(np.eye(3))

    estimate = estimator(
        make_samples(),
        t_ns=1_000_000_000,
    )

    np.testing.assert_allclose(
        estimate.down_cam,
        [0.0, -1.0, 0.0],
        atol=1e-6,
    )

    assert estimate.stationarity > 0.8
    assert estimate.source == "accel"

def test_stationary_gravity_reaches_floor_fitter(monkeypatch):
    """A valid gravity estimate is supplied to fit_floor."""
    import slam_for_ada.floor_pool as module

    cfg, stack = setup_bag_mocks(
        monkeypatch,
        make_samples(),
    )

    received = []

    def fake_fit_floor(depth, floor_cfg, gravity=None):
        received.append(gravity)
        return "floor_plane"

    monkeypatch.setattr(module, "fit_floor", fake_fit_floor)

    estimator = ImuGravityEstimator(np.eye(3))

    planes = _fit_bag_floors(
        cfg,
        stack,
        Path("fake.bag"),
        stride=1,
        estimator=estimator,
    )

    assert planes == ["floor_plane"]
    assert len(received) == 1
    assert received[0] is not None

    np.testing.assert_allclose(
        received[0].down_cam,
        [0.0, -1.0, 0.0],
        atol=1e-6,
    )

def test_moving_imu_falls_back_to_visual_floor(monkeypatch):
    """Low-stationarity IMU data must not guide floor fitting."""
    import slam_for_ada.floor_pool as module

    cfg, stack = setup_bag_mocks(
        monkeypatch,
        make_samples(gyro=(1.0, 0.0, 0.0)),
    )

    received = []

    def fake_fit_floor(depth, floor_cfg, gravity=None):
        received.append(gravity)
        return "visual_floor"

    monkeypatch.setattr(module, "fit_floor", fake_fit_floor)

    estimator = ImuGravityEstimator(np.eye(3))

    planes = _fit_bag_floors(
        cfg,
        stack,
        Path("fake.bag"),
        stride=1,
        estimator=estimator,
    )

    assert planes == ["visual_floor"]
    assert received == [None]

def test_default_mode_does_not_read_imu(monkeypatch):
    """The default stereo-only path never accesses the IMU."""
    import slam_for_ada.floor_pool as module

    cfg, stack = setup_bag_mocks(
        monkeypatch,
        make_samples(),
    )

    received = []

    def forbidden_imu_reader(bag):
        raise AssertionError("IMU should not be read")

    def fake_fit_floor(depth, floor_cfg, gravity=None):
        received.append(gravity)
        return "visual_floor"

    monkeypatch.setattr(
        module,
        "iter_imu",
        forbidden_imu_reader,
    )
    monkeypatch.setattr(
        module,
        "fit_floor",
        fake_fit_floor,
    )

    planes = _fit_bag_floors(
        cfg,
        stack,
        Path("fake.bag"),
        stride=1,
    )

    assert planes == ["visual_floor"]
    assert received == [None]

def test_unverified_calibration_is_rejected(tmp_path):
    """IMU mode must require explicit calibration verification."""
    cfg = make_config(tmp_path)

    with pytest.raises(
        ValueError,
        match="physically verified",
    ):
        pooled_floor(
            cfg,
            stack=None,
            use_cache=False,
            use_imu=True,
        )

def test_invalid_stride_is_rejected(tmp_path):
    cfg = make_config(tmp_path)

    with pytest.raises(ValueError, match="stride"):
        pooled_floor(
            cfg,
            stack=None,
            stride=0,
        )

def test_invalid_stationarity_threshold_is_rejected(tmp_path):
    cfg = make_config(tmp_path)

    with pytest.raises(
        ValueError,
        match="min_stationarity",
    ):
        pooled_floor(
            cfg,
            stack=None,
            min_stationarity=1.5,
        )

def test_imu_and_visual_modes_use_separate_caches(
    monkeypatch,
    tmp_path,
):
    """The IMU-assisted result must not overwrite floor.json."""
    import slam_for_ada.floor_pool as module

    cfg = make_config(tmp_path)
    fake_plane = object()
    cache_paths = []

    monkeypatch.setattr(
        module,
        "list_bags",
        lambda bag_dir: [Path("fake.bag")],
    )

    monkeypatch.setattr(
        module,
        "_fit_bag_floors",
        lambda *args, **kwargs: [fake_plane],
    )

    monkeypatch.setattr(
        module,
        "combine_floors",
        lambda planes: fake_plane,
    )

    monkeypatch.setattr(
        module,
        "_write_cached_floor",
        lambda path, plane: cache_paths.append(path),
    )

    monkeypatch.setattr(
        module,
        "load_imu_to_cam0_rotation",
        lambda cameras, imu: np.eye(3),
    )

    visual = pooled_floor(
        cfg,
        stack=None,
        use_cache=False,
    )

    assisted = pooled_floor(
        cfg,
        stack=None,
        use_cache=False,
        use_imu=True,
        calibration_verified=True,
    )

    assert visual is fake_plane
    assert assisted is fake_plane

    assert [path.name for path in cache_paths] == [
        "floor.json",
        "floor_imu.json",
    ]
