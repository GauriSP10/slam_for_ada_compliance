"""Shared fixtures: a pinhole camera and synthetic scenes whose correct answers are known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from slam_for_ada.datatypes import DepthMap, Plane3D
from tests.helpers import CAM_HEIGHT, CX, CY, F, H, W, make_depth


@pytest.fixture
def K() -> np.ndarray:
    return np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])


@pytest.fixture
def floor_plane() -> Plane3D:
    """Level camera at 1.3 m: camera y points down, so up is -y and the plane is  -y + 1.3 = 0."""
    return Plane3D(np.array([0.0, -1.0, 0.0]), CAM_HEIGHT, 1000, 0.5, 0.005)


@pytest.fixture
def scene_depth() -> np.ndarray:
    """Floor below the horizon (rows > CY) and a fronto-parallel wall 3 m away; the floor meets the wall at z = 3 m."""
    v = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))
    with np.errstate(divide="ignore"):
        floor_z = CAM_HEIGHT * F / (v - CY)
    return np.where((v > CY) & (floor_z < 3.0), floor_z, 3.0)


@pytest.fixture
def depth_map(K, scene_depth) -> DepthMap:
    return make_depth(K, scene_depth)


@pytest.fixture
def floor_depth_map(K) -> DepthMap:
    """A level floor seen from 1.3 m. With a 33 degree vertical field of view it only appears beyond about 4.4 m."""
    v = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(v > CY, CAM_HEIGHT * F / (v - CY), np.nan)
    z[z > 9.0] = np.nan  # beyond the working range
    return make_depth(K, z)
