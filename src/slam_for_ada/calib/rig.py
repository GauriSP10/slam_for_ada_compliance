"""Parse data/calibration/cams_calib.yaml into the 5-camera rig model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from slam_for_ada.datatypes import CameraId, CameraIntrinsics, Mat4, RigCalibration


def load_rig(path: Path) -> RigCalibration:
    """Read intrinsics, distortion and the chain of neighbour-to-neighbour transforms.

    Kalibr convention: ``T_cn_cnm1`` maps points from camera n-1 into camera n. Camera 0 has none.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cameras: dict[CameraId, CameraIntrinsics] = {}
    transforms: dict[CameraId, Mat4] = {}
    for key, entry in raw.items():
        cam = int(key.removeprefix("cam"))
        fx, fy, cx, cy = entry["intrinsics"]
        w, h = entry["resolution"]
        d = entry["distortion_coeffs"]
        if entry["distortion_model"] != "radtan" or entry["camera_model"] != "pinhole":
            raise ValueError(f"cam{cam}: only pinhole + radtan is supported")
        cameras[cam] = CameraIntrinsics(cam, fx, fy, cx, cy, int(w), int(h), (d[0], d[1], d[2], d[3]))
        if "T_cn_cnm1" in entry:
            transforms[cam] = np.array(entry["T_cn_cnm1"], dtype=np.float64)
    return RigCalibration(cameras, transforms, str(path))


def relative_transform(rig: RigCalibration, i: CameraId, j: CameraId) -> Mat4:
    """4x4 transform mapping points in camera ``i`` coordinates into camera ``j`` coordinates (i < j)."""
    if not i < j:
        raise ValueError("need i < j")
    T = np.eye(4)
    for n in range(i + 1, j + 1):
        T = rig.T_cn_cnm1[n] @ T
    return T


def baseline_m(rig: RigCalibration, i: CameraId, j: CameraId) -> float:
    """Straight-line distance between the optical centres of two cameras."""
    T = relative_transform(rig, i, j)
    return float(np.linalg.norm(T[:3, 3]))
