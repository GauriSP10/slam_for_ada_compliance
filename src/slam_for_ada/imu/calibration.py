
"""Transform VectorNav IMU directions into the cam0 reference frame."""

from pathlib import Path

import numpy as np
import yaml

from slam_for_ada.calib.rig import load_rig, relative_transform

def load_imu_to_cam0_rotation(
    cameras_yaml: Path,
    imu_yaml: Path,
) -> np.ndarray:
    """Return rotation mapping IMU-frame vectors into cam0."""

    # Camera-to-camera calibration
    rig = load_rig(cameras_yaml)

    # T_cam2_cam0 maps points from cam0 to cam2
    T_cam2_cam0 = relative_transform(rig, 0, 2)

    # Read the IMU-to-camera-2 calibration
    with Path(imu_yaml).open(encoding="utf-8") as f:
        calibration = yaml.safe_load(f)

    entry = calibration["cam0"]

    if entry["rostopic"] != "/camera_array/cam2/image_raw":
        raise ValueError("Expected the camera 2 IMU calibration")

    # Kalibr T_cam_imu maps IMU coordinates into camera coordinates
    T_cam2_imu = np.asarray(entry["T_cam_imu"], dtype=float)

    if T_cam2_imu.shape != (4, 4):
        raise ValueError("T_cam_imu must be a 4x4 matrix")

    R_cam2_imu = T_cam2_imu[:3, :3]
    R_cam2_cam0 = T_cam2_cam0[:3, :3]

    # Inverse of cam0 -> cam2 rotation, then IMU -> cam2
    R_cam0_imu = R_cam2_cam0.T @ R_cam2_imu

    if not np.allclose(
        R_cam0_imu.T @ R_cam0_imu,
        np.eye(3),
        atol=1e-3,
    ):
        raise ValueError("Invalid combined camera/IMU rotation")

    if not np.isclose(np.linalg.det(R_cam0_imu), 1, atol=1e-3):
        raise ValueError("Combined rotation must be proper")

    return R_cam0_imu
