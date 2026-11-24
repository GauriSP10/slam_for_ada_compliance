# Calibration files (ISEC dataset)

Downloaded from the public Drive links listed in the NUFR-M3F README
(https://github.com/neufieldrobotics/NUFR-M3F), section "ISEC dataset calibration".

| File | Contents | Drive file id |
|---|---|---|
| `cams_calib.yaml` | cam0-cam4: pinhole + radtan intrinsics, distortion, resolution 720x540, `T_cn_cnm1` (cam n-1 -> cam n). Adjacent baseline ~0.164 m | 1lxLkwcEMKf5TeIiJWZ2X6-NtqaPyxaO- |
| `cam2_imu_calib.yaml` | camera_2 <-> IMU transform and time shift | 1OkmgSuuP0SIqZA20dMNP2pWIjLqgDBre |
| `imu_params.yaml` | Vectornav noise densities and random walks, 200 Hz, topic `/imu/imu_compensated` | 1EOVYdOzPHtPbLP-UCKShVpp_N6YuehTd |

Not downloaded (not needed for Level 2): `cam5_imu_calib.yaml`, `cam6_imu_calib.yaml` (side cameras).

## Verified so far

- Image size and encoding in the bags match `cams_calib.yaml` (720x540, bgr8).
- Rectifying cam0-cam1 / cam0-cam2 / cam0-cam4 on one frame of bag 33 gave median
  vertical error 0.0 / 0.5 / 1.2 px. Single frame, ORB matches including outliers; the
  proper multi-frame check is still to do.

## NOT yet verified

- The rotation convention of `cam2_imu_calib.yaml` (camera->IMU or IMU->camera).
  Check before using it in `imu/gravity.py`.
- The ZED2i has no ISEC calibration (README: ZED calibrated for Snell only).
