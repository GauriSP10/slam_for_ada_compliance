# data/

```
data/
  raw/bags/            5 ROS1 bags, ~43 GB total. GIT-IGNORED. Never edit.
  calibration/         camera + IMU calibration YAMLs (small, tracked)
  ground_truth/
    photos/            photos of the handwritten tape measurements (metadata stripped)
    ground_truth.yaml  typed-up values, inches (23 features, all confirmed)
  cache/               extracted frames, depth maps, etc. GIT-IGNORED, regenerable
```

## Where it came from

All data is from the **NUFR-M3F** dataset (Northeastern Field Robotics Lab):
https://github.com/neufieldrobotics/NUFR-M3F
Kaveti et al., "Challenges of Indoor SLAM: A Multi-Modal Multi-Floor Dataset for SLAM Evaluation",
IEEE CASE 2023, doi:10.1109/CASE56687.2023.10260618. Cite this in the report.

The 5 bags are that dataset's `4th_floor` sequence (131 s, ~66 m, one loop,
some dynamic content toward the end). Recorded 2023-03-14.

| Bag | Length | Notes |
|---|---|---|
| 2023-03-14-12-12-59_31b | 17 s | elevator, atrium railing, kitchen/lounge |
| 2023-03-14-12-13-30_32 | 32 s | lounge, start of curved hallway |
| 2023-03-14-12-14-02_33 | 32 s | hallway with wooden office doors |
| 2023-03-14-12-14-34_34 | 32 s | hallway end, back to atrium |
| 2023-03-14-12-15-06_35a | 18 s | atrium to elevators, people walking, ZED frames dropped |

## Topics we use

| Topic | Rate | Use |
|---|---|---|
| `/camera_array/cam0..cam4/image_raw` | 20 Hz, 720x540 bgr8 | stereo (hardware-synced, ~1.6 ms spread) |
| `/imu/imu_compensated` | 200 Hz | gravity direction (Vectornav 100) |

Present in the bags but not used in Level 2: `cam5`, `cam6` (side cameras), `/ouster/*`
(LiDAR), `/zed2i/*` (no ISEC calibration exists for it).

## Calibration files

See `calibration/README.md`.
