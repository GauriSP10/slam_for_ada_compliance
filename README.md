# Stereo Visual SLAM for ADA Compliance: Level 2

EECE5554 project. Measures doors, hallways and wall-object heights from a calibrated
5-camera rig walking ISEC 4th floor, then reports PASS / FAIL against ADA thresholds and
compares every measurement to tape-measure ground truth.

- Results, limitations and every figure: `docs/LEVEL2_VISION_RESULTS.md`
- Data, provenance, calibration: `data/README.md`
- The original proposal PDF is kept out of the repo (it lists team members' contact details)

## Setup

Python 3.12 (not 3.14).

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Layout

```
config/                default.yaml, ada_rules.yaml          (all tunable numbers)
data/                  raw bags, calibration, ground truth, cache   (see data/README.md)
docs/                  results write-up
outputs/               generated tables, figures, videos (git-ignored)
scripts/               command-line entry points
src/slam_for_ada/
  datatypes.py config.py                       shared types and settings
  pipeline.py                                  orchestrator: floor, per-frame depth, measure, fuse
  features.py fusion.py floor_pool.py          per-kind measurement, repeated-observation fusion, shared floor
  tracking.py auto.py                          visual-odometry tracking; measuring from detection masks
  dataio/              bag_reader, frame_cache, annotations
  calib/               rig, rectify, calib_check
  vision/              frame_quality, stereo_depth, learned_stereo, hybrid_depth, multibaseline,
                       floor_plane, door_detector, open_vocab, auto_measure, mono_depth,
                       homography, width, height, protrusion, temporal_fusion
  imu/                 gravity
  ada/                 rules, evaluate
  eval/                ground_truth, minimal_scorer, sampling, floor_scatter  (metrics, ablation, report: still stubs)
  viz/                 overlays
tests/                 one test file per module
```

Level 3 (SLAM map) is out of scope here.

## Run

```
pip install -e .                                  # once, so scripts and tests find the package
python scripts/extract_frames.py                  # read the bags once -> data/cache (about 4 minutes)
python scripts/run_level2.py                      # measure, score, write outputs/level2_results.csv
python scripts/make_overlays.py                   # annotated frames and outputs/demo_hallway.mp4
python scripts/run_level2.py --stereo hybrid      # learned + classical matcher (best heights)
python scripts/run_level2.py --auto               # height features from detection masks (one click per feature)
pytest                                            # 86 unit tests
ruff check src scripts tests && ruff format --check src scripts tests
pyright                                           # type check (CI runs all of these)
```

Features are located by hand in `data/annotations/rois.yaml` (view frames with `scripts/annotate_view.py`).
