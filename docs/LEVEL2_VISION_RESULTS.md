# Level 2: vision side results

Everything below is reproducible from the repo: `python scripts/run_level2.py` regenerates the table, and the figures come from the other scripts listed at the end. All lengths are inches against the tape-measure ground truth.

## Pipeline

```
bag  ->  keyframes (sharpest of every 10 sets, blur + people filter)          dataio/, vision/frame_quality.py
     ->  rectify 4 camera pairs (0-1, 0-2, 0-3, 0-4), calibration checked       calib/
     ->  SGBM depth per pair, resampled onto cam0's undistorted grid            vision/stereo_depth.py
     ->  multi-baseline fusion (inverse-variance, outlier-gated)                vision/multibaseline.py
     ->  pooled floor plane (RANSAC per frame, outliers rejected, median)       vision/floor_plane.py
     ->  measurements at annotated features, each with an uncertainty           vision/{width,height,protrusion}.py
     ->  fuse repeated observations -> one estimate per feature and method      pipeline.py
     ->  verdict vs ADA rule, error vs tape                                     ada/, eval/minimal_scorer.py
```

## What was verified

| Check | Result |
|---|---|
| Frame quality | 263 keyframes from 5 bags; 225 usable (9 rejected for blur, 29 for people) |
| Rectification | 49 frames across all bags: median vertical error 0.00 px for every pair, mean within 0.1 px, so no systematic offset |
| Floor | 55 of 60 frames agree; camera height 1.270 m, spread 1.5 cm and 0.46 deg tilt (rigid mount). 5 frames locked onto another surface and are rejected |
| Depth precision (floor scatter on pixels all pairs share) | fused 1.6 / 1.8 / 2.7 cm at 4 / 5 / 7 m, against 2.0 / 2.4 / 3.4 cm for the narrowest pair: about 20 percent better. The textbook gain from a 4x wider baseline is not reached, because a floor of about 1.5 cm remains (matching noise, carpet texture) |
| Tests | 86 unit tests on synthetic scenes and fake models with exactly known answers, all passing |

## Results

Nine of the 23 ground-truth features are measured. The other 14 were not annotated: several never appear on the route (water fountain, microwave, sink counter), some are hidden or not frontal (bathroom door, signs), and the kitchen items would need more annotation time.

| Feature | Method | Obs | Measured | Tape | Error | Verdict (measured / tape) |
|---|---|---|---|---|---|---|
| door_hardware_wooden | plane offset | 2 | 38.9 | 39.5 | -0.6 | pass / pass |
| plug_point_height | plane offset | 1 | 20.3 | 19.3 | +1.0 | no rule |
| thermostat_height | plane offset | 1 | 53.3 | 51.25 | +2.1 | fail / fail |
| handrail_height | plane offset | 2 | 33.8 | 35.5 | -1.7 | borderline / pass |
| card_reader_height | plane offset | 1 | 44.2 | 47.0 | -2.8 | pass / pass |
| door_hardware_entrance | plane offset | 1 | 37.6 | 41.5 | -3.9 | pass / pass |
| door_interior_width | homography | 1 | 33.2 | 35.6 | -2.4 | pass / pass |
| door_entrance_width | homography | 1 | 32.0 | 35.0 | -3.0 | borderline / pass |
| door_interior_width | stereo depth | 3 | 32.6 | 35.6 | -3.1 | borderline / pass |
| door_entrance_width | stereo depth | 1 | 40.0 | 35.0 | +5.0 | pass / pass |
| hallway_width_bathroom | stereo planes | 1 | 72.4 | 70.0 | +2.4 | pass / pass |
| hallway_width_bathroom | floor homography | 1 | 75.9 | 70.0 | +5.9 | pass / pass |

By method, mean absolute error: heights 2.0 in (max 3.9), the three stereo-depth widths 3.5 in (max 5.0), the three homography widths 3.7 in (max 5.9).

**Of 11 estimates that have an ADA rule, 8 gave the same verdict as the tape value and 3 were BORDERLINE. None was confidently wrong** (a PASS where the tape says FAIL, or the reverse). BORDERLINE never counts as correct, so the decision score is 8 of 11.

## What worked and what did not

- **Heights are the strongest result.** Errors of 0.6 to 3.9 in, mostly under 2 in. The thermostat is correctly a FAIL (above the 48 in reach limit), and the reach-band overlay shows it sitting visibly above the tinted band.
- **Quality gates mattered more than the matcher.** Without them, one depth patch that matched background through a glass sidelight (12 m instead of 1.5 m) was averaged with a good observation and moved a result by 7 in; a close oblique door gave a 69 in "width". Each failure now declines to report instead of reporting garbage: depth patches are rejected if empty, scattered or if the camera pairs disagree; plane intersections are rejected at grazing incidence; widths outside 20 to 60 in are rejected.
- **Stereo reads of a door edge are unreliable.** Depth near a vertical depth edge is wrong over roughly 30 px, so the direct stereo width is the least stable (errors of +5.0 and -3.1 in, and the three interior-door observations scatter by 6 in). The plane-based homography method is better conditioned when it is allowed to run, but it declines for strongly oblique doors, which is why only one interior-door observation survives.
- **Floor homography for hallway width overestimates by 5.9 in**, more than the wall-plane method (+2.4 in). The hallway curves, so a straight-line fit of each wall boundary is an approximation.
- **Door detection (Harris corners + long vertical edges, metric size from depth) recovers 1 of 2 annotated doors.** The oblique wooden door is found (rank 5 of 6); the entrance door's dark leaf has weak edges and the detector locks onto the outer frame instead. Two doors is too few to claim a detection rate.

## Limitations to state in the report

1. **Feature locations are hand-annotated**, not detected. The Tier 1 pipeline measures a feature once its pixels are given; automatic detection exists only for doors and only partly works. The Tier 2 open-vocabulary detector is the intended fix for objects.
2. **Ground-truth conventions are ambiguous.** Tape values do not say whether a height was to the centre, top or bottom of a device. The thermostat reads +2.1 in at its centre but would match its bottom edge; the entrance pull bar reads 3.9 in below the tape and may not be the handle that was measured. Treat errors of a few inches on heights as partly annotation basis, not only measurement error.
3. **Door width is the leaf edge, tape is the full opening**, so a small negative bias (-2 to -3 in) is expected for the homography widths.
4. **Single-frame results are fragile.** Most features have one observation. The handrail's two observations disagree by about 2.4 in.
5. **The stereo baseline gain is about 20 percent**, not the theoretical factor.
6. **Doors seen at a steep angle and close up cannot be measured** by plane rectification; this is a geometric limit of the method, not a bug.

## Tier 2 results

All Tier 2 models load from plain-data formats (ONNX, `safetensors`); no pickled third-party code is executed.

### A11. Learned stereo (Fast-FoundationStereo, ONNX)

On the same floor pixels, 16 frames, pairs 0-2 and 0-4 (`scripts/evaluate_learned_stereo.py`, `outputs/learned_vs_sgbm_floor.png`):

| Pair | Floor coverage SGBM / learned | Scatter at 3.5-4.5 / 4.5-6 / 6-8 m, SGBM (cm) | Learned (cm) |
|---|---|---|---|
| 0-2 (0.33 m) | 57% / 98% | 1.87 / 1.97 / 2.65 | 1.75 / 1.67 / 2.33 |
| 0-4 (0.66 m) | 43% / 98% | 1.78 / 1.83 / 2.25 | 1.49 / 1.63 / 1.94 |

The learned model fills almost the whole floor and is 8 to 17 percent more precise where both give an answer. But at the **measurement** level it is not simply better (`outputs/level2_results{,_learned,_hybrid}.csv`):

| Matcher | Estimates | Mean abs error | Height error | Verdicts correct | Confidently wrong |
|---|---|---|---|---|---|
| SGBM (Tier 1) | 12 | 3.0 in | 2.0 in | 7 of 11 | none |
| Learned | 11 | 6.0 in | 7.1 in | 6 of 10 | handrail (reads 7 in, tape 35.5), interior-door width |
| **Hybrid** (learned, cross-checked against SGBM) | 11 | 3.7 in | **1.7 in** | 6 of 10 | interior-door stereo width |

A dense network fills in thin or transparent structures (a rail in front of glass) with a plausible but wrong depth, and because it answers almost everywhere the coverage and scatter gates cannot catch it. The hybrid keeps the learned value only where it agrees with block matching within its error bars (`vision/hybrid_depth.py`), which fixes the handrail and gives the best heights of the three. It does not fix door widths, where learned depth changes the leaf-plane fit for oblique doors. **Recommendation: hybrid for heights, SGBM for widths.**

### A12. Open-vocabulary detection (OWLv2 + SAM)

- Finds **10 of 11** hand-annotated height features with no training data; measuring straight from its masks gives mean error 1.20 in against the tape, versus 1.46 in for hand-placed points (6 features with both). It also measures three lever observations that the quality gate had rejected for the hand placements.
- **Automatic survey** (`scripts/survey_open_vocab.py`): 690 measurements from all 225 usable keyframes, no annotation. These are population checks, since many objects share a class:

| Prompt | n | Median (in) | Tape (in) | Result |
|---|---|---|---|---|
| door lever lock | 418 | 40.0 (IQR 39.0-41.2) | 39.5 | +0.5 in, strong |
| electrical outlet (top of plate) | 38 | 20.4 | 19.3 | +1.1 in |
| handrail | 156 | 36.8 (IQR 36.3-78.6) | 35.5 | median fine, but the IQR shows many false detections (stair rails) |
| card reader | 3 | 43.4 | 47.0 | too few |
| wall thermostat | 75 | 78.1 | 51.25 | **not meaningful**: the prompt fires on fire-alarm strobes and other wall objects |

### Automatic height measurement (`run_level2.py --auto`)

The tape notes carry no location, so something must say which object a ground-truth entry refers to. With `--auto`, that is reduced to **one point per feature**: the detector finds the mask under the point, and both the measured pixel and the depth patch come from the mask. Of 11 height annotations, **9 were measured from a detection mask, 2 fell back to the hand-drawn depth box, 0 were unmeasurable**. Accuracy is unchanged (mean height error 2.02 in against 2.00 in for hand-drawn boxes, no confidently wrong verdicts), and the wooden-door lever gets 4 observations instead of 2. Doors and hallways still use hand-drawn geometry.

### A13. Temporal fusion (visual odometry)

Features were tracked through up to 3 keyframes each side (ORB + depth + PnP, then a fresh depth reading at the predicted pixel). The same object varies by only **0.2 to 0.35 in** between frames, so the random error is already small and the remaining error is systematic (the thermostat is +2 in in every frame). Fusing frames therefore does not reduce the mean error (1.69 in single, 1.81 in fused over 7 features). A handrail track drifted (9 in scatter), because a point on a thin rail slides along it.

### A14. Monocular metric depth (Depth Anything V2 indoor)

On 24 frames the stereo/monocular depth ratio is stable (median 1.43, IQR 1.37-1.48) but not 1: the monocular floor puts the camera at 0.87 m, stereo at 1.27 m. **The monocular model therefore does not corroborate the stereo scale on its own.** The stronger evidence for the stereo scale is the tape: heights, which depend on it, agree within about 1 in over 418 lever measurements. A plausible cause is that this camera's narrow field of view (44 degrees) looks "zoomed in" to a model trained on wider views; this is not verified.

### Tier 2 caveats

- The learned model runs from ONNX at about 14 s per camera pair per frame (the PyTorch release is about 300 times faster, but it needs loading pickled weights, so it was not used). Disparities are cached, so reruns are free.
- The stereo door-width uncertainty includes a 10 percent edge-bias term (`stereo_edge_bias_frac`) that I calibrated on the same two annotated doors it is evaluated on. It is a rough error model, not a validated one.
- The learned-vs-classical comparison uses 16 frames and two pairs.

## Reproduce

| Command | Produces |
|---|---|
| `python scripts/extract_frames.py` | the keyframe cache in `data/cache` |
| `python scripts/check_calibration.py` | rectification error per pair |
| `python scripts/evaluate_depth.py` | `outputs/depth_precision_by_baseline.png` |
| `python scripts/evaluate_door_detector.py` | door detector hit rate and overlays |
| `python scripts/run_level2.py` | `outputs/level2_results.csv` and `.md` |
| `python scripts/make_overlays.py` | `outputs/overlays/` and `outputs/demo_hallway.mp4` |
| `python scripts/annotate_view.py <bag> <key> <out.jpg>` | an undistorted frame with a pixel grid, for annotating |
| `python scripts/evaluate_learned_stereo.py` | `outputs/learned_vs_sgbm_floor.png` (A11) |
| `python scripts/run_level2.py --stereo learned` or `hybrid` | `outputs/level2_results_learned.csv`, `_hybrid.csv` |
| `python scripts/evaluate_open_vocab.py` | detection hit rate and mask-based measurements (A12) |
| `python scripts/survey_open_vocab.py` | `outputs/survey_open_vocab.csv`, 690 automatic measurements |
| `python scripts/evaluate_temporal_fusion.py` | `outputs/temporal_fusion.csv` (A13) |
| `python scripts/evaluate_mono_depth.py` | monocular scale check (A14) |
| `pytest` | the unit tests |
