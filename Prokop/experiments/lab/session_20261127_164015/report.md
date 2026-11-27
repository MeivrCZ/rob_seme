# Lab check session 2026-11-27T16:40:15

- host `rob-crs93`, git `3f39f72` (added lab checking scripts)
- calibration: `/home/dvorapro/Desktop/semestralka/rob_seme/Prokop/data/calibration/homography_20261127_165215.json`
- guide: `Prokop/docs/LAB_CHECK_GUIDE.md`; the constants are NOT changed here

## Overview

| # | Check | Result | Folder |
|---|---|---|---|
| 1 | dims: the model's dimensions (caliper) | skipped (by the operator) | `Prokop/experiments/lab/20261127_164019_dims` |
| 2 | calibrate: camera calibration (homography) | skipped (by the operator) | `Prokop/experiments/lab/20261127_165009_calibrate` |
| 3 | timing: timing model on the real robot | skipped (by the operator) | `Prokop/experiments/lab/20261127_165230_timing` |
| 4 | turntable: perception on the spinning turntable | skipped (by the operator) | `Prokop/experiments/lab/20261127_165339_turntable` |
| 5 | fov: where the camera sees the base | no verdict | `Prokop/experiments/lab/20261127_165640_fov` |
| 6 | dryrun: solver dry run (perception + planning, no motion) | **pass** | `Prokop/runs/20261127_170400` |

## 1. dims: the model's dimensions (caliper)

| | Measured | Pass | |
|---|---|---|---|
| puzzle bar: side of the square cross-section | - | 10.0 +-1 |  |
| puzzle: top bar centreline above the pad (base top) | 155.0 (+0.6) | 154.4 +-1 | ok |
| puzzle: open end from the stem's axis (horizontal) | - | 85.0 +-1 |  |
| puzzle: bend radius (centreline) | - | 50.0 +-1 |  |
| puzzle: jog top above the pad | 44.8 (+0.4) | 44.4 +-1 | ok |
| puzzle: jog bottom above the pad | 30.0 (+0.6) | 29.4 +-1 | ok |
| hoop: hole diameter | - | 60.0 +-1 |  |
| hoop: ring outer diameter | 69.0 (-0.6) | 69.6 +-1 | ok |
| hoop: ring thickness | - | 6.6 +-1 |  |
| base: marker plane above the table | 67.0 (+2.4) | 64.6 +-1 | **FAIL** |
| ring centre (mid-thickness) above the table at q_home | - | 977.4 +-1 |  |
| ring centre in front of the robot axis at q_home | - | 323.0 +-1 |  |

## 2. calibrate: camera calibration (homography)

| | Measured | Pass | |
|---|---|---|---|
| points recorded | 22 of 27 | >= 15 | ok |
| held-out RMS [px] | 5.28 | < 3 | **FAIL** |
| held-out RMS in the table plane [mm] | 1.24 | < 2 | ok |
| RANSAC rejected | [] | - |  |

## 3. timing: timing model on the real robot

| | Measured | Pass | |
|---|---|---|---|
| run 1 (no camera): lateness per move [ms] | -53, -56, -57, -58, -59 | abs(median) <= 30 | **FAIL** |
| run 1 (no camera): lateness median / spread [ms] | -57 / 6 | the same for all moves |  |
| run 1 (no camera): move_to_q [ms] | 0-10 | <= 60 | ok |
| run 1 (no camera): get_q period [ms] | 51 | 40-80 | ok |
| run 2 (camera grabbing): lateness per move [ms] | -64, -67, -67, -69, -70 | abs(median) <= 30 | **FAIL** |
| run 2 (camera grabbing): lateness median / spread [ms] | -67 / 6 | the same for all moves |  |
| run 2 (camera grabbing): move_to_q [ms] | 0-0 | <= 60 | ok |
| run 2 (camera grabbing): get_q period [ms] | 51 | 40-80 | ok |
| run 2 (camera grabbing): grab duration [ms] | 184 | note only |  |
| run 2 - run 1: lateness / move_to_q [ms] | -10 / -10 | within ~20 | ok |

## 4. turntable: perception on the spinning turntable

| | Measured | Pass | |
|---|---|---|---|
| detections | 10 of 10 | >= 8 | ok |
| period [s] | 27.89 | 27.9-28.1 | **FAIL** |
| xy residual [mm] | 0.22 | < 2 | ok |
| yaw residual [deg] | 0.19 | < 1 | ok |
| tracker - fit drift [deg/s] | 0.0641 | < 0.03 | **FAIL** |
| markers lost | 0 | any |  |
| axis [mm] / offset [mm] / direction | (505.2, -99.3) / 36.7 / -1 | matches by eye |  |

## 5. fov: where the camera sees the base

| | Measured | Pass | |
|---|---|---|---|
| images with markers | 245 of 388 (4 with both) | - |  |
| base seen at x [mm] | 335 to 662 | work area 337-637 |  |
| base seen at y [mm] | -249 to 254 | work area -242..+242 |  |

Markers lost at: on the edge dubass

## 6. dryrun: solver dry run (perception + planning, no motion)

| | Measured | Pass | |
|---|---|---|---|
| Phase 1: turntable model | ok, 8.2 s | ok | ok |
| Phase 2: plan | ok, 20.98 s | ok | ok |
| result | all phases ran | all phases ran |  |
| orbit | P1.3: fit_orbit | - |  |
| margins from the uncertainties | P2: margins from the uncertainties: hoop-puzzle 4.6 mm, joint limits 6.0 deg (k = 2: angle sd 0.99 deg incl. speed sd 0.049 deg/s over 12 s, position sd 0.36 mm, reach 128 mm, dimensions sd 0.58 mm) | - |  |
| reachability map | P2: reachability map (36 ms): checkpoints reachable in 78, 83, 100, 100, 100, 100, 100 % of the turntable's angles; start angles from which all are reachable in time: 73 % | - |  |
| predicted end | predicted end 48.6 s after the start (limit 120 s) | - |  |
| predicted end [s] | 48.6 | < 120 minus the margin | ok |

## Deviations for Prokop

Constants by check (guide): dims -> `solver/wire.py` (`BAR_HALF`, `TOP_Z`, `OPEN_END_X`, `ARC_RADIUS`, `JOG_*`, `RING_*`), marker height -> Vojtěch's `MARKER_HEIGHT_ABOVE_TABLE_M`; timing offset -> `COMMAND_DELAY` in `solver/timing.py`; period -> `SPIN_PERIOD` in `solver/contracts.py`; residuals and drift -> `solver/uncertainty.py`.

- step 1 (dims): base: marker plane above the table: 67.0 (+2.4) (pass: 64.6 +-1)
- step 2 (calibrate): held-out RMS [px]: 5.28 (pass: < 3)
- step 3 (timing): run 1 (no camera): lateness per move [ms]: -53, -56, -57, -58, -59 (pass: abs(median) <= 30)
- step 3 (timing): run 2 (camera grabbing): lateness per move [ms]: -64, -67, -67, -69, -70 (pass: abs(median) <= 30)
- step 4 (turntable): period [s]: 27.89 (pass: 27.9-28.1)
- step 4 (turntable): tracker - fit drift [deg/s]: 0.0641 (pass: < 0.03)
