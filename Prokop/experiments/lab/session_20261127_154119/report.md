# Lab check session 2026-11-27T15:41:19

- host `rob-crs93`, git `3f39f72` (added lab checking scripts)
- calibration: `/home/dvorapro/Desktop/semestralka/rob_seme/Prokop/data/calibration/homography_20261127_154349.json`
- guide: `Prokop/docs/LAB_CHECK_GUIDE.md`; the constants are NOT changed here

## Overview

| # | Check | Result | Folder |
|---|---|---|---|
| 1 | dims: the model's dimensions (caliper) | skipped (by the operator) | - |
| 2 | calibrate: camera calibration (homography) | **NOT PASSED** | `Prokop/experiments/lab/20261127_154135_calibrate` |
| 3 | timing: timing model on the real robot | **NOT PASSED** | `Prokop/experiments/lab/20261127_154421_timing` |
| 4 | turntable: perception on the spinning turntable | **NOT PASSED** | `Prokop/experiments/lab/20261127_154542_turntable` |
| 5 | fov: where the camera sees the base | no verdict | `Prokop/experiments/lab/20261127_154659_fov` |
| 6 | dryrun: solver dry run (perception + planning, no motion) | **NOT PASSED** | `Prokop/runs/20261127_155008` |

## 2. calibrate: camera calibration (homography)

| | Measured | Pass | |
|---|---|---|---|
| points recorded | 18 of 27 | >= 15 | ok |
| held-out RMS [px] | 4.21 | < 3 | **FAIL** |
| held-out RMS in the table plane [mm] | 0.93 | < 2 | ok |
| RANSAC rejected | [] | - |  |

## 3. timing: timing model on the real robot

| | Measured | Pass | |
|---|---|---|---|
| run 1 (no camera): lateness per move [ms] | -54, -57, -58, -59, -61 | abs(median) <= 30 | **FAIL** |
| run 1 (no camera): lateness median / spread [ms] | -58 / 6 | the same for all moves |  |
| run 1 (no camera): move_to_q [ms] | 0-10 | <= 60 | ok |
| run 1 (no camera): get_q period [ms] | 51 | 40-80 | ok |
| run 2 (camera grabbing): lateness per move [ms] | -64, -67, -68, -69, -70 | abs(median) <= 30 | **FAIL** |
| run 2 (camera grabbing): lateness median / spread [ms] | -68 / 6 | the same for all moves |  |
| run 2 (camera grabbing): move_to_q [ms] | 0-0 | <= 60 | ok |
| run 2 (camera grabbing): get_q period [ms] | 51 | 40-80 | ok |
| run 2 (camera grabbing): grab duration [ms] | 185 | note only |  |
| run 2 - run 1: lateness / move_to_q [ms] | -10 / -10 | within ~20 | ok |

## 4. turntable: perception on the spinning turntable

| | Measured | Pass | |
|---|---|---|---|
| detections | 10 of 10 | >= 8 | ok |
| period [s] | 28.70 | 27.9-28.1 | **FAIL** |
| xy residual [mm] | 0.17 | < 2 | ok |
| yaw residual [deg] | 0.50 | < 1 | ok |
| tracker - fit drift [deg/s] | 0.2612 | < 0.03 | **FAIL** |
| markers lost | 0 | any |  |
| axis [mm] / offset [mm] / direction | (527.9, -90.0) / 13.2 / -1 | matches by eye |  |

## 5. fov: where the camera sees the base

| | Measured | Pass | |
|---|---|---|---|
| images with markers | 100 of 144 (25 with both) | - |  |
| base seen at x [mm] | 376 to 640 | work area 337-637 |  |
| base seen at y [mm] | -221 to 203 | work area -242..+242 |  |

Markers lost at: along the edges, the work area is a bit smaller I guess

## 6. dryrun

```
  File "/home/dvorapro/Desktop/semestralka/rob_seme/.venv/lib/python3.12/site-packages/ctu_crs/crs_robot.py", line 130, in initialize
    assert self._mars.check_ready()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dvorapro/Desktop/semestralka/rob_seme/.venv/lib/python3.12/site-packages/ctu_mars_control_unit/mars_control_unit.py", line 166, in check_ready
    raise Exception("Check ready: %s." % s[:-2])
Exception: Check ready: arm power is off.
```

## Deviations for Prokop

Constants by check (guide): dims -> `solver/wire.py` (`BAR_HALF`, `TOP_Z`, `OPEN_END_X`, `ARC_RADIUS`, `JOG_*`, `RING_*`), marker height -> Vojtěch's `MARKER_HEIGHT_ABOVE_TABLE_M`; timing offset -> `COMMAND_DELAY` in `solver/timing.py`; period -> `SPIN_PERIOD` in `solver/contracts.py`; residuals and drift -> `solver/uncertainty.py`.

- step 2 (calibrate): held-out RMS [px]: 4.21 (pass: < 3)
- step 3 (timing): run 1 (no camera): lateness per move [ms]: -54, -57, -58, -59, -61 (pass: abs(median) <= 30)
- step 3 (timing): run 2 (camera grabbing): lateness per move [ms]: -64, -67, -68, -69, -70 (pass: abs(median) <= 30)
- step 4 (turntable): period [s]: 28.70 (pass: 27.9-28.1)
- step 4 (turntable): tracker - fit drift [deg/s]: 0.2612 (pass: < 0.03)
