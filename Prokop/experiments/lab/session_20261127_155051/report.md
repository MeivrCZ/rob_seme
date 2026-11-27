# Lab check session 2026-11-27T15:50:51

- host `rob-crs93`, git `3f39f72` (added lab checking scripts)
- calibration: `/home/dvorapro/Desktop/semestralka/rob_seme/Prokop/data/calibration/homography_20261127_154349.json`
- guide: `Prokop/docs/LAB_CHECK_GUIDE.md`; the constants are NOT changed here

## Overview

| # | Check | Result | Folder |
|---|---|---|---|
| 1 | dims: the model's dimensions (caliper) | skipped (by the operator) | - |
| 2 | calibrate: camera calibration (homography) | skipped (by the operator) | - |
| 3 | timing: timing model on the real robot | skipped (by the operator) | - |
| 4 | turntable: perception on the spinning turntable | skipped (by the operator) | - |
| 5 | fov: where the camera sees the base | skipped (by the operator) | - |
| 6 | dryrun: solver dry run (perception + planning, no motion) | **pass** | `Prokop/runs/20261127_155059` |

## 6. dryrun: solver dry run (perception + planning, no motion)

| | Measured | Pass | |
|---|---|---|---|
| Phase 1: turntable model | ok, 8.24 s | ok | ok |
| Phase 2: plan | ok, 23.84 s | ok | ok |
| result | all phases ran | all phases ran |  |
| orbit | P1.3: fit_orbit | - |  |
| margins from the uncertainties | P2: margins from the uncertainties: hoop-puzzle 3.7 mm, joint limits 6.0 deg (k = 2: angle sd 0.99 deg incl. speed sd 0.049 deg/s over 12 s, position sd 0.09 mm, reach 101 mm, dimensions sd 0.58 mm) | - |  |
| reachability map | P2: reachability map (34 ms): checkpoints reachable in 75, 83, 100, 100, 100, 100, 100 % of the turntable's angles; start angles from which all are reachable in time: 71 % | - |  |
| predicted end | predicted end 56.0 s after the start (limit 120 s) | - |  |
| predicted end [s] | 56.0 | < 120 minus the margin | ok |

## Deviations for Prokop

Constants by check (guide): dims -> `solver/wire.py` (`BAR_HALF`, `TOP_Z`, `OPEN_END_X`, `ARC_RADIUS`, `JOG_*`, `RING_*`), marker height -> Vojtěch's `MARKER_HEIGHT_ABOVE_TABLE_M`; timing offset -> `COMMAND_DELAY` in `solver/timing.py`; period -> `SPIN_PERIOD` in `solver/contracts.py`; residuals and drift -> `solver/uncertainty.py`.

- none so far
