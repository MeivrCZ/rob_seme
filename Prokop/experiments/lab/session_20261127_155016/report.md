# Lab check session 2026-11-27T15:50:16

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
| 6 | dryrun: solver dry run (perception + planning, no motion) | **NOT PASSED** | `Prokop/runs/20261127_155030` |

## 6. dryrun

```
                     [--speed SPEED] [--tracker {camera,fitted}] [--frames FRAMES] [--interval INTERVAL]
                     [--pre-entry PRE_ENTRY] [--continuity [ALPHA BETA_DEG [CAP_DEG] ...]]
                     [--margin-floor MARGIN_FLOOR] [--rank {closest,margin}] [--speed-cap] [--phases PHASES]
                     [--planner {prokop,natka,copy}] [--stop-after {1,2,3}] [--calibration CALIBRATION]
                     [--port PORT] [--hold] [--real]
run_solver.py: error: the robot must stand at q_home (home it by hand first)
```

## Deviations for Prokop

Constants by check (guide): dims -> `solver/wire.py` (`BAR_HALF`, `TOP_Z`, `OPEN_END_X`, `ARC_RADIUS`, `JOG_*`, `RING_*`), marker height -> Vojtěch's `MARKER_HEIGHT_ABOVE_TABLE_M`; timing offset -> `COMMAND_DELAY` in `solver/timing.py`; period -> `SPIN_PERIOD` in `solver/contracts.py`; residuals and drift -> `solver/uncertainty.py`.

- none so far
