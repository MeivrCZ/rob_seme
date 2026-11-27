# Lab check session 2026-11-27T15:40:52

- host `rob-crs93`, git `3f39f72` (added lab checking scripts)
- calibration: `none yet`
- guide: `Prokop/docs/LAB_CHECK_GUIDE.md`; the constants are NOT changed here

## Overview

| # | Check | Result | Folder |
|---|---|---|---|
| 1 | dims: the model's dimensions (caliper) | skipped (by the operator) | - |
| 2 | calibrate: camera calibration (homography) | **NOT PASSED** | `Prokop/experiments/lab/20261127_154054_calibrate` |
| 3 | timing: timing model on the real robot | not run | - |
| 4 | turntable: perception on the spinning turntable | not run | - |
| 5 | fov: where the camera sees the base | not run | - |
| 6 | dryrun: solver dry run (perception + planning, no motion) | not run | - |

## 2. calibrate

```
the robot is 176.7 deg from q_home: home it by hand first (this script never homes)
```

## Deviations for Prokop

Constants by check (guide): dims -> `solver/wire.py` (`BAR_HALF`, `TOP_Z`, `OPEN_END_X`, `ARC_RADIUS`, `JOG_*`, `RING_*`), marker height -> Vojtěch's `MARKER_HEIGHT_ABOVE_TABLE_M`; timing offset -> `COMMAND_DELAY` in `solver/timing.py`; period -> `SPIN_PERIOD` in `solver/contracts.py`; residuals and drift -> `solver/uncertainty.py`.

- none so far
