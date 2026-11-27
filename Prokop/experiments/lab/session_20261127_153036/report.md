# Lab check session 2026-11-27T15:30:36

- host `rob-crs93`, git `3f39f72` (added lab checking scripts)
- calibration: `none yet`
- guide: `Prokop/docs/LAB_CHECK_GUIDE.md`; the constants are NOT changed here

## Overview

| # | Check | Result | Folder |
|---|---|---|---|
| 1 | dims: the model's dimensions (caliper) | **NOT PASSED** | `Prokop/experiments/lab/20261127_153116_dims` |
| 2 | calibrate: camera calibration (homography) | **NOT PASSED** | `Prokop/experiments/lab/20261127_153557_calibrate` |
| 3 | timing: timing model on the real robot | not run | - |
| 4 | turntable: perception on the spinning turntable | not run | - |
| 5 | fov: where the camera sees the base | not run | - |
| 6 | dryrun: solver dry run (perception + planning, no motion) | not run | - |

## 1. dims: the model's dimensions (caliper)

| | Measured | Pass | |
|---|---|---|---|
| puzzle bar: side of the square cross-section | 10.0 (+0.0) | 10.0 +-1 | ok |
| puzzle: top bar centreline above the pad (base top) | 157.0 (+2.6) | 154.4 +-1 | **FAIL** |
| puzzle: open end from the stem's axis (horizontal) | - | 85.0 +-1 |  |
| puzzle: bend radius (centreline) | - | 50.0 +-1 |  |
| puzzle: jog top above the pad | - | 44.4 +-1 |  |
| puzzle: jog bottom above the pad | - | 29.4 +-1 |  |
| hoop: hole diameter | - | 60.0 +-1 |  |
| hoop: ring outer diameter | - | 69.6 +-1 |  |
| hoop: ring thickness | - | 6.6 +-1 |  |
| base: marker plane above the table | - | 64.6 +-1 |  |
| ring centre (mid-thickness) above the table at q_home | - | 977.4 +-1 |  |
| ring centre in front of the robot axis at q_home | - | 323.0 +-1 |  |

## 2. calibrate

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

- step 1 (dims): puzzle: top bar centreline above the pad (base top): 157.0 (+2.6) (pass: 154.4 +-1)
