# Working on the real robot

How to run the measurement tools at the CRS93 station, and what is still to be
checked there. The finished experiments are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Safety rules

- **Home by hand, never from a script.** The hard home drives every joint to its
  home switch without any collision checking. Started from a bad pose, it can
  wreck the arm or hit the camera rail (we were warned). No tool in `Prokop/`
  homes the robot; they check that it stands in `q_home` and refuse otherwise.
- **Low speed and a hand on the E-stop** for anything new.
- **Keep the work area free** when a tool moves the arm over it
  (`camera_survey.py --delay`).
- **Check new poses in the sim first.** `collision.ArmChecker` tests the cage,
  camera rail, camera and table, but not self-collision and not the turntable
  with the puzzle.

## Running the tools at the station

```bash
git pull && uv sync
uv run python                                   # homing, by hand:
>>> from ctu_crs import CRS93; import numpy as np
>>> r = CRS93(); r.initialize(home=False)
>>> r.hard_home(); r.soft_home(); print(np.rad2deg(r.get_q()).round(1))   # 0 0 -45 0 -45 0
>>> r.close()

uv run Prokop/tools/log_motion.py --real                # motion profile, delays, blending (E5, E6)
uv run Prokop/tools/log_motion.py --real --tests NAME   # only some tests
uv run Prokop/tools/camera_survey.py                    # camera timing, no motion (E7)
uv run Prokop/tools/camera_survey.py --delay            # image delay from the survey pose (E8)
```

The results land in `Prokop/experiments/`. Commit and push them, then analyse
them on any computer with `tools/analyze_motion.py` and
`tools/analyze_survey.py`.

## Still open

| Check | Why | How |
|---|---|---|
| Camera intrinsics and lens distortion | the focal length (17.0 mm) is an estimate (E9) | Vojtěch's camera calibration; then `cell.CAMERA_K` |
| Hard home paths | the sim moves straight to the hard home pose; the real joints first drive to their switches | watch which way each joint group moves (B, then A + C, then D + E + F), which starting poses are safe, how long it takes |
| Hoop ring centre in `q_home` | sanity check of the adapter turn and the hoop model | ruler from the flange: the model has the ring centre 10 mm in front of the flange and 135 mm above it (x 0.320, z 0.978 m) |
| Camera body depth | 42 mm is the ace 2 family value, not confirmed | ruler; `cell.CAMERA_SIZE` |
| Queue limit of the control unit | never reached; `move_to_q` would block ("coordmv queue full - waiting") | only matters for long via chains; watch for the message |
| Accelerations | known only to ~×2 (E5) | a better test would need faster sampling than `get_q` allows (51 ms) |
