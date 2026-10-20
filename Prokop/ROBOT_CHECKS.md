# What to check on the real robot

Things the simulation assumes or does not know yet. Low speed, hand on the
E-stop, arm up and away from the table for all motion tests.

**Homing is done by hand, never by a script.** The hard home has no collision
checking: started from a bad pose it can wreck the arm (we were warned). The same
holds for any motion that dips into the table or raises the arm into the camera
bar. The scripts below never home; they check that the robot is in q_home and
only move joints 1, 4 and 6 a little from there.

## 1. Cage, camera bar, turntable (tape measure)

In the robot frame: x forward (direction of the arm in q_home), y to the left,
origin on the robot's vertical axis, z up from the ctu base frame (the table is
at z = -55.8 mm).

- The cage: the table area inside it where the turntable can stand (x/y limits).
  Sim: `CAGE_XY` = +-0.8 m, a placeholder; random placement draws the turntable
  centre from it.
- The camera bar: position, height and size (an obstacle for the arm, not in the
  model yet).
- Turntable: diameter and height of the top surface; period of one turn at the
  slowest speed in both directions (time ~3 turns; is the speed constant?).
  Sim: `SPIN_PERIOD` = 16 s, a placeholder.

## 2. Homing

- Where the hard home actually ends and which way each joint group moves
  (sequence B, then A + C, then D + E + F). From the ctu_crs parameters, if the
  control unit zeroes the encoder at the home switch, it ends at about
  q = (163.5, 0.3, -56.0, 176.8, 14.9, -172.0) deg - an inference, to be checked
  (`hard_home()` alone, then `get_q()`). Joint 2 goes first and ends nearly
  vertical: with a raised elbow, can that hit the camera bar?
- Which starting poses are safe for it, and how long it takes.
- Soft home is a coordinated move to q_home = (0, 0, -45, 0, -45, 0) deg, the
  same joint values as in the sim; check that the real arm looks like the sim
  there (the sim draws the robot from the ctu_crs kinematics).

The sim's hard home (each joint group to q = 0) is a placeholder until then.

## 3. Motion profile: `uv run Prokop/tools/log_motion.py --real`

Robot homed by hand and standing in q_home. Runs five short tests (joints 1, 4,
6 only) and logs `get_q` to `Prokop/motion_logs/real/*.csv`. Run it without
`--real` too and compare with the `sim/` logs. What each test answers:

| Test | Question | Sim assumes |
| --- | --- | --- |
| `j1_30_speed025`, `_050` | peak joint speed, ramp duration (speed up / slow down) | speed IRC*256/ms; ramps from the acceleration setting (units are a guess) |
| `j6_90_speed050` | the same for a wrist joint (different defaults) | as above |
| `j1_queued_15_15` | two queued moves: does the joint stop at 15 deg or pass through smoothly? | stops in between |
| `j1_30_j4_60_coord` | do both joints start and end together? | yes, same profile |

It also prints the delays to put into `CRS93Sim(command_delay=..., query_delay=...)`:
- `get_q` = round trip of a state query = `query_delay`,
- start latency = `command_delay + query_delay / 2`, +- half a sampling period
  (a bit more due to the speed-up ramp); repeat for an average.

Also note whether `move_to_q` ever prints "coordmv queue full - waiting".

## 4. Camera: `uv run Prokop/tools/camera_survey.py [--delay]`

- Without `--delay`: resolution and time per `grab_image()`, no motion.
- `--delay` (robot in q_home, arm visible in the camera): joint 1 moves +10 deg
  and back at low speed while frames are grabbed; the image delay is the shift at
  which the image change matches the joint motion on the way out and back.
  Goes into `CRS93Sim(camera_delay=...)`.
- Results in `Prokop/camera_survey/<date-time>/`: zip and send.
- Pose of the camera in the robot frame and its intrinsics: from Vojtěch's
  calibration.

## 5. Hoop on the adapter

- Decided: the hoop sits triangle-on-triangle (sim: `HOOP_MOUNT` = identity) and
  a slight shift on its magnets is accepted and ignored, unless it turns out to
  be a problem.
- Ring centre position in q_home (ruler from the flange) as a sanity check of
  the model: sim says ring centre at x 0.320, y 0.000, z 0.709 m
  (flange at x 0.310, z 0.843 m).
