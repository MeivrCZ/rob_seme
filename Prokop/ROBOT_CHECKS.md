# What to check on the real robot

Things the simulation assumes or does not know yet. Low speed, hand on the
E-stop, arm up and away from the table for all motion tests.

**Homing is done by hand, never by a script.** The hard home has no collision
checking: started from a bad pose it can wreck the arm (we were warned). The same
holds for any motion that dips into the table or raises the arm into the camera
bar. The scripts below never home; they check that the robot is in q_home and
only move joints 1, 4 and 6 a little from there.

## 1. Cage, camera, turntable (tape measure)

In the robot frame: x forward (direction of the arm in q_home), y to the left,
origin on the robot's vertical axis.

Done (2026-10-21, in `cell.py`): table height (25 mm below the top of the
robot's square foot), work area, octagonal cage (75 cm inner sides, flat panel in
front, 40 mm frame), camera rail (40 x 40 mm, top 131 cm above the table, along
x, on a post standing on the front of the cage), camera on the axis 50 cm from
the outer face of the front wall.

Also done: cage height 108 cm, rail over the whole cage, camera Basler
a2A1920-51gcBAS (1920 x 1200 px, 3.45 um), image top edge towards the robot,
camera ~4 cm closer to the robot than the work area centre (by eye).

Lens: a 28 mm tall cylinder, focal length not printed; the sim estimates
17.0 mm from the image just fitting the sticker rectangle (image on the table:
x 0.295-0.596 m, y +-0.241 m). The real intrinsics come from a camera
calibration (ChArUco in tilted poses): a straight-down homography of the table
cannot separate the focal length from the camera height.

- Done: turntable period at the slowest speed 28.0 s per turn, the same both
  ways (CW 28.00 +- 0.05 s over 5 turns, CCW 28.06 +- 0.02 s over 4).

## 2. Homing

Done (2026-10-21): the hard home ends at q = (163.4, 0.3, -56.0, 176.7, 14.9,
-172.0) deg, where the control unit zeroes the encoders (as predicted from the
ctu_crs parameters; the sim's hard home now ends there). Soft home reaches
q_home = (0, 0, -45, 0, -45, 0) deg. In q_home the hoop points UP: the adapter
is turned by 180 deg on this station (`ADAPTER_TURN` in tools/robodk2urdf.py,
now in models/crs93.urdf).

Still open:
- Which way each joint group moves on its way to the switches (sequence B, then
  A + C, then D + E + F), which starting poses are safe for it, and how long it
  takes. Sim: the hard home pose is 29 cm from the camera rail, the arm's
  highest point 99 cm above the table (rail bottom at 127 cm).

## 3. Motion profile: `uv run Prokop/tools/log_motion.py --real`

Robot homed by hand and standing in q_home; joints 1, 4, 6 only. Logs `get_q` to
`Prokop/motion_logs/real/*.csv`.

Done (2026-10-21, first five tests):
- Speeds match the sim within 0.1 deg/s (speed units confirmed); accelerations
  roughly (the ramps last 1-2 samples at 18 Hz, so only to ~x2).
- Joints start and stop together (coordinated move), as in the sim.
- `get_q` round trip 51 ms (41-154): `query_delay` = 0.05 s; `in_motion` ~10 ms:
  `status_delay` = 0.01 s; motion starts 93-147 ms (fit, mean 126) after
  `move_to_q`: `command_delay` = 0.12 s; `move_to_q` itself blocks ~25 ms (the
  serial write): `send_delay` = 0.025 s. A loop of `get_q` + `in_motion` runs at
  ~18 Hz. These are the sim's defaults.
- At the end the arm overshoots ~0.3 deg and settles within ~0.25 s (not
  simulated).
- Queued moves (second batch, 2026-10-21): blended only when the next move
  continues in exactly the same joint-space direction (`j1_queued_15_15`,
  `j1_vias_10x3`: one smooth move); a corner (`j1_j4_corner`) or a reversal
  (`j1_reverse`) stops first; via points on a bending path (`j1_j4_arc_vias`)
  nearly stop at each via (speed down to a few deg/s). The sim follows this
  rule (stop unless the same direction); the arc takes 2.99 s in the sim, 2.95 s
  on the robot.

## 4. Camera: `uv run Prokop/tools/camera_survey.py [--delay]`

Done (2026-10-21):
- 1920 x 1200 x 3, `grab_image()` takes 186 ms (the first 0.8 s): ctu_crs starts
  and stops the camera on every call, so at most ~5 images/s.
- Image delay (age when `grab_image` returns) 108 ms (`--delay` from the survey
  pose; mismatch 30x lower than with no delay): `camera_delay` = 0.11 s.
- `survey.png`: the hoop, seen face-on, appeared 3.0 cm nearer to the robot
  than modelled, so the camera is at x ~0.475 m (~47 cm from the outer face of
  the front cage wall, not 50); the ring size fits the modelled height to ~3 %.
  `cell.py` uses this; the model now predicts the hoop within ~2 mm.

Pose of the camera in the robot frame and its intrinsics: from Vojtěch's
calibration.

## 5. Hoop on the adapter

- Decided: the hoop sits triangle-on-triangle (sim: `HOOP_MOUNT` = identity) and
  a slight shift on its magnets is accepted and ignored, unless it turns out to
  be a problem.
- Ring centre position in q_home (ruler from the flange) as a sanity check of
  the model: sim says ring centre at x 0.320, y 0.000, z 0.978 m, i.e. 10 mm in
  front of the flange and 135 mm above it (flange at x 0.310, z 0.843 m).
