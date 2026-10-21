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
- Queued moves BLEND: in `j1_queued_15_15` joint 1 passes 15 deg at full speed;
  the sim still stops in between (to be changed).
- `get_q` round trip 51 ms (41-154): `query_delay` ~ 0.05 s; motion starts
  93-147 ms after `move_to_q`: `command_delay` ~ 0.12 s. A loop of `get_q` +
  `in_motion` runs at ~18 Hz.
- At the end the arm overshoots ~0.3 deg and settles within ~0.25 s.

To run now, how the blending works beyond moving on straight:

    uv run Prokop/tools/log_motion.py --real --tests j1_j4_corner j1_reverse j1_vias_10x3 j1_j4_arc_vias

| Test | Question |
| --- | --- |
| `j1_j4_corner` | j1 to 20 deg, then j4 to 40 deg, queued: is the corner cut (j4 starts before j1 stops) or taken with a stop? |
| `j1_reverse` | j1 to 20 deg, then back to 5 deg, queued: does it stop at 20 deg or turn early? |
| `j1_vias_10x3` | ten queued via points 3 deg apart: one smooth move, or slowing at each? any "coordmv queue full"? |
| `j1_j4_arc_vias` | nine via points on a curve in j1/j4 (like a planned path): is the curve followed, how much is cut? |

## 4. Camera: `uv run Prokop/tools/camera_survey.py [--delay]`

Done (2026-10-21): 1920 x 1200 x 3, `grab_image()` takes 186 ms (the first
0.8 s): ctu_crs starts and stops the camera on every call, so each image is
taken during the call (age < 186 ms) and at most ~5 images/s are possible.
The first delay test failed: 2-3 frames per leg, and in q_home the arm is
hardly in the camera view.

To run now (robot homed, in q_home, work area free):

    uv run Prokop/tools/camera_survey.py --delay

It prints and checks a survey pose (flange pointing down at x 0.35 m, hoop
centre 49 cm above the work area, in the camera view; >= 8 cm from the table and
the cell in the pose, on the way there and during the swing), moves there at low
speed, swings joint 1 by 15 deg and back at the lowest speed (~5 deg/s, ~12
frames per leg), returns to q_home. Check `survey.png`: the hoop must be in it.
Goes into `CRS93Sim(camera_delay=...)`.

Pose of the camera in the robot frame and its intrinsics: from Vojtěch's
calibration.

## 5. Hoop on the adapter

- Decided: the hoop sits triangle-on-triangle (sim: `HOOP_MOUNT` = identity) and
  a slight shift on its magnets is accepted and ignored, unless it turns out to
  be a problem.
- Ring centre position in q_home (ruler from the flange) as a sanity check of
  the model: sim says ring centre at x 0.320, y 0.000, z 0.978 m, i.e. 10 mm in
  front of the flange and 135 mm above it (flange at x 0.310, z 0.843 m).
