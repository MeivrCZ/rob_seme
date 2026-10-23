# Notes for the team (2026-10-20, updated 2026-10-23)

## Vojtěch: changes in Prokop/ that affect your current code

Your current `rob_crs93_v4` code will break against `Prokop/` as it is now. As you
are rewriting it anyway, here is what changed:

- **Joint-6 view offset removed, adapter turned in the model instead.** On the
  CRS93 station the adapter is screwed on turned by 180 deg, so the hoop points
  up in q_home: that is what your `JOINT6_VIEW_OFFSET_RAD` compensated. The
  offset (and `display_q()`) are gone from `view_models.py`, because they made
  the drawn flange disagree with `fk(q)`; the turn is now in the robot model
  itself (`models/crs93.urdf`, `ADAPTER_TURN` in `tools/robodk2urdf.py`), so no
  joint offset is needed anywhere. `tests/test_homography.py` checks for the
  removed code.
- **Files moved.** URDFs and meshes are now in `Prokop/models/`, generators in
  `Prokop/tools/`. Code that loads `Prokop/*.urdf` (e.g. `app/robot.py`,
  `tests/test_table_guard_and_scan.py`) needs the new path; `placement.MODELS`
  points to it.
- **`view_models.py` refactored.** The scene is now the `SceneViewer` class;
  `main()` keeps the same command line (`--port`), so running it as a script
  still works.
- **CRS97 -> CRS93.** We use the CRS93 station, so everything in `Prokop/` is
  built on `ctu_crs.CRS93` now (same kinematics as CRS97; joint limits differ by
  up to 3 deg, speeds/accelerations differ): `models/crs97.urdf` is now
  `models/crs93.urdf` (meshes in `models/meshes/crs93/`), the simulated robot
  is `CRS93Sim`.
- **Simulated robot.** `Prokop/simulation.py` has `CRS93Sim`, a drop-in for
  `ctu_crs.CRS93` shown in the Viser scene. Scripts written against the
  `CRS93` interface can be tested in it first and run on the real robot by
  swapping one line. See `Prokop/README.md`.
- **Measurements and experiments.** Everything measured at the station (cell
  geometry, camera pose, delays, motion behaviour) is in
  `Prokop/docs/MEASUREMENTS.md` and `Prokop/docs/EXPERIMENTS.md`; the cell
  geometry and the camera model (pose, estimated intrinsics) are importable from
  `Prokop/cell.py`.

## Vojtěch: simulated camera (2026-10-23)

`CRS93Sim.grab_image()` now returns a rendered camera image: 1920 x 1200 BGR
uint8 like the real one, with the real timing (blocks ~0.19 s, the image is
~0.11 s old). The base markers are the real ones (`DICT_4X4_50`, ids 1 and 2),
so your ArUco detection can run on it unchanged. Flat colours, no noise: good
for testing the pipeline's logic, not for tuning detection. Details in
`Prokop/README.md`.

## Everyone: hard collisions in the simulation (2026-10-23)

`CRS93Sim` now stops on a collision of the exact meshes (arm with itself, the
cell, the table, the turntable, base and puzzle; the hoop touching the puzzle
counts). The robot and turntable stop, the bodies turn red, and the next robot
call raises `simulation.CollisionError`; catch it if your script should go on.
Motion commands keep raising until someone presses **Release** in the viewer
("Collision" panel), which puts the robot back where the colliding move
started. On the real robot there is no such check: a collision there is a crash.

## Natka: how queued moves blend (2026-10-23)

The robot (and `CRS93Sim`, E6 in `Prokop/docs/EXPERIMENTS.md`) blends a queued
move into the previous one only if it continues in exactly the same
joint-space direction; at any corner or reversal it (nearly) stops first.

- What counts is joint space: a path curved in Cartesian space runs without
  stopping if it is a straight line in joint space (`q0 + s * d`). Via points on
  such a line add nothing; send the end point as one move.
- Chain moves from the previous target, not from `get_q()`, which returns the
  configuration mid-move: `move_to_q(q + d1)` then `move_to_q(q + d1 + d2)`.
  With `get_q() + d2` the second move bends or reverses relative to the first
  and starts from standstill.
- The second command has to arrive before the first move starts braking.
