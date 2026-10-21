# Notes for the team (2026-10-20, updated 2026-10-21)

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
