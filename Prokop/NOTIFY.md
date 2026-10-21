# Notes for the team (2026-10-20)

## Vojtěch: changes in Prokop/ that affect your current code

Your current `rob_crs93_v4` code will break against `Prokop/` as it is now. As you
are rewriting it anyway, here is what changed:

- **Joint-6 view offset removed.** `JOINT6_VIEW_OFFSET_RAD` and `display_q()` are
  gone from `view_models.py`. The URDF was right without it: in `q_home` the
  adapter triangle and the hoop point down, as on the real robot, and the URDF
  flange equals `ctu_crs` `fk(q)` exactly. The offset rotated the drawn hoop
  by 180 deg (pointing up), away from the FK. `tests/test_homography.py` checks
  for the removed code.
- **Hoop up in q_home (2026-10-21).** On the CRS93 station the adapter is
  screwed on turned by 180 deg, so the hoop points up in q_home: that is what
  your joint-6 view offset compensated. It is now in the robot model itself
  (`models/crs93.urdf`, `ADAPTER_TURN` in `tools/robodk2urdf.py`); `fk(q)` is
  still the real flange, so no joint offset is needed anywhere.
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
