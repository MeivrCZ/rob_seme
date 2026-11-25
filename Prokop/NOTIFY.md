# Notes for the team (2026-10-20, updated 2026-11-18)

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

## Everyone: the solver plan and the contracts to review (2026-11-18)

The plan for our solution of variant D is in `Prokop/docs/SOLVER_PLAN.md`:
the pipeline, the tasks with their lab methods and acceptance tests, the
approved split, and a status table. Please review the shared types in
`Prokop/solver/contracts.py` (checklist in section 4 of the plan; open
decisions C1-C12):

- Vojtěch: `BaseMeasurement` (`yaw` is the base frame's x axis in robot
  coordinates, `t_capture` the time the image was taken) and the
  `PhaseTracker` protocol for the Kalman filter.
- Natka: `TurntableModel`, `PuzzlePath`, `Plan`/`PlanStep` (incl. the timing
  meaning of `t_rel`, C12, and the roll convention).

Already usable from `Prokop/solver/`: the turntable chain (FK of every
checkpoint at any turntable angle), the robot timing model (arrival times of
queued moves), approach paths (RRT), phase corrections and the executor. A new
task P2.8 (feasibility funnel: reach check, IK feasibility map, start-phase
screening before the expensive search) was added to the planning phase.

## Natka: what blocks the planner (P2.2-P2.6), and the hoop side (2026-11-22)

I ran the whole pipeline in the sim (`Prokop/run_solver.py`, see below) and a
sweep of 8 random placements x both spin directions. Perception works, and
Prokop's feasibility funnel (P2.8, since removed: see below) found a path in 12
of the 16. **Your planner found no plan in any of the 16 cases.** Your
files are untouched; to find out what blocks, I used a copy of them in
`Prokop/natka_copy/` (changes marked `# PROKOP COPY:`, currently only one
line). Three things block, in this order:

1. **Hoop side (decided, contract C13).** `hoop_targets.checkpoint_ring_target`
   puts the hoop's +z along the direction of travel, so *side A* leads (the
   flat side of the ring facing the robot in `q_home`; the URDF's z axis
   points out of it). Nothing in the contracts said which side, so this was a
   gap in the spec, not a mistake. Prokop decided: **only side B leads** (the
   side facing away from the robot in `q_home`), because with side A the arm
   ends far too close to the table at the bottom of the path. In all 12
   placements where the funnel (then searching both sides) found a path, it
   used side B at every checkpoint; with side A your planner had no IK
   solution at checkpoint 0. The convention (contracts docstring, C13): ring
   frame = `checkpoint_T @ Rx(pi) @ Rz(roll)` (handle along the checkpoint's
   -y at roll 0). Your `hoop_targets` is now the only implementation, and in
   it that is one line in `checkpoint_ring_target`:

   ```python
   ring[:3, :3] = checkpoint[:3, :3] @ so3_rx(np.pi) @ so3_rx(tilt) @ so3_rz(roll)
   ```

   With it, your IK finds the same solutions as the funnel (checked: same
   flange target to 1e-9).

2. **The traversal starts at `q_home`** (`timed_plan(q_start=None)` falls back
   to `robot.q_home`). Then the first step is a move of ~100 deg from home
   straight into the puzzle: it takes ~12 s, the turntable turns ~150 deg
   meanwhile, and in the next fixed-point iteration checkpoint 0 is out of
   reach. The contracts mean `q_start` to be the pre-entry configuration
   (C12, P2.7): the robot gets there first on the approach line and waits for
   theta_start; checkpoint 0 is already 15 mm outside the bar, so its
   configuration can be `q_start` (one per start phase: an IK solution of
   checkpoint 0 at theta_start).

3. **The selection (P2.5) does not keep continuity.** `_compare_scores`
   compares (worst margin, -transition cost) lexicographically; margins are
   floats and practically never equal, so the joint cost and the branch-switch
   penalty never decide anything. Even with `q_start` = the funnel's
   checkpoint-0 configuration, the selected path starts 220 deg away from it
   and jumps 160 deg (a wrist flip) at checkpoint 7; in another run 174 deg.
   Such a jump takes seconds (so the fixed point diverges) and between the two
   checkpoints the hoop leaves the path. Needed: a hard limit on the joint
   change between checkpoints (the funnel used 25 deg) and
   a combined objective (or margin rounded / thresholded first), so that
   continuity counts. How to weigh margin against continuity is your call.

Smaller things seen on the way:

- `fixed_point_arrivals` returns at the first infeasible iteration; with a
  diverging first guess (all arrivals 0) that ends the search for that phase.
- **Speed, the main cause (profiled 2026-11-24):** `hoop_targets._fixed_mount_frames`
  loads and parses `crs93.urdf` and `hoop.urdf` on every call, twice per
  `flange_target`: 86 % of the planner's time (1540 s of a 1788 s run). The
  frames never change, so `@functools.cache` on it gives identical targets
  (checked: difference 0) about 15x faster (76 -> 5 ms per target).
- Speed: one `select_roll_branch` call over 25 checkpoints takes ~130 s here
  (36 rolls x all IK branches x the margin with finite-difference Jacobians);
  a whole `timed_plan` over 6 start phases took 30-140 s, once 825 s. The
  turntable keeps spinning meanwhile (the model predicts it, but the 2-minute
  attempt limit counts). Precomputing all IK solutions per checkpoint x
  turntable angle x roll once (the funnel did that in ~2 s, vectorized; in
  git history, `Prokop/solver/ik_batch.py`) could replace the per-call IK.
- Without a collision checker (`checker=None`) the margin counts only the
  Jacobian and the joint limits.
- `PlanStep` has no field for the hoop side; with C13 there is only one side,
  so none is needed.

How to see it yourself (from the repo root, viewer on http://localhost:8080):

```
uv run Prokop/run_solver.py --seed 1                    # your planner
uv run Prokop/run_solver.py --seed 1 --planner copy     # the copy (side B)
uv run Prokop/run_solver.py --seed 1 --stop-after 2     # perception + planning only
```

It prints every phase (measurements vs the sim's truth, the plan, the
approach) and stops at the first failing phase;
log, camera frames and `summary.json` in `Prokop/runs/<time>/`.

## Everyone: approach paths are straight lines now (2026-11-22)

P2.7 no longer uses RRT: the work area is right below `q_home`, so
`solver.approach.plan_line(robot, q_from, q_to, checker)` moves the ring centre
on a straight task-space line (position linear, rotation R0 exp(s log(R0^T R1)),
separately), one IK solution per 2 cm / 5 deg on one branch, collision-checked;
it returns the path or the reason it cannot. `RRT` and `plan_approach` are gone.

## Everyone: Prokop's duplicates of Natka's planning code removed (2026-11-24)

We build on Natka's planning code, so Prokop's parallel versions are gone:

- `solver/feasibility.py` (P2.8 funnel) and `solver/ik_batch.py`: the funnel
  did its own ring targets, IK and start-phase walk, i.e. P2.2-P2.6 again.
  Still in git history (commit 8bf7cc9).
- `solver/correction.py`: its `so3_log`, `pose_error`,
  `jacobian_finite_difference` and `ik_numerical` are gone; corrections (P3.4)
  use `plan_ik.ik_numerical`.
- `solver/contracts.py`: `checkpoint_frame`, `example_path()` (the fake path)
  and `ring_target`/`RING_FLIP` are gone. The puzzle path is
  `crs_puzzle_d_hoop_path.puzzle_path_contract()` everywhere (tests: the
  `puzzle_path` fixture), the side-B convention (C13) stays in the docstring
  and is implemented by `hoop_targets`.
- `solver/approach.py` (`plan_line`) now takes its IK from
  `plan_ik.all_solutions` / `select_continuous_branch` and the ring offset from
  `hoop_targets.hoop_ring_pose_from_flange`.

`import solver` (any `solver.*` module) appends `Natka/` to `sys.path`, so
`plan_ik`, `hoop_targets`, ... import by their plain names. This means
**Prokop/solver now depends on these Natka functions; please tell Prokop before
renaming or changing them**: `plan_ik.ik_numerical`, `all_solutions`,
`select_continuous_branch`, `so3_log`, `hoop_targets.hoop_ring_pose_from_flange`,
`crs_puzzle_d_hoop_path.puzzle_path_contract`.

`run_solver.py` hands the planner an even grid of start phases (`--phases`,
default 6) instead of the funnel's candidates. `Prokop/natka_copy/` stays as
the side-B experiment until `hoop_targets` has the flip.

## Everyone: soft collision model for planning (2026-11-25)

Planning code should not use the simulator's exact meshes (`collision.py`):
they are the environment's ground truth. There is now a model for planning:
`solver/soft_collision.py`, documented in `docs/SOFT_COLLISION.md`.

- Every body is a few boxes in separate soft URDFs (`models/soft/*_soft.urdf`:
  robot, hoop, puzzle D, base, turntable, cell); the original URDFs are
  unchanged. The task is posed from the perceived `TurntableModel`.
- `SoftScene(model).clearance(q, theta)` returns the least distance of the
  arm (with the hoop) to the cell, table and task, and between arm parts, as a
  lower bound with the margins subtracted (10 mm; hoop vs puzzle 3 mm), in
  about 1.5 ms (exact meshes: 8.6 ms).
- Validated against the exact meshes: the boxes contain every mesh vertex,
  and in 500 random states the soft distance never exceeded the exact one.
  Near the bar it accepts 94 % of the truly free configurations (67 % with
  the margins).
- The hoop's hole is 30.0 mm in radius in the CAD, not 31.2 mm (62.4 mm
  diameter): about 23 mm of room around the bar.
