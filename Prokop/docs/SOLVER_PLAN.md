# Solver plan: puzzle D on the spinning turntable

Plan for the team's solution of variant D (hoop through puzzle D on the
spinning turntable, both spin directions), split into tasks that the three of
us can work on in parallel, each with an AI assistant. **Status: agreed
plan (2026-11-18); contracts drafted, under review.**

Read first: `Prokop/README.md` (simulation, conventions), `Prokop/docs/MEASUREMENTS.md`
(every measured value), `Prokop/docs/EXPERIMENTS.md` (how the robot really behaves).

## 1. Goal and success criteria (assignment)

- The robot passes the hoop over puzzle D while the turntable spins at its
  slowest speed (28.0 s/rev); at the end the hoop is less than 4 cm above the
  base with the ArUco markers.
- At most 2 minutes per attempt, at most 3 failed attempts (touching the puzzle).
- Disqualification: any collision of the robot with the table, the turntable or
  itself.
- An unreachable placement must be **detected and reported** instead of moving.
- Bonus: both spin directions (the examiner chooses).
- Report: experiments, including a map of where in the workspace the solution
  works; each member's technical contribution; use of AI.

## 2. Principles

1. **The pipeline uses only the `ctu_crs.CRS93` interface** (`move_to_q`,
   `get_q`, `in_motion`, `wait_for_motion_stop`, `grab_image`, `fk`, `ik`,
   `in_limits`, speed settings), plus the camera calibration. Running on the real
   robot means `robot = CRS93()` instead of `robot = CRS93Sim()`, nothing else.
   The simulator's internals (`robot.viewer`, `_now`, `_collider`, the true
   placement) may only be used by the evaluation harness (task I2), never by the
   pipeline.
2. **Everyone works in their own directory.** The shared data types live in one
   contracts module (`Prokop/solver/contracts.py`, task C0); changing it needs
   agreement of all three.
3. **Time is explicit.** All times are `time.monotonic()` seconds on the robot
   computer. A camera measurement carries the time the image was *taken*
   (`grab_image` return time − camera delay, 0.11 s).
4. **Every task is testable alone** against fakes / synthetic data, before
   integration.
5. **Units and frames**: SI units; the `ctu_crs` robot base frame (x forward, y
   left, z up, table at z ≈ +1 mm); joint values exactly as in `move_to_q`.

## 3. Pipeline

```
Phase 1  Puzzle on the turntable        Phase 2  Plan                         Phase 3  Execution
──────────────────────────────          ──────────────────────────            ─────────────────────────────
camera image ─► P1.2 measurement ─┐     P2.1 path in puzzle frame ──┐          P3.1 phase tracker (Kalman)
                                  ├► P1.3 fit ─► TurntableModel ─────► P2.6 ─► Plan ─► P3.3 executor ─► robot
P1.1 turntable kinematic chain ───┘     P2.2–P2.5 targets, IK,      │          ▲   P3.4 corrections
     (FK(θ) → puzzle, checkpoints)           safety, roll choice ───┘          │   P3.2 robot timing model
                                        P2.7 approach (RRT) ───────────────────┘
```

- **Phase 1** takes images for ~a quarter turn and fits the
  `TurntableModel`: the turntable as a one-joint robot (revolute θ), whose FK
  gives the pose of the puzzle and of every checkpoint for any θ, plus θ(t).
- **Phase 2** searches for the safest plan: start configuration, start phase θ₀,
  the hoop roll at every checkpoint, the IK branch; simulated over the whole
  timed traversal (the puzzle keeps turning). Output: a `Plan`, or "unreachable".
- **Phase 3** approaches, waits for θ₀, walks the checkpoints with targets placed
  at the puzzle's predicted pose at arrival (Kalman filter on θ, aware of the
  camera delay), corrects small errors, and lowers the hoop at the end.

## 4. Shared contracts (task C0)

`Prokop/solver/contracts.py` (drafted 2026-11-18, **for review by all three**;
tests in `Prokop/tests/test_contracts.py`). Data types only, plus a reference
implementation of the turntable chain and fakes. The conventions (time, the
turntable angle θ, the puzzle frame, the checkpoint frame and the roll
reference) are in its module docstring.

| Type | Content | Produced by → used by |
|---|---|---|
| `BaseMeasurement` | `t_capture` [s, time the image was taken], `xy` (base centre) [m], `yaw` (base frame x axis) [rad], `n_markers` (1/2), `quality` | P1.2 → P1.3, P3.1 |
| `TurntableModel` | `axis_xy`, `base_offset`, `base_yaw` (base on the turntable top), `direction` (±1), `omega`, `theta_ref` at `t_ref`, `covariance`; `theta_at(t)`, `puzzle_T(θ)`, `checkpoint_T(path, i, θ)`, `measurement_at(t)` | P1.3 → P2, P3 |
| `PuzzlePath` | checkpoint `frames` (N×4×4, puzzle frame), `clearance` per checkpoint, `roll_limits` (or None = free), `final_index` | P2.1 → P2 |
| `RobotTiming` (protocol) | `move_duration(q0, q1, speed)`, `blends(dq_prev, dq_next)` | P3.2 → P2.6, P3.3 |
| `PhaseEstimate` | `t`, `theta`, signed `omega`, 2×2 covariance; `predict(t)` → θ, σ | P3.1 → P3.3 |
| `Plan`, `PlanStep` | `feasible`/`reason`, `speed`, `direction`, `theta_start`, `q_start`, steps (`index`, `q`, `roll`, `t_rel`, `margin`), `approach`, `min_margin` | P2.6, P2.7 → P3.3 |

Fakes, so that nobody waits for anybody: `example_model()`,
`fake_measurements(model, times, noise, …)` (with one-marker and missing
detections), `example_path()` (a simplified puzzle-D-like path; the real one is
P2.1), `example_plan()`.

`TurntableModel.puzzle_T(θ)` is the reference chain: robot frame → axis on the
table → rotation θ → turntable top → base (offset, yaw, on its foam) →
`puzzle_mount`. It equals the simulator's placement of the puzzle exactly
(tested), which also gives the evaluation harness the ground truth: in the sim,
θ = `SceneViewer.turntable_angle(t)`.

Review checklist: Vojtěch the `BaseMeasurement` (note: `yaw` is the base frame's
x axis in the robot frame, not an image angle), Natka the `TurntableModel`,
`PuzzlePath` and `Plan`, everyone the conventions.

## 5. Tasks

Each task: what, which lab homework approach it uses, inputs → outputs,
acceptance tests, and dependencies. "Fake" means the C0 fakes are enough to
start.

### Phase 1: puzzle on the turntable

**P1.1 Turntable kinematic chain** (HW02 URDF + FK, HW01 SE3)
- The turntable + base + puzzle as a 1-DOF robot, generated as a URDF from a
  `TurntableModel` (as in HW02): world → axis (fixed, `axis_xy` on the table) →
  revolute θ about z → turntable top → base (fixed: offset, yaw) →
  `puzzle_mount` → one fixed frame per checkpoint of a `PuzzlePath`. FK gives the
  pose of every checkpoint at any θ; the URDF can also be shown in the viewer.
- Tests: the FK equals the contracts' reference `puzzle_T` / `checkpoint_T` to
  1e-9 for random parameters and θ; θ is periodic.
- Depends on: C0 (the reference chain exists there, so nobody waits for P1.1).

**P1.2 Base measurement from one image** (HW03 perception: ArUco, homography)
- `measure(image, t_capture) -> BaseMeasurement | None`, from Vojtěch's ArUco
  detector and camera calibration. Pixel → robot plane **at the markers' height**
  (base top, 64.6 mm above the table), not at table height (see E9).
- One marker visible → still a measurement (weaker); none → None.
- Tests: on simulated images (`CRS93Sim.grab_image`) at known placements, error
  of xy and yaw against the truth (harness I2 provides the truth); statistics
  feed the noise model of P1.3 and P3.1. Later the same on real images.
- Depends on: C0, the camera calibration. Owner's own directory.

**P1.3 Turntable fit** (least squares; HW01 SE2/SO2 for the angles)
- From N timed measurements (~7 s, a quarter turn) fit `axis_xy`, `base_offset`,
  `base_yaw`, `theta_ref`, `direction`, with |ω| = 2π/28.0 s fixed. The marker
  centre moves on a circle around the axis; its yaw advances at ω.
- Handle the degenerate case of the base near the axis (circle radius → 0: the
  axis is ill-determined but then hardly matters) and outliers.
- Output: `TurntableModel` with covariance.
- Tests: synthetic measurements (fake) with the noise of P1.2: recovered
  parameters within tolerance; both directions; degenerate case; outliers.
- Depends on: C0 (fake measurements), P1.1 for the model class.

### Phase 2: plan

**P2.1 Puzzle path** (geometry; puzzle CAD)
- Checkpoints of the hoop's ring centre through puzzle D in the puzzle frame,
  with tangents; from `Natka/crs_puzzle_d_hoop_path.profile_points` (mm, `D.step`
  frame = puzzle URDF origin) as a starting point.
- Per checkpoint: the clearance hoop ↔ bar as a function of tilt (ring Ø 62.4 mm
  inner around a 10 mm bar), checked against the puzzle mesh; the end point
  satisfies "hoop < 4 cm above the base".
- Tests: the ring at each checkpoint (no tilt, any roll) does not intersect the
  puzzle mesh; spacing between checkpoints small enough that a straight segment
  stays within the clearance.
- Depends on: nothing (puzzle frame only).

**P2.2 Hoop targets** (HW01 SE3)
- `flange_target(checkpoint_T, roll, tilt=0) -> T_flange`: ring centre on the
  checkpoint, ring axis along the tangent (rotated by `tilt`), rotated by `roll`
  about the ring axis; uses the hoop mount (`placement.HOOP_MOUNT`, the adapter
  turned 180°, `models/crs93.urdf`).
- Tests: FK of the URDF with the hoop at IK(target) puts the ring centre on the
  checkpoint and its axis along the tangent; roll does not move the ring centre.
- Depends on: P2.1 types (fake checkpoints are enough).

**P2.3 IK layer** (HW05 IK: analytical + numerical)
- All solutions (`ctu_crs` analytical IK), within limits, ±2π variants of the
  wrist joints, numerical refinement near singularities (the analytical IK sets
  q5 = 0 exactly there); choice of the solution continuing a given branch.
- Tests: FK(IK(T)) = T; continuity along a smooth target sequence; behaviour at
  q5 ≈ 0.
- Depends on: nothing.

**P2.4 Safety metrics** (HW04 Jacobian; collision module)
- For a configuration: manipulability / smallest singular value of the CRS93
  Jacobian (analytical from the DH, or finite differences as in HW04), distance
  to joint limits, and (via `Prokop/collision.py` with the *estimated* model)
  clearance arm ↔ cell/table/turntable and hoop ↔ puzzle.
- For a segment q0 → q1: largest joint step, the same metrics sampled along it,
  with the puzzle rotating during the segment (timing from P3.2).
- Tests: σ_min → 0 at known singular configurations; analytical vs
  finite-difference Jacobian; clearance against known cases.
- Depends on: P2.3 for test configurations (optional), P3.2 for segment timing
  (fake timing until then).

**P2.5 Roll / branch selection along the path** (HW04 null-space motion; dynamic programming)
- The task (ring centre + ring axis) has 5 DOF; the roll about the ring axis is
  free and costs no clearance: the robot's 6th DOF is a null-space direction to
  optimize. Discretize roll (e.g. 10°) × IK branch per checkpoint and pick the
  sequence maximizing the worst margin with smooth transitions (Viterbi / DP);
  optionally refine with null-space gradient steps as in HW04. Small tilt allowed,
  its clearance loss counted.
- Tests: on `example_model()`: avoids a constructed singular checkpoint by
  rolling; result continuous (no branch switch mid-path).
- Depends on: P2.2, P2.3, P2.4 (fakes for the metric until P2.4 is ready).

**P2.6 Timed plan and start phase** (simulation of the traversal)
- For a start phase θ₀ and speed: arrival times of the checkpoints from
  `RobotTiming` (including the near-stops at bends), puzzle at θ₀ + ω·t, targets
  at the predicted poses (fixed-point iteration: arrival time ↔ target), P2.5 on
  that moving target sequence, margins along all segments. Search θ₀ (e.g. 5°
  grid) for both directions; pick the max-min margin plan. No feasible θ₀ →
  `Plan(feasible=False, reason=...)` (the assignment's "report unreachable").
- Tests: on `example_model()` returns a feasible plan; on a placement out of
  reach returns infeasible with a reason; the plan's own replay (FK at the
  predicted times) keeps the ring on the path within the clearance.
- Depends on: P2.1–P2.5, P1.1, P3.2.

**P2.7 Approach and retreat paths** (HW06 RRT + random shortcut)
- Collision-free joint-space path q_home → `q_start` (and back), checked with
  `collision.ArmChecker` plus the turntable's swept volume
  (`placement.TaskScene.swept_task_mesh`), since the puzzle rotates while we
  approach. A straight joint-space move when it is free; RRT + shortcut
  otherwise.
- Tests: free straight case; a case needing a detour; respects joint limits.
- Depends on: C0; independent of the rest of P2.

### Phase 3: execution

**P3.1 Phase tracker** (Kalman filter; not in the labs)
- State [θ, ω]; measurements of θ from `BaseMeasurement.yaw` applied **at
  t_capture** (delayed measurements: propagate from the capture time; keep a
  short history to accept out-of-order measurements); prior from P1.3; ω prior
  2π/28.0 s ± 0.5 %; outlier gating; prediction to any future t. Gaps (markers
  covered by the arm) are normal.
- Tests: synthetic measurements with delay, noise, gaps and outliers: prediction
  error at +0.3 s within tolerance; recovers after a gap.
- Depends on: C0 (fake measurements), P1.2 noise statistics (assumed until then).

**P3.2 Robot timing model** (measurements E5, E6)
- Public module with the measured behaviour: duration of a coordinated move at a
  speed setting (trapezoidal profile), `command_delay`, `send_delay`, blending
  only for the same joint-space direction. Same numbers as `CRS93Sim`.
- Tests: agrees with `CRS93Sim` and with the real logs
  (`Prokop/experiments/motion_logs`).
- Depends on: nothing.

**P3.3 Executor** (state machine on the CRS93 interface)
- Approach (P2.7) → wait at the pre-entry pose until the predicted θ reaches θ₀
  (minus the command lead) → for each checkpoint: target at the predicted puzzle
  pose at arrival (P3.1 + P1.1), `move_to_q` → finish: lower the hoop below 4 cm
  above the base → retreat. Abort and retreat if the predicted error exceeds the
  plan's margin or a measurement disagrees with the prediction.
- Tests: in the sim with `example_plan()` and a perfect tracker first; then with
  the real tracker; timing log of every command.
- Depends on: C0 (fake plan), P3.2; integrates P3.1, P3.4.

**P3.4 Online corrections** (HW05 numerical IK, HW04 Jacobian)
- Retarget the next checkpoint from the latest prediction: a small IK step seeded
  with the planned q (stay on the planned branch), with limits on the change.
- Tests: corrected targets converge; never switch branch; respect limits.
- Depends on: P2.2, P2.3.

### Integration

**I1 Application** (`robot = CRS93Sim()` ↔ `CRS93()`, CLI, logging of every
measurement / estimate / command to a run folder for the report).

**I2 Evaluation harness** (sim only, may use the sim's truth)
- Random placements in the work area × both directions × speeds; per run:
  success criteria, min hoop↔puzzle clearance, total time, phase/estimation
  errors. Output: the success map over the work area for the report.

**I3 Robot procedure** (`Prokop/docs/ROBOT_CHECKS.md` style): static puzzle at
low speed first, then spinning; one direction, then both.

## 6. Dependencies and parallel work

```
C0 contracts + fakes ─┬─► P1.1 chain ─────────────┬─► P1.3 fit ─────────┐
                      ├─► P1.2 measurement ───────┘   (noise model) ──► P3.1 Kalman ──┐
                      ├─► P2.1 path ─► P2.2 targets ─┐                                 │
                      ├─► P2.3 IK ───────────────────┼─► P2.5 roll/branch ─► P2.6 plan ─┼─► I1/I2 ─► I3
                      ├─► P2.4 metrics ──────────────┘                      ▲          │
                      ├─► P3.2 timing ──────────────────────────────────────┴─► P3.3 executor ◄─ P3.4
                      └─► P2.7 approach (RRT)
```

After C0, everything except P1.3, P2.5, P2.6, P3.3 can start at once; those
four start on fakes and switch to the real inputs when available. Critical path:
P2.1 → P2.2/P2.3 → P2.5 → P2.6 → integration.

## 7. Proposed split (to be decided)

Balanced by size and matched to what everyone already does:

| | Vojtěch (perception & estimation) | Natka (planning) | Prokop (simulation, models, integration) |
|---|---|---|---|
| 1st | P1.2 measurement | P2.1 path | C0 contracts + fakes (with all) |
| 2nd | P1.3 turntable fit | P2.3 IK layer | P1.1 chain, P3.2 timing |
| 3rd | P3.1 Kalman tracker | P2.2 targets, P2.4 metrics | P2.7 approach (RRT) |
| 4th | camera calibration on the robot | P2.5 roll/branch, P2.6 plan | P3.3 executor, P3.4 |
| then | estimation accuracy experiments | planning experiments | I1, I2, I3 |

Each person keeps their code in their own directory (`Vojtěch/…`, `Natka/…`,
`Prokop/solver/…`); only `Prokop/solver/contracts.py` is shared.

## 8. Working with AI on a task

Give the assistant: this plan, the task's section, `contracts.py`, and the docs
listed at the top. Each task is done when its acceptance tests pass and it has
Google-style docstrings; results that affect others go to `Prokop/NOTIFY.md`.

## 9. Open questions

1. Speed setting (decided in testing; everything takes it as a parameter).
2. Hoop tilt limit (roll is free; tilt costs clearance).
3. Checkpoint spacing vs. the controller's near-stops at bends (E6): fewer,
   longer straight segments in joint space run smoother.
4. Calibration format from Vojtěch (`Homography` 3×4 projection at marker height).
5. The review of the contracts (section 4) by all three.
