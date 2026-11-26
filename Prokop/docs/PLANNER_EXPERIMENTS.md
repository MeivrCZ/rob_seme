# Planning: experiments, tests and solved problems (2026-11-22 .. 11-25)

What was tried for planning the hoop's path through puzzle D, how it was
tested, what each experiment showed and how the problems were solved. Written
for the report's *Experiments* chapter and for the team. Experiments at the
station (timing, camera, …) are in [EXPERIMENTS.md](EXPERIMENTS.md); measured
values in [MEASUREMENTS.md](MEASUREMENTS.md); the soft collision model in
[SOFT_COLLISION.md](SOFT_COLLISION.md).

All runs are in the simulator (`CRS93Sim`, exact meshes as the ground truth).
"Seed N" is `TaskScene.random_placement(default_rng(N), WORK_AREA_XY)`; seed 1
puts the turntable axis at (0.595, 0.183) m with the base 45 mm off the axis.
"Facing" means the puzzle's open end (puzzle +x) points at the robot.

Contents:

1. [Natka's planner with fixes (copy)](#1-natkas-planner-with-fixes-the-copy-in-natka_copy)
2. [A static puzzle as a sanity check](#2-a-static-puzzle-as-a-sanity-check)
3. [Our own planner, static puzzle](#3-our-own-planner-static-puzzle)
4. [The soft collision model](#4-the-soft-collision-model-for-planning)
5. [Planner on the soft model](#5-the-planner-on-the-soft-model)
6. [Waiting pose, approach, tilts](#6-waiting-pose-approach-tilt-grid)
7. [Simulator as a standalone process, perception, demo](#7-the-simulator-as-a-standalone-process-perception-demo)
8. [Reachability over the work area (static)](#8-reachability-over-the-work-area-static-puzzle)
9. [The spinning puzzle](#9-the-spinning-puzzle)
10. [Open ideas](#10-open-ideas-not-built-yet)
11. [Scripts](#11-scripts-to-reproduce)

---

## 1. Natka's planner with fixes (the copy in `natka_copy/`)

Natka's P2.2–P2.6 (roll/branch selection by dynamic programming, a fixed-point
loop between arrival times and moving checkpoint targets) found **no plan** in
any of our runs. To find out why without touching `Natka/`, a copy with
changes marked `PROKOP COPY` was run with `run_solver.py --planner copy`.

| Problem found | Evidence | Change in the copy |
|---|---|---|
| Wrong hoop side (side A leads; with A the arm ends too close to the table) | 0/16 plans | side B (contract C13) |
| Traversal timed from `q_home` (a 12 s move, the turntable turns 150°) | fixed point diverged | start at checkpoint 0, wait there for the start angle |
| Very slow: **1540 s of a 1788 s run** reparsing the URDFs in `hoop_targets._fixed_mount_frames` (profiled) | cProfile | `@functools.cache`: identical targets, 76 → 4.9 ms each |
| Lexicographic ranking (worst margin first): joint travel never counts → 160–220° jumps between checkpoints | per-pass logs | closest configurations first, margin as a filter (> 0, > 0.02) |
| Branch jumps | 70–209° steps | first-order continuity ‖(q−q_prev) − J⁺Δx‖ ≤ α‖J⁺Δx‖ + β (α 0.5, β 2°), later capped at 20° |

**Result (seed 1, spinning):** still no plan. Each fix moved the failure; the
real blocker was the fixed-point loop between arrival times and targets:
the arrival times kept changing by 1–6 s per pass (no convergence), the worst
margin drifted towards 0, and eventually a checkpoint became unreachable. The
continuity bound also had a loophole: near the wrist singularity J⁺ blows up
and so does the allowed deviation.

## 2. A static puzzle as a sanity check

To separate "the planner cannot plan" from "the timing cannot converge", the
turntable was stopped and the puzzle placed facing the robot on a 3×3 grid
around the work-area centre (x 0.407–0.567, y ±0.10).

- Natka's planner (copy, continuity only): **6/9** placements planned; the near
  row (x 0.407) failed at checkpoint 0 (too close to the robot). Every plan had
  a 55° wrist step at the 45° jog: Natka's path tilts the hoop 37° there and
  back (the ring follows the wire's tangent).
- **Conclusion:** planning itself works; with spin, the timing loop was the problem.

## 3. Our own planner, static puzzle

Built from scratch (`solver/wire.py`, `solver/planner.py`, `solver/hoop_pose.py`).

**The wire** (`solver/wire.py`): the centreline measured on the puzzle mesh —
top bar z = 155 mm from the open end x = 85 to 35 mm, R50 bend, vertical run
to z = 45, 45° jog to (0, 30), stem. A 10 × 10 mm square bar. Checked against
the mesh: the centreline is 5.0 mm inside the bar everywhere (except at the
open end face).

**Tiers of checkpoints:** tier 0 = only the segment ends (the bend is one
move); every tier halves each interval (6, 11, 21, 41, 81 checkpoints).
Selection: every IK solution over a roll grid = a candidate; dynamic
programming picks the least total joint travel; every move is checked along
the way; the first collision-free tier wins.

| Experiment | Result | What it led to |
|---|---|---|
| Centre placement, exact collision check | tier 0 collides (hoop–puzzle in the bend), tier 1 **worse** (link 4 hits the puzzle) | the candidates themselves were never collision checked → filter candidates first; then tier 1 free (11 checkpoints, 13.4 s) |
| Both hoop sides allowed (the hoop cannot turn over around the bar, so one side per plan) | side A: IK finds 120–192 solutions per checkpoint, **0 within the joint limits** (the flange would point back at the robot) | side B confirmed for a facing puzzle |
| End point | the hoop must end < 4 cm above the pad (assignment); the jog's top is above that | the path ends 5 mm into the stem (z = 25 mm, ring 24.4 mm above the pad); the whole ring's height is checked (tilt ≤ ~25° there) |
| **Position-only checkpoints** (user idea): the checkpoint fixes the ring centre, orientation free on a tilt × direction × roll grid | near-row placement, which failed every tier, plans at **tier 0**: steps ≤ 15.5°, joint 1 constant | orientation constraints caused most failures |
| Tilt penalty? | not needed (user) | — |
| Finer tiers to "soften" the tilt (centre) | tilts stay (30° on the top bar, 45° on the jog = ring kept horizontal); traversal 6.9 → 8.0 → 11.9 → 14.2 s (tiers 0–3), clearance 14.4 → 8.9 mm | tier 0 is best; the tilt is what least joint motion prefers |

## 4. The soft collision model for planning

The simulator's exact meshes are the environment's ground truth; planning code
must not use them. Built `solver/soft_collision.py` (details:
[SOFT_COLLISION.md](SOFT_COLLISION.md)): boxes fitted once to the CAD
(`tools/fit_soft_boxes.py` → `models/soft/*_soft.urdf`), task posed from the
perceived `TurntableModel`, box–box distance by the 15-axis separating test
(a lower bound), a box hierarchy refined only near contact.

**Validation** (`experiments/validate_soft_collision.py`, 500 random states):

| Check | Result |
|---|---|
| every mesh vertex inside its boxes | yes (all parts, hoop 3310, puzzle 604 vertices) |
| soft distance > exact (table, cell, task, self, hierarchy) | **0** violations |
| threaded rings accepted (exactly free candidates) | 94 % without margins, 67 % with |
| speed | 1.4 ms (2.4 ms around the bar) vs 8.6 ms exact |

**Problems solved on the way:**

- The hoop's hole is **30.0 mm** in radius in the CAD, not 31.2 (62.4 mm nominal).
- One box per long link: `link4`–`link6` boxes overlapped in most wrist poses
  (41 false self-contacts in 200 states vs 1 real; 85 % of the false
  rejections of threaded rings). Fixed by cutting long parts into slabs and by
  excluding `link4`–`link6` from the self-check: the exact meshes stay
  ≥ 7.9 mm apart over the whole q5 × q6 range.
- 6 ms per query → batched matrices and the hierarchy → 1.4 ms.
- The soft URDFs in `models/` broke the viewer's model gallery (no visual
  meshes) → moved to `models/soft/`. Found by running the full test suite on
  a clean checkout before committing (commit `2787b92`, merged to main).

## 5. The planner on the soft model

- **Clearance-based move check:** after each sample the next one is placed so
  no arm point can have moved further than the clearance (per-joint lever
  arms × the move's peak speed; checked: the real motion of the box corners
  stayed ≤ 79 % of the bound over 300 random 2° steps).
- **Moves inside the selection** (user choice): a colliding move rules out
  that pair of configurations and the selection runs again (instead of finer
  tiers or a smaller tilt grid).
- **Lazy candidates:** only candidates the selection picks are checked;
  centre placement 83 → 54 s.
- **3×3 grid, static, facing:** all 9 placements plan at tier 0; every plan
  replayed against the **exact** meshes: no contact (least clearance 4.3–14.5 mm).

## 6. Waiting pose, approach, tilt grid

| Finding | Fix |
|---|---|
| Joint 1 turned ~165° from `q_home`: the plan used the "over the top" branch (wrist centre 419–541 mm from the base axis, not a singularity) because the approach from `q_home` cost nothing | the joint travel from `q_home` counts in the selection → forward branch (q1 ≈ ±21°) |
| Checkpoint 0 must be a pose where the robot can wait while the puzzle turns | the soft model got the swept volume (each task box sweeps a cylinder, enclosed by 4 boxes rotated 22.5°); checkpoint 0 must clear it; the entry distance moves out (15, 30, 50, 80 mm) until a pose can wait. Result: tilts 30/30/15/15/15/0, clearance 14.4 mm |
| The entry distance was chosen once for both sides (side A could wait at 15 mm, side B needed 30) | chosen per side |
| Tilt grid 0–60° in 15° steps → 0–50° in 10° (user) | 9/9 grid placements; traversal 4.6–7.2 s |
| The approach (demo): straight line fails (other IK branch), direct move hits the top bar | `plan_approach`: direct, or through a pre-entry pose back / up / back-and-up; every move clear of the **swept volume** (user: the approach must not cross the spinning puzzle's volume) |

## 7. The simulator as a standalone process; perception; demo

- **Demo** (`experiments/show_plan.py`): plan → approach → traversal → back,
  looped in the viewer; the simulator's collision monitor is the independent
  judge. 20+ loops without a collision.
- **Perception added** (user): 5 camera images → Vojtěch's ArUco
  `measure_base` → a static model; the plan is made on the estimate and run
  against the truth. Error 0.01 mm / 0.02° (the simulated camera is perfect).
- **Planning took 11 min inside the demo vs 27 s alone:** the simulator's
  threads (renderer, collision monitor, viewer) starved the planner of the
  interpreter lock. Fixed by running the simulator as a **standalone process**
  (`sim_process.SimProcess`, user design): the solver gets only a
  `ctu_crs.CRS93` (local kinematics; commands and `grab_image` forwarded);
  scene setup and ground truth only on the `SimProcess` handle. Planning
  34 s. `run_solver.py` moved to it (phase 1 verified: axis error 0.1 mm).
  Tests: `tests/test_sim_process.py`.

## 8. Reachability over the work area (static puzzle)

`experiments/reachability.py`: 25 placements, facing, the whole pipeline
(perception → planning → approach) and an exact replay of every plan found.

| x [m] at y = 0 | result |
|---|---|
| 0.29, 0.31 (outside) | unreachable at **perception** (no markers) |
| 0.337 | unreachable at planning (no pose can wait clear of the swept volume) |
| 0.35–0.37 | unreachable at planning (side B cannot wait, side A no IK) |
| 0.38–0.40 | unreachable at the approach (traversal planned) |
| **0.42–0.60** | **plan**, exact clearance 8.8–12.5 mm |
| 0.637 (edge), 0.68 | unreachable at perception (half the base out of the image) |

Sides (y ±0.2): plans at x 0.487; perception or approach failures at the
corners (marker visibility depends on the base's rotation). **No plan ever
collided in the exact replay.** Placements near the robot base pass perception
but are infeasible, and are reported before anything moves.

## 9. The spinning puzzle

**Perception** (images while spinning, Vojtěch's `fit_orbit`): axis within
0.1–0.4 mm, period 27.95–28.11 s vs 28.00 s, base orientation error
0.13–0.27°. A bug in our comparison (the fit measures the angle from its own
zero and puts the base yaw into `base_yaw`) gave "−102°"; compared by the base
orientation (angle + base yaw) it is 0.2°.

**Execution:** the existing `Executor` (P3.3) with a `PhaseTracker`; the
Kalman tracker (P3.1, Vojtěch) is not done yet, so `ModelTracker` on the
fitted model (user choice) or a perfect tracker from the simulator.

**Start angle** (user): one fixed angle per placement (the puzzle facing the
robot half way through the traversal); the robot waits at checkpoint 0 for it.
No search over start angles.

### 9.1 Tier planner with timing passes

Pass k computes the targets at pass k−1's arrival times. One start angle took
> 24 min: every pass recomputed every candidate from scratch (no warm start),
and the passes did not converge (as with Natka's loop).

### 9.2 RRT between checkpoints (user design, lab 6)

`solver/rrt.py`: joint-space RRT with time-aware edges + random shortcuts;
checkpoints = segment ends + 3 or 5 on the bend; one configuration per
checkpoint from the selection, the RRT connects consecutive ones (straight
first).

- **Static centre:** collision free, the shortcuts reduce the whole traversal
  to **3 waypoints (2 moves)**, 6.4 s, exact clearance 6.6 mm, planned in 24 s
  (3 bend points; 5 gave 6.1 s / 70 s). Valid because the ring cannot leave
  the bar without touching it.
- **Spinning:** failed. RRT budgets were too generous (1200 nodes × 2
  continuous edge checks); then lazy edges + a 5 s budget; then a time
  mismatch: targets computed for the pass's estimated arrival, checked at the
  tree's real time → the goal pose no longer fit the turned puzzle.

### 9.3 Space-time tree (user design)

One tree from checkpoint 0 at t = 0 (puzzle at the start angle): nodes (q, t),
a node's time = parent's + move duration, every move checked against the
puzzle turning during it; checkpoints as subgoals solved at the arrival time.
`plan_spacetime` in `solver/planner.py`.

Iterations (each found by a run or a diagnostic):

| Problem (seed 1 unless noted) | Evidence | Fix |
|---|---|---|
| Checkpoint 1 unreachable | the direct move is free (8.6 mm) — diagnostic; the fallback orientations were all tilt 50° | fallbacks spread over the tilts |
| The tree does not grow inside the bar (3–6 nodes / 15 s) | uniform samples push the ring into the bar (narrow passage) | **samples guided along the wire** + a cheap configuration check before the continuous one (user OK) |
| A long 4.9 s direct move (puzzle turned 63°) led into a dead end | log | **no long moves** (user) |
| Short 5° steps: the tree stalled at 25–45 % of the top bar, also at speed 0.5 | `follow_bar.py` (below) | **moves limited by duration (≤ 1.2 s), large steps (≤ 25°)**; robot speed 0.2 → 0.5 (user OK) |
| End of the bend unreachable | `checkpoint_timeline.py`: free poses at the next checkpoint shrink ~460 → 400 → 180 → **50** as the puzzle turns | — (see below) |
| The ring turned with the puzzle (roll fixed to the puzzle) | user remark | **roll free** at every moment (the ring is round) |
| Tree got 88–100 % of the way but "not reached" | the static plan's exact pose was demanded | **position-only checkpoints** in the tree (ring centre within 5 mm) |
| 2° sampling noise ≈ 2 cm at the ring | log | noise 0.5°, 30 % of guided steps aim at the checkpoint itself |
| Dead ends depend on how a checkpoint was reached | different arrivals → different outcomes | **backtracking**: up to 3 different arrivals per checkpoint |

**`follow_bar.py` (the key diagnostic).** Ring stepped along the top bar in 30
steps, each solved at its arrival time:

| along the bar | frozen puzzle | turning puzzle |
|---|---|---|
| 3 % | 1.0° step, t 1.10 s, 10.1 mm | 4.8°, t 1.00 s, 10.5 mm |
| 43 % | 1.1°, t 2.62 s | 8.9° (joint 6), t 3.48 s |
| 53 % | 1.2°, t 3.02 s | **19.5°**, t 4.88 s |
| 67 % | 1.2°, t 3.55 s | **no IK solution** (t ≈ 7.4 s) |

Never a collision: many short stop-and-go moves (the robot blends only moves
in the same joint direction) let the puzzle turn away until the pose leaves
the arm's reach. Following a turning puzzle needs few, fast, long moves.

**Start-angle sweep** (`checkpoint_timeline.py`, ±60° in 15° steps, greedy
direct moves): no angle gets round the bend (best: +60° up to checkpoint 4);
results noisy because the planners' time budgets are wall-clock based.

**Seed 1 vs the work-area centre** (user: "could it be the position?"): seed 1
is at the edge of the work area (axis 0.62 m from the robot, base 45 mm off
the axis, puzzle up to ~0.67 m out). At the centre (0.487, 0), base 20 mm off,
the tree **gets round the whole bend** (checkpoint 4 at 4.26–5.00 s) and the
vertical run became the problem.

**Why the vertical run fails** (user: "the puzzle running into the arm?"):
rejections logged by body pair on the way to checkpoint 5 (1039):

| count | rejected | bodies |
|---|---|---|
| 404 + 260 | configuration | forearm (`link4`) — **top bar** |
| 94 + 51 | configuration | forearm — start of the bend |
| 35 / 27 / 25 | move | ring — stem / vertical run / jog |
| 29 | configuration | ring — base |

**~80 % is the forearm hit by the turning top bar** (it sticks out 85 mm at a
height of 155 mm; facing the robot, it is under the forearm while the ring
goes down). Status: not solved forwards — see 9.4 (backwards).

### 9.4 Planning backwards in time (user design) — first success

`plan_backward` in `solver/planner.py`: the root is the ring in the stem at
t = 0 with the puzzle at an end angle (`end_angles`: the top bar pointing away
from the robot first, then 30° steps); every new node is earlier than its
parent by the move's duration (the move new → parent is checked in forward
time); checkpoints in reverse as subgoals down to checkpoint 0, which must be
a waiting pose. The puzzle's angle there is the start angle (the robot waits
for it). Otherwise as 9.3 (wire-guided, free roll, position-only, ≤ 1.2 s
moves, backtracking). `plan_spinning` uses it for a spinning puzzle.

| Run | Result |
|---|---|
| central placement (0.487, 0), offset 20 mm, speed 0.5, true model (`backward_plan.py`) | stem → checkpoints 5, 4, 3 in single moves (5 s of planning) — the forearm/top-bar problem of 9.3 gone; the top bar (back to checkpoint 1) is now the hard part (138 rejections, one backtrack) |
| same, first version | full path found, but **collided after shortcutting** (bug): each shortcut was checked alone, yet it makes every later move earlier (another puzzle angle). Fix: a shortcut is kept only if the whole path stays free |
| same, fixed | **collision free**: 8 waypoints, traversal 5.3 s, start angle −123° (top bar 59° from the robot), least soft clearance 0.6 mm, 96 s |
| **end to end in the simulator** (`spin_plan.py --axis 0.487 0`): perception → plan on the fitted model → approach → executor (ModelTracker) | **SUCCESS**: ring 24.4 mm above the pad, no collision. End angles −104°, −74° failed (budget 120 s), −134° planned in 29 s; 5 waypoints, traversal 4.2 s planned / **6.1 s driven**; the fitted model **7.4° off** at the start (period 27.95 vs 28.00 s, 318 s after the fit) — worked anyway |

**Sanity sweep, 6 placements** (`spin_sanity.py`: true model, then an
independent replay against the **exact** meshes with the puzzle turning, every
0.5° of joint motion). Goal share of the tree raised 20 → 50 % (user: as in
the homework; checkpoint-to-checkpoint moves are mostly direct), wire-guided
40 %, uniform 10 %.

| Placement (axis, base offset, yaw, spin) | first sweep | after the entry fix |
|---|---|---|
| (0.487, 0), (20, 0) mm, 0°, + | plan 132 s, exact 8.2 mm | plan 318 s (2 end angles failed), exact 4.8 mm |
| (0.45, +0.10), (0, 30) mm, 60°, + | plan 46 s, exact 4.4 mm | plan 73 s, exact 4.7 mm |
| (0.43, −0.05), (0, −20) mm, 0°, − | plan 79 s, exact 4.9 mm | plan 164 s, exact 6.4 mm |
| (0.52, −0.10), (−30, 0) mm, −90°, + | **no plan** (771 s) | plan 77 s, exact 6.5 mm |
| (0.55, +0.08), (25, 25) mm, 150°, − | **no plan** (797 s) | plan 59 s, exact 5.5 mm |
| seed 1 (0.595, 0.183), (28, −35) mm, 0°, + | **no plan** (763 s) | plan 50 s, exact 9.3 mm |

**Every plan replayed without contact** (soft clearance 0.5–3.4 mm, exact
4.4–9.3 mm: the soft model stays conservative). Why the three failed
(`backward_plan.py`): the trees got the ring back along the whole top bar
(100 % of the way) but never "arrived" — checkpoint 0 was only the 15 mm
entry, where no pose could wait clear of the swept volume; the larger
entries (30, 50, 80 mm) came after it in a loop and never got time. **Fix:**
checkpoint 0 accepts any entry (the wire laid out from the largest; the ring
leaves the bar backwards along the straight top-bar axis; a pose at any entry
counts if it can wait), the plan records the entry it waits at. The entry
cannot be chosen first as in the static planner: where a pose can wait
depends on the start angle, known only at the end.

Problem left: **planning time, 302 s** (failing end angles use their whole
budget). Options: end angles in parallel, smaller budgets, the plan library,
faster checks.

### 9.5 Camera tracker, the end pose, parallel end angles

**Vojtěch's `CameraPhaseTracker`** (only `camera_phase_tracker.py` +
`advanced_aruco.py` + its test copied unchanged from his branch
`AdvAruco-SimSolve`, commit `1a08f98`; nothing else of the branch merged):
keeps measuring the base in camera images while the robot works. Default
tracker of `spin_plan.py` (`--tracker camera`). A watcher in `spin_plan.py`
prints once "lost aruco, believing model" (latest measurement > 1 s old) and
once "found aruco again, updating model" (user request; his file unchanged).

End to end, central placement: **SUCCESS** (ring 24.4 mm); plan 103 s; the
fitted model alone would have been **5.3° off** at the start, the camera
tracker **0.86°**; traversal 5.7 s driven vs 5.2 s planned (the camera
thread did not visibly slow the commands); the tracker had no new
measurement for the last 4.3 s (arm over the markers, probably).

**Collision after the end:** the robot holds the final pose while the puzzle
keeps turning; the top bar came round and hit the forearm (simulator monitor,
a few seconds after SUCCESS). The user: prefer a final pose that stays free
for a whole turn, else plan anyway and warn (harmless: hollow plastic puzzle
on foam, magnetic hoop). Implemented: the 40 final poses with most clearance
are checked over a full turn (5° steps); those that stay free are tried first;
`result.end_waits`, warning in `spin_plan.py`. In the runs so far no such pose
exists (the ring is threaded on the off-axis stem, which carries it round).

**Parallel end angles** (`BW_WORKERS = 3` processes, first collision-free
plan wins, the others stopped):

| Placement | one after another | parallel (3) | exact replay |
|---|---|---|---|
| (0.487, 0), central | 318 s | **44 s** | no contact, 9.8 mm |
| (0.43, −0.05), spin − | 164 s | **86 s** | no contact, 6.4 mm |

**Workers:** one process per end angle by default (`BW_WORKERS =
BW_END_ANGLES`, capped at the core count) — the user: 3 attempts in the
assignment, so even 120 s of planning per attempt is too long; the lab PC
runs it, memory can be freed.

**Profile of one backward search** (`profile_breakdown.py`; central
placement, first end angle, one process; 75 s under the profiler — about 1.5×
slower than without): plan found, 11 tree searches (5 of them failed, each
using its whole 6 s budget), then shortcuts.

| Block | calls | time | share |
|---|---|---|---|
| tree searches between checkpoints | 11 | 40.9 s | 54 % |
| — continuous move checks in the tree | 85 | 23.1 s (272 ms each) | 31 % |
| — pose before a node on the wire (IK over the roll grid) | 492 | 12.0 s | 16 % |
| — direct tries to the checkpoint | 32 | 3.3 s | 4 % |
| — configuration checks | 104 | 0.3 s | 0 % |
| **shortcuts (whole path rechecked per shortcut)** | 8 | **22.2 s** | **29 %** |
| final poses that stay free for a turn | 40 | 1.6 s | 2 % |
| *building blocks:* soft clearance of one configuration | 15 895 | 49.2 s (3.1 ms each) | 65 % |
| *building blocks:* analytic IK | 15 336 | 11.0 s (0.7 ms each) | 15 % |
| *building blocks:* forward kinematics | 45 343 | 5.0 s | 7 % |

So **~70 % of the time is the continuous checks** (the soft clearance at
every sample; inside it the 15-axis box test `box_gaps` and NumPy's `cross`
dominate), the rest mostly IK. The tree searches are bounded by wall-clock
budgets, so a faster check means more search per budget rather than less
time; the shortcuts are fixed work (each rechecks the whole remaining path).

### 9.6 The whole pipeline in run_solver.py; why execution aborted

`run_solver.py` now runs Prokop's planner (`--planner prokop`, default) and
the camera tracker (`--tracker camera`, default), speed 0.5, `--axis/--offset/
--yaw` for a chosen placement. Default run (seed 0): plan in 35–48 s, but the
executor **aborted at step 7 of 9** every time ("correction of −4.9 … −7.2°
not acceptable"; the user also saw 16.2°), the ring 116 mm above the pad.

Diagnosed with `ExecutionProbe` in `run_solver.py` (simulator only: tracker
estimates vs the truth, link call timings, step arrivals vs the truth, which
check of the correction failed) and `experiments/execution_replay.py` (re-renders
every tracker frame offline and feeds it through the tracker):

* the abort: rotating step 7's target by the correction put **joint 3 outside
  its limits**; the correction came from a **tracker error of −4 to −5°**;
* the measurements are accurate (< 1 mm, < 1°, single marker included; a
  −1.4…−1.7° blip for ~0.4 s when the second marker reappears); the tracker's
  logic is not the cause;
* **the cause: the simulated camera returns late while the arm moves** — up
  to 0.34 s (stamp late = the wait for the collision monitor before rendering;
  transfer ~1 ms). The tracker stamps return − 0.11 s, so the image is older
  than it believes: up to 4.3° of apparent lag. Rendered at the true times,
  the replay reproduces the live errors exactly (−4.22°, −3.66°). At rest the
  stamps are ~12 ms late. A simulator artefact (no monitor on the real
  camera), but the real robot would fail the same way with late images;
* the tracker takes a measurement within 6° of its prediction entirely (no
  blending) and it becomes the reference, so errors walk in steps; it does
  not update the speed (≈ 0.15° over a 7 s traversal);
* the robot arrives 0.07–0.12 s late per step (~1.2–1.7° of correction
  missed); robot commands waited up to 385 ms on the simulator link behind
  camera grabs (mostly absorbed: the executor sends ahead).

**Fixes** (user: do all three, then the executor):

1. *Simulated camera keeps its timing:* `grab_image` stops waiting for the
   collision monitor `RENDER_RESERVE` (0.09 s) before it must return →
   stamps at most 30 ms late (was 338 ms). (`test_grab_image_timing_and_age`
   failed once in a full-file run under load, then passed 3 alone + 2 full runs.)
2. *Tracker frozen for the traversal* (`FreezingTracker` / `FreezingExecutor`
   in `run_solver.py`; Vojtěch's file unchanged): the camera tracker measures
   until LEAD (0.5 s) before the start, is then stopped (no grabs compete
   with robot commands; late images cannot pull the angle off), and the
   traversal follows the fitted model re-anchored on its last angle (with the
   ModelTracker's zero uncertainty: the camera tracker's placeholder speed
   uncertainty, 0.8°/s, would cross the executor's 3° limit after ~4 s).
   Stopping it at the start itself delayed step 0 by ~0.09 s (it waits for
   the grab in progress), hence LEAD before.
3. *Lateness:* the timing model's command delay equals the simulator's
   (0.12 s); the 0.1 s came from commands waiting behind grabs (fixed by 2).

**Executor bugs** (`solver/executor.py`, found with the replay and an
offline replay of each step's correction rounds):

* the fixed point kept the arrival of the round before the configuration it
  sent → with growing corrections it believed itself late and over-corrected
  (step 5: +4.4° applied, −4.0° needed);
* repeating x ← f(x) does not converge: a correction changes the move's
  duration — at some steps f creeps (+1.1 → +4.5° after 12 rounds), at others
  it swings between two values (+6.1° ↔ −1.0°); **now solved as f(x) = x by
  secant steps**, warm-started from the previous step's correction;
* the numerical IK of a correction takes ~35 ms, which silently delayed every
  command (arrivals −35 ms off) → **the send time is fixed COMPUTE_BUDGET
  (80 ms) ahead, the step solved for it and sent exactly then** (arrivals
  within 3 ms of the simulator); new test
  `test_corrected_arrivals_are_timed_for_the_configuration_sent` (tracker
  speed 15 % off).

**Result: default pipeline SUCCESS** (seed 0): plan 40 s (15 waypoints,
traversal 12.4 s), driven in 12.8 s, ring 24.4 mm above the pad; tracker
−0.18° at the freeze, every correction ≈ 0 (the frozen model and the plan
agree; the truth needed ≤ 0.15°).

### 9.7 Twenty random placements, the 120 s limit, margins from the uncertainties

`experiments/batch_summary.py` over `run_solver.py --seed 1..20` (spin alternating):

| Batch | what changed | success | unreachable: perception | planning | aborted |
|---|---|---|---|---|---|
| 1 | (9.6 fixes) | 8 (5 of them > 120 s) | 9 | 2 (no approach) | 1 (freeze skipped) |
| 2 | fixed time budget (planning ≈ 55 s, worst-case reserve) + approach in the workers | 1 | 9 | 10 | 0 |
| 3 | dynamic deadline (a plan counts if it ends ≤ 117 s) + uncertainty margins + ring approach | 3 (96, 108, 114 s) | 9 | 8 | 0 |

* **Perception (9/20 every time)**: the edge placements (|y| ≳ 0.2 or x ≳ 0.6):
  markers in 0–3 of 8 images. Correct by the user's rule (no markers = outside
  the working area); the random placements cover more than the camera sees.
* **The 120 s limit** (user: program start → hoop < 4 cm, threaded): times per
  phase of batch 1: perception 8 s, planning 29–97 s, execution 21–43 s (approach
  + tracker ~10 s, the wait for the start angle 3–27 s, traversal 6–13 s); 5 of 11
  successes were over. Now: one deadline from the program start (the simulator's
  own start-up excluded), planning until a plan could still end in time, the plan
  that ends soonest chosen (singularity score as tie-break), the executor stops at
  the deadline.
* **Freeze fix** (batch 1 seed 5): the camera tracker is frozen unconditionally at
  the start of the final wait.
* **Approach counted in the plan choice** (seeds 7, 18: the first plan had no
  approach) and rebuilt (user design): the turning task as 6 coaxial **rings**
  (turntable, base + the puzzle's foot, stem, vertical run + jog, bend, top bar;
  inner and outer radius, height), the long move home → a **pre-pre-entry pose**
  checked against the rings only (no table, no cell), the last move into the
  waiting pose with the full swept check. Found on the way: seed 18's waiting
  pose was 3° from two joint limits, so every pre-entry pose with the same hoop
  orientation was out of limits → the pre-pre-entry pose fixes only the ring's
  position; candidates tried by the shortest last move. Approach 20–58 s → 2–4 s.
* **Margins from the uncertainties** (`solver/uncertainty.py`, k = 2, ±2 mm
  dimensions, user): angle sd 0.88° (tracker 0.8° one marker + speed sd × 12 s),
  hoop–puzzle **5.5–6.8 mm** (was 3), every configuration **≥ 5.5° from the joint
  limits** (room for the executor's corrections; was none).
* **What limits now: planning speed.** With the margins and 6 workers sharing the
  CPU with the simulator, one end angle takes 65–91 s (standalone, 3 mm margin:
  23–56 s): plans arrive too late (predicted ends 126–150 s) or not within 98 s.
  Next: the plan library / seeding trees (user idea, section 10).

## 10. Open ideas (not built yet)

- **Parallel end angles** (planning time): one process per end angle, first plan wins.
- **Null-space steps** (user): with only the ring centre fixed the arm can
  reconfigure (tilt, tilt direction, roll) while the ring stays put, e.g. to
  lift the forearm out of the top bar's way.
- **Start angle by the top bar's direction during the descent** (sweep started,
  2 angles done: at offset 0 the bar points 20° from the robot at t = 5 s).
- **Seeding trees / plan library** (user; now the main limit, 9.7): keep the trees of
  precomputed placements; load the nearest, prune what collides, grow from it. Cheaper
  first step: repair and recheck the nearest stored final path. Rotation about the robot axis and the free start
  time reduce a placement to 3 parameters (axis distance, offset length,
  offset direction relative to the bar) + spin direction; precompute, refine
  the nearest online.
- **Kalman phase tracker** (P3.1, Vojtěch); **uncertainty margins** k·σ from
  it, plus the ruler-measured dimensions (the 3 mm hoop margin is a placeholder).
- **Blending at corners**: the controller's `COORDISCONT` (not exposed by
  `move_to_q`) might let the robot pass waypoints without stopping — to be
  measured on the robot.

## 11. Scripts to reproduce

All run from the repo root (`uv run python Prokop/experiments/<script>`).

| Script | What |
|---|---|
| `static_puzzle.py` | Natka's planner (copy) on a static facing grid |
| `static_plan.py [--axis X Y] [--rrt N]` | our planner, static, + exact replay |
| `validate_soft_collision.py` | the soft model vs the exact meshes |
| `show_plan.py` | perception → plan → run in the viewer (standalone simulator) |
| `reachability.py` | the 25-placement sweep (section 8) |
| `spin_plan.py [--seed N \| --axis X Y] [--tracker fitted\|truth]` | spinning: perception, plan, executor, evaluation |
| `spacetime_plan.py [YAW SPEED AXIS OFFSET START]` | the space-time tree on a model, with rejection tallies |
| `follow_bar.py [SPEED]` | following the top bar step by step, frozen vs turning |
| `checkpoint_timeline.py [START_OFFSET]` | free poses per checkpoint against the clock |
| `first_spin_move.py` | the first spinning move traced |
| `backward_plan.py [YAW SPEED AXIS OFFSET [N]]` | the backward planner on a model, per end angle, with rejection tallies |
| `execution_replay.py RUN/execution.pkl` | a simulated execution offline: tracker frame by frame (re-rendered), stamp lateness, step timing |
| `profile_breakdown.py RUN.prof` | where a profiled planning run's time goes, by planner block |
| `spin_sanity.py AX AY OX OY [YAW_DEG [DIR [SPEED]]]` | spinning plan on the true model + exact replay with the puzzle turning |
| `../run_solver.py --planner copy` | Natka's planner (copy) in the full pipeline |
