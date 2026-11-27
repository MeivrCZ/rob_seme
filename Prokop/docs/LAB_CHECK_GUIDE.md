# Lab check guide (for a Claude instance at the station)

Written 2026-11-26 for the lab session on 2026-11-27. You are helping Prokop at the
CRS93 station (or on another PC) **without the conversation history** that produced
the solver. This page is that history's short form: what the project is, the rules,
and what to check, step by step, with the expected values and what to do when they
differ.

## The project in five lines

ROB1 semestral project, variant D: a hoop held by the CRS93 arm is threaded over a
bent square bar (the "puzzle") that stands on a base on a **spinning turntable**
(28 s per turn). The camera above the work area sees ArUco markers on the base. The
pipeline is `Prokop/run_solver.py`:

1. **perception**: a whole turn of images (29, 1 s apart), `fit_orbit` gives the turntable's axis, the base's offset, its direction and speed;
2. **planning**: `solver/planner.py` plans backwards in time from several end angles in parallel;
3. **execution**: `solver/executor.py` queues the moves. Vojtěch's camera tracker keeps correcting the phase until 0.5 s before the start.

The whole attempt must end in **under 120 s** from the start of the program, with the
hoop less than 4 cm above the base.

Everything in the solver was developed and tested **only in the simulator**
(`Prokop/simulation.py`). The October lab session measured the cell, the
turntable period, the motion profile, the serial delays and the camera timing
(E1–E9 in [EXPERIMENTS.md](EXPERIMENTS.md), values in [MEASUREMENTS.md](MEASUREMENTS.md)).
**What this session checks** is everything else the solver assumes about the real
station.

## Rules (from Prokop, all standing)

- **Never home the robot from a script.** Hard homing drives the joints to their switches with no collision checking: from a bad pose it can wreck the arm. Prokop homes by hand (below). Every tool refuses to move unless the robot is in `q_home`.
- **Someone at the e-stop for every motion.** Every motion in `lab_check.py` asks for Enter first. Never answer those prompts yourself: let the operator press Enter. Run motion commands only when Prokop says so.
- **Never modify `Natka/` or `Vojtěch/`.** Those are the teammates' folders. Prokop works only in `Prokop/`.
- **Prokop designs the algorithms; you implement and test.** No unsolicited architecture changes.
- **When something fails or a value is off, present the findings and ask.** Don't change the solver's constants or its approach on your own: write down the measured value, which constant it corresponds to (tables below) and your suggestion.
- **Commit and push only when asked.**
- **Keep the user's personal details out of anything external.**

## Setup

```bash
cd <repo>                      # git@github.com:MeivrCZ/rob_seme.git, branch main
git pull && uv sync            # numba is a dependency now (the collision checks)
uv run Prokop/tools/lab_check.py dims        # no robot: checks the install
```

**Optional: try the tools against the simulator first** (5–10 min, no robot). Every
check except `dims` takes `--sim` and then runs against a simulator at
`http://localhost:8080`, with no prompts:

```bash
uv run Prokop/tools/lab_check.py calibrate --sim
uv run Prokop/tools/lab_check.py timing --sim
uv run Prokop/tools/lab_check.py turntable --sim
uv run Prokop/tools/lab_check.py fov --sim --images 4
```

The simulator's reference values are in the tables below. On a WSL machine, run heavy
jobs under a memory cap, otherwise an out-of-memory freezes WSL:
`systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 uv run ...`.

**Homing, by hand, done by Prokop** (from `ROBOT_CHECKS.md`). Only Prokop does this,
looking at the arm:

```python
uv run python
>>> from ctu_crs import CRS93; import numpy as np
>>> r = CRS93(); r.initialize(home=False)
>>> r.hard_home(); r.soft_home(); print(np.rad2deg(r.get_q()).round(1))   # 0 0 -45 0 -45 0
>>> r.close()
```

Every check writes `Prokop/experiments/lab/<time>_<check>/`, containing `log.txt` (everything
printed), `summary.json` and the images.

## The checks, in order

| # | Check | Robot | Takes | Needs |
|---|---|---|---|---|
| 1 | `dims` | none | 10 min (caliper) | the puzzle, the hoop, the base |
| 2 | `calibrate` | moves | 3 min | the work area **empty** (turntable removed) |
| 3 | `timing` | moves | 2 min | the arm's surroundings free (it stays up near `q_home`) |
| 4 | `turntable` | no motion | 1.5 min | the turntable spinning, the base and puzzle on it; the calibration from 2 |
| 5 | `fov` | no motion | 5 min | someone moving the base around the work area |
| 6 | dry run of the solver | no motion | 2 min | as 4 |

### 1. `dims`: the model's dimensions

```bash
uv run Prokop/tools/lab_check.py dims
```

The command prints a table: what to measure, the model's value, and how to measure
it. Measure each item with a caliper or a tape and write the values next to the
table, in the session notes (see "Recording").

- **Pass:** every item within ±1 mm. The safety margins (`solver/uncertainty.py`, `DIMENSION_SD`) assume each dimension is right to its 1 mm rounding (sd 1/√12 mm, four parts in quadrature).
- **Off by more than 1 mm:** report it.
  - The puzzle's numbers are constants in `Prokop/solver/wire.py` (`BAR_HALF`, `TOP_Z`, `OPEN_END_X`, `ARC_RADIUS`, `JOG_*`).
  - The hoop's numbers are `RING_*` in the same file.
  - The marker height is `MARKER_HEIGHT_ABOVE_TABLE_M` in Vojtěch's `Homografy/turntable_orbit.py`. That one isn't ours to change: tell Prokop.
  - The two "at `q_home`" rows check the adapter and hoop model (`solver/hoop_pose.py`) together with the kinematics. The model puts the ring's centre at 977 mm above the table and 323 mm in front of the robot's axis.

### 2. `calibrate`: the camera calibration (homography)

```bash
uv run Prokop/tools/lab_check.py calibrate
```

Prompts:
1. "Remove the turntable, the base and the puzzle": the hoop goes down to 75 mm above the table.
2. The route summary, then Enter to start.

What happens:
- The hoop is held flat. Its centre visits a 3 × 3 grid (x 0.39–0.58 m, y ±0.15 m) at 75, 130 and 190 mm above the table, at speed 0.2. The route is checked against the table and the cell first; if it would collide, nothing moves.
- At each point, the ring's centre comes from the forward kinematics of the measured joints, and its pixel from Vojtěch's hole detector (`rob_crs93_v4/hoop_detection.py`).
- A 3×4 projection matrix is fitted (Vojtěch's `Homografy/homography.py`, with RANSAC and 25 % of the points held out). It is saved to `Prokop/data/calibration/homography_<time>.json`: **this file is the `--calibration` argument for steps 4–6.**

Pass:
- at least 15 of 27 points recorded;
- on the held-out points, RMS < 3 px and < 2 mm in the table plane.

The simulator gives 21 points, 0.12 px and 0.03 mm. There, the hole detector misses the 6 points nearest the camera's far edge (x 0.58 m, the upper two heights).

If it fails:
- **Few points found:** the lighting, or the ring is out of view. Look at `point_XX.png`.
  - `--ring-fallback` also uses a coarser blue-ring fit, 2–13 px off in the simulator. Use it only if the hole detector alone gives fewer than 10 points, and say so in the notes.
- **A large RMS:** look at which points RANSAC rejected. A point that is off by a lot usually means the detector picked the wrong circle (check its image).
- The log also prints where the **simulator's camera model** predicts each point. Large, systematic differences between that and the detection are expected (the camera model in `Prokop/cell.py` is an estimate) and don't matter for the solver, which uses only the homography. Still, note the typical offset.

### 3. `timing`: does the executor's timing model hold on the real robot?

```bash
uv run Prokop/tools/lab_check.py timing
```

Two prompts, one per run. The arm makes 5 small moves near `q_home` (joints 1, 4
and 6 only; at most 12° on joint 1; the arm stays up), at speed 0.5, the solver's.
Each move is sent 0.15 s before the previous one is predicted to end, exactly like
the executor sends a traversal. `RobotTiming.schedule` (`Prokop/solver/timing.py`)
predicts every move. The check measures how late each real move is, as the median
over all of its `get_q` samples of (sample time − predicted time at that progress).

Run 1 is without the camera. Run 2 has a second thread grabbing images all the time.
That is what happens during a real attempt, because the camera tracker runs until
0.5 s before the start. Run 2 answers the open question: **does a camera grab hold up
`move_to_q` / `get_q`** (a held GIL, a blocking driver)?

| Measured | Simulator | Pass | If not |
|---|---|---|---|
| lateness, run 1 | +10 ms, sd 2 ms | \|median\| ≤ 30 ms, the same for all 5 moves | a constant offset → `COMMAND_DELAY` (0.12 s) in `solver/timing.py` is off by about that much. Report the median; don't change it yourself |
| `move_to_q` duration | 26–47 ms | ≤ 60 ms (E5: ~25 ms) | — |
| `get_q` period | 54 ms | 40–80 ms (E5: 51 ms) | — |
| run 2 against run 1 | not comparable: the simulator's process passes every call through one pipe, so a 300 ms render holds everything | lateness and `move_to_q` within ~20 ms of run 1 | the camera holds the robot's commands: report the numbers. The fix (grabbing in another process, for example) is Prokop's call |
| grab duration | 293 ms (rendering) | E7 measured the real one; just note it | — |

At speed 0.5 the turntable turns 12.9°/s. So 30 ms of timing error moves the puzzle
by about 0.4°, roughly 0.7 mm at the hoop. That is well inside the margins (5.5–7
mm), and it is why 30 ms is the pass limit.

### 4. `turntable`: perception on the real spinning turntable

```bash
uv run Prokop/tools/lab_check.py turntable --calibration Prokop/data/calibration/homography_<time>.json
```

Set-up: the turntable at the slowest speed, the base with the puzzle on it, somewhere
in the work area. Nothing moves the robot. The check:

1. takes 10 images 1 s apart, measures the base in each (Vojtěch's `measure_base`) and fits the orbit (`fit_orbit`), the same as phase 1 of `run_solver`;
2. runs Vojtěch's `CameraPhaseTracker` for 30 s and compares its phase with the fit's prediction.

| Measured | Simulator | Pass | Constant / if not |
|---|---|---|---|
| detections | 10 of 10 | ≥ 8 | lighting or the field of view: see `fov` |
| period | 28.00 s | 27.9–28.1 s (E3 stopwatch: 28.00 ± 0.05 s) | `SPIN_PERIOD` in `solver/contracts.py` (only the fallback; the solver uses the fitted speed) |
| xy residual | 0.18 mm | < 2 mm | feeds the margins (`solver/uncertainty.py`); larger is allowed but the margins grow: note it |
| yaw residual | 0.09° | < 1° | as above |
| tracker − fit drift | 0.005°/s | < 0.03°/s (`MIN_OMEGA_SD`) | the fitted speed is off; the tracker fixes it until the freeze, after that it's drift. Report it |
| markers lost | 1 (the puzzle hides one) | any; but "the latest measurement at most N s old" should stay < 5 s | — |

Also check by eye that the printed axis matches where the turntable really stands.
The positions are in mm in the robot's frame: x forwards from the robot's axis, y to
the left.

### 5. `fov`: where the camera sees the base

```bash
uv run Prokop/tools/lab_check.py fov --calibration Prokop/data/calibration/homography_<time>.json
```

Every second it prints which markers (IDs 1, 2) it sees and where the base is. Move
the base (turntable stopped, or by hand without the turntable) along the edges and
into the corners of the work area. In the robot's frame, the work area is
x 0.337–0.637 m and y −0.2425 to +0.2425 m (`cell.WORK_AREA_XY`). Ctrl+C stops it.

Write down where the markers stop being detected. In the simulator, 9 of 20 random
placements were never seen during the run's perception phase. Those are the
"unreachable at perception" failures, and whether the real camera sees more matters
for which placements the solver can handle at all.

### 6. Dry run of the whole solver (no motion)

```bash
uv run Prokop/run_solver.py --real --calibration Prokop/data/calibration/homography_<time>.json --stop-after 2
```

The set-up is as in step 4. The command runs perception and planning on the real
data and stops before any motion. Read the log for:
- the orbit fit;
- the margins line ("hoop-puzzle … mm, joint limits … deg");
- the reachability line;
- the plan ("predicted end … s"), which must stay under 120 s minus the margin;
- the planning time on the station PC. Plans use as many worker processes as there are end angles, capped by the CPU count.

On this laptop's simulator runs, planning had a median of 35 s and the whole run
ended at a median of 66 s (20 seeds, 9 of the 11 visible placements succeeded).

**The real execution (`--stop-after 3`, the default) is NOT part of these checks.**
`--real` is marked untested in the code. Run it only if Prokop explicitly asks,
after 1–6 pass, with someone at the e-stop.

## Recording

1. Append a section to [EXPERIMENTS.md](EXPERIMENTS.md), the next number after E9, for example **E10: lab checks 2026-11-27**. In the style of E1–E9: what was measured, how, the values, and what they mean for the code.
2. Measured values that replace estimates go into [MEASUREMENTS.md](MEASUREMENTS.md), in the relevant table, with "measured" and the experiment number.
3. If any value differs from the code's constant, **list it for Prokop**: the measured value, the constant (file and name) and the suggested new value. **Don't change the constants yourself.**
4. Results are in `Prokop/experiments/lab/` and the calibration in `Prokop/data/calibration/`. Commit them only when Prokop asks.

## If something goes wrong

| Symptom | Likely cause / what to do |
|---|---|
| "the robot is N deg from q_home" | not homed, or moved: Prokop homes by hand (above) |
| "the calibration route collides in the soft model" / "the timing chain collides" | the robot's `q_home` differs from the model's? Print `np.rad2deg(r.get_q())`, report; don't loosen the check |
| `ModuleNotFoundError` for a teammate's module | `git pull`; the paths are set at the top of `lab_check.py` (`Vojtěch/ArucoFinder`, `Vojtěch/Homografy`, `Natka`, and `Vojtěch/rob_crs93_v4` last, since it has its own `homography` module) |
| the camera returns no image | the Basler is in use by another process (close Vojtěch's / Natka's tools); `ctu_crs` opens it on the first grab |
| "coordmv queue full - waiting" printed | the control unit's queue limit (never seen before): note it, with which check |
| a motion looks wrong | e-stop. Then the log in the run folder; report before retrying |

## Where things are

| | |
|---|---|
| `Prokop/tools/lab_check.py` | these checks |
| `Prokop/run_solver.py` | the whole pipeline (`--real`, `--calibration`, `--stop-after`) |
| `Prokop/solver/` | planner, executor, timing model (`timing.py`), soft collision model, margins (`uncertainty.py`), puzzle geometry (`wire.py`) |
| `Prokop/simulation.py`, `Prokop/cell.py` | the simulator and the cell's model (table, cage, camera) |
| `Prokop/docs/PLANNER_EXPERIMENTS.md` | everything tried on the planner, and why |
| `Prokop/docs/ROBOT_CHECKS.md` | the older station tools (`log_motion.py`, `camera_survey.py`) and the open checks |
| `Prokop/NOTIFY.md` | notices for the teammates |
| `Vojtěch/` | camera tracker, ArUco, homography (read only for us) |
