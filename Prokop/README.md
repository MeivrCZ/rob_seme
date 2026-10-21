# CRS93 hoop task: simulation

Prokop's part of the ROB1 semester project (CTU FEE): a simulation of the CRS93
station, so that the team can develop and test without booking the robot. The
task is variant D: the robot moves a hoop along puzzle D standing on a spinning
turntable, guided by the camera above the table.

The core is `CRS93Sim`, a drop-in replacement for `ctu_crs.CRS93`. A script
written against the robot's interface runs in the sim and on the real robot by
changing one line. The sim behaves like the real station as measured there:
joint speeds, delays of the serial line, blending of queued moves, the hard
home pose, and the cell geometry around it.

![Hoop seen by the station camera, with the camera model's predictions](docs/img/survey_hoop_annotated.jpg)

## Quick start

From the repo root:

```bash
uv sync
uv run Prokop/demo_sim.py        # simulated robot homes, moves, traces a square
uv run Prokop/view_models.py     # interactive scene: joint sliders, turntable
uv run pytest Prokop/tests       # ~100 tests, ~1 min
```

Both programs serve the scene at http://localhost:8080.

```python
from simulation import CRS93Sim     # from ctu_crs import CRS93
robot = CRS93Sim()                  # robot = CRS93()
robot.initialize()                  # hard home, then q_home
robot.move_to_q(q)                  # returns after ~25 ms, like the real call
robot.wait_for_motion_stop()
```

The extra keyword arguments exist in the sim only:
- `q0`, `port`, `time_scale` (sim seconds per wall second), `keep_alive`;
- the hardware delays `command_delay`, `send_delay`, `query_delay`,
  `status_delay`, `camera_delay`. They default to the values measured on the
  CRS93; pass `0` for an ideal robot.

## How the simulation works

- **Same kinematics.** `CRS93Sim` subclasses `ctu_crs.CRS93`: `fk`, `ik`, joint
  limits, `q_home` and the unit conversions are the driver's own. Only the
  calls that talk to the MARS control unit are replaced.
- **Motion** (`simulation.Segment`).
  - Every `move_to_q` is a coordinated move: all joints follow one trapezoidal
    speed profile, with the real speed settings. Targets are rounded to encoder
    counts like the real commands.
  - Queued moves run one after another. A move continuing in exactly the same
    joint-space direction is blended into the previous one; any other starts
    from standstill. That's the real control unit's behaviour (E6).
- **Delays.** `move_to_q` blocks for the serial write, and the motion starts
  `command_delay` later. `get_q` and `in_motion` block for their round trips
  and return the state from its middle.
- **One clock.** The robot, the delays and the turntable spin all run on one
  sim clock, scaled by `time_scale`. The viewer draws the whole scene at one
  instant.
- **Scene** (`view_models.SceneViewer`). Built from:
  - the robot URDF (generated from RoboDK, kinematics checked against
    `ctu_crs`),
  - the hoop, base, puzzles and turntable (from the CAD files of the hardware
    repo),
  - the measured cell (`cell.py`).
- **Collision checking.** `collision.ArmChecker` gives the clearance of the arm
  and hoop to the table and the cell, for a pose or along the joint-space path
  of a move.
- **Not simulated yet:** the camera image (`grab_image` raises), the turntable
  and puzzle as obstacles for the arm, self-collision, the control unit's queue
  limit, and the small overshoot at the end of a real move.

## Documentation

| Document | Contents |
|---|---|
| [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) | every measured, estimated or assumed value: what, how, where in the code |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | the experiments at the station (E1–E9): why, what, results, what changed |
| [docs/ROBOT_CHECKS.md](docs/ROBOT_CHECKS.md) | safety rules, running the tools at the station, open checks |
| [NOTIFY.md](NOTIFY.md) | notes for the team about changes that affect their code |

The modules and tools have Google-style docstrings; `--help` on any tool prints
its usage.

## Layout

| Path | Contents |
|---|---|
| `simulation.py` | `CRS93Sim`, the simulated robot |
| `view_models.py` | `SceneViewer` (the Viser scene) and the interactive viewer |
| `cell.py` | the static cell: table, work area, cage, camera rail, camera (pose, intrinsics) |
| `collision.py` | `ArmChecker`: clearance of the arm + hoop to the table and the cell |
| `placement.py` | task models; random turntable / base placement with tip-over and clearance checks |
| `demo_sim.py` | smoke test of the simulation |
| `models/` | URDFs and STL meshes: robot with adapter, hoop, base, puzzles A–E, turntable |
| `tools/robodk2urdf.py`, `step2stl.py`, `mesh_repair.py` | model generators (robot URDF from RoboDK, meshes from STEP) |
| `tools/log_motion.py`, `camera_survey.py` | measurements on the real robot (also run against the sim) |
| `tools/analyze_motion.py`, `analyze_survey.py` | analysis of the recorded experiments |
| `experiments/` | recorded data: `motion_logs/real/`, `camera_survey/<date-time>/` |
| `docs/` | the documents above, `img/` |
| `tests/` | pytest suite: models, cell, collision, placement, sim interface, viewer, tools |

## Conventions

- **Units:** SI (m, rad, s); degrees only in printouts and docs.
- **Frame:** the `ctu_crs` robot base frame. x forward (where the arm points
  in `q_home`), y left, z up; the table lies at z = +1 mm.
- **Joint values** are exactly the `q` of `robot.move_to_q()`; the URDF
  `flange` frame equals `robot.fk(q)`.
- **Hoop:** in `q_home` it points up (the adapter is turned 180° on this
  station).

## Interfaces with the rest of the team

- **Camera (Vojtěch):** outputs poses (and matrices if needed) and should accept
  an input image. Then either the real camera image or a rendered image from the
  sim's camera (`cell.CAMERA_T`, `CAMERA_K`; not implemented yet) can be fed in.
  A rendered image is good for finding logic errors, not for tuning detection.
- **Path planning (Natka):** outputs flange poses and joint configurations that
  go straight into `move_to_q`, on the sim or the robot. Mind the blending rule:
  bending chains of via points run stop-and-go (E6).

## Owner

`Prokop/` is maintained by Prokop; the other team members' directories are
theirs. Changes that affect their code are announced in `NOTIFY.md`.
