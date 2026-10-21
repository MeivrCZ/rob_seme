# Prokop: simulation of the CRS93 hoop task

Run everything from the repo root with `uv run`.

| Command | What it does |
| --- | --- |
| `uv run Prokop/demo_sim.py` | smoke test: simulated robot homes, moves, traces a square |
| `uv run Prokop/view_models.py` | interactive scene: joint sliders, turntable |

Both serve the scene at http://localhost:8080.

## Simulated robot

`simulation.CRS93Sim` is a drop-in for `ctu_crs.CRS93` (same methods and
arguments) that moves the robot in the Viser scene:

```python
from simulation import CRS93Sim     # from ctu_crs import CRS93
robot = CRS93Sim()                  # robot = CRS93()
```

Extra keyword arguments (simulation only): `q0`, `port`, `time_scale`,
`keep_alive` and the hardware delays `command_delay`, `send_delay`,
`query_delay`, `status_delay`, `camera_delay` (defaults measured on the CRS93;
0 = ideal). Queued moves blend only when they continue in the same joint-space
direction, as on the real robot. What is and is not simulated is listed in the `simulation.py`
docstring.

## Layout

| Path | Contents |
| --- | --- |
| `simulation.py` | `CRS93Sim`, the simulated robot |
| `view_models.py` | `SceneViewer` (the Viser task scene) and the interactive viewer |
| `cell.py` | the static work cell: table, work area, cage, camera rail, camera |
| `collision.py` | clearance of the arm + hoop to the table and the cell (pose, joint path) |
| `placement.py` | random turntable / base placement with tip-over and clearance checks |
| `demo_sim.py` | smoke test of the simulation |
| `models/` | URDFs and STL meshes (robot, adapter, hoop, base, puzzles, turntable) |
| `tools/` | model generators: `robodk2urdf.py` (robot URDF), `step2stl.py` (meshes); measurements on the real robot: `log_motion.py` (motion logs, real or sim), `camera_survey.py` (camera timing and image delay) |
| `ROBOT_CHECKS.md` | what to measure on the real robot to calibrate the simulation |

Measured: table height, work area, cage, camera rail and camera position
(`cell.py`), `BASE_FOAM` (placement.py), turntable size. Placeholders still to be
replaced by measurements: camera intrinsics (`cell.py`, focal length estimated
from the work area), motion profile units (simulation.py). Turntable period `SPIN_PERIOD` = 28.0 s
(view_models.py), measured, the same both ways.
Seen on the robot: the adapter is turned by 180 deg, the hoop points up in
q_home (`ADAPTER_TURN`, tools/robodk2urdf.py); the hard home ends where the
control unit zeroes the encoders (simulation.py).
Decided, not measured: `HOOP_MOUNT` (placement.py) stays the nominal CAD seating
(triangle in triangle); the hoop can sit slightly off on its magnets, which the
team accepts and ignores unless it proves to be a problem.
