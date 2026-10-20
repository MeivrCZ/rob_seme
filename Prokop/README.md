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
`keep_alive` and the hardware delays `command_delay`, `query_delay`,
`camera_delay`. What is and is not simulated is listed in the `simulation.py`
docstring.

## Layout

| Path | Contents |
| --- | --- |
| `simulation.py` | `CRS93Sim`, the simulated robot |
| `view_models.py` | `SceneViewer` (the Viser task scene) and the interactive viewer |
| `placement.py` | random turntable / base placement with tip-over and clearance checks |
| `demo_sim.py` | smoke test of the simulation |
| `models/` | URDFs and STL meshes (robot, adapter, hoop, base, puzzles, turntable) |
| `tools/` | model generators: `robodk2urdf.py` (robot URDF), `step2stl.py` (meshes); measurements on the real robot: `log_motion.py` (motion logs, real or sim), `camera_survey.py` (camera timing and image delay) |
| `ROBOT_CHECKS.md` | what to measure on the real robot to calibrate the simulation |

Measured: `TABLE_Z`, `WORK_AREA_XY` (area seen by the camera, used by the viable
placement), `BASE_FOAM` (placement.py), turntable size.
Placeholders still to be replaced by measurements: `CAGE_XY` (placement.py, where
the random placement puts the turntable), the robot's steel base plate and the
cage walls (not modelled yet), `SPIN_PERIOD` (view_models.py), motion profile
units (simulation.py).
Decided, not measured: `HOOP_MOUNT` (placement.py) stays the nominal CAD seating
(triangle in triangle); the hoop can sit slightly off on its magnets, which the
team accepts and ignores unless it proves to be a problem.
