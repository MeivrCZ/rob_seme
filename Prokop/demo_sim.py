"""Smoke test of the simulated CRS93: homing, a few joint moves and a square traced
by the flange (IK), then back home.

Run from the repo root:
    uv run Prokop/demo_sim.py               # open http://localhost:8080, press Enter

Uses only the ctu_crs CRS93 interface, so swapping the two robot lines below runs
it on the real robot.
"""

import numpy as np
from simulation import CRS93Sim

robot = CRS93Sim()  # from ctu_crs import CRS93; robot = CRS93()
input("Open the viewer, then press Enter to start ")

robot.initialize()  # hard home (q = 0 joint group by group), then q_home
print("homed, q [deg]:", np.rad2deg(robot.get_q()).round(1))

# Joint moves; queued moves run one after another.
for dq_deg in ([45, 0, 0, 0, 0, 0], [-45, 15, 0, 0, 0, 0], [0, 0, 0, 90, 0, 0]):
    robot.move_to_q(robot.q_home + np.deg2rad(dq_deg))
robot.move_to_q(robot.q_home)
robot.wait_for_motion_stop()

# Square 20 x 20 cm in front of the robot, flange orientation of q_home.
T = robot.fk(robot.q_home)
centre = T[:3, 3] + [0.05, 0.0, -0.15]
q = robot.q_home
for dy, dz in [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1), (0.1, 0.1)]:
    T[:3, 3] = centre + [0.0, dy, dz]
    sols = [s for s in robot.ik(T) if robot.in_limits(s)]
    if not sols:
        raise SystemExit(f"unreachable flange position {T[:3, 3].round(3)}")
    q = min(sols, key=lambda s: np.linalg.norm(s - q))  # closest to the last one
    robot.move_to_q(q)
robot.wait_for_motion_stop()
print("square done, flange [m]:", robot.fk(robot.get_q())[:3, 3].round(3))

robot.move_to_q(robot.q_home)
robot.wait_for_motion_stop()
robot.close()
