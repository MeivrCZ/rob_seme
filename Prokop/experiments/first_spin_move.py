"""Why the first spinning move (checkpoint 0 -> 1) fails: trace the check."""
import sys
from dataclasses import replace
sys.path[:0] = ["Prokop"]
import numpy as np
from ctu_crs import CRS93
from solver import planner
from solver.contracts import TurntableModel, OMEGA
from solver.planner import plan_rrt, start_angles, _q_at, check_moves
from solver.hoop_pose import flange_target
from solver.soft_collision import SoftScene
from solver.timing import RobotTiming
from solver.wire import wire_path

kin = CRS93(tty_dev=None); timing = RobotTiming(kin, 0.2)
model = TurntableModel(axis_xy=np.array([0.5954, 0.1826]), base_offset=np.array([0.028, -0.035]),
                       base_yaw=0.0, direction=1, omega=OMEGA)
theta = start_angles(model, count=1)[0]
scene = SoftScene(model)
static = plan_rrt(kin, timing, replace(model, omega=0.0), theta, 3, scene=scene)
print("static:", static.collision_free, "face", static.face, "entry", static.entry,
      "tilts", np.round(np.rad2deg(static.orientations[:, 1])).tolist(), flush=True)
path = wire_path(0, static.entry, 3)
angle = lambda t: theta + model.omega * t
q0 = static.qs[0]
roll, tilt, dirn = static.orientations[1]
for frozen in (True, False):
    t_arr = 0.0
    for _ in range(3):
        target = flange_target(model.checkpoint_T(path, 1, theta if frozen else angle(t_arr)),
                               roll, static.face, tilt, dirn)
        sols = [q for q in kin.ik(target) if kin.in_limits(q)]
        goal = min(sols, key=lambda s: np.abs(s - q0).max())
        t_arr = timing.move_duration(q0, goal)
    ang = (lambda t: theta) if frozen else angle
    c_goal = scene.clearance(goal, ang(t_arr))
    chk = check_moves(scene, ang, 0.0 if frozen else model.omega, timing, np.array([q0, goal]),
                      np.array([-1, 0]))[0][0]
    print(f"{'frozen ' if frozen else 'turning'}: move {t_arr:.2f} s, goal jump "
          f"{np.rad2deg(np.abs(goal - q0).max()):.1f} deg; goal clearance {1000*c_goal.distance:.1f} mm "
          f"({c_goal.pair}); move: {chk.samples} samples, least {1000*chk.clearance:.1f} mm, hits {chk.hits} "
          f"at t {chk.t_hit}", flush=True)
    # clearance along the move
    _, moves = timing.schedule(q0, [goal])
    for t in np.linspace(0, moves[0].t_end, 9):
        c = scene.clearance(_q_at(moves, q0, t), ang(t))
        print(f"    t {t:4.2f} s: {1000*c.distance:6.1f} mm {c.pair}, self {1000*c.self_gap:.1f} mm")
