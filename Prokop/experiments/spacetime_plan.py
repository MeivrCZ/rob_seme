"""plan_spacetime (the spinning puzzle's space-time tree) on a model, without the simulator.

Prints a line per checkpoint search and, per stage, why the tree rejected
extensions (which bodies collide). Run from the repo root:

    uv run python Prokop/experiments/spacetime_plan.py [YAW SPEED AXIS OFFSET START_OFFSET_DEG]
    e.g. ... spacetime_plan.py 0 0.5 0.487,0.0 0.02,0.0 0     (central placement)
         ... spacetime_plan.py 0 0.5                         (seed 1's placement)
"""
import sys, time
sys.path[:0] = ["Prokop"]
import numpy as np
from ctu_crs import CRS93
from solver import planner
from solver.contracts import TurntableModel, OMEGA
from solver.timing import RobotTiming

if __name__ == "__main__":
    kin = CRS93(tty_dev=None); timing = RobotTiming(kin, float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
    yaw = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    axis = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.5954, 0.1826]
    offset = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.028, -0.035]
    print(f"axis {axis}, base offset {offset}", flush=True)
    model = TurntableModel(axis_xy=np.array(axis), base_offset=np.array(offset),
                           base_yaw=yaw, direction=1, omega=OMEGA)
    t0 = time.monotonic()
    planner.PASS_LOG = lambda s: print(f"[{time.monotonic() - t0:5.0f} s] {s}", flush=True)
    offset = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    theta = planner.start_angles(model, count=1)[0] + np.deg2rad(offset)
    # Where the top bar's overhang (puzzle +x) points at t = 5 s, relative to the robot.
    T5 = model.puzzle_T(theta + model.direction * model.omega * 5.0)
    rel = np.degrees(np.arctan2(T5[1, 0], T5[0, 0]) - np.arctan2(-T5[1, 3], -T5[0, 3]))
    print(f"start offset {offset:+.0f} deg: at t = 5 s the top bar points "
          f"{(rel + 180) % 360 - 180:+.0f} deg from 'at the robot'", flush=True)
    import collections
    rejects = collections.defaultdict(collections.Counter)
    def note(k, kind, pair, d, t):
        a, b = pair
        a = "hoop_ring" if a.startswith("hoop_ring") else a
        b = "hoop_ring" if b.startswith("hoop_ring") else b
        rejects[k][f"{kind}: {a} - {b}"] += 1
    planner.REJECT_LOG = note
    r = planner.plan_spacetime(kin, timing, model, theta)
    for k in sorted(rejects):
        total = sum(rejects[k].values())
        print(f"rejections on the way to checkpoint {k} ({total}):", flush=True)
        for what, count in rejects[k].most_common(8):
            print(f"   {count:5d}  {what}", flush=True)
    bad = [m for m in r.moves if m.hits]
    print(f"result: {'collision free' if r.collision_free else r.reason or bad}, {len(r.qs) if r.qs is not None else 0} "
          f"waypoints, traversal {r.arrivals[-1] if r.arrivals is not None else float('nan'):.1f} s, "
          f"least soft clearance {1000 * r.min_clearance:.1f} mm ({r.seconds:.0f} s)", flush=True)
