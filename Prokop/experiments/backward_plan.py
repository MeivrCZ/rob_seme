"""plan_backward (the spinning puzzle planned backwards in time) on a model, without the simulator.

For each end angle (the top bar away from the robot first): the backward tree's
log, why it rejected extensions (per checkpoint, which bodies), and the result
(the start angle that falls out, the traversal, the soft clearance). Stops at
the first end angle that gives a plan. Run from the repo root:

    uv run python Prokop/experiments/backward_plan.py [YAW SPEED AXIS OFFSET [END_ANGLES]]
    e.g. ... backward_plan.py 0 0.5 0.487,0.0 0.02,0.0      (central placement)
         ... backward_plan.py 0 0.5                         (seed 1's placement)
"""
import collections
import sys
import time

sys.path[:0] = ["Prokop"]
import numpy as np  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from solver import planner  # noqa: E402
from solver.contracts import OMEGA, TurntableModel  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402


def bar_direction(model: TurntableModel, theta: float) -> float:
    """Where the top bar's overhang (puzzle +x) points relative to 'at the robot' [deg]."""
    T = model.puzzle_T(theta)
    rel = np.degrees(np.arctan2(T[1, 0], T[0, 0]) - np.arctan2(-T[1, 3], -T[0, 3]))
    return (rel + 180) % 360 - 180


if __name__ == "__main__":
    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
    yaw = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    axis = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.5954, 0.1826]
    offset = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.028, -0.035]
    count = int(sys.argv[5]) if len(sys.argv) > 5 else planner.BW_END_ANGLES
    model = TurntableModel(axis_xy=np.array(axis), base_offset=np.array(offset),
                           base_yaw=yaw, direction=1, omega=OMEGA)
    print(f"axis {axis}, base offset {offset}, speed {timing.speed}", flush=True)
    t0 = time.monotonic()
    planner.PASS_LOG = lambda s: print(f"[{time.monotonic() - t0:5.0f} s] {s}", flush=True)
    for theta_end in planner.end_angles(model, count):
        print(f"end angle {np.rad2deg(theta_end):+.0f} deg: at the end the top bar points "
              f"{bar_direction(model, theta_end):+.0f} deg from 'at the robot'", flush=True)
        rejects = collections.defaultdict(collections.Counter)

        def note(k, kind, pair, d, t):
            a, b = ("hoop_ring" if x.startswith("hoop_ring") else x for x in pair)
            rejects[k][f"{kind}: {a} - {b}"] += 1
        planner.REJECT_LOG = note
        theta_start, r = planner.plan_backward(kin, timing, model, theta_end)
        for k in sorted(rejects, reverse=True):
            total = sum(rejects[k].values())
            print(f"rejections on the way back to checkpoint {k} ({total}):", flush=True)
            for what, n in rejects[k].most_common(6):
                print(f"   {n:5d}  {what}", flush=True)
        if theta_start is None:
            print(f"result: {r.reason} ({r.seconds:.0f} s)", flush=True)
            continue
        print(f"result: {'collision free' if r.collision_free else 'collides'}, start angle "
              f"{np.rad2deg(theta_start):+.0f} deg (top bar {bar_direction(model, theta_start):+.0f} "
              f"deg from the robot), {len(r.qs)} waypoints, traversal {r.arrivals[-1]:.1f} s, "
              f"entry {1000 * r.entry:.0f} mm, least soft clearance {1000 * r.min_clearance:.1f} mm "
              f"({r.seconds:.0f} s)", flush=True)
        if r.collision_free:
            break
