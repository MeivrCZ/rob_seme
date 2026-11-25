"""Reachability over the work area: perception, planning and an independent check.

For each placement of a static puzzle facing the robot (a grid over the work
area and just beyond it), on the standalone simulator (sim_process):

1. perception: camera images -> ArUco base measurement -> static model; no
   markers raises PerceptionError: outside the camera's view, unreachable;
2. planning on the perceived model: traversal (solver.planner.plan_tiers) and
   approach (plan_approach); a failure is reported as unreachable, with why;
3. a plan found is replayed against the exact meshes at the TRUE placement:
   a contact there would be a plan the solver wrongly thinks is fine.

    uv run python Prokop/experiments/reachability.py [--port P] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))
sys.path.insert(0, str(PROKOP / "experiments"))

from cell import WORK_AREA_XY  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from placement import Placement  # noqa: E402
from show_plan import PerceptionError, perceive  # noqa: E402
from sim_process import SimProcess  # noqa: E402
from static_plan import replay_exact  # noqa: E402
from static_puzzle import facing_model  # noqa: E402

from solver.planner import plan_approach, plan_tiers  # noqa: E402
from solver.soft_collision import SoftScene  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402

# x across the work area (denser near the robot), at the centre line; then the sides.
X_LINE = (0.29, 0.31, 0.337, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.42, 0.45, 0.487, 0.55,
          0.60, 0.637, 0.68)
SIDES = [(x, y) for x in (0.36, 0.40, 0.487, 0.60) for y in (-0.2, 0.2)] + [(0.487, 0.28)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--quick", action="store_true", help="only the centre line")
    args = parser.parse_args()

    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, args.speed)
    placements = [(x, 0.0) for x in X_LINE] + ([] if args.quick else SIDES)
    (x0, x1), (y0, y1) = WORK_AREA_XY
    print(f"work area x {x0:.3f}..{x1:.3f} m, y {y0:.3f}..{y1:.3f} m; puzzle facing the robot",
          flush=True)
    rows = []
    with SimProcess(port=args.port) as sim:
        robot = sim.robot
        robot.initialize(home=False)
        for x, y in placements:
            inside = x0 <= x <= x1 and y0 <= y <= y1
            true_model, true_theta = facing_model(np.array([x, y]))
            sim.set_placement(Placement(np.array([x, y]), np.zeros(2), 0.0))
            sim.set_turntable(true_theta, 0.0)
            t0 = time.monotonic()
            head = f"({x:.3f}, {y:+.2f}) {'in ' if inside else 'OUT'}"
            try:
                model, theta, found = perceive(robot, images=4)  # capture_series needs >= 4
            except PerceptionError as error:
                rows.append((x, y, inside, "unreachable (perception)", str(error)))
                print(f"{head}: UNREACHABLE at perception: {error}", flush=True)
                continue
            result = plan_tiers(kin, timing, model, theta)[-1]
            if not result.collision_free:
                why = result.reason or "moves collide at every tier"
                rows.append((x, y, inside, "unreachable (planning)", why))
                print(f"{head}: UNREACHABLE at planning ({time.monotonic() - t0:.0f} s): {why}",
                      flush=True)
                continue
            approach, how = plan_approach(kin, SoftScene(model), model, theta, timing,
                                          kin.q_home, result.qs[0], result.entry)
            if approach is None:
                rows.append((x, y, inside, "unreachable (approach)", how))
                print(f"{head}: UNREACHABLE at the approach ({time.monotonic() - t0:.0f} s): {how}",
                      flush=True)
                continue
            hits, least = replay_exact(true_model, true_theta, timing, result, quiet=True)
            status = "PLAN, but CONTACT at the truth" if hits else "plan"
            rows.append((x, y, inside, status, f"{least * 1000:.1f} mm"))
            print(f"{head}: {status}: tier {result.tier}, entry {1000 * result.entry:.0f} mm, "
                  f"traversal {result.arrivals[-1]:.1f} s, approach {how.split(',')[0]}; exact "
                  f"clearance {1000 * least:.1f} mm{' CONTACTS ' + str(hits) if hits else ''} "
                  f"({time.monotonic() - t0:.0f} s)", flush=True)
    print("\nsummary:")
    for x, y, inside, status, detail in rows:
        print(f"  ({x:.3f}, {y:+.2f}) {'in ' if inside else 'out'}  {status:32s} {detail}")


if __name__ == "__main__":
    main()
