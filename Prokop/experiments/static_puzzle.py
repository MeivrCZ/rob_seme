"""Sanity check of the copy planner on a static puzzle (no spin).

The puzzle stands still (omega = 0) with its entry side (puzzle +x, where
checkpoint 0 lies) facing the robot base, at a grid of positions around the
middle of the work area. The plan uses only the continuity limit (no speed
cap), margin > 0, closest configurations first. If this fails, something is
wrong in the planner or the targets rather than in the timing.

    uv run python Prokop/experiments/static_puzzle.py [--grid X0 X1 NX Y0 Y1 NY]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))

import solver  # noqa: E402,F401  (puts Natka/ on the path)
from cell import WORK_AREA_XY  # noqa: E402
from crs_puzzle_d_hoop_path import puzzle_path_contract  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from natka_copy.planning import timed_plan  # noqa: E402
from solver.contracts import TurntableModel  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402


def facing_model(axis_xy: np.ndarray) -> tuple[TurntableModel, float]:
    """Static model with the puzzle's +x pointing at the robot base, and its theta."""
    model = TurntableModel(axis_xy=np.asarray(axis_xy, float), base_offset=np.zeros(2),
                           base_yaw=0.0, direction=1, omega=0.0)
    theta = 0.0
    for _ in range(3):  # the puzzle origin may sit off the axis: iterate
        T = model.puzzle_T(theta)
        heading = np.arctan2(T[1, 0], T[0, 0])
        to_robot = np.arctan2(-T[1, 3], -T[0, 3])
        theta += to_robot - heading
    return model, float(np.arctan2(np.sin(theta), np.cos(theta)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    centre = WORK_AREA_XY.mean(axis=1)
    parser.add_argument("--grid", type=float, nargs=6,
                        default=[centre[0] - 0.08, centre[0] + 0.08, 3, -0.10, 0.10, 3],
                        metavar=("X0", "X1", "NX", "Y0", "Y1", "NY"),
                        help="axis positions [m] (default: +-8 cm x, +-10 cm y around "
                             "the work-area centre)")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--continuity", type=float, nargs=3, default=[0.5, 2.0, 20.0],
                        metavar=("ALPHA", "BETA_DEG", "CAP_DEG"))
    parser.add_argument("--margin-floor", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true", help="per-pass progress lines")
    args = parser.parse_args()

    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, args.speed)
    path = puzzle_path_contract()
    a, b, cap = args.continuity
    continuity = (a, np.deg2rad(b), np.deg2rad(cap))
    x0, x1, nx, y0, y1, ny = args.grid
    print(f"work-area centre {np.round(centre, 3)}; continuity {a} ||dq*|| + {b} deg, "
          f"cap {cap} deg; margin > {args.margin_floor}; {len(path.frames)} checkpoints")

    results = []
    for x in np.linspace(x0, x1, int(nx)):
        for y in np.linspace(y0, y1, int(ny)):
            model, theta = facing_model(np.array([x, y]))
            t0 = time.monotonic()
            plan = timed_plan(kin, timing, model, path, start_phases=[theta],
                              directions=(1,), speeds=(args.speed,), q_start=kin.q_home,
                              pre_entry=0.0, continuity=continuity,
                              margin_floor=args.margin_floor, closest=True,
                              progress=print if args.verbose else None)
            seconds = time.monotonic() - t0
            if plan.feasible:
                qs = np.array([plan.q_start] + [s.q for s in plan.steps])
                steps = np.rad2deg(np.abs(np.diff(qs, axis=0)).max(axis=1))
                q5 = np.rad2deg(np.abs(qs[:, 4]).min())
                line = (f"OK   min margin {plan.min_margin:.3f}, largest joint step "
                        f"{steps[1:].max():.1f} deg (step {int(np.argmax(steps[1:])) + 1}), "
                        f"min |q5| {q5:.1f} deg")
            else:
                line = f"FAIL {plan.reason}"
            print(f"axis ({x:.3f}, {y:+.3f}) theta {np.rad2deg(theta):+6.1f} deg: {line} "
                  f"({seconds:.0f} s)", flush=True)
            results.append(plan.feasible)
    print(f"{sum(results)}/{len(results)} placements planned")


if __name__ == "__main__":
    main()
