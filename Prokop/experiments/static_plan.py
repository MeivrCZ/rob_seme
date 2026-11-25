"""Prokop's own planner (solver.planner) on a static puzzle facing the robot.

Runs the granularity tiers 0, 1, ... (solver.wire) until every move is
collision free, and prints per tier: the checkpoints, the collisions along
the moves (where on the wire, which bodies), the least clearance, the
traversal time and the largest joint step. The planner checks collisions with
its soft model only; the chosen plan is then replayed against the simulator's
exact meshes (densely sampled) as an independent check.

    uv run python Prokop/experiments/static_plan.py [--axis X Y] [--max-tier N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))
sys.path.insert(0, str(PROKOP / "experiments"))

from ctu_crs import CRS93  # noqa: E402
from static_puzzle import facing_model  # noqa: E402

from collision import SceneCollider  # noqa: E402
from placement import Placement, TaskScene  # noqa: E402

from solver.planner import (FACE_NAMES, FACES, MAX_TIER, ROLL_STEP, TILT_DIRECTION_STEP,  # noqa: E402
                            TierResult, _q_at, orientation_grid, plan_rrt, plan_tiers)
from solver.timing import RobotTiming  # noqa: E402


def report(r: TierResult) -> None:
    """Prints one tier's result."""
    head = f"tier {r.tier}: {r.checkpoints} checkpoints, side {FACE_NAMES[r.face]}"
    if not r.feasible:
        print(f"{head}: {r.reason} ({r.seconds:.0f} s)")
        return
    steps = np.rad2deg(np.abs(np.diff(r.qs, axis=0)).max(axis=1))
    bad = [m for m in r.moves if m.hits]
    print(f"{head}: {'collision free' if not bad else f'{len(bad)} of {len(r.moves)} moves collide'}, "
          f"traversal {r.arrivals[-1]:.1f} s, largest joint step {steps.max():.1f} deg "
          f"(into checkpoint {int(np.argmax(steps)) + 1}), tilts "
          f"{np.round(np.rad2deg(r.orientations[:, 1])).astype(int).tolist()} deg, q1 range "
          f"{np.ptp(np.rad2deg(r.qs[:, 0])):.1f} deg, {r.banned} candidates and {r.ruled_out} moves ruled out in "
          f"{r.selections} selections "
          f"({r.seconds:.0f} s)")
    for m in bad:
        pairs = ", ".join(f"{a}-{b}" for a, b in m.hits)
        print(f"    move into {m.index:2d} ({m.segment}): {pairs} at t {m.t_hit:.2f} s")
    free = [m.clearance for m in r.moves if not m.hits]
    if free:
        print(f"    least soft clearance of the free moves {1000 * min(free):.1f} mm (margins "
              f"subtracted), {sum(m.samples for m in r.moves)} samples")


def replay_exact(model, theta: float, timing, r: TierResult, step: float = np.deg2rad(0.5),
                 quiet: bool = False) -> tuple[list, float]:
    """Replays a plan against the exact meshes (independent check), every step [rad].

    Returns:
        The contacts (move index, pairs) and the least exact clearance [m].
    """
    exact = SceneCollider(TaskScene())
    p = Placement(np.asarray(model.axis_xy, float), np.asarray(model.base_offset, float),
                  model.base_yaw)
    _, moves = timing.schedule(r.qs[0], list(r.qs[1:]))
    least, hits = np.inf, []
    for k in range(1, len(r.qs)):
        t0, t1 = r.arrivals[k - 1], r.arrivals[k]
        n = max(2, int(np.ceil(2 * np.abs(r.qs[k] - r.qs[k - 1]).max() / step)) + 1)
        for t in np.linspace(t0, t1, n):
            h, d = exact.check(_q_at(moves, r.qs[0], t), p, theta)
            least = min(least, d)
            if h:
                hits.append((k, h))
                break
    if not quiet:
        print(f"exact replay: {'no contact' if not hits else f'CONTACT in moves {hits}'}, "
              f"least exact clearance {1000 * least:.1f} mm")
    return hits, least


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", type=float, nargs=2, default=[0.487, 0.0],
                        metavar=("X", "Y"), help="turntable axis [m] (default: work-area centre)")
    parser.add_argument("--max-tier", type=int, default=MAX_TIER)
    parser.add_argument("--min-tier", type=int, default=0,
                        help="first tier tried (finer ones even if coarser ones work)")
    parser.add_argument("--roll-step", type=float, default=np.rad2deg(ROLL_STEP),
                        help="hoop roll grid [deg]")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--max-tilt", type=float, default=50.0,
                        help="largest tilt of the ring axis from the wire direction [deg]; 0 = "
                             "orientation fixed up to the roll (the old behaviour)")
    parser.add_argument("--tilt-step", type=float, default=10.0, help="tilt grid [deg]")
    parser.add_argument("--rrt", type=int, metavar="BEND_POINTS",
                        help="RRT between checkpoints with this many on the bend (3 or 5) "
                             "instead of the tiers")
    parser.add_argument("--faces", choices=("both", "A", "B"), default="both",
                        help="leading side(s) of the hoop")
    args = parser.parse_args()

    kin = CRS93(tty_dev=None)
    model, theta = facing_model(np.array(args.axis))  # omega = 0: no spin
    print(f"axis {np.round(args.axis, 3)}, theta {np.rad2deg(theta):+.1f} deg (puzzle facing "
          f"the robot, no spin), speed {args.speed}, roll grid {args.roll_step:g} deg")
    tilts = np.deg2rad(np.arange(0.0, args.max_tilt + 1e-9, args.tilt_step))
    grid = orientation_grid(np.deg2rad(args.roll_step), tilts, TILT_DIRECTION_STEP)
    print(f"{len(grid)} ring orientations per checkpoint and side (tilts up to "
          f"{args.max_tilt:g} deg)")
    timing = RobotTiming(kin, args.speed)
    if args.rrt:
        r = plan_rrt(kin, timing, model, theta, bend_points=args.rrt, orientations=grid,
                     faces={"both": FACES, "A": (1,), "B": (-1,)}[args.faces])
        if not r.feasible:
            print(f"RRT: {r.reason} ({r.seconds:.0f} s)")
            return
        steps = np.rad2deg(np.abs(np.diff(r.qs, axis=0)).max(axis=1))
        bad = [m for m in r.moves if m.hits]
        print(f"RRT, {args.rrt} bend points: {r.checkpoints} checkpoints, side "
              f"{FACE_NAMES[r.face]}, entry {1000 * r.entry:.0f} mm: "
              f"{'collision free' if not bad else f'{len(bad)} moves collide'}, {len(r.qs)} "
              f"waypoints after shortcutting, traversal {r.arrivals[-1]:.1f} s, largest step "
              f"{steps.max():.1f} deg, checkpoint tilts "
              f"{np.round(np.rad2deg(r.orientations[:, 1])).astype(int).tolist()} deg, "
              f"{r.banned} candidates and {r.ruled_out} links ruled out, least soft clearance "
              f"{1000 * r.min_clearance:.1f} mm ({r.seconds:.0f} s)")
        if not bad:
            replay_exact(model, theta, timing, r)
        return
    results = plan_tiers(kin, timing, model, theta,
                         max_tier=args.max_tier, orientations=grid,
                         faces={"both": FACES, "A": (1,), "B": (-1,)}[args.faces],
                         progress=report, min_tier=args.min_tier)
    last = results[-1]
    print("collision-free plan at tier", last.tier if last.collision_free else "none")
    if last.collision_free:
        replay_exact(model, theta, timing, last)


if __name__ == "__main__":
    main()
