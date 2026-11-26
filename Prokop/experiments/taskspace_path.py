"""Task-space interpolation between the checkpoints (user experiment, 2026-11-26).

Between two consecutive checkpoints the ring's pose relative to the puzzle follows the
SE(3) geodesic T_a exp(s log(T_a^-1 T_b)) at discrete s (about one sample per
SAMPLE_STEP of wire). For frames whose z axis is the wire's tangent this geodesic IS
the wire: a straight line on the straight stretches, the circular arc on the R50 bend
(a screw motion about the bend's axis), so the ring centre stays on the wire by
construction. Every sample is solved by IK at its time on the turning puzzle (planned
backwards from the end, like the planner: t = the next sample's time - the move's
duration), the solution closest to the next sample's; the robot stops at every sample.
No decimation, planning speed ignored (user). Then checked: the whole path forward,
continuously against the turning puzzle (soft model, the run's margins).

The ring at every checkpoint: the planner's convention (solver.hoop_pose.flange_target:
side B leads, --tilt, one roll for the whole path); all rolls of the grid are tried,
the best kept.

    uv run python Prokop/experiments/taskspace_path.py [--axis 0.43 -0.05]
        [--offset 0 -0.02] [--yaw 0] [--direction -1] [--end 0] [--tilt 0] [--step 0.005]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROKOP), str(PROKOP.parent / "Natka")]
from ctu_crs import CRS93  # noqa: E402

from solver import planner  # noqa: E402
from solver.contracts import OMEGA, TurntableModel  # noqa: E402
from solver.hoop_pose import ROLL_STEP, flange_target, ring_in_flange  # noqa: E402
from solver.soft_collision import SoftScene  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from solver.wire import checkpoint_arclengths, wire_frame_at, wire_path  # noqa: E402

ENTRY = 0.015
HOOP_MARGIN = 0.006
JOINT_MARGIN = np.deg2rad(5.5)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", type=float, nargs=2, default=(0.43, -0.05))
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, -0.02))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--direction", type=int, default=-1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--tilt", type=float, default=0.0, help="ring tilt [deg]")
    ap.add_argument("--step", type=float, default=0.005, help="wire per sample [m]")
    ap.add_argument("--from-plan", action="store_true",
                    help="the planner's waypoints as the poses, task-space moves between them")
    args = ap.parse_args()

    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, 0.5)
    model = TurntableModel(axis_xy=np.array(args.axis), base_offset=np.array(args.offset),
                           base_yaw=np.deg2rad(args.yaw), direction=args.direction,
                           omega=OMEGA)
    scene = SoftScene(model, hoop_puzzle_margin=HOOP_MARGIN)
    theta_end = planner.end_angles(model)[args.end]
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN
    flange_of_ring = np.linalg.inv(ring_in_flange())
    tilt = np.deg2rad(args.tilt)

    def angle(t):  # t <= 0, the end at 0
        return theta_end + model.direction * model.omega * t

    path = wire_path(0, ENTRY, 3)
    s_cp = checkpoint_arclengths(0, ENTRY, 3)
    # The samples (puzzle frame): per stretch the geodesic between its checkpoints.
    def samples(roll):
        # The planner's convention: side B leads (face -1), tilt, roll (flange_target);
        # the ring pose (puzzle frame) from that flange pose.
        ring_at = [flange_target(F, float(roll), -1, tilt, 0.0) @ ring_in_flange()
                   for F in path.frames]
        poses = [ring_at[0]]
        for k in range(len(ring_at) - 1):
            n = max(1, int(np.ceil((s_cp[k + 1] - s_cp[k]) / args.step)))
            for j in range(1, n + 1):
                poses.append(planner.se3_interpolate(ring_at[k], ring_at[k + 1], j / n))
        return poses

    # How far a sample's ring centre is from the wire (should be ~0 by construction).
    s_fine = np.arange(0.0, checkpoint_arclengths(0, 0.08, 3)[-1], 0.0005)
    wire_pts = np.array([wire_frame_at(x, 0.08)[:3, 3] for x in s_fine])  # from 80 mm out

    def off_wire(pose):
        return float(np.linalg.norm(wire_pts - pose[:3, 3], axis=1).min())

    def solve(poses):
        """Backwards from the end: IK per sample at its time, closest to the next one's."""
        qs, ts = [None] * len(poses), [0.0] * len(poses)
        q_next, t_next = None, 0.0
        for i in range(len(poses) - 1, -1, -1):
            q = q_next
            for _ in range(3):
                t = t_next - (timing.move_duration(q, q_next) if q_next is not None else 0.0)
                target = model.puzzle_T(angle(t)) @ poses[i] @ flange_of_ring
                sols = [x for x in kin.ik(target)
                        if np.all(x >= lo) and np.all(x <= hi)]
                if not sols:
                    return None, i
                ref = q_next if q_next is not None else kin.q_home
                q = min(sols, key=lambda x: float(np.abs(x - ref).max()))
            qs[i], ts[i] = q, t
            q_next, t_next = q, t
        return (np.array(qs), np.array(ts)), None

    def check(qs, ts):
        theta_start = angle(ts[0])

        def fwd(t):
            return theta_start + model.direction * model.omega * t

        checks, arrivals = planner.check_moves(scene, fwd, model.omega, timing, qs,
                                               np.zeros(len(qs), int))
        hits = [(i + 1, c.hits[0]) for i, c in enumerate(checks) if c.hits]
        steps = np.rad2deg(np.abs(np.diff(qs, axis=0)).max(axis=1))
        return dict(n=len(qs), hits=hits, least=min(c.clearance for c in checks),
                    traversal=float(arrivals[-1]), step=float(steps.max()),
                    waits=planner.waits_free(scene, qs[0], theta_start))

    def describe(label, r, off=None):
        hit = ("no collision" if not r["hits"] else f"{len(r['hits'])} of {r['n'] - 1} moves "
               f"collide, e.g. " + ", ".join(f"{i}: {p[0]} - {p[1]}" for i, p in r["hits"][:3]))
        print(f"  {label}: {r['n']} waypoints, traversal {r['traversal']:.1f} s, largest joint "
              f"step {r['step']:.1f} deg, "
              + (f"ring off the wire <= {1000 * off:.2f} mm, " if off is not None else "")
              + f"least soft clearance {1000 * r['least']:.1f} mm, start "
              f"{'is' if r['waits'] else 'is NOT'} a waiting pose; {hit}")

    if args.from_plan:
        # The planner's checkpoint poses (the subdivision planner), task-space small moves
        # between them: the geodesic between consecutive waypoints' ring poses relative
        # to the puzzle, one sample per --step of ring travel.
        t0 = time.monotonic()
        theta_start, plan = planner.plan_backward(kin, timing, model, theta_end, scene=scene,
                                                  joint_margin=JOINT_MARGIN)
        if theta_start is None or not plan.collision_free:
            print(f"the planner found no path: {plan.reason}")
            return
        print(f"planner ({time.monotonic() - t0:.0f} s):")
        describe("straight joint moves", check(np.array(plan.qs),
                                                np.array(plan.arrivals) - plan.arrivals[-1]))
        L = [np.linalg.inv(model.puzzle_T(theta_start + model.direction * model.omega * t))
             @ kin.fk(q) @ ring_in_flange() for q, t in zip(plan.qs, plan.arrivals)]
        poses = [L[0]]
        for a, b in zip(L, L[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b[:3, 3] - a[:3, 3]) / args.step)))
            poses += [planner.se3_interpolate(a, b, j / n) for j in range(1, n + 1)]
        # The same samples at the planner's own times (interpolated along each move):
        # is it the stop-and-go timing (the puzzle turns further) or the geodesic itself?
        t_plan = np.array(plan.arrivals) - plan.arrivals[-1]
        t_samples = [t_plan[0]]
        for k, (a_, b_) in enumerate(zip(L, L[1:])):
            n = max(1, int(np.ceil(np.linalg.norm(b_[:3, 3] - a_[:3, 3]) / args.step)))
            t_samples += [t_plan[k] + (t_plan[k + 1] - t_plan[k]) * j / n for j in range(1, n + 1)]
        missing = [i for i, (pose, t) in enumerate(zip(poses, t_samples))
                   if not any(np.all(x >= lo) and np.all(x <= hi) for x in
                              kin.ik(model.puzzle_T(angle(t)) @ pose @ flange_of_ring))]
        print(f"  the same {len(poses)} samples at the planner's times: "
              + ("IK in the joint room at every one" if not missing else
                 f"no IK at {len(missing)} of them (e.g. sample {missing[0]})"))
        solved, bad = solve(poses)
        if solved is None:
            print(f"  task-space small moves (stop at each, re-timed): no IK at sample {bad} of "
                  f"{len(poses)}")
            return
        describe(f"task-space small moves ({1000 * args.step:.0f} mm)", check(*solved),
                 max(off_wire(p) for p in poses))
        return

    t0 = time.monotonic()
    rolls = np.arange(-np.pi, np.pi, ROLL_STEP)
    results = []
    for roll in rolls:
        poses = samples(roll)
        solved, bad = solve(poses)
        if solved is None:
            results.append((roll, None, f"no IK at sample {bad} of {len(poses)}"))
            continue
        qs, ts = solved
        theta_start = angle(ts[0])

        def fwd(t, theta_start=theta_start):
            return theta_start + model.direction * model.omega * t

        checks, arrivals = planner.check_moves(scene, fwd, model.omega, timing, qs,
                                               np.zeros(len(qs), int))
        hits = [(i + 1, c.hits[0]) for i, c in enumerate(checks) if c.hits]
        steps = np.rad2deg(np.abs(np.diff(qs, axis=0)).max(axis=1))
        results.append((roll, dict(n=len(qs), hits=hits,
                                   least=min(c.clearance for c in checks),
                                   traversal=float(arrivals[-1]), step=float(steps.max()),
                                   off=max(off_wire(p) for p in poses),
                                   waits=planner.waits_free(scene, qs[0], theta_start)), ""))
    print(f"{len(rolls)} rolls in {time.monotonic() - t0:.0f} s; {len(path.frames)} checkpoints, "
          f"one sample per {1000 * args.step:.0f} mm, tilt {args.tilt:.0f} deg")
    ok = [r for r in results if r[1] is not None]
    for roll, why in [(r[0], r[2]) for r in results if r[1] is None][:3]:
        print(f"  roll {np.rad2deg(roll):+5.0f} deg: {why}")
    print(f"  {len(ok)} of {len(rolls)} rolls have IK along the whole path")
    ok.sort(key=lambda r: (len(r[1]["hits"]), -r[1]["least"]))
    for roll, r, _ in ok[:6]:
        hit = ("no collision" if not r["hits"] else f"{len(r['hits'])} of {r['n'] - 1} moves "
               f"collide, e.g. " + ", ".join(f"{i}: {p[0]} - {p[1]}" for i, p in r["hits"][:3]))
        print(f"  roll {np.rad2deg(roll):+5.0f} deg: {r['n']} waypoints, traversal "
              f"{r['traversal']:.1f} s, largest joint step {r['step']:.1f} deg, ring off the wire "
              f"<= {1000 * r['off']:.2f} mm, least soft clearance {1000 * r['least']:.1f} mm, "
              f"start {'is' if r['waits'] else 'is NOT'} a waiting pose; {hit}")


if __name__ == "__main__":
    main()
