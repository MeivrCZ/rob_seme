"""A baseline (user, 2026-11-26): walk the wire backwards point by point.

The wire split into points SPACING apart, from the end in the stem back to the
entry. From an initial configuration at the end (the end pose with the most clearance,
the ring low enough, side B), for every earlier point:

1. predict the time: the joint move from the next point's configuration estimated with
   the Jacobian (dq = J+ dx of the ring centre), its duration from the measured
   trapezoid profile (RobotTiming.profile) -> the predicted moment and so the puzzle's
   angle;
2. the IK at the wire point at that angle, the solution closest to the next point's
   configuration: damped least squares from it on the ring centre and the ring axis
   along the wire's tangent, the roll free (the orientation carried exactly along the
   tangent left the joint room within a few points);
3. the true time: the duration of that actual joint move; the puzzle's angle then is
   not the predicted one, so the ring is off-centre;
4. correction: the IK again at the true time (closest to the first solution), the true
   time again;
5. compensation: the off-centre left over is carried into the next point's target
   (that point is aimed at minus it), so it does not add up along the walk.

Then checked: the off-centre at every point (before / after the correction), the
whole path forward continuously against the turning puzzle (soft model, the run's
margins), the traversal (the robot stops at every point), the largest joint step.

    uv run python Prokop/experiments/baseline_walk.py [--spacing 0.005 0.01 0.02 0.04]
        [--axis 0.43 -0.05] [--offset 0 -0.02] [--yaw 0] [--direction -1] [--end 0]
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
from solver.wire import (SUCCESS_HEIGHT, checkpoint_arclengths, ring_top_above_pad,  # noqa: E402
                         wire_frame_at)

ENTRY = 0.015
AXIS_WEIGHT = 0.05  # m per unit of axis misalignment in the IK (1 rad ~ 5 cm)
HOOP_MARGIN = 0.006
JOINT_MARGIN = np.deg2rad(5.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacing", type=float, nargs="+", default=(0.005, 0.01, 0.02, 0.04))
    ap.add_argument("--axis", type=float, nargs=2, default=(0.43, -0.05))
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, -0.02))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--direction", type=int, default=-1)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, 0.5)
    model = TurntableModel(axis_xy=np.array(args.axis), base_offset=np.array(args.offset),
                           base_yaw=np.deg2rad(args.yaw), direction=args.direction,
                           omega=OMEGA)
    scene = SoftScene(model, hoop_puzzle_margin=HOOP_MARGIN)
    theta_end = planner.end_angles(model)[args.end]
    to_ring = ring_in_flange()
    flange_of_ring = np.linalg.inv(to_ring)
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN

    def angle(t):  # t <= 0, the end at 0
        return theta_end + model.direction * model.omega * t

    def in_room(q):
        return bool(np.all(q >= lo) and np.all(q <= hi))

    s_end = checkpoint_arclengths(0, ENTRY)[-1]

    # The initial configuration: the end pose (side B, any tilt / roll of the grid) with
    # the ring low enough and the most clearance.
    from solver.hoop_pose import orientation_grid
    end_frame = model.puzzle_T(angle(0.0)) @ wire_frame_at(s_end, ENTRY)
    best = None
    for o in orientation_grid():
        for q in kin.ik(flange_target(end_frame, float(o[0]), -1, float(o[1]), float(o[2]))):
            if not in_room(q):
                continue
            ring = np.linalg.inv(model.puzzle_T(angle(0.0))) @ kin.fk(q) @ to_ring
            if ring_top_above_pad(ring) >= SUCCESS_HEIGHT:
                continue
            c = scene.clearance(q, angle(0.0))
            if c.free and (best is None or c.distance > best[0]):
                best = (c.distance, q)
    if best is None:
        print("no free end pose")
        return
    q_end = best[1]

    def ring_T(q):
        return kin.fk(q) @ to_ring

    def jac_pos(q):
        r = ring_T(q)[:3, 3]
        J = np.empty((3, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-6
            J[:, j] = (ring_T(q + dq)[:3, 3] - r) / 1e-6
        return J

    def ik_closest(T_ring_world, q_ref):
        """The IK closest to q_ref: damped least squares from q_ref on the ring centre
        (onto the target's position) and the ring axis (onto the target's z, the wire's
        tangent), the roll free; it walks to the nearest solution by construction."""
        target, tangent = T_ring_world[:3, 3], T_ring_world[:3, 2]

        def err(q):
            T = ring_T(q)
            axis = T[:3, 2] * (1.0 if T[:3, 2] @ tangent >= 0 else -1.0)
            return np.concatenate((target - T[:3, 3], AXIS_WEIGHT * np.cross(axis, tangent)))

        q = np.array(q_ref, float)
        for _ in range(200):
            e = err(q)
            if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < AXIS_WEIGHT * 1e-3:
                break
            J = np.empty((6, 6))
            for j in range(6):
                dq = np.zeros(6)
                dq[j] = 1e-6
                J[:, j] = -(err(q + dq) - e) / 1e-6
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 ** 2 * np.eye(6), e)
            dq *= min(1.0, 0.1 / max(np.abs(dq).max(), 1e-12))
            q = np.clip(q + dq, lo, hi)
        if np.linalg.norm(err(q)[:3]) > 0.002:  # not on the wire point (limits, reach)
            return None
        return q

    def walk(spacing):
        s = np.arange(s_end, -1e-9, -spacing)  # from the end back to the entry
        if s[-1] > 1e-6:
            s = np.append(s, 0.0)
        qs, ts = [q_end], [0.0]
        off_pred, off_corr = [], []
        comp = np.zeros(3)  # the compensation carried into the next target (puzzle frame)
        for k in range(1, len(s)):
            q_next, t_next = qs[-1], ts[-1]
            F = wire_frame_at(s[k], ENTRY)
            F_next = wire_frame_at(s[k - 1], ENTRY)
            # the orientation: the next point's ring orientation (puzzle frame) carried
            # along the wire's tangent
            P_next = model.puzzle_T(angle(t_next))
            R_rel = F_next[:3, :3].T @ (np.linalg.inv(P_next) @ ring_T(q_next))[:3, :3]
            local = np.eye(4)
            local[:3, :3] = F[:3, :3] @ R_rel
            local[:3, 3] = F[:3, 3] - comp
            # 1. the predicted time: dq from the Jacobian, the profile's duration
            x_now = ring_T(q_next)[:3, 3]
            target_now = (model.puzzle_T(angle(t_next)) @ local)[:3, 3]
            dq = np.linalg.pinv(jac_pos(q_next)) @ (target_now - x_now)
            t_pred = t_next - timing.profile(dq)[0]
            # 2. the IK at the predicted angle, the closest to the next configuration
            q = ik_closest(model.puzzle_T(angle(t_pred)) @ local, q_next)
            if q is None:
                return None, f"no IK at s {1000 * s[k]:.0f} mm (point {k} of {len(s) - 1})"
            # 3. the true time; how far off-centre the ring then is
            t_true = t_next - timing.move_duration(q, q_next)

            def off(q, t):
                ring = (np.linalg.inv(model.puzzle_T(angle(t))) @ ring_T(q))[:3, 3]
                return ring - F[:3, 3]
            off_pred.append(float(np.linalg.norm(off(q, t_true))))
            # 4. correction: the IK at the true time, the true time again
            q2 = ik_closest(model.puzzle_T(angle(t_true)) @ local, q)
            if q2 is not None:
                q = q2
                t_true = t_next - timing.move_duration(q, q_next)
            e = off(q, t_true)
            off_corr.append(float(np.linalg.norm(e)))
            # 5. compensation: the next point is aimed at minus what is left over
            comp = e
            qs.append(q)
            ts.append(t_true)
        return (np.array(qs[::-1]), np.array(ts[::-1]), off_pred[::-1], off_corr[::-1]), ""

    print(f"placement axis {args.axis}, offset {args.offset}, spin {args.direction:+d}; end "
          f"pose clearance {1000 * best[0]:.1f} mm")
    for spacing in args.spacing:
        t0 = time.monotonic()
        out, why = walk(spacing)
        if out is None:
            print(f"spacing {1000 * spacing:4.0f} mm: {why} ({time.monotonic() - t0:.1f} s)")
            continue
        qs, ts, off_pred, off_corr = out
        theta_start = angle(ts[0])

        def fwd(t, theta_start=theta_start):
            return theta_start + model.direction * model.omega * t

        checks, arrivals = planner.check_moves(scene, fwd, model.omega, timing, qs,
                                               np.zeros(len(qs), int))
        hits = [(i + 1, c.hits[0]) for i, c in enumerate(checks) if c.hits]
        steps = np.rad2deg(np.abs(np.diff(qs, axis=0)).max(axis=1))
        print(f"spacing {1000 * spacing:4.0f} mm: {len(qs)} points, traversal "
              f"{arrivals[-1]:.1f} s, largest joint step {steps.max():.1f} deg; off-centre "
              f"with the predicted time max {1000 * max(off_pred):.1f} mm, after the "
              f"correction max {1000 * max(off_corr):.1f} mm (mean "
              f"{1000 * np.mean(off_corr):.1f}); least soft clearance "
              f"{1000 * min(c.clearance for c in checks):.1f} mm; "
              + ("no collision" if not hits else f"{len(hits)} of {len(checks)} moves collide, "
                 f"e.g. " + ", ".join(f"{i}: {p[0]} - {p[1]}" for i, p in hits[:3]))
              + f"; start {'is' if planner.waits_free(scene, qs[0], theta_start) else 'is NOT'}"
              f" a waiting pose ({time.monotonic() - t0:.1f} s)")


if __name__ == "__main__":
    main()
