"""How far does a plan carry to a nearby placement by the Jacobian's first order?

For the plan library (user, 2026-11-27): how dense must the grid of placements be?
1. One placement (default the centre of the work area: axis (0.487, 0), base offset
   (0.02, 0), yaw 0, spin +1); the "ideal" plan with no time limit: plan_backward at
   12 end angles (every 30 deg, a full turn), no per-angle limit, the best by the
   planner's own rank (_plan_rank: joint-limit room, then sigma_min). Cached in
   experiments/data/jacobian_reach_plan.npz (--replan to redo).
2. The placement perturbed in the library's parameters (rotation about the robot axis
   and the start time are free, so 3 remain):
     rho  the turntable axis' distance from the robot axis [mm],
     b    the base's offset from the turntable axis [mm],
     phi  the base's yaw about its own centre (relative to the offset) [deg],
     dth  the turntable angle along the whole timeline (the same placement, the plan
          started dth later in the turn: the puzzle's heading relative to the robot
          at every waypoint; the planner searches this as its end angle) [deg].
   Each waypoint keeps its pose relative to the puzzle at its time: the ring target
   moves rigidly with the puzzle, T' = P'(t) P(t)^-1 T. First order: q' = q + J^-1 xi
   (J the body Jacobian of the ring frame, xi the body twist T^-1 T'). Measured:
   * the pose error of the first-order waypoints (position mm, rotation deg),
   * the exact waypoints (Newton from there) and how far they are from the
     first-order ones in joint space,
   * whether the path is still collision free in the soft model with the planning
     margins (both the first-order and the exact one; timing re-scheduled), its least
     clearance, and whether every waypoint stays within the joint limits minus the
     margin.

    uv run python Prokop/experiments/jacobian_reach.py [--replan] [--workers 6]
        [--seed N | --axis X Y --offset X Y --yaw DEG] [--direction 1|-1]
        [--angle-limit S] [--only rho|b|phi|dth]

--seed N: run_solver's random placement N (the batch's), direction as in batch 6
(odd +1, even -1) unless --direction. The plan is cached per placement.
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROKOP), str(PROKOP.parent / "Natka")]

from ctu_crs import CRS93  # noqa: E402

from solver import planner  # noqa: E402
from solver.contracts import OMEGA, TurntableModel  # noqa: E402
from solver.hoop_pose import ring_in_flange  # noqa: E402
from solver.soft_collision import SoftScene  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from solver.uncertainty import Margins  # noqa: E402

CACHE_DIR = PROKOP / "experiments" / "data"
SPEED = 0.5
HOOP_MARGIN, JOINT_MARGIN = 0.005, np.deg2rad(6.0)  # the revised margins at 140 mm reach
AXIS, OFFSET, YAW, DIRECTION = np.array([0.487, 0.0]), np.array([0.02, 0.0]), 0.0, 1
END_ANGLES = 12
STEPS_MM = (1, 2, 5, 10, 20, 50)
STEPS_DEG = (1, 2, 5, 10, 20, 45)
STEPS_THETA = (1, 2, 5, 10, 20, 30, 45, 90)
H = 1e-6  # rad, finite-difference step of the Jacobian
FAMILY_GAP = 2.0  # deg, exact - first order beyond this: not the same family


def model_of(axis=None, offset=None, yaw=None) -> TurntableModel:
    return TurntableModel(np.asarray(AXIS if axis is None else axis, float),
                          np.asarray(OFFSET if offset is None else offset, float),
                          float(YAW if yaw is None else yaw), DIRECTION, OMEGA)


def margins() -> Margins:
    return Margins(hoop_puzzle=HOOP_MARGIN, joint_limit=JOINT_MARGIN, sd_angle=0.0,
                   sd_xy=0.0, sd_omega=0.0, reach=0.0)


def _set_limit(limit: float) -> None:
    planner.BW_TOTAL_LIMIT = limit  # per end angle (the planner's own: 120 s)


def plan_ideal(workers: int, limit: float, cache: Path) -> dict:
    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, SPEED)
    model = model_of()
    angles = planner.end_angles(model, count=END_ANGLES)
    jobs = [(timing, model, a, 3, i, None, margins()) for i, a in enumerate(angles)]
    t0 = time.monotonic()
    found = []
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_set_limit,
                                                   initargs=(limit,)) as pool:
        for theta_end, theta, result in pool.imap_unordered(planner._backward_job, jobs):
            ok = theta is not None and result.collision_free
            q = planner.plan_quality(kin, result.qs) if ok else None
            print(f"  end angle {np.rad2deg(theta_end):7.1f} deg: "
                  + (f"plan, {len(result.qs)} waypoints, traversal {result.arrivals[-1]:.1f} s, "
                     f"sigma_min {1000 * q[0]:.1f} mm, joint room {np.rad2deg(q[1]):.1f} deg"
                     if ok else f"none ({result.reason})")
                  + f" ({result.seconds:.0f} s; {time.monotonic() - t0:.0f} s total)", flush=True)
            if ok:
                result.quality = q
                found.append((theta, result))
    if not found:
        raise SystemExit("no plan at any end angle")
    theta, best = max(found, key=lambda u: planner._plan_rank(u[1]))
    data = {"theta_start": theta, "qs": best.qs, "arrivals": best.arrivals,
            "sigma_min": best.quality[0], "joint_room": best.quality[1],
            "plans": len(found), "seconds": time.monotonic() - t0}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def ring_T(kin, to_ring, q):
    return kin.fk(q) @ to_ring


def twist(T):
    w, v = planner.se3_log(T)
    return np.concatenate((w, v))


def body_jacobian(kin, to_ring, q):
    T = ring_T(kin, to_ring, q)
    Ti = np.linalg.inv(T)
    J = np.zeros((6, 6))
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = H
        J[:, k] = twist(Ti @ ring_T(kin, to_ring, q + dq)) / H
    return J


def pose_error(T_target, T):
    E = np.linalg.inv(T_target) @ T
    w, _ = planner.se3_log(E)
    return 1000 * float(np.linalg.norm(E[:3, 3])), float(np.rad2deg(np.linalg.norm(w)))


def carry(kin, to_ring, qs, targets, exact: bool):
    """First-order waypoints (and, if exact, Newton to convergence from them)."""
    out = []
    for q, T1 in zip(qs, targets):
        q1 = q + np.linalg.solve(body_jacobian(kin, to_ring, q),
                                 twist(np.linalg.inv(ring_T(kin, to_ring, q)) @ T1))
        if exact:
            for _ in range(20):
                xi = twist(np.linalg.inv(ring_T(kin, to_ring, q1)) @ T1)
                if np.linalg.norm(xi) < 1e-10:
                    break
                q1 = q1 + np.linalg.solve(body_jacobian(kin, to_ring, q1), xi)
        out.append(q1)
    return np.array(out)


def check(kin, timing, model, theta_start, qs):
    scene = SoftScene(model, hoop_puzzle_margin=HOOP_MARGIN, restricted=True)
    angle = lambda t: theta_start + model.direction * model.omega * t  # noqa: E731
    checks, _ = planner.check_moves(scene, angle, model.omega, timing, qs,
                                    np.zeros(len(qs), int))
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN
    in_room = bool(np.all((qs >= lo) & (qs <= hi)))
    room = float(np.rad2deg(np.min(np.minimum(qs - kin.q_min, kin.q_max - qs))))
    return (not any(c.hits for c in checks)), min(c.clearance for c in checks), in_room, room


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replan", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", help="only the cases starting with this (rho, b, phi, dth)")
    ap.add_argument("--seed", type=int, help="run_solver's random placement")
    ap.add_argument("--axis", type=float, nargs=2)
    ap.add_argument("--offset", type=float, nargs=2)
    ap.add_argument("--yaw", type=float, help="deg")
    ap.add_argument("--direction", type=int, choices=(-1, 1))
    ap.add_argument("--angle-limit", type=float, default=1e9,
                    help="s per end angle (default none; the failing ones can take 15 min)")
    args = ap.parse_args()
    global AXIS, OFFSET, YAW, DIRECTION
    name = "centre"
    if args.seed is not None:
        from cell import WORK_AREA_XY
        from placement import TaskScene
        pl = TaskScene().random_placement(np.random.default_rng(args.seed), WORK_AREA_XY)
        AXIS, OFFSET = np.asarray(pl.turntable_xy, float), np.asarray(pl.base_xy, float)
        YAW, DIRECTION = float(pl.base_yaw), 1 if args.seed % 2 else -1
        name = f"seed{args.seed}"
    if args.axis is not None:
        AXIS, name = np.array(args.axis), "custom"
    if args.offset is not None:
        OFFSET = np.array(args.offset)
    if args.yaw is not None:
        YAW = np.deg2rad(args.yaw)
    if args.direction is not None:
        DIRECTION = args.direction
    if name == "custom":
        name = (f"axis{1000 * AXIS[0]:.0f}_{1000 * AXIS[1]:.0f}_off{1000 * OFFSET[0]:.0f}_"
                f"{1000 * OFFSET[1]:.0f}_yaw{np.rad2deg(YAW):.0f}")
    cache = CACHE_DIR / f"jacobian_reach_{name}_dir{DIRECTION:+d}.npz"
    if name == "centre" and DIRECTION == 1 and (CACHE_DIR / "jacobian_reach_plan.npz").exists():
        cache = CACHE_DIR / "jacobian_reach_plan.npz"  # the first run's
    print(f"placement {name}: axis {np.round(1000 * AXIS).astype(int).tolist()} mm (distance "
          f"{1000 * np.linalg.norm(AXIS):.0f} mm), offset {np.round(1000 * OFFSET).astype(int).tolist()} mm "
          f"({1000 * np.linalg.norm(OFFSET):.0f} mm), yaw {np.rad2deg(YAW):.0f} deg, spin "
          f"{DIRECTION:+d}", flush=True)
    if args.replan or not cache.exists():
        print(f"planning the ideal plan ({END_ANGLES} end angles, limit per angle "
              f"{args.angle_limit:.0f} s):", flush=True)
        data = plan_ideal(args.workers, args.angle_limit, cache)
    else:
        data = dict(np.load(cache))
    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, SPEED)
    to_ring = ring_in_flange()
    qs, t, theta0 = np.asarray(data["qs"]), np.asarray(data["arrivals"]), float(data["theta_start"])
    m0 = model_of()
    ok0, cl0, room_ok0, room0 = check(kin, timing, m0, theta0, qs)
    print(f"\nthe plan: {len(qs)} waypoints, traversal {t[-1]:.1f} s, sigma_min "
          f"{1000 * float(data['sigma_min']):.1f} mm, start angle {np.rad2deg(theta0):.1f} deg "
          f"(best of {int(data['plans'])} plans, {float(data['seconds']):.0f} s); on its own "
          f"placement: {'free' if ok0 else 'COLLIDES'}, least clearance {1000 * cl0:.1f} mm "
          f"(beyond the {1000 * HOOP_MARGIN:.0f} mm margin), joint room {room0:.1f} deg")
    angle = lambda tt: theta0 + DIRECTION * OMEGA * tt  # noqa: E731
    T = np.array([ring_T(kin, to_ring, q) for q in qs])
    P = np.array([m0.puzzle_T(angle(tt)) for tt in t])
    unit_axis = AXIS / np.linalg.norm(AXIS)
    unit_off = (OFFSET / np.linalg.norm(OFFSET) if np.linalg.norm(OFFSET) > 1e-3
                else np.array([1.0, 0.0]))
    cases = []
    for s in STEPS_MM:
        for sign in (1, -1):
            cases.append((f"rho {sign * s:+d} mm", model_of(axis=AXIS + sign * s * 1e-3 * unit_axis)))
    for s in STEPS_MM:
        for sign in (1, -1):
            off = OFFSET + sign * s * 1e-3 * unit_off
            if np.dot(off, unit_off) < 0:
                continue  # beyond the axis: a different placement, not a perturbation
            cases.append((f"b   {sign * s:+d} mm", model_of(offset=off)))
    for s in STEPS_DEG:
        for sign in (1, -1):
            cases.append((f"phi {sign * s:+d} deg", model_of(yaw=YAW + np.deg2rad(sign * s))))
    cases = [(n, m, 0.0) for n, m in cases]
    for s in STEPS_THETA:
        for sign in (1, -1):
            cases.append((f"dth {sign * s:+d} deg", m0, np.deg2rad(sign * s)))
    if args.only:
        cases = [c for c in cases if c[0].startswith(args.only)]
    print(f"\n{'perturbation':16s} {'ring moves':>10s} | {'1st order error':>17s} | "
          f"{'exact - 1st':>11s} | {'1st order path':>26s} | {'exact path':>26s}")
    print(f"{'':16s} {'max [mm]':>10s} | {'pos mm':>8s} {'rot deg':>8s} | {'max deg':>11s} | "
          f"{'free? clearance, room':>26s} | {'free? clearance, room':>26s}")
    rows = []
    for name, m1, dth in cases:
        P1 = np.array([m1.puzzle_T(angle(tt) + dth) for tt in t])
        targets = np.array([p1 @ np.linalg.inv(p) @ Ti for p1, p, Ti in zip(P1, P, T)])
        moved = max(1000 * np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(targets, T))
        q1 = carry(kin, to_ring, qs, targets, exact=False)
        qe = carry(kin, to_ring, qs, targets, exact=True)
        errs = np.array([pose_error(Tt, ring_T(kin, to_ring, q)) for Tt, q in zip(targets, q1)])
        gap = float(np.rad2deg(np.abs(qe - q1).max()))
        r1 = check(kin, timing, m1, theta0 + dth, q1)
        re = check(kin, timing, m1, theta0 + dth, qe)

        def fmt(r):
            return (f"{'free' if r[0] else 'HITS':>4s} {1000 * r[1]:+6.1f} mm, "
                    f"{r[3]:4.1f} deg{'' if r[2] else '!'}")
        print(f"{name:16s} {moved:10.1f} | {errs[:, 0].max():8.3f} {errs[:, 1].max():8.3f} | "
              f"{gap:11.3f} | {fmt(r1):>26s} | {fmt(re):>26s}", flush=True)
        rows.append({"case": name, "param": name.split()[0], "step": float(name.split()[1]),
                     "moved_mm": moved, "pos_err_mm": errs[:, 0].max(),
                     "rot_err_deg": errs[:, 1].max(), "exact_gap_deg": gap,
                     "first_free": r1[0], "first_clearance_mm": 1000 * r1[1], "first_room": r1[2],
                     "exact_free": re[0], "exact_clearance_mm": 1000 * re[1], "exact_room": re[2],
                     "converged": bool(np.isfinite(qe).all() and gap < 90)})
    print("\n(clearance = beyond the hoop margin; room = the least joint-limit distance, '!' = "
          f"inside the {np.rad2deg(JOINT_MARGIN):.0f} deg margin)")
    # Reach per parameter and side: the largest step up to which every step is
    # "same family" (exact - 1st < FAMILY_GAP deg, within the joint margin) / free.
    print(f"\nreach (every step up to it passes; family = exact - 1st < {FAMILY_GAP:g} deg "
          f"and within the joint margin):")
    for prm in ("rho", "b", "phi", "dth"):
        sel = [r for r in rows if r["param"] == prm]
        if not sel:
            continue
        unit = "mm" if prm in ("rho", "b") else "deg"
        parts = []
        for sign in (1, -1):
            side = sorted((r for r in sel if np.sign(r["step"]) == sign), key=lambda r: abs(r["step"]))
            fam = free = 0.0
            for r in side:
                if r["converged"] and r["exact_gap_deg"] < FAMILY_GAP and r["exact_room"]:
                    fam = abs(r["step"])
                else:
                    break
            for r in side:
                if r["exact_free"] and r["exact_room"]:
                    free = abs(r["step"])
                else:
                    break
            parts.append(f"{'+' if sign > 0 else '-'}: family {fam:g}, free {free:g}")
        print(f"  {prm:4s} [{unit}]  " + ";  ".join(parts))


if __name__ == "__main__":
    main()
