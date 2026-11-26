"""The traversal as one optimisation problem (user, 2026-11-26), separate from the planner.

The 4D path: the wire at N + 1 points from the entry (i = 0) to the end in the stem
(i = N), each point at the time the robot gets there, planned backwards from the end
(t_N = 0, t_(i-1) = t_i - the move's duration). Unknowns: the joint configurations
q_0 .. q_N (bounds: the joint limits minus a margin). Target function:

    sum_i |ring centre(q_i) - wire point_i at t_i|^2  +  lambda * sum_i |q_i - q_(i-1)|^2

The times are unknowns too (user): the gaps dt_i between the points, with the robot
allowed to wait: dt_i >= the move's duration (a smooth upper bound). A small cost per
second of traversal (waiting is not free: the 120 s). Solved with scipy's SLSQP
(bounds: the joint limits minus a margin, 0 <= dt <= MAX_GAP). (An earlier version
alternated: times fixed, configurations optimised, times recomputed: the times then
jumped by seconds between rounds, the targets with them, and it diverged.)

No discrete orientations anywhere (user: the point is to relax the roll and tilt
grids): the unknowns are the joint configurations themselves, and the starting guess
is a continuous position-only IK (damped least squares). The ring against the wire:
two walls on the distance (user), a log barrier -log(1 - (d/D)^2) that is +inf where
the ring would touch the bar. The walls close in as the ring tilts against the wire's
local tangent (user): the rim comes within R cos(tilt) of the bar's axis, so
D(tilt) = R cos(tilt) - the bar's half-diagonal - the ring's half-thickness - margin
(no room left beyond ~57 deg with the 6 mm margin). Afterwards checked: the continuous soft
collision check with the real timing (the whole arm), the largest distance from the
wire, the ring's tilt against the wire.

    uv run python Prokop/experiments/optimize_path.py [--n 20] [--lam-mm-per-deg 1.0]
        [--axis 0.487 0.0] [--offset 0.02 0.0] [--yaw 0] [--direction 1] [--end 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PROKOP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROKOP), str(PROKOP.parent / "Natka")]
from ctu_crs import CRS93  # noqa: E402

from solver import planner  # noqa: E402
from solver.contracts import OMEGA, TurntableModel  # noqa: E402
from solver.hoop_pose import ring_in_flange  # noqa: E402
from solver.soft_collision import SoftScene  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from solver.wire import checkpoint_arclengths, wire_frame_at  # noqa: E402

ENTRY = 0.015  # m before the open end
JOINT_MARGIN = np.deg2rad(5.5)
HOOP_MARGIN = 0.006  # m
RING_INNER = 0.030  # m, the hoop's hole (CAD)
BAR_HALF_DIAGONAL = 0.005 * np.sqrt(2.0)  # m, the 10 mm square bar
RING_HALF_THICKNESS = 0.0033  # m (solver.wire.RING_THICKNESS / 2)
WALL_MARGIN = HOOP_MARGIN  # m, kept inside the walls (the same as the collision check's)
BARRIER_KNEE = 0.99  # the log barrier up to here, then a steep quadratic
DLS_DAMPING = 0.01  # m, damped least squares of the starting guess
DLS_MAX_STEP = 0.1  # rad, largest joint change of one IK step
AXIS_WEIGHT = 0.05  # m per unit of axis misalignment in the starting IK (1 rad ~ 5 cm)
DQ_EPS = 1e-4  # rad, smoothing of |dq| in the durations
TAU = 0.02  # s, temperature of the smooth maximum over the joints
MAX_GAP = 5.0  # s, longest time between two points (move + wait)
ARM_MARGIN = 0.010  # m, the arm's gap to the task and the table (the soft model's)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="wire intervals (N + 1 points)")
    ap.add_argument("--lam-mm-per-deg", type=float, default=1.0,
                    help="lambda: 1 deg of joint change weighs as much as this many mm")
    ap.add_argument("--axis", type=float, nargs=2, default=(0.487, 0.0))
    ap.add_argument("--offset", type=float, nargs=2, default=(0.02, 0.0))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--direction", type=int, default=1)
    ap.add_argument("--end", type=int, default=0, help="which of planner.end_angles")
    ap.add_argument("--maxiter", type=int, default=300, help="SLSQP iterations")
    ap.add_argument("--arm-weight", type=float, default=100.0,
                    help="weight of the arm's soft constraint (per m^2 of missing gap)")
    ap.add_argument("--time-weight", type=float, default=1e-6,
                    help="cost of one second of traversal (m^2): waiting is not free")
    ap.add_argument("--barrier-weight", type=float, default=1e-4,
                    help="weight of the walls' log barrier (m^2 per unit)")
    args = ap.parse_args()

    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, 0.5)
    model = TurntableModel(axis_xy=np.array(args.axis), base_offset=np.array(args.offset),
                           base_yaw=np.deg2rad(args.yaw), direction=args.direction,
                           omega=OMEGA)
    theta_end = planner.end_angles(model)[args.end]
    to_ring = ring_in_flange()
    lam = (args.lam_mm_per_deg * 1e-3 / np.deg2rad(1.0)) ** 2
    mu = args.barrier_weight
    nu = args.arm_weight
    kappa = args.time_weight
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN

    def angle(t):  # t <= 0: before the end
        return theta_end + model.direction * model.omega * t

    s_end = checkpoint_arclengths(0, ENTRY)[-1]
    s = np.linspace(0.0, s_end, args.n + 1)  # i = 0 the entry, i = N the end
    wire_local = np.array([wire_frame_at(x, ENTRY) for x in s])

    def targets(times):
        return np.array([(model.puzzle_T(angle(t)) @ F)[:3, 3]
                         for t, F in zip(times, wire_local)])

    def ring_pos(q):
        return (kin.fk(q) @ to_ring)[:3, 3]

    def times_of(qs):
        t = np.zeros(len(qs))
        for i in range(len(qs) - 2, -1, -1):
            t[i] = t[i + 1] - timing.move_duration(qs[i], qs[i + 1])
        return t

    # The starting guess, continuous too (no orientation grid), and inside the walls (a
    # barrier needs a feasible start): a damped least-squares IK of the ring centre onto
    # the wire and the ring axis onto the tangent, the roll free; from q_home to the end
    # point, then each earlier point from the next point's configuration, at the time
    # it would be reached.
    def jac(q, r):
        J = np.empty((3, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-6
            J[:, j] = (ring_pos(q + dq) - r) / 1e-6
        return J

    def ik_threaded(frame, q):
        """Damped least squares on the ring centre (onto the wire point) and the ring's
        axis (onto the wire's tangent, either way round); the roll about the ring axis
        stays free: a continuous IK, no orientation grid. Returns q and the remaining
        position error [m]."""
        target, tangent = frame[:3, 3], frame[:3, 2]

        def err(q):
            T = kin.fk(q) @ to_ring
            axis = T[:3, 2] * (1.0 if T[:3, 2] @ tangent >= 0 else -1.0)
            return np.concatenate((target - T[:3, 3], AXIS_WEIGHT * np.cross(axis, tangent)))

        for _ in range(1000):
            e = err(q)
            if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < AXIS_WEIGHT * 1e-4:
                break
            J = np.empty((6, 6))
            for j in range(6):
                dq = np.zeros(6)
                dq[j] = 1e-6
                J[:, j] = -(err(q + dq) - e) / 1e-6
            dq = J.T @ np.linalg.solve(J @ J.T + DLS_DAMPING ** 2 * np.eye(6), e)
            dq *= min(1.0, DLS_MAX_STEP / max(np.abs(dq).max(), 1e-12))  # walk, not jump
            q = np.clip(q + dq, lo, hi)
        return q, float(np.linalg.norm(err(q)[:3]))

    qs = [None] * (args.n + 1)
    t_next, q_next = 0.0, np.clip(kin.q_home, lo, hi)
    for i in range(args.n, -1, -1):
        q, t_i = q_next, t_next
        for _ in range(3):  # its time depends on its move to the next point
            t_i = t_next - (timing.move_duration(q, q_next) if i < args.n else 0.0)
            q, err = ik_threaded(model.puzzle_T(angle(t_i)) @ wire_local[i], q)
        if err > 0.001:
            print(f"starting guess: point {i} (s {1000 * s[i]:.0f} mm) only within "
                  f"{1000 * err:.1f} mm of the wire")
        qs[i] = q
        t_next, q_next = t_i, q
    qs = np.array(qs)

    def barrier(rho2):
        """The two walls in rho^2 = (d / D(tilt))^2: -log(1 - rho^2), +inf at rho = 1; a
        steep quadratic beyond rho = BARRIER_KNEE so the line search never sees inf."""
        k2 = BARRIER_KNEE ** 2
        if rho2 < k2:
            return -np.log(1.0 - rho2)
        v0, s0 = -np.log(1.0 - k2), 1.0 / (1.0 - k2)
        x = rho2 - k2
        return v0 + s0 * x + 5.0 * (s0 * x) ** 2

    def room(cos_tilt):
        """D(tilt) [m]: how far the ring centre may be from the wire before the rim (at
        R cos(tilt) from the bar's axis at the closest) touches the bar, margin kept."""
        return RING_INNER * cos_tilt - BAR_HALF_DIAGONAL - RING_HALF_THICKNESS - WALL_MARGIN

    # The arm as a soft constraint (user): convex boxes apart = some separating plane
    # exists = a union of halfspaces; its separating-axis gap (soft_collision.box_gaps:
    # the best of 15 axes) is a max of mins of linear functions. Penalty per pair:
    # max(0, margin - gap)^2. Pairs: the arm (not the robot base, not the ring) against
    # the base and the puzzle, the ring against the base, every moving part against the
    # turntable (inside both its boxes: the larger gap) and above the table (one
    # halfspace). Not the ring against the puzzle: the walls. Not the cell (user).
    from solver.soft_collision import ARM_ORDER, box_gaps, lowest_z
    arm_scene = SoftScene(model)
    leaves = arm_scene._arm_leaves
    part = arm_scene._arm_part[leaves]
    names = [str(arm_scene.arm.name[k]) for k in leaves]
    moving = leaves[part != ARM_ORDER.index("base_link")]
    ring_idx = np.array([k for k, nm in zip(leaves, names) if nm.startswith("hoop_ring")])
    body = np.array([k for k in moving if k not in set(ring_idx)])
    n_obs = len(arm_scene._obs_leaves)
    tags = np.array(arm_scene.obs.tag[:n_obs])
    base_obs = np.flatnonzero(tags == "base")
    task_obs = np.flatnonzero((tags == "base") | (tags == "puzzle"))
    p_body = (np.repeat(body, len(task_obs)), np.tile(task_obs, len(body)))
    p_ring = (np.repeat(ring_idx, len(base_obs)), np.tile(base_obs, len(ring_idx)))
    n_tt = len(arm_scene._turntable)
    p_tt = (np.repeat(moving, n_tt), np.tile(np.arange(n_tt), len(moving)))

    def arm_penalty(q, theta):
        A = arm_scene.arm_boxes(q)
        O = arm_scene.obstacle_boxes(theta)
        gaps = [box_gaps(A, O, *p_body), box_gaps(A, O, *p_ring),
                box_gaps(A, arm_scene._tt_boxes, *p_tt).reshape(-1, n_tt).max(axis=1),
                lowest_z(type(A)(A.c[moving], A.R[moving], A.h[moving])) - arm_scene.table_z]
        g = np.concatenate(gaps)
        short = np.maximum(0.0, ARM_MARGIN - g)
        return float(np.sum(short * short)), float(g.min())

    def point_term(q, w, tangent, theta):
        """|ring centre - wire point|^2 + mu * the walls + nu * the arm, for one point."""
        T = kin.fk(q) @ to_ring
        e = T[:3, 3] - w
        d2 = float(e @ e)
        cos_tilt = abs(float(T[:3, 2] @ tangent))
        D_t = room(cos_tilt)
        arm_pen = nu * arm_penalty(q, theta)[0]
        if D_t <= 1e-3:  # tilted so far there is no room at all: a smooth steep bowl
            return d2 + arm_pen + mu * (barrier(4.0) + 1e7 * (1e-3 - D_t) ** 2)
        return d2 + arm_pen + mu * barrier(d2 / D_t ** 2)

    # --- Times as unknowns (user): the time gaps dt_1 .. dt_N, t_N = 0 at the end,
    # t_i = -(dt_(i+1) + ... + dt_N). The robot may wait: dt_i >= the move's duration
    # (it arrives early and stands at the stop until t_i). The duration as a smooth
    # upper bound: per joint the trapezoid time of sqrt(dq^2 + eps^2), then log-sum-exp
    # over the joints (both never below the true values: a met constraint means the
    # real move fits).
    ramp = timing.v_max / timing.a_max

    def smooth_duration(dq):
        d = np.sqrt(dq * dq + DQ_EPS ** 2)
        T = np.where(d > timing.v_max * ramp, d / timing.v_max + ramp,
                     2.0 * np.sqrt(d / timing.a_max))
        m = T.max()
        return float(m + TAU * np.log(np.sum(np.exp((T - m) / TAU))))

    n_pts = args.n + 1
    n_q = 6 * n_pts

    def unpack(x):
        Q = x[:n_q].reshape(-1, 6)
        dt = x[n_q:]
        times = np.concatenate((-np.cumsum(dt[::-1])[::-1], [0.0]))
        return Q, dt, times

    def frame_at(i, t):
        return model.puzzle_T(angle(t)) @ wire_local[i]

    def term(q, i, t):
        F = frame_at(i, t)
        return point_term(q, F[:3, 3], F[:3, 2], angle(t))

    def objective(x):
        Q, dt, times = unpack(x)
        f, gq, gt = 0.0, np.zeros_like(Q), np.zeros(n_pts)
        for i, (q, t) in enumerate(zip(Q, times)):
            v = term(q, i, t)
            f += v
            for j in range(6):
                dq = np.zeros(6)
                dq[j] = 1e-6
                gq[i, j] = (term(q + dq, i, t) - v) / 1e-6
            gt[i] = (term(q, i, t + 1e-5) - v) / 1e-5
        d = np.diff(Q, axis=0)
        f += lam * float(np.sum(d * d)) + kappa * float(np.sum(dt))
        gq[1:] += 2.0 * lam * d
        gq[:-1] -= 2.0 * lam * d
        # t_i = -(dt_(i+1) + ... + dt_N): d t_i / d dt_k = -1 for k > i (dt index k - 1)
        g_dt = -np.cumsum(gt)[:-1] + kappa  # dt_k affects t_0 .. t_(k-1)
        return f, np.concatenate((gq.ravel(), g_dt))

    def waits(x):
        """dt_i - duration(q_(i-1), q_i) >= 0, one per move."""
        Q, dt, _ = unpack(x)
        return np.array([dt[i] - smooth_duration(Q[i + 1] - Q[i]) for i in range(args.n)])

    def waits_jac(x):
        Q, _, _ = unpack(x)
        J = np.zeros((args.n, len(x)))
        for i in range(args.n):
            dq = Q[i + 1] - Q[i]
            base = smooth_duration(dq)
            for j in range(6):
                e = np.zeros(6)
                e[j] = 1e-7
                g = (smooth_duration(dq + e) - base) / 1e-7
                J[i, 6 * (i + 1) + j] -= g  # d / d q_(i+1)
                J[i, 6 * i + j] += g  # d / d q_i
            J[i, n_q + i] = 1.0
        return J

    def report(label, Q, times):
        dist = np.array([np.linalg.norm(ring_pos(q) - frame_at(i, t)[:3, 3])
                         for i, (q, t) in enumerate(zip(Q, times))])
        steps = np.rad2deg(np.abs(np.diff(Q, axis=0)).max(axis=1))
        print(f"{label}: distance from the wire max {1000 * dist.max():.1f} mm, mean "
              f"{1000 * dist.mean():.1f} mm; joint steps max {steps.max():.1f} deg; "
              f"traversal {-times[0]:.1f} s")

    t0 = time.monotonic()
    dt0 = np.array([max(smooth_duration(qs[i + 1] - qs[i]), timing.move_duration(qs[i], qs[i + 1]))
                    for i in range(args.n)]) + 1e-3
    x0 = np.concatenate((qs.ravel(), dt0))
    report("start", *unpack(x0)[0:1], unpack(x0)[2])
    bounds = ([(a, b) for _ in range(n_pts) for a, b in zip(lo, hi)]
              + [(0.0, MAX_GAP)] * args.n)
    res = minimize(objective, x0, jac=True, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": waits, "jac": waits_jac}],
                   options={"maxiter": args.maxiter, "ftol": 1e-12})
    Q, dt, times = unpack(res.x)
    print(f"SLSQP: {res.message} ({res.nit} iterations), objective {res.fun:.3e}; "
          f"least wait slack {1000 * waits(res.x).min():+.1f} ms")
    report("optimised", Q, times)
    arm_gaps = [arm_penalty(q, angle(t))[1] for q, t in zip(Q, times)]
    print(f"arm's least gap at the points (soft constraint): {1000 * min(arm_gaps):.1f} mm "
          f"(margin {1000 * ARM_MARGIN:.0f} mm)")
    waited = [float(dt[i] - timing.move_duration(Q[i], Q[i + 1])) for i in range(args.n)]
    print(f"waits at the stops: {sum(w > 0.01 for w in waited)} of {args.n} moves wait, "
          f"{sum(max(w, 0) for w in waited):.1f} s in all; ({time.monotonic() - t0:.1f} s)")

    # Checks, not in the target function: the whole arm against the scene along the
    # real motion, with the waits (each move leaves at its point's time, the robot
    # then stands at the next stop until that stop's time while the puzzle turns).
    scene = SoftScene(model, hoop_puzzle_margin=HOOP_MARGIN)
    hits, least = [], np.inf
    for i in range(args.n):
        t_dep = times[i]
        chk = planner.check_moves(scene, lambda x: angle(t_dep + x), model.omega, timing,
                                  np.array([Q[i], Q[i + 1]]), np.array([-1, 0]))[0][0]
        least = min(least, chk.clearance)
        if chk.hits:
            hits.append((i + 1, "move", chk.hits[0]))
        t_arr = t_dep + timing.move_duration(Q[i], Q[i + 1])
        for t in np.arange(t_arr, times[i + 1], 0.05):  # standing while it turns
            c = scene.clearance(Q[i + 1], angle(t))
            least = min(least, min(c.distance, c.self_gap))
            if not c.free:
                hits.append((i + 1, "standing", c.pair))
                break
    print(f"collisions (soft model, {1000 * HOOP_MARGIN:.0f} mm hoop margin): "
          + ("none" if not hits else f"{len(hits)} ({len(set(h[0] for h in hits))} of "
             f"{args.n} moves), e.g. " + ", ".join(f"{k} {w}: {p[0]} - {p[1]}"
                                                   for k, w, p in hits[:4])))
    print(f"least soft clearance {1000 * least:.1f} mm")
    tilts = []
    for i, (q, t) in enumerate(zip(Q, times)):
        axis_ring = (kin.fk(q) @ to_ring)[:3, 2]
        tilts.append(np.degrees(np.arccos(np.clip(abs(axis_ring @ frame_at(i, t)[:3, 2]),
                                                  -1, 1))))
    print(f"ring axis against the wire's tangent: {np.round(tilts).astype(int).tolist()} deg")


if __name__ == "__main__":
    main()
