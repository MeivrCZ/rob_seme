"""Refining a planned path by optimisation (user, 2026-11-26).

The planner (plan_backward with the binary subdivision) gives a collision-free path:
waypoints q_0 .. q_N with arrival times, the start angle. Its waypoints and the time
gaps between them (the robot may wait: gap >= the move's duration, a smooth upper
bound) are then optimised with scipy's SLSQP, the start angle kept (time runs forward
from the waiting pose, t_0 = 0):

    sum over the terms [ d^2 + mu * walls(d / D(tilt)) + nu * arm penalty ]
      + lambda * sum |q_i - q_(i-1)|^2 + kappa * traversal time

The terms cover the whole motion, not only the waypoints: each waypoint when it is
left, and again when it is reached if the robot waits there (the puzzle turns while
it stands), and 3 samples along each move (1/4, 1/2, 3/4 of the way, at their real
times on the trapezoid profile). d = the ring centre's distance from the wire's
nearest point within the stretch between the neighbouring waypoints; D(tilt) = the
room the walls leave (R cos tilt - the bar's half-diagonal - the ring's half-thickness
- margin); the arm penalty = sum over box pairs of max(0, margin - separating-axis
gap)^2 (convex boxes apart = a union of halfspaces), the ring against the puzzle left
to the walls. The gradient block-wise: a waypoint affects its own terms and its two
moves; a time gap shifts every later term alike.

    uv run python Prokop/experiments/refine_path.py [--axis 0.43 -0.05] [--offset 0 -0.02]
        [--yaw 0] [--direction -1] [--end 0]
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
from solver.soft_collision import ARM_ORDER, SoftScene, box_gaps, lowest_z  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from solver.wire import checkpoint_arclengths, wire_frame_at  # noqa: E402

HOOP_MARGIN = 0.006  # m (the run's margin)
JOINT_MARGIN = np.deg2rad(5.5)
RING_INNER, RING_HALF_THICKNESS = 0.030, 0.0033
BAR_HALF_DIAGONAL = 0.005 * np.sqrt(2.0)
ARM_MARGIN = 0.010  # m
BARRIER_KNEE = 0.99
DQ_EPS, TAU = 1e-4, 0.02  # smooth duration: |dq| smoothing [rad], max temperature [s]
SAMPLES = (0.25, 0.5, 0.75)  # along each move
WIRE_STEP = 0.001  # m, the wire sampled for the nearest point
MAX_GAP = 5.0  # s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", type=float, nargs=2, default=(0.43, -0.05))
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, -0.02))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--direction", type=int, default=-1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--lam-mm-per-deg", type=float, default=1.0)
    ap.add_argument("--barrier-weight", type=float, default=1e-4)
    ap.add_argument("--arm-weight", type=float, default=100.0)
    ap.add_argument("--time-weight", type=float, default=1e-5,
                    help="cost of one second of traversal (m^2)")
    ap.add_argument("--maxiter", type=int, default=100)
    args = ap.parse_args()

    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, 0.5)
    model = TurntableModel(axis_xy=np.array(args.axis), base_offset=np.array(args.offset),
                           base_yaw=np.deg2rad(args.yaw), direction=args.direction,
                           omega=OMEGA)
    scene = SoftScene(model, hoop_puzzle_margin=HOOP_MARGIN)
    to_ring = ring_in_flange()
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN
    lam = (args.lam_mm_per_deg * 1e-3 / np.deg2rad(1.0)) ** 2
    mu, nu, kappa = args.barrier_weight, args.arm_weight, args.time_weight

    # 1. The planner's path (the subdivision first, the tree as fallback).
    t_plan = time.monotonic()
    theta_end = planner.end_angles(model)[args.end]
    theta_start, plan = planner.plan_backward(kin, timing, model, theta_end, scene=scene,
                                              joint_margin=JOINT_MARGIN)
    if theta_start is None or not plan.collision_free:
        print(f"the planner found no path ({time.monotonic() - t_plan:.0f} s): {plan.reason}")
        return
    Q0 = np.array(plan.qs)
    print(f"planner: {len(Q0)} waypoints in {time.monotonic() - t_plan:.0f} s, traversal "
          f"{plan.arrivals[-1]:.2f} s, least soft clearance {1000 * plan.min_clearance:.1f} mm")
    entry = plan.entry

    def angle(t):  # forward time from the waiting pose
        return theta_start + model.direction * model.omega * t

    s_wire = np.arange(0.0, checkpoint_arclengths(0, max(planner.ENTRIES))[-1] + 1e-9,
                       WIRE_STEP)
    wire_frames = [wire_frame_at(x, max(planner.ENTRIES)) for x in s_wire]
    wire_pts = np.array([F[:3, 3] for F in wire_frames])
    wire_tan = np.array([F[:3, 2] for F in wire_frames])

    def nearest(ring_world, t, window):
        """Distance to the wire's nearest point within the window of indices, its tangent
        (world), and the index."""
        P = model.puzzle_T(angle(t))
        local = (np.linalg.inv(P) @ np.append(ring_world, 1.0))[:3]
        a, b = window
        d = np.linalg.norm(wire_pts[a:b] - local, axis=1)
        k = a + int(np.argmin(d))
        return float(np.linalg.norm(wire_pts[k] - local)), P[:3, :3] @ wire_tan[k], k

    # The waypoints' places on the wire (each projected at its time, in order), and so
    # each term's window: between the neighbouring waypoints' places.
    places = []
    for q, t in zip(Q0, plan.arrivals):
        lo_k = places[-1] if places else 0
        places.append(nearest((kin.fk(q) @ to_ring)[:3, 3], t, (lo_k, len(s_wire)))[2])
    n_way = len(Q0)

    def window(i0, i1):  # indices between waypoints i0 .. i1 (with a little slack)
        a = places[max(i0, 0)] - 10
        b = places[min(i1, n_way - 1)] + 10
        return max(a, 0), min(b, len(s_wire))

    # 2. The terms.
    leaves = scene._arm_leaves
    part = scene._arm_part[leaves]
    names = [str(scene.arm.name[k]) for k in leaves]
    moving = leaves[part != ARM_ORDER.index("base_link")]
    ring_idx = np.array([k for k, nm in zip(leaves, names) if nm.startswith("hoop_ring")])
    body = np.array([k for k in moving if k not in set(ring_idx)])
    n_obs = len(scene._obs_leaves)
    tags = np.array(scene.obs.tag[:n_obs])
    base_obs = np.flatnonzero(tags == "base")
    task_obs = np.flatnonzero((tags == "base") | (tags == "puzzle"))
    p_body = (np.repeat(body, len(task_obs)), np.tile(task_obs, len(body)))
    p_ring = (np.repeat(ring_idx, len(base_obs)), np.tile(base_obs, len(ring_idx)))
    n_tt = len(scene._turntable)
    p_tt = (np.repeat(moving, n_tt), np.tile(np.arange(n_tt), len(moving)))

    def arm_gaps(q, theta):
        A = scene.arm_boxes(q)
        O = scene.obstacle_boxes(theta)
        return np.concatenate([
            box_gaps(A, O, *p_body), box_gaps(A, O, *p_ring),
            box_gaps(A, scene._tt_boxes, *p_tt).reshape(-1, n_tt).max(axis=1),
            lowest_z(type(A)(A.c[moving], A.R[moving], A.h[moving])) - scene.table_z])

    def barrier(rho2):
        k2 = BARRIER_KNEE ** 2
        if rho2 < k2:
            return -np.log(1.0 - rho2)
        v0, s0 = -np.log(1.0 - k2), 1.0 / (1.0 - k2)
        x = rho2 - k2
        return v0 + s0 * x + 5.0 * (s0 * x) ** 2

    def term(q, t, win):
        """One moment: the ring against the wire (distance + walls) and the arm."""
        T = kin.fk(q) @ to_ring
        d, tangent, _ = nearest(T[:3, 3], t, win)
        D_t = (RING_INNER * abs(float(T[:3, 2] @ tangent)) - BAR_HALF_DIAGONAL
               - RING_HALF_THICKNESS - HOOP_MARGIN)
        walls = (barrier(4.0) + 1e7 * (1e-3 - D_t) ** 2 if D_t <= 1e-3
                 else barrier(d * d / D_t ** 2))
        short = np.maximum(0.0, ARM_MARGIN - arm_gaps(q, angle(t)))
        return d * d + mu * walls + nu * float(np.sum(short * short))

    def frac_time(u, T, tau):
        """Time at which a trapezoid move (duration T, ramp tau) has covered fraction u."""
        v = 1.0 / max(T - tau, 1e-9)
        if u <= 0.5 * v * tau:
            return np.sqrt(2.0 * u * tau / v)
        if u <= 1.0 - 0.5 * v * tau:
            return tau + (u - 0.5 * v * tau) / v
        return T - np.sqrt(2.0 * (1.0 - u) * tau / v)

    ramp = timing.v_max / timing.a_max

    def smooth_duration(dq):
        d = np.sqrt(dq * dq + DQ_EPS ** 2)
        Tj = np.where(d > timing.v_max * ramp, d / timing.v_max + ramp,
                      2.0 * np.sqrt(d / timing.a_max))
        m = Tj.max()
        return float(m + TAU * np.log(np.sum(np.exp((Tj - m) / TAU))))

    n_q = 6 * n_way

    def unpack(x):
        Q = x[:n_q].reshape(-1, 6)
        dt = x[n_q:]
        return Q, dt, np.concatenate(([0.0], np.cumsum(dt)))

    def move_terms(Q, times, i):
        """The terms of move i -> i + 1 (samples along it, the arrival if it waits),
        each with the time it is at relative to t_i (for the time derivative)."""
        T, tau = timing.profile(Q[i + 1] - Q[i])
        out = []
        for u in SAMPLES:
            out.append((Q[i] + u * (Q[i + 1] - Q[i]), frac_time(u, T, tau), window(i, i + 1)))
        if times[i] + T < times[i + 1] - 0.02:  # it waits at i + 1: also when it arrives
            out.append((Q[i + 1], T, window(i, i + 2)))
        return out

    def energy(x, which=None):
        """The objective, or only the terms touching waypoint `which`; with the list of
        (value, absolute time index) for the time derivative."""
        Q, dt, times = unpack(x)
        f = 0.0
        idx = range(n_way) if which is None else [k for k in (which - 1, which, which + 1)
                                                  if 0 <= k < n_way]
        for i in idx:
            if which is None or i == which:
                f += term(Q[i], times[i], window(i - 1, i + 1))
            if i + 1 < n_way and (which is None or i in (which - 1, which)):
                for q, dtm, win in move_terms(Q, times, i):
                    f += term(q, times[i] + dtm, win)
        if which is None:
            d = np.diff(Q, axis=0)
            f += lam * float(np.sum(d * d)) + kappa * float(np.sum(dt))
        return f

    def objective(x):
        Q, dt, times = unpack(x)
        f = energy(x)
        g = np.zeros_like(x)
        h = 1e-6
        for i in range(n_way):  # block-wise: waypoint i's own terms and its moves
            base = energy(x, i)
            for j in range(6):
                xp = x.copy()
                xp[6 * i + j] += h
                g[6 * i + j] = (energy(xp, i) - base) / h
        d = np.diff(Q, axis=0)
        gq = g[:n_q].reshape(-1, 6)
        gq[1:] += 2.0 * lam * d
        gq[:-1] -= 2.0 * lam * d
        # A gap dt_k shifts every term from waypoint k + 1 on (and move k's arrival
        # term, which ends there): the derivative of those terms' sum in time.
        for k in range(len(dt)):
            xp = x.copy()
            xp[n_q + k] += 1e-5
            g[n_q + k] = (energy(xp) - f) / 1e-5
        return f, g

    def waits(x):
        Q, dt, _ = unpack(x)
        return np.array([dt[i] - smooth_duration(Q[i + 1] - Q[i]) for i in range(n_way - 1)])

    def waits_jac(x):
        Q, _, _ = unpack(x)
        J = np.zeros((n_way - 1, len(x)))
        for i in range(n_way - 1):
            dq = Q[i + 1] - Q[i]
            base = smooth_duration(dq)
            for j in range(6):
                e = np.zeros(6)
                e[j] = 1e-7
                gj = (smooth_duration(dq + e) - base) / 1e-7
                J[i, 6 * (i + 1) + j] -= gj
                J[i, 6 * i + j] += gj
            J[i, n_q + i] = 1.0
        return J

    # 3. Evaluation: the real motion with its waits, against the soft model.
    def evaluate(label, Q, times):
        least, hits, worst_d, tilts = np.inf, [], 0.0, []
        for i in range(n_way - 1):
            chk = planner.check_moves(scene, lambda s: angle(times[i] + s), model.omega,
                                      timing, np.array([Q[i], Q[i + 1]]), np.array([-1, 0]))[0][0]
            least = min(least, chk.clearance)
            if chk.hits:
                hits.append((i + 1, "move", chk.hits[0]))
            t_arr = times[i] + timing.move_duration(Q[i], Q[i + 1])
            for t in np.arange(t_arr, times[i + 1], 0.05):
                c = scene.clearance(Q[i + 1], angle(t))
                least = min(least, min(c.distance, c.self_gap))
                if not c.free:
                    hits.append((i + 1, "standing", c.pair))
                    break
        for i, (q, t) in enumerate(zip(Q, times)):
            T = kin.fk(q) @ to_ring
            d, tangent, _ = nearest(T[:3, 3], t, window(i - 1, i + 1))
            worst_d = max(worst_d, d)
            tilts.append(np.degrees(np.arccos(min(1.0, abs(float(T[:3, 2] @ tangent))))))
        waited = sum(max(0.0, times[i + 1] - times[i] - timing.move_duration(Q[i], Q[i + 1]))
                     for i in range(n_way - 1))
        print(f"{label}: traversal {times[-1]:.2f} s (waiting {waited:.2f} s); least soft "
              f"clearance {1000 * least:.1f} mm; "
              + ("no collision" if not hits else f"{len(hits)} collisions, e.g. "
                 + ", ".join(f"{k} {w}: {p[0]} - {p[1]}" for k, w, p in hits[:3]))
              + f"; ring from the wire at the waypoints max {1000 * worst_d:.1f} mm; tilts "
              f"{np.round(tilts).astype(int).tolist()} deg")

    times0 = np.array(plan.arrivals) - plan.arrivals[0]
    evaluate("planner  ", Q0, times0)
    dt0 = np.maximum(np.diff(times0), [smooth_duration(Q0[i + 1] - Q0[i]) + 1e-3
                                       for i in range(n_way - 1)])
    x0 = np.concatenate((Q0.ravel(), dt0))
    bounds = ([(a, b) for _ in range(n_way) for a, b in zip(lo, hi)]
              + [(0.0, MAX_GAP)] * (n_way - 1))
    t_opt = time.monotonic()
    res = minimize(objective, x0, jac=True, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": waits, "jac": waits_jac}],
                   options={"maxiter": args.maxiter, "ftol": 1e-12})
    Q, dt, times = unpack(res.x)
    print(f"SLSQP: {res.message} ({res.nit} iterations, {time.monotonic() - t_opt:.0f} s); "
          f"objective {energy(x0):.3e} -> {res.fun:.3e}")
    evaluate("refined  ", Q, times)


if __name__ == "__main__":
    main()
