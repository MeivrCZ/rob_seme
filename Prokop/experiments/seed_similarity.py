"""Does a placement near a seed get a path similar to the seed's? (the plan library's premise)

The seed: the centre placement's ideal plan (jacobian_reach.py's cache). A seed grid
around it with the spacing jacobian_reach suggested there (rho 20 mm, b 20 mm, phi 5
deg); two placements:

    close   rho +2 mm,  b +2 mm,  phi +0.5 deg  (a tenth of a cell from the seed)
    middle  rho +10 mm, b +10 mm, phi +2.5 deg  (the cell's corner: farthest from every
                                                  neighbouring seed)

For each, the planner as run_solver runs it (plan_spinning: 6 end angles, the planner's
own time limits, margins 5 mm / 6 deg; with no limit if that finds nothing) and the
planner at the seed's own end angle only (where a seed-guided planner would start). Each
plan is compared with the seed and with the seed carried to first order (the Jacobian
shift): both paths sampled every 20 ms along their timed moves, each sample paired with
the other path's sample whose ring is nearest in the puzzle frame; the joint difference
of the pairs (max over joints): largest and median; plus the start angle, the traversal
time and the waypoints.

    uv run python Prokop/experiments/seed_similarity.py
"""
from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROKOP), str(PROKOP / "experiments"), str(PROKOP.parent / "Natka")]

import jacobian_reach as jr  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from solver import planner  # noqa: E402
from solver.hoop_pose import ring_in_flange  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402

CASES = {"close": (0.002, 0.002, 0.5), "middle": (0.010, 0.010, 2.5)}  # rho m, b m, phi deg
DT = 0.02


def perturbed(d_rho, d_b, d_phi):
    unit_axis = jr.AXIS / np.linalg.norm(jr.AXIS)
    unit_off = jr.OFFSET / np.linalg.norm(jr.OFFSET)
    return jr.model_of(axis=jr.AXIS + d_rho * unit_axis, offset=jr.OFFSET + d_b * unit_off,
                       yaw=jr.YAW + np.deg2rad(d_phi))


def samples(kin, timing, to_ring, model, theta_start, qs):
    """(ring positions in the puzzle frame, configurations) every DT along the timed path."""
    _, moves = timing.schedule(qs[0], list(qs[1:]))
    t_end = moves[-1].t_end
    ts = np.arange(moves[0].t0, t_end + 1e-9, DT)
    pos, conf = [], []
    for t in ts:
        m = next((mv for mv in moves if t <= mv.t_end), moves[-1])
        q = m.q_at(t)
        P = model.puzzle_T(theta_start + model.direction * model.omega * t)
        pos.append((np.linalg.inv(P) @ kin.fk(q) @ to_ring)[:3, 3])
        conf.append(q)
    return np.array(pos), np.array(conf)


def compare(a, b) -> tuple[float, float, float]:
    """Joint difference [deg] of the ring-nearest pairs (both ways): max, median; and the
    largest ring distance of a pair [mm] (how well the pairing matched)."""
    diffs, dists = [], []
    for (pa, qa), (pb, qb) in ((a, b), (b, a)):
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        j = np.argmin(d, axis=1)
        diffs.append(np.rad2deg(np.abs(qa - qb[j]).max(axis=1)))
        dists.append(d[np.arange(len(pa)), j])
    diffs, dists = np.concatenate(diffs), np.concatenate(dists)
    return float(diffs.max()), float(np.median(diffs)), 1000 * float(dists.max())


def plan(model, kin, timing, candidates=None, unlimited=False):
    t0 = time.monotonic()
    if unlimited:
        angles = candidates or planner.end_angles(model, count=jr.END_ANGLES)
        jobs = [(timing, model, a, 3, i, None, jr.margins()) for i, a in enumerate(angles)]
        found = []
        with multiprocessing.get_context("spawn").Pool(6, initializer=jr._set_limit,
                                                       initargs=(1e9,)) as pool:
            for _, theta, result in pool.imap_unordered(planner._backward_job, jobs):
                if theta is not None and result.collision_free:
                    result.quality = planner.plan_quality(kin, result.qs)
                    found.append((theta, result))
        if not found:
            return None, None, time.monotonic() - t0
        theta, result = max(found, key=lambda u: planner._plan_rank(u[1]))
        return theta, result, time.monotonic() - t0
    theta, result = planner.plan_spinning(
        kin, timing, model, candidates=candidates, margins=jr.margins(),
        workers=1 if candidates else None)
    return theta, result, time.monotonic() - t0


def main() -> None:
    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, jr.SPEED)
    to_ring = ring_in_flange()
    seed = dict(np.load(jr.CACHE_DIR / "jacobian_reach_plan.npz"))
    s_qs, s_t, s_th = np.asarray(seed["qs"]), np.asarray(seed["arrivals"]), float(seed["theta_start"])
    m0 = jr.model_of()
    omega = jr.DIRECTION * m0.omega
    s_end = s_th + omega * s_t[-1]
    print(f"seed (centre): {len(s_qs)} waypoints, traversal {s_t[-1]:.1f} s, start angle "
          f"{np.rad2deg(s_th):.1f} deg, end angle {np.rad2deg(s_end):.1f} deg", flush=True)
    seed_s = samples(kin, timing, to_ring, m0, s_th, s_qs)
    P0 = [m0.puzzle_T(s_th + omega * t) for t in s_t]
    T0 = [kin.fk(q) @ to_ring for q in s_qs]
    for name, (d_rho, d_b, d_phi) in CASES.items():
        m1 = perturbed(d_rho, d_b, d_phi)
        print(f"\n=== {name}: rho {1000 * d_rho:+.0f} mm, b {1000 * d_b:+.0f} mm, phi "
              f"{d_phi:+.1f} deg", flush=True)
        # the seed carried to first order (the same timeline)
        targets = [m1.puzzle_T(s_th + omega * t) @ np.linalg.inv(P) @ T
                   for t, P, T in zip(s_t, P0, T0)]
        carried = jr.carry(kin, to_ring, s_qs, targets, exact=False)
        ok, cl, room_ok, room = jr.check(kin, timing, m1, s_th, carried)
        print(f"  seed carried to first order: {'free' if ok else 'COLLIDES'} (clearance "
              f"{1000 * cl:+.1f} mm beyond the margin), joint room {room:.1f} deg", flush=True)
        carried_s = samples(kin, timing, to_ring, m1, s_th, carried)
        for label, cands in (("planner as in run_solver", None),
                             ("planner at the seed's end angle", [float(s_end)])):
            theta, result, secs = plan(m1, kin, timing, cands)
            how = "with the time limits"
            if result is None:
                print(f"  {label}: nothing with the time limits ({secs:.0f} s) -> no limit", flush=True)
                theta, result, secs = plan(m1, kin, timing, cands, unlimited=True)
                how = "WITHOUT time limits"
            if result is None:
                print(f"  {label}: no plan even without limits ({secs:.0f} s)")
                continue
            qs, t = result.qs, result.arrivals
            end = theta + omega * t[-1]
            ps = samples(kin, timing, to_ring, m1, theta, qs)
            vs_seed = compare(ps, seed_s)
            vs_carried = compare(ps, carried_s)
            sig, rm = planner.plan_quality(kin, qs)
            print(f"  {label} ({how}, {secs:.0f} s): {len(qs)} waypoints, traversal "
                  f"{t[-1]:.1f} s, start {np.rad2deg(theta):.1f} deg, end {np.rad2deg(end):.1f} "
                  f"deg (seed {np.rad2deg(s_end):.1f}), sigma_min {1000 * sig:.1f} mm, joint "
                  f"room {np.rad2deg(rm):.1f} deg")
            print(f"     vs the seed:         joints differ max {vs_seed[0]:5.1f} deg, median "
                  f"{vs_seed[1]:5.1f} deg (ring pairing within {vs_seed[2]:.1f} mm)")
            print(f"     vs the carried seed: joints differ max {vs_carried[0]:5.1f} deg, median "
                  f"{vs_carried[1]:5.1f} deg (ring pairing within {vs_carried[2]:.1f} mm)",
                  flush=True)
            print("     waypoints (deg):")
            for q, tt in zip(qs, t):
                print(f"       t {tt:5.2f}  {np.round(np.rad2deg(q), 1).tolist()}")
    print("\nseed waypoints (deg):")
    for q, tt in zip(s_qs, s_t):
        print(f"       t {tt:5.2f}  {np.round(np.rad2deg(q), 1).tolist()}")


if __name__ == "__main__":
    main()
