"""Analyzes the motion logs of tools/log_motion.py (experiments E5, E6).

Fits each move with a trapezoidal profile (motion start, joint speed,
acceleration), extracts the delays of the serial line and, with --sim, runs the
same tests in CRS93Sim for comparison.

Usage:
    From the repo root::

        uv run Prokop/tools/analyze_motion.py               # the real logs
        uv run Prokop/tools/analyze_motion.py --sim         # + the same tests in the sim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

PROKOP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROKOP))
sys.path.insert(0, str(PROKOP / "tools"))
LOGS = PROKOP / "experiments" / "motion_logs" / "real"
MOVING = 1.0  # deg/s, a joint counts as moving above this speed
GAP = 0.3  # s, a pause this long ends the out leg of a test


def load_log(path: Path) -> np.ndarray:
    """Loads a log_motion CSV.

    Args:
        path: The CSV file.

    Returns:
        Rows [t, q1..q6, get_q duration] (s, rad, s).
    """
    return np.loadtxt(path, delimiter=",", skiprows=1)


def trapezoid(t: np.ndarray, t0: float, v: float, a: float, d: float) -> np.ndarray:
    """Position along a trapezoidal (or triangular) move from rest to rest.

    Args:
        t: Times [s].
        t0: Start of the move [s].
        v: Max speed.
        a: Acceleration.
        d: Distance (> 0), in the units of v and a.

    Returns:
        The distance travelled at @t.
    """
    t = np.clip(t - t0, 0, None)
    ta = v / a
    if d < v * ta:  # never reaches v
        ta = np.sqrt(d / a)
        v = a * ta
    T = d / v + ta
    s = np.where(t < ta, 0.5 * a * t**2,
                 np.where(t < T - ta, 0.5 * a * ta**2 + v * (t - ta),
                          d - 0.5 * a * np.clip(T - t, 0, None) ** 2))
    return np.where(t >= T, d, s)


def fit_first_move(log: np.ndarray, joint: int) -> dict:
    """Fits the first move of a joint (from rest to its farthest point) with a trapezoid.

    Args:
        log: Rows from load_log.
        joint: Joint index (0-based).

    Returns:
        start_s (motion start after the first move_to_q), speed_deg_s,
        accel_deg_s2 and rms_deg of the fit.
    """
    t, q = log[:, 0], log[:, 1 + joint]
    disp = np.abs(q - q[0])
    d = disp.max()
    k = int(np.argmax(disp >= 0.999 * d)) + 3
    res = least_squares(lambda p: trapezoid(t[:k], p[0], p[1], p[2], d) - disp[:k],
                        [0.1, 0.5, 2.0], bounds=([0, 0.01, 0.05], [1, 5, 100]))
    t0, v, a = res.x
    return {"start_s": float(t0), "speed_deg_s": float(np.rad2deg(v)),
            "accel_deg_s2": float(np.rad2deg(a)),
            "rms_deg": float(np.rad2deg(np.sqrt(np.mean(res.fun**2))))}


def out_leg(log: np.ndarray, joints: list[int]) -> dict:
    """Times the out leg of a test (until the first pause longer than GAP).

    Args:
        log: Rows from load_log.
        joints: Joints (0-based) that move in the test.

    Returns:
        start_s, end_s and duration_s of the leg, and min_via_speed_deg_s: the
        lowest speed of the moving joints inside the leg (away from its ends),
        near 0 if the arm (nearly) stops at a via point.
    """
    t = log[:, 0]
    v = np.abs(np.gradient(np.rad2deg(log[:, 1:7]), t, axis=0))[:, joints].max(axis=1)
    moving = np.flatnonzero(v > MOVING)
    end = moving[0]
    for a, b in zip(moving[:-1], moving[1:]):
        if t[b] - t[a] > GAP:
            break
        end = b
    inner = v[moving[0] + 3:end - 3]
    return {"start_s": float(t[moving[0]]), "end_s": float(t[end]),
            "duration_s": float(t[end] - t[moving[0]]),
            "min_via_speed_deg_s": float(inner.min()) if len(inner) else float("nan")}


def delays(logs: dict[str, np.ndarray]) -> dict:
    """Delays of the serial line from all logs.

    Args:
        logs: Logs by test name.

    Returns:
        get_q_s (median round trip), loop_s (median sample period; one get_q
        plus one in_motion per sample) and in_motion_s (their difference).
    """
    get_q = np.median(np.concatenate([log[:, 7] for log in logs.values()]))
    loop = np.median(np.concatenate([np.diff(log[:, 0]) for log in logs.values()]))
    return {"get_q_s": float(get_q), "loop_s": float(loop),
            "in_motion_s": float(loop - get_q)}


# Joints (0-based) moving in each test of log_motion.TESTS.
JOINTS = {"j1_30_speed025": [0], "j1_30_speed050": [0], "j6_90_speed050": [5],
          "j1_queued_15_15": [0], "j1_30_j4_60_coord": [0, 3],
          "j1_j4_corner": [0, 3], "j1_reverse": [0], "j1_vias_10x3": [0],
          "j1_j4_arc_vias": [0, 3]}


def report(logs: dict[str, np.ndarray], title: str) -> None:
    """Prints the fits, the out legs and the delays of a set of logs.

    Args:
        logs: Logs by test name.
        title: Heading of the printout.
    """
    print(f"=== {title}")
    print("profile fits (first move of each moving joint):")
    for name in ("j1_30_speed025", "j1_30_speed050", "j6_90_speed050",
                 "j1_30_j4_60_coord"):
        if name in logs:
            for j in JOINTS[name]:
                f = fit_first_move(logs[name], j)
                print(f"  {name:18s} j{j + 1}: start {1000 * f['start_s']:4.0f} ms, "
                      f"speed {f['speed_deg_s']:6.1f} deg/s, accel "
                      f"{f['accel_deg_s2']:6.0f} deg/s^2, rms {f['rms_deg']:.2f} deg")
    print("out legs (blending):")
    for name, log in logs.items():
        leg = out_leg(log, JOINTS[name])
        print(f"  {name:18s} {leg['duration_s']:.2f} s, lowest speed at the vias "
              f"{leg['min_via_speed_deg_s']:5.1f} deg/s")
    d = delays(logs)
    print(f"delays: get_q {1000 * d['get_q_s']:.0f} ms, loop {1000 * d['loop_s']:.0f} ms"
          f" -> in_motion {1000 * d['in_motion_s']:.0f} ms")


def main() -> None:
    """Analyzes the real logs and, with --sim, the same tests in the simulation."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true",
                        help="also run the tests in CRS93Sim (real time, ~40 s)")
    args = parser.parse_args()
    real = {p.stem: load_log(p) for p in sorted(LOGS.glob("*.csv")) if p.stem in JOINTS}
    report(real, f"real CRS93 ({LOGS.relative_to(PROKOP.parent)})")
    if args.sim:
        import log_motion
        from simulation import CRS93Sim
        robot = CRS93Sim(port=8123, keep_alive=False)
        robot.initialize(home=False)
        sim = {name: log_motion.run_test(robot, *log_motion.TESTS[name])
               for name in real}
        report(sim, "CRS93Sim, default delays")


if __name__ == "__main__":
    main()
