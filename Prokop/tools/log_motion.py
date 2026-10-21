"""Records joint trajectories of short test moves, on the real robot or the sim.

Measures the real motion profile, the delays of the serial line and how queued
moves are blended (experiments E5 and E6 in docs/EXPERIMENTS.md).

Usage:
    From the repo root::

        uv run Prokop/tools/log_motion.py                 # simulation
        uv run Prokop/tools/log_motion.py --real          # real CRS93
        uv run Prokop/tools/log_motion.py --real --tests j1_j4_corner j1_reverse

The real robot is NOT homed by this script (homing is not collision checked and
can wreck the arm from a bad pose): home it by hand and leave it in q_home.

Every test starts and ends in q_home and keeps the arm up (joints 1, 4 and 6
only, at most 30 deg on joint 1). The queued tests send several moves without
waiting, to see how the control unit blends them (same direction, corner,
reversal, chains of via points like a planned path).

Writes one CSV per test to Prokop/experiments/motion_logs/<real|sim>/ with the
columns t [s], q1..q6 [rad], get_q_duration [s], and prints the duration, the
peak joint speeds and the delays for CRS93Sim: the start latency (move_to_q
until the joints move, about command_delay +- half a sampling period) and the
get_q round trip (query_delay); the loop period minus get_q is about
status_delay (in_motion). tools/analyze_motion.py fits the logs in detail.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROKOP))

SETTLE = 0.5  # s, logging continues after in_motion() turns False
SAMPLE_DT = 0.01  # s, at most 100 Hz (the real get_q is a serial query, slower)
D = np.deg2rad

# name: (relative speed, list of joint offsets from q_home [deg] sent without
# waiting in between; the return to q_home is logged as part of the test)
MOVED = 2e-4  # rad, a joint counts as moving after this change (~10 IRC)
TESTS = {
    "j1_30_speed025": (0.25, [[30, 0, 0, 0, 0, 0]]),
    "j1_30_speed050": (0.5, [[30, 0, 0, 0, 0, 0]]),
    "j6_90_speed050": (0.5, [[0, 0, 0, 0, 0, 90]]),
    "j1_queued_15_15": (0.5, [[15, 0, 0, 0, 0, 0], [30, 0, 0, 0, 0, 0]]),
    "j1_30_j4_60_coord": (0.5, [[30, 0, 0, 60, 0, 0]]),
    "j1_j4_corner": (0.5, [[20, 0, 0, 0, 0, 0], [20, 0, 0, 40, 0, 0]]),
    "j1_reverse": (0.5, [[20, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]]),
    "j1_vias_10x3": (0.5, [[3 * (i + 1), 0, 0, 0, 0, 0] for i in range(10)]),
    "j1_j4_arc_vias": (0.5, [[30 * np.sin(a), 0, 0, 60 * (1 - np.cos(a)), 0, 0]
                             for a in np.linspace(0, np.pi / 2, 10)[1:]]),
}


def log_until_stopped(robot, t0: float, rows: list) -> None:
    """Samples the configuration until the motion stops, plus SETTLE.

    Samples every SAMPLE_DT, or as fast as get_q allows.

    Args:
        robot: CRS93 or CRS93Sim.
        t0: Time origin (time.monotonic) of the test [s].
        rows: List to append [t, q1..q6, get_q duration] to; t is the middle of
            the get_q call.
    """
    stop = None
    while stop is None or time.monotonic() < stop:
        t = time.monotonic()
        q = robot.get_q()
        t_done = time.monotonic()
        rows.append([(t + t_done) / 2 - t0, *q, t_done - t])
        if stop is None and not robot.in_motion():
            stop = time.monotonic() + SETTLE
        time.sleep(max(0.0, t + SAMPLE_DT - time.monotonic()))


def run_test(robot, speed: float, offsets: list) -> np.ndarray:
    """Runs one test from q_home and back, logging the motion.

    Args:
        robot: CRS93 or CRS93Sim, standing in q_home.
        speed: Relative speed (set_speed_relative).
        offsets: Targets as offsets from q_home [deg], sent without waiting.

    Returns:
        Log rows [t, q1..q6, get_q duration] of the moves out and back.
    """
    robot.set_speed_relative(speed)
    rows: list = []
    t0 = time.monotonic()
    for dq in offsets:
        robot.move_to_q(robot.q_home + D(dq))
    log_until_stopped(robot, t0, rows)
    robot.move_to_q(robot.q_home)
    log_until_stopped(robot, t0, rows)
    return np.array(rows)


def summarize(log: np.ndarray) -> dict:
    """Summarizes a test log.

    Args:
        log: Rows [t, q1..q6, get_q duration] from run_test.

    Returns:
        duration_s, samples, sample_rate_hz, start_latency_s (midpoint between
        the last still and the first moving sample), get_q_s (median) and
        peak_speed_deg_s per moving joint.
    """
    t, q, query = log[:, 0], log[:, 1:7], log[:, 7]
    i = max(1, int(np.argmax(np.abs(q - q[0]).max(axis=1) > MOVED)))
    moving = np.flatnonzero(np.ptp(q, axis=0) > 1e-4)
    speeds = np.abs(np.diff(q[:, moving], axis=0)) / np.diff(t)[:, None]
    return {
        "duration_s": float(t[-1]),
        "samples": len(t),
        "sample_rate_hz": float(len(t) / t[-1]),
        "start_latency_s": float((t[i - 1] + t[i]) / 2),  # last still / first moving
        "get_q_s": float(np.median(query)),
        "peak_speed_deg_s": {f"j{j + 1}": float(np.rad2deg(v))
                             for j, v in zip(moving, speeds.max(axis=0))},
    }


def save_test(out: Path, name: str, log: np.ndarray) -> dict:
    """Writes the CSV of a test and prints its summary.

    Args:
        out: Output directory.
        name: Test name (CSV file name).
        log: Rows from run_test.

    Returns:
        The summary (see summarize).
    """
    np.savetxt(out / f"{name}.csv", log, delimiter=",",
               header="t,q1,q2,q3,q4,q5,q6,get_q_duration", comments="")
    s = summarize(log)
    print(f"{name}: {s['duration_s']:.2f} s incl. return and {2 * SETTLE:.1f} s "
          f"settle, {s['samples']} samples ({s['sample_rate_hz']:.0f} Hz), start "
          f"latency {1000 * s['start_latency_s']:.0f} ms, get_q "
          f"{1000 * s['get_q_s']:.1f} ms, peak speed [deg/s] "
          + ", ".join(f"{j} {v:.1f}" for j, v in s["peak_speed_deg_s"].items()))
    return s


def main() -> None:
    """Runs the selected tests on the real robot or the simulation."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--real", action="store_true",
                        help="use the real CRS93 (already homed, standing in q_home)")
    parser.add_argument("--tests", nargs="+", choices=list(TESTS), default=list(TESTS),
                        help="run only these tests (default: all)")
    args = parser.parse_args()

    if args.real:
        from ctu_crs import CRS93
        robot = CRS93()
    else:
        from simulation import CRS93Sim
        robot = CRS93Sim(keep_alive=False)
    out = PROKOP / "experiments" / "motion_logs" / ("real" if args.real else "sim")
    out.mkdir(parents=True, exist_ok=True)

    robot.initialize(home=False)
    if not np.allclose(robot.get_q(), robot.q_home, atol=np.deg2rad(2)):
        raise SystemExit(f"the robot is not in q_home (q [deg] = "
                         f"{np.rad2deg(robot.get_q()).round(1)}); move it there first")
    input("Robot in q_home: runs the tests (joints 1, 4, 6 only). Enter to start ")

    for name in args.tests:
        speed, offsets = TESTS[name]
        save_test(out, name, run_test(robot, speed, offsets))
    robot.close()
    print("CSV logs in", out)


if __name__ == "__main__":
    main()
