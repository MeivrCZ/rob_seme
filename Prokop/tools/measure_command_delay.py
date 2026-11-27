"""Measures the robot's command delay and writes it into the config (COMMAND_DELAY).

COMMAND_DELAY is the time from a move_to_q call to the joints starting to move: the
executor's timing model (solver/timing.py) predicts every arrival from it, and the
simulator (simulation.py) uses the same value. The lab check of 2026-11-27 found the
real robot about 57 ms ahead of the model with 0.12 s (tools/lab_check.py timing).

What it does (the same moves as `lab_check.py timing`: 5 small moves near q_home,
joints 1, 4 and 6 only, at speed 0.5, each sent 0.15 s before the previous one is
predicted to end, exactly as the executor sends a traversal):

1. measure: the chain with the current COMMAND_DELAY; the lateness of every move
   (median over its get_q samples of sample time - predicted time); the new delay is
   the current one plus the median lateness;
2. verify: the chain again with the new delay; it must come out within VERIFY_TOL;
3. write: COMMAND_DELAY in solver/timing.py and simulation.py (the two must agree,
   tests/test_timing.py), rounded to 1 ms, only if the measurement is consistent (all
   moves within SPREAD_TOL of each other, the value within PLAUSIBLE) and the
   verification passed.

Every run (and every motion) asks for Enter first; the robot must stand in q_home,
homed by hand (this script never homes). Results: Prokop/experiments/lab/
<time>_command_delay/ (log.txt, summary.json). Nothing is committed.

Usage, from the repo root on the station PC:

    uv run Prokop/tools/measure_command_delay.py            # measure, verify, write
    uv run Prokop/tools/measure_command_delay.py --dry-run  # measure and verify only
    uv run Prokop/tools/measure_command_delay.py --sim      # against the simulator (never writes)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP / "tools"))
import lab_check  # noqa: E402  (sets up the paths, the robot connection and the checks)

CONFIG_FILES = (PROKOP / "solver" / "timing.py", PROKOP / "simulation.py")
PATTERN = re.compile(r"^(COMMAND_DELAY = )([0-9.]+)(\s*#.*)$", re.M)
SPREAD_TOL = 0.015  # s, the moves' lateness must agree to within this (a constant offset)
VERIFY_TOL = 0.010  # s, |median lateness| with the new delay
PLAUSIBLE = (0.02, 0.25)  # s, a delay outside this is a measuring problem, not a delay


def current_delay() -> float:
    from solver.timing import COMMAND_DELAY
    return float(COMMAND_DELAY)


def write_delay(value: float, log) -> None:
    for path in CONFIG_FILES:
        text = path.read_text(encoding="utf-8")
        new, n = PATTERN.subn(lambda m: f"{m.group(1)}{value:.3f}{m.group(3)}", text, count=1)
        if n != 1:
            raise SystemExit(f"COMMAND_DELAY not found in {path}: nothing written")
        path.write_text(new, encoding="utf-8")
        log(f"  written: {path.relative_to(PROKOP.parent)}: COMMAND_DELAY = {value:.3f}")


def measure(robot, delay: float, label: str, args, log) -> dict:
    """One chain (no camera thread) with the given delay: the moves' lateness [s]."""
    from solver.timing import RobotTiming
    timing = RobotTiming(robot, lab_check.TIMING_SPEED, command_delay=delay)
    lab_check.confirm(args, f"{label}: {len(lab_check.TIMING_CHAIN)} small moves near q_home "
                            f"at speed {lab_check.TIMING_SPEED} (COMMAND_DELAY {delay:.3f} s), "
                            f"back to q_home.")
    robot.set_speed_relative(lab_check.TIMING_SPEED)
    rows = [r for r in lab_check.run_chain(robot, timing, False, log) if "move" in r]
    late = np.array([r["error_s"] for r in rows])
    log(f"\n{label} (COMMAND_DELAY {delay:.3f} s): lateness per move "
        + ", ".join(f"{1000 * x:+.0f}" for x in late) + " ms")
    good = late[np.isfinite(late)]
    return {"delay": delay, "late_s": late.tolist(),
            "median_s": float(np.median(good)) if len(good) else float("nan"),
            "spread_s": float(np.ptp(good)) if len(good) else float("nan"), "moves": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="against the simulator (never writes)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dry-run", action="store_true", help="measure and verify, don't write")
    args = ap.parse_args()
    folder = lab_check.OUT / f"{datetime.now():%Y%m%d_%H%M%S}_command_delay{'_sim' if args.sim else ''}"
    log = lab_check.Log(folder)
    log(f"command delay ({'simulator' if args.sim else 'REAL ROBOT'}); results in {folder}")
    old = current_delay()
    summary: dict = {"old": old}
    robot, sim = lab_check.connect(args, log)
    try:
        chain = [robot.q_home] + [robot.q_home + lab_check.D(np.array(o, float))
                                  for o in lab_check.TIMING_CHAIN]
        from solver.timing import RobotTiming
        ok, least = lab_check.path_free(lab_check.free_scene(),
                                        RobotTiming(robot, lab_check.TIMING_SPEED), chain)
        log(f"the chain is {'free' if ok else 'NOT free'} of the table and the cell "
            f"({1000 * least:.0f} mm)")
        if not ok:
            raise SystemExit("the chain collides in the soft model: not moving")

        first = measure(robot, old, "1. measure", args, log)
        summary["measure"] = first
        new = round(old + first["median_s"], 3)
        log(f"  median {1000 * first['median_s']:+.0f} ms, spread {1000 * first['spread_s']:.0f} ms"
            f" -> COMMAND_DELAY {old:.3f} -> {new:.3f} s")
        problems = []
        if not np.isfinite(first["median_s"]):
            problems.append("no lateness measured")
        if first["spread_s"] > SPREAD_TOL:
            problems.append(f"the moves disagree by {1000 * first['spread_s']:.0f} ms "
                            f"(> {1000 * SPREAD_TOL:.0f}): not a constant delay")
        if not PLAUSIBLE[0] <= new <= PLAUSIBLE[1]:
            problems.append(f"{new:.3f} s is outside {PLAUSIBLE} s")
        if not problems:
            second = measure(robot, new, "2. verify", args, log)
            summary["verify"] = second
            log(f"  median {1000 * second['median_s']:+.0f} ms with the new delay "
                f"(must be within +-{1000 * VERIFY_TOL:.0f} ms)")
            if not abs(second["median_s"]) <= VERIFY_TOL:
                problems.append(f"the verification is off by {1000 * second['median_s']:+.0f} ms")
        summary.update(new=new, problems=problems)
        if problems:
            log("\nNOT written: " + "; ".join(problems))
        elif args.sim or args.dry_run:
            log(f"\nnot written ({'simulator' if args.sim else '--dry-run'}): COMMAND_DELAY would "
                f"be {new:.3f} s")
        elif abs(new - old) < 0.001:
            log(f"\nCOMMAND_DELAY {old:.3f} s confirmed: nothing to change")
        else:
            log("\n3. write")
            write_delay(new, log)
            summary["written"] = True
            log("\nCommit it when Prokop says so:\n  git add Prokop/solver/timing.py "
                "Prokop/simulation.py && git commit -m \"COMMAND_DELAY measured\" && git push")
    finally:
        (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        if sim is not None:
            sim.stop()


if __name__ == "__main__":
    main()
