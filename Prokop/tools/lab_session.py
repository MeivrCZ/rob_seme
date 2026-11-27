"""The whole lab check session (docs/LAB_CHECK_GUIDE.md) in one run, recorded.

Runs the checks of tools/lab_check.py in the guide's order, then the solver's dry
run (run_solver.py --real --stop-after 2), each as its own process with the
terminal attached, so every motion still asks the operator for Enter. Between the
checks it says what to set up, asks whether to run, skip or quit, collects what
only a person can give (the caliper values, what was seen by eye, notes), judges
each result against the guide's pass limits, and keeps a report up to date:

    Prokop/experiments/lab/session_<time>/report.md     results, deviations, notes
    Prokop/experiments/lab/session_<time>/session.json  everything, for --resume

The checks' own folders (log.txt, summary.json, images) stay where lab_check.py
and run_solver.py put them; the report links them. Nothing in the solver is
changed and nothing is committed.

Usage, from the repo root on the station PC:

    git pull && uv sync
    uv run Prokop/tools/lab_session.py              # the whole session
    uv run Prokop/tools/lab_session.py --resume     # go on with the latest session
    uv run Prokop/tools/lab_session.py --only 4 5   # only some steps
    uv run Prokop/tools/lab_session.py --sim        # the same against the simulator

Before the motion steps the robot must stand in q_home, homed BY HAND by Prokop
(the checks refuse otherwise; this script never homes). Keep a hand on the e-stop.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROKOP = Path(__file__).resolve().parents[1]
REPO = PROKOP.parent
LAB_CHECK = PROKOP / "tools" / "lab_check.py"
RUN_SOLVER = PROKOP / "run_solver.py"
LAB_OUT = PROKOP / "experiments" / "lab"
RUNS = PROKOP / "runs"
CALIBRATION_DIR = PROKOP / "data" / "calibration"

DIMS_TOLERANCE_MM = 1.0
RUN_LIMIT_S = 120.0

STEPS = {
    1: ("dims", "the model's dimensions (caliper)", False,
        "No robot. Have a caliper and a tape ready: the puzzle, the hoop and the base."),
    2: ("calibrate", "camera calibration (homography)", True,
        "THE ROBOT MOVES. Robot homed by hand, in q_home. Work area EMPTY: turntable, base "
        "and puzzle removed (the hoop goes down to 75 mm above the table). Someone at the "
        "e-stop."),
    3: ("timing", "timing model on the real robot", True,
        "THE ROBOT MOVES (5 small moves near q_home, twice). Robot in q_home, the arm's "
        "surroundings free. Someone at the e-stop."),
    4: ("turntable", "perception on the spinning turntable", False,
        "No motion. Put the turntable back, the base with the puzzle on it, spinning at the "
        "slowest speed, in the camera's view."),
    5: ("fov", "where the camera sees the base", False,
        "No motion. Someone moves the base (turntable stopped, or by hand) along the edges "
        "and into the corners of the work area (x 0.337-0.637 m, y -0.2425..+0.2425 m). "
        "Ctrl+C ends the check."),
    6: ("dryrun", "solver dry run (perception + planning, no motion)", False,
        "No motion. Set-up as in step 4 (turntable spinning, base and puzzle on it). The "
        "robot must stand in q_home."),
}
MOTION_STEPS = {2, 3}


# --- terminal helpers ---

def ask(prompt: str, default: str = "") -> str:
    try:
        return input(prompt).strip() or default
    except EOFError:
        return default


def ask_float(prompt: str) -> float | None:
    while True:
        text = ask(prompt)
        if not text:
            return None
        try:
            return float(text.replace(",", "."))
        except ValueError:
            print("    a number in mm, or blank to skip")


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


# --- running the checks ---

def run_child(cmd: list[str]) -> tuple[int, str]:
    """Runs a child with the terminal (stdin/stdout) attached; stderr is shown live and
    kept (the checks report their refusals there). Ctrl+C goes to the child (fov ends
    that way); this script waits for it to finish."""
    print(f"$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    err_lines: list[str] = []

    def pump():
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            err_lines.append(line)
    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    while True:
        try:
            code = proc.wait()
            break
        except KeyboardInterrupt:
            continue  # the child got the Ctrl+C too and ends on its own
    pump_thread.join(timeout=2.0)
    return code, "".join(err_lines)


def newest_folder(parent: Path, pattern: str, since: float) -> Path | None:
    found = [p for p in parent.glob(pattern) if p.is_dir() and p.stat().st_mtime >= since - 1]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def read_summary(folder: Path | None) -> dict | None:
    if folder is None or not (folder / "summary.json").exists():
        return None
    try:
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def error_tail(stderr: str, lines: int = 6) -> str:
    return "\n".join(stderr.strip().splitlines()[-lines:])


# --- verdicts: (what, measured, limit, ok) rows ---

def judge(rows: list[tuple]) -> tuple[bool | None, list[dict]]:
    out = [{"what": w, "measured": m, "limit": lim, "ok": ok} for w, m, lim, ok in rows]
    judged = [r["ok"] for r in out if r["ok"] is not None]
    return (all(judged) if judged else None), out


def verdict_calibrate(s: dict) -> list[tuple]:
    val = s.get("validation", {}) or {}
    px, mm = val.get("pixel_rmse"), val.get("xy_rmse_mm")
    return [("points recorded", f"{s.get('recorded')} of 27", ">= 15", s.get("recorded", 0) >= 15),
            ("held-out RMS [px]", fmt(px, 2), "< 3", None if px is None else px < 3),
            ("held-out RMS in the table plane [mm]", fmt(mm, 2), "< 2",
             None if mm is None else mm < 2),
            ("RANSAC rejected", str((s.get("fit") or {}).get("rejected_ids")), "-", None)]


def timing_stats(rows: list[dict]) -> dict:
    moves = [r for r in rows if "move" in r]
    errs = [1000 * r["error_s"] for r in moves if r["error_s"] == r["error_s"]]
    mtq = [1000 * r["move_to_q_s"] for r in moves]
    grabs = [r for r in rows if "grabs" in r]
    errs_sorted = sorted(errs)
    return {"err_ms": errs, "median_ms": errs_sorted[len(errs) // 2] if errs else float("nan"),
            "spread_ms": (max(errs) - min(errs)) if errs else float("nan"),
            "move_to_q_ms": mtq, "period_ms": 1000 * moves[0]["sample_period_s"] if moves else float("nan"),
            "grab_ms": 1000 * grabs[0]["grab_s_median"] if grabs else None}


def verdict_timing(s: dict) -> list[tuple]:
    rows = []
    stats = {k: timing_stats(v) for k, v in s.items() if k in ("plain", "camera")}
    for key, label in (("plain", "run 1 (no camera)"), ("camera", "run 2 (camera grabbing)")):
        if key not in stats:
            continue
        st = stats[key]
        rows += [
            (f"{label}: lateness per move [ms]", ", ".join(f"{e:+.0f}" for e in st["err_ms"]),
             "abs(median) <= 30", abs(st["median_ms"]) <= 30),
            (f"{label}: lateness median / spread [ms]",
             f"{st['median_ms']:+.0f} / {st['spread_ms']:.0f}", "the same for all moves", None),
            (f"{label}: move_to_q [ms]", f"{min(st['move_to_q_ms']):.0f}-{max(st['move_to_q_ms']):.0f}",
             "<= 60", max(st["move_to_q_ms"]) <= 60),
            (f"{label}: get_q period [ms]", f"{st['period_ms']:.0f}", "40-80",
             40 <= st["period_ms"] <= 80),
        ]
        if st["grab_ms"] is not None:
            rows.append((f"{label}: grab duration [ms]", f"{st['grab_ms']:.0f}", "note only", None))
    if "plain" in stats and "camera" in stats:
        p, c = stats["plain"], stats["camera"]
        d_late = c["median_ms"] - p["median_ms"]
        d_mtq = max(c["move_to_q_ms"]) - max(p["move_to_q_ms"])
        rows.append(("run 2 - run 1: lateness / move_to_q [ms]", f"{d_late:+.0f} / {d_mtq:+.0f}",
                     "within ~20", abs(d_late) <= 20 and abs(d_mtq) <= 20))
    return rows


def verdict_turntable(s: dict, track_s: float) -> list[tuple]:
    d = s.get("tracker_minus_fit_deg") or []
    rate = abs(d[-1] - d[0]) / track_s if len(d) > 1 else None
    axis = s.get("axis_mm") or [float("nan")] * 2
    return [("detections", f"{s.get('detections')} of 10", ">= 8", s.get("detections", 0) >= 8),
            ("period [s]", fmt(s.get("period_s"), 2), "27.9-28.1",
             27.9 <= s.get("period_s", 0) <= 28.1),
            ("xy residual [mm]", fmt(s.get("xy_rmse_mm"), 2), "< 2", s.get("xy_rmse_mm", 99) < 2),
            ("yaw residual [deg]", fmt(s.get("yaw_rmse_deg"), 2), "< 1", s.get("yaw_rmse_deg", 99) < 1),
            ("tracker - fit drift [deg/s]", fmt(rate, 4), "< 0.03", None if rate is None else rate < 0.03),
            ("markers lost", str(s.get("lost")), "any", None),
            ("axis [mm] / offset [mm] / direction",
             f"({axis[0]:.1f}, {axis[1]:.1f}) / {fmt(s.get('offset_mm'), 1)} / {s.get('direction')}",
             "matches by eye", None)]


def verdict_fov(s: dict) -> list[tuple]:
    samples = s.get("samples") or []
    seen = [r for r in samples if r.get("markers", 0) > 0]
    both = [r for r in samples if r.get("markers", 0) >= 2]
    rows = [("images with markers", f"{len(seen)} of {len(samples)} ({len(both)} with both)", "-", None)]
    pos = [r for r in seen if "x_mm" in r]
    if pos:
        xs, ys = [r["x_mm"] for r in pos], [r["y_mm"] for r in pos]
        rows.append(("base seen at x [mm]", f"{min(xs):.0f} to {max(xs):.0f}", "work area 337-637", None))
        rows.append(("base seen at y [mm]", f"{min(ys):.0f} to {max(ys):.0f}", "work area -242..+242", None))
    return rows


def verdict_dryrun(s: dict, log_text: str) -> list[tuple]:
    phases = s.get("phases", {})
    rows = []
    for name, rec in phases.items():
        rows.append((name, f"{'ok' if rec.get('ok') else 'FAILED: ' + str(rec.get('reason'))}, "
                           f"{rec.get('seconds')} s", "ok", bool(rec.get("ok"))))
    rows.append(("result", str(s.get("result")), "all phases ran", None))
    for key in ("orbit", "margins from the uncertainties", "reachability map", "predicted end",
                "theta_start"):
        for line in log_text.splitlines():
            if key in line:
                rows.append((key, line.strip(), "-", None))
                break
    end = None
    for line in log_text.splitlines():
        if "predicted end" in line:
            try:
                end = float(line.split("predicted end")[1].split()[0])
            except (IndexError, ValueError):
                pass
    if end is not None:
        rows.append(("predicted end [s]", f"{end:.1f}", f"< {RUN_LIMIT_S:.0f} minus the margin",
                     end < RUN_LIMIT_S))
    return rows


def fmt(x, digits: int) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


# --- the session ---

class Session:
    def __init__(self, folder: Path, args) -> None:
        self.folder = folder
        self.path = folder / "session.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            folder.mkdir(parents=True, exist_ok=True)
            self.data = {"started": datetime.now().isoformat(timespec="seconds"),
                         "sim": args.sim, "host": platform.node(), "git": git_state(),
                         "calibration": None, "steps": {}, "notes": []}
        if args.calibration:
            self.data["calibration"] = str(args.calibration)
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        (self.folder / "report.md").write_text(report(self.data), encoding="utf-8")

    def step(self, n: int) -> dict:
        return self.data["steps"].setdefault(str(n), {})


def git_state() -> dict:
    def git(*a):
        r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
        return r.stdout.strip()
    return {"head": git("rev-parse", "--short", "HEAD"), "subject": git("log", "-1", "--format=%s"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def lab_cmd(check: str, sim: bool, *extra: str) -> list[str]:
    return [sys.executable, str(LAB_CHECK), check, *(["--sim"] if sim else []), *extra]


def need_calibration(sess: Session) -> str | None:
    cal = sess.data.get("calibration")
    if cal and Path(cal).exists():
        return cal
    if sess.data["sim"]:
        return None  # the checks use the simulator's own calibration
    files = sorted((p for p in CALIBRATION_DIR.glob("homography_*.json") if "_sim" not in p.name),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        print("No calibration: run step 2 first (or pass --calibration FILE).")
        return None
    answer = ask(f"No calibration from this session. Use the newest, {files[-1].name}? [Y/n] ", "y")
    if answer.lower().startswith("y"):
        sess.data["calibration"] = str(files[-1])
        return str(files[-1])
    return None


def do_step(n: int, sess: Session, args) -> None:
    check, title, moves, setup = STEPS[n]
    rec = sess.step(n)
    rec.update(check=check, title=title, started=datetime.now().isoformat(timespec="seconds"))
    t0 = time.time()
    sim = sess.data["sim"]

    if check == "dims":
        code, err = run_child(lab_cmd("dims", False))
        folder = newest_folder(LAB_OUT, "*_dims", t0)
        s = read_summary(folder)
        rec.update(exit=code, folder=str(folder) if folder else None, stderr=error_tail(err))
        if s is None:
            rec["verdict"] = None
            return
        print("\nEnter the measured values in mm (blank = not measured):")
        rows, measured = [], rec.setdefault("measured", {})
        for row in s["rows"]:
            old = measured.get(row["what"])
            hint = f" [{old}]" if old is not None else ""
            value = ask_float(f"  {row['what']} (model {row['model_mm']:.1f}){hint}: ")
            value = old if value is None else value
            measured[row["what"]] = value
            if value is None:
                rows.append((row["what"], "-", f"{row['model_mm']:.1f} +-1", None))
            else:
                diff = value - row["model_mm"]
                rows.append((row["what"], f"{value:.1f} ({diff:+.1f})", f"{row['model_mm']:.1f} +-1",
                             abs(diff) <= DIMS_TOLERANCE_MM))
        rec["verdict"], rec["rows"] = judge(rows)
        return

    extra: list[str] = []
    if check in ("turntable", "fov", "dryrun"):
        cal = need_calibration(sess)
        if cal is None and not sim and check in ("turntable", "dryrun"):
            rec.update(verdict=None, skipped="no calibration")
            return
        if cal:
            extra += ["--calibration", cal]
    if check == "turntable":
        extra += ["--track", str(args.track)]

    if check == "dryrun":
        cmd = [sys.executable, str(RUN_SOLVER), "--stop-after", "2", *([] if sim else ["--real"]), *extra]
        code, err = run_child(cmd)
        folder = newest_folder(RUNS, "*", t0)
        s = read_summary(folder)
        log_text = (folder / "log.txt").read_text(encoding="utf-8", errors="replace") \
            if folder and (folder / "log.txt").exists() else ""
        rec.update(exit=code, folder=str(folder) if folder else None, stderr=error_tail(err))
        rec["verdict"], rec["rows"] = judge(verdict_dryrun(s, log_text)) if s else (False, [])
        return

    code, err = run_child(lab_cmd(check, sim, *extra))
    folder = newest_folder(LAB_OUT, f"*_{check}{'_sim' if sim else ''}", t0)
    s = read_summary(folder)
    rec.update(exit=code, folder=str(folder) if folder else None, stderr=error_tail(err))
    if s is None:
        rec["verdict"] = False if code else None
        rec["rows"] = []
        print(f"\n(no summary: the check stopped. {error_tail(err, 2)})")
        return
    rows = {"calibrate": verdict_calibrate, "timing": verdict_timing, "fov": verdict_fov}.get(check)
    rows = verdict_turntable(s, args.track) if check == "turntable" else rows(s)
    rec["verdict"], rec["rows"] = judge(rows)
    if check == "calibrate" and s.get("path"):
        use = "y" if rec["verdict"] else ask(
            "The calibration did not pass. Use it for steps 4-6 anyway? [y/N] ", "n")
        if use.lower().startswith("y"):
            sess.data["calibration"] = s["path"]
    if check == "turntable":
        rec["by_eye"] = ask("Does the printed axis match where the turntable stands? "
                            "(y / n / what's off): ")
    if check == "fov":
        rec["edges"] = ask("Where did the markers stop being detected? (free text): ")


def print_verdict(rec: dict) -> None:
    for r in rec.get("rows", []):
        mark = {True: "ok  ", False: "FAIL", None: "    "}[r["ok"]]
        print(f"  [{mark}] {r['what']}: {r['measured']}  (pass: {r['limit']})")
    v = rec.get("verdict")
    print(f"  -> {'PASS' if v else 'NOT PASSED' if v is False else 'no verdict'}"
          + (f" (exit code {rec.get('exit')})" if rec.get("exit") else ""))


# --- the report ---

def report(data: dict) -> str:
    lines = [f"# Lab check session {data['started']}" + (" (SIMULATOR)" if data["sim"] else ""), "",
             f"- host `{data['host']}`, git `{data['git']['head']}` ({data['git']['subject']})"
             + (", **working tree modified**" if data["git"]["dirty"] else ""),
             f"- calibration: `{data.get('calibration') or 'none yet'}`",
             "- guide: `Prokop/docs/LAB_CHECK_GUIDE.md`; the constants are NOT changed here", ""]
    lines += ["## Overview", "", "| # | Check | Result | Folder |", "|---|---|---|---|"]
    for n, (check, title, _, _) in STEPS.items():
        rec = data["steps"].get(str(n))
        if not rec:
            res = "not run"
        elif rec.get("skipped"):
            res = f"skipped ({rec['skipped']})"
        else:
            v = rec.get("verdict")
            res = "**pass**" if v else "**NOT PASSED**" if v is False else "no verdict"
        folder = (rec or {}).get("folder")
        rel = Path(folder).relative_to(REPO).as_posix() if folder and Path(folder).is_relative_to(REPO) else folder
        lines.append(f"| {n} | {check}: {title} | {res} | {f'`{rel}`' if rel else '-'} |")
    deviations = []
    for n, (check, title, _, _) in STEPS.items():
        rec = data["steps"].get(str(n))
        if not rec or not rec.get("rows"):
            if rec and rec.get("stderr"):
                lines += ["", f"## {n}. {check}", "", "```", rec["stderr"], "```"]
            continue
        lines += ["", f"## {n}. {check}: {title}", "", "| | Measured | Pass | |", "|---|---|---|---|"]
        for r in rec["rows"]:
            mark = {True: "ok", False: "**FAIL**", None: ""}[r["ok"]]
            lines.append(f"| {r['what']} | {r['measured']} | {r['limit']} | {mark} |")
            if r["ok"] is False:
                deviations.append(f"- step {n} ({check}): {r['what']}: {r['measured']} "
                                  f"(pass: {r['limit']})")
        for key, label in (("by_eye", "Axis by eye"), ("edges", "Markers lost at"), ("note", "Note")):
            if rec.get(key):
                lines += ["", f"{label}: {rec[key]}"]
        if rec.get("stderr") and rec.get("exit"):
            lines += ["", "```", rec["stderr"], "```"]
    lines += ["", "## Deviations for Prokop", "",
              "Constants by check (guide): dims -> `solver/wire.py` (`BAR_HALF`, `TOP_Z`, "
              "`OPEN_END_X`, `ARC_RADIUS`, `JOG_*`, `RING_*`), marker height -> Vojtěch's "
              "`MARKER_HEIGHT_ABOVE_TABLE_M`; timing offset -> `COMMAND_DELAY` in "
              "`solver/timing.py`; period -> `SPIN_PERIOD` in `solver/contracts.py`; residuals "
              "and drift -> `solver/uncertainty.py`.", ""]
    lines += deviations or ["- none so far"]
    if data.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {t}: {x}" for t, x in data["notes"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="every check against the simulator")
    ap.add_argument("--resume", action="store_true", help="continue the latest session")
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(STEPS), help="run only these steps")
    ap.add_argument("--calibration", type=Path, help="an existing homography JSON for steps 4-6")
    ap.add_argument("--track", type=float, default=30.0, help="turntable: tracker seconds")
    args = ap.parse_args()

    LAB_OUT.mkdir(parents=True, exist_ok=True)
    folder = None
    if args.resume:
        old = sorted(LAB_OUT.glob("session_*" + ("_sim" if args.sim else "")))
        old = [p for p in old if args.sim or not p.name.endswith("_sim")]
        folder = old[-1] if old else None
        if folder is None:
            print("no session to resume: starting a new one")
    if folder is None:
        folder = LAB_OUT / f"session_{datetime.now():%Y%m%d_%H%M%S}{'_sim' if args.sim else ''}"
    sess = Session(folder, args)
    args.sim = sess.data["sim"]
    print(f"Lab check session{' (SIMULATOR)' if args.sim else ''}: {folder}\n"
          f"report: {folder / 'report.md'}")
    if sess.data["git"]["dirty"]:
        print("NOTE: the working tree has local changes (recorded in the report).")

    steps = args.only or sorted(STEPS)
    if args.sim and 1 in steps and not args.only:
        steps = [n for n in steps if n != 1]  # dims has nothing to simulate
    warned_home = False
    for n in steps:
        check, title, moves, setup = STEPS[n]
        banner(f"Step {n}: {check} - {title}")
        print(setup)
        if moves and not args.sim and not warned_home:
            print("\n  The robot must be homed BY HAND by Prokop and stand in q_home (this "
                  "script never homes;\n  the check refuses otherwise). Every motion asks for "
                  "Enter: the OPERATOR presses it.")
            warned_home = True
        rec = sess.data["steps"].get(str(n))
        if rec and rec.get("verdict") is not None:
            print(f"\n  (already run in this session: {'pass' if rec['verdict'] else 'NOT PASSED'})")
        choice = ask("\nEnter = run, s = skip, n = add a note, q = quit: ").lower()
        while choice == "n":
            sess.data["notes"].append((datetime.now().strftime("%H:%M"), ask("  note: ")))
            sess.save()
            choice = ask("Enter = run, s = skip, n = add a note, q = quit: ").lower()
        if choice == "q":
            break
        if choice == "s":
            sess.step(n).setdefault("skipped", "by the operator")
            sess.save()
            continue
        sess.step(n).pop("skipped", None)
        try:
            do_step(n, sess, args)
        finally:
            sess.save()
        rec = sess.step(n)
        print(f"\nStep {n} result:")
        print_verdict(rec)
        note = ask("Note for this step (blank = none): ")
        if note:
            rec["note"] = note
        sess.save()

    sess.save()
    banner("Session recorded")
    print(f"{folder / 'report.md'}\n\nResume with: uv run Prokop/tools/lab_session.py --resume"
          + (" --sim" if args.sim else "") + "\nNothing was committed: commit the results when "
          "Prokop asks (experiments/lab/, data/calibration/, runs/).")


if __name__ == "__main__":
    main()
