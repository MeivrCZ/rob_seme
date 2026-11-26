"""Outcome table of several run_solver.py runs (their summary.json).

    uv run python Prokop/experiments/batch_summary.py Prokop/runs/<a> Prokop/runs/<b> ...

Each run is put in one class: UNREACHABLE (perception), UNREACHABLE (planning:
plan or approach), ABORTED (the executor stopped sending), COLLISION, NOT DONE
(ran through but the ring is too high), SUCCESS.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def classify(summary: dict) -> tuple[str, str]:
    phases = summary.get("phases", {})
    p1 = phases.get("Phase 1: turntable model", {})
    p2 = phases.get("Phase 2: plan", {})
    p3 = phases.get("Phase 3: execution", {})
    if not p1.get("ok"):
        return "UNREACHABLE (perception)", p1.get("reason", "")
    if not p2.get("ok"):
        return "UNREACHABLE (planning)", p2.get("reason", "")
    if "collision" in p3:
        return "COLLISION", p3["collision"]
    if not p3.get("ok"):
        return "ABORTED", p3.get("reason", "")
    top = p3.get("ring_top_above_pad_mm")
    t = p3.get("end_after_start_s")
    if top is not None and top < 40.0:
        if t is not None and t > 120.0:
            return "SUCCESS but OVER 120 s", f"ring {top:.1f} mm, {t:.1f} s"
        return "SUCCESS", f"ring {top:.1f} mm" + (f", {t:.1f} s" if t is not None else "")
    return "NOT DONE", f"ring {top:.1f} mm" if top is not None else ""


def main() -> None:
    rows = []
    for folder in sys.argv[1:]:
        path = Path(folder) / "summary.json"
        if not path.exists():
            rows.append((folder, "?", "?", "NO SUMMARY", "the run did not finish"))
            continue
        s = json.loads(path.read_text())
        args, pl = s.get("args", {}), s.get("placement", {})
        where = (f"axis ({pl['turntable_xy'][0]:.3f}, {pl['turntable_xy'][1]:+.3f}), offset "
                 f"{1000 * (pl['base_xy'][0] ** 2 + pl['base_xy'][1] ** 2) ** 0.5:.0f} mm"
                 if pl else "")
        plan = s.get("phases", {}).get("Phase 2: plan", {}).get("plan", {})
        took = f"plan {plan['seconds']:.0f} s" if "seconds" in plan else ""
        outcome, why = classify(s)
        rows.append((f"seed {args.get('seed')} dir {int(args.get('direction', 0)):+d}", where,
                     took, outcome, why))
    for r in rows:
        print(f"{r[0]:14s} {r[1]:36s} {r[2]:11s} {r[3]:26s} {r[4]}")
    print()
    for outcome, n in Counter(r[3] for r in rows).most_common():
        print(f"{n:3d}  {outcome}")


if __name__ == "__main__":
    main()
