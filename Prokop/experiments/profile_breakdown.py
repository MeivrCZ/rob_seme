"""Where the time of a planning run goes: a breakdown of a cProfile file.

    uv run python -m cProfile -o run.prof Prokop/experiments/backward_plan.py 0 0.5 0.487,0.0 0.02,0.0 1
    uv run python Prokop/experiments/profile_breakdown.py run.prof

Prints, for the planner's building blocks, the calls and the cumulative time
(a function's own time plus everything it calls; the blocks overlap, e.g. the
continuous move check calls the soft clearance), then the 15 functions with
the most own time.
"""
import pstats
import sys

# (label, file part, function name), in the order printed.
BLOCKS = [
    ("whole backward search (plan_backward)", "planner.py", "plan_backward"),
    ("  final poses: stays free for a turn", "planner.py", "stays_free"),
    ("  waiting pose at checkpoint 0 (waiting_entry)", "planner.py", "waiting_entry"),
    ("  subdivision per checkpoint step (bisect)", "planner.py", "bisect"),
    ("    a pose on the wire (solve_at)", "planner.py", "solve_at"),
    ("      the candidate flange poses (targets)", "planner.py", "targets"),
    ("      the orientation carried along the wire (carried)", "planner.py", "carried"),
    ("    the projection onto the wire (project)", "planner.py", "project"),
    ("  tree search between checkpoints (reach)", "planner.py", "reach"),
    ("    direct try to the checkpoint (try_goal)", "planner.py", "try_goal"),
    ("    pose before a node on the wire (before)", "planner.py", "before"),
    ("      IK over the roll grid (ik_any_roll)", "planner.py", "ik_any_roll"),
    ("    configuration check (config_free)", "planner.py", "config_free"),
    ("    continuous move check (move_free)", "planner.py", "move_free"),
    ("    at a checkpoint? (at_checkpoint)", "planner.py", "at_checkpoint"),
    ("      waiting pose clear of the swept volume (waits_free)", "planner.py", "waits_free"),
    ("  shortcuts: whole path rechecked (path_free)", "planner.py", "path_free"),
    ("building blocks, wherever called:", None, None),
    ("  analytic IK (CRS93.ik)", "", "ik"),
    ("  forward kinematics (CRS93.fk)", "", "fk"),
    ("  soft clearance of one configuration (SoftScene.clearance)", "soft_collision.py", "clearance"),
    ("  continuous checks (check_moves)", "planner.py", "check_moves"),
    ("  move timing (RobotTiming.move_duration)", "timing.py", "move_duration"),
    ("  flange pose for a ring pose (flange_target)", "", "flange_target"),
]


def main() -> None:
    stats = pstats.Stats(sys.argv[1])
    table = stats.stats  # (file, line, name) -> (prim calls, calls, own, cumulative, callers)
    total = max(v[3] for v in table.values())
    print(f"run: {total:.1f} s in all\n")
    print(f"{'block':62s} {'calls':>9s} {'time':>8s} {'share':>6s}  per call")
    for label, file_part, name in BLOCKS:
        if name is None:
            print(label)
            continue
        rows = [v for (f, _, n), v in table.items() if n == name and file_part in f
                and (name not in ("ik", "fk") or "ctu_crs" in f or "crs" in f.lower())]
        if not rows:
            print(f"{label:62s} {'-':>9s}")
            continue
        calls = sum(v[1] for v in rows)
        cum = max(v[3] for v in rows)  # recursive/nested entries: the outermost
        print(f"{label:62s} {calls:9d} {cum:7.1f}s {100 * cum / total:5.0f}%  "
              f"{1000 * cum / max(calls, 1):.2f} ms")
    print("\nthe 15 functions with the most own time:")
    stats.sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
