"""The soft model's restricted pairs (planning) against the full check, near the task.

Configurations near the task: random puzzle placements over the work area
(tools/build_reachability_map.cell_model), random points along the wire, random ring
orientations, every IK solution within the joint limits minus 5.5 deg; and the same
with random joint offsets (+-10 deg) to get poses close to colliding. For each, the
full clearance and the restricted one (SoftScene(restricted=True)): a missed collision
(full: collides, restricted: free) is the failure that matters; also how often an
excluded pair is the closest within 20 mm, and the time per query.

    uv run python Prokop/experiments/validate_restricted_pairs.py [N_PLACEMENTS]
"""
import collections
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROKOP), str(PROKOP / "tools"), str(PROKOP.parent / "Natka")]
from ctu_crs import CRS93  # noqa: E402

from build_reachability_map import cell_model  # noqa: E402
from solver.hoop_pose import flange_target  # noqa: E402
from solver.soft_collision import SoftScene  # noqa: E402
from solver.wire import checkpoint_arclengths, wire_frame_at  # noqa: E402

JOINT_MARGIN = np.deg2rad(5.5)


def main() -> None:
    n_place = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(0)
    kin = CRS93(tty_dev=None)
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN
    s_end = checkpoint_arclengths(0, 0.015)[-1]
    n, missed, missed_pairs, closest_excluded = 0, 0, collections.Counter(), 0
    t_full = t_restr = 0.0
    full_hits = 0
    for _ in range(n_place):
        model = cell_model(float(rng.uniform(0.35, 0.70)), float(rng.uniform(-np.pi, np.pi)))
        full_scene = SoftScene(model, hoop_puzzle_margin=0.006)
        restr_scene = SoftScene(model, hoop_puzzle_margin=0.006, restricted=True)
        for _ in range(6):
            F = model.puzzle_T(0.0) @ wire_frame_at(float(rng.uniform(0.0, s_end)), 0.015)
            tilt = float(rng.choice(np.deg2rad([0, 10, 20, 30, 40, 50])))
            target = flange_target(F, float(rng.uniform(-np.pi, np.pi)), -1, tilt,
                                   float(rng.uniform(0, 2 * np.pi)))
            for q in kin.ik(target):
                if not (np.all(q >= lo) and np.all(q <= hi)):
                    continue
                for qq in (q, np.clip(q + rng.uniform(-0.17, 0.17, 6), lo, hi)):
                    t0 = time.perf_counter()
                    a = full_scene.clearance(qq, 0.0)
                    t1 = time.perf_counter()
                    b = restr_scene.clearance(qq, 0.0)
                    t2 = time.perf_counter()
                    t_full += t1 - t0
                    t_restr += t2 - t1
                    n += 1
                    if not a.free:
                        full_hits += 1
                    if not a.free and b.free:
                        missed += 1
                        pair = a.pair if a.distance <= a.self_gap else a.self_pair
                        missed_pairs[pair] += 1
                    full_d = min(a.distance, a.self_gap)
                    restr_d = min(b.distance, b.self_gap)
                    if full_d < 0.02 and full_d < restr_d - 1e-9:
                        closest_excluded += 1
    print(f"{n} configurations near the task ({n_place} placements): the full check finds "
          f"{full_hits} collisions")
    print(f"missed collisions (full: collides, restricted: free): {missed}")
    for pair, c in missed_pairs.most_common(10):
        print(f"   {c:5d}  {pair[0]} - {pair[1]}")
    print(f"an excluded pair is the closest within 20 mm: {closest_excluded} times")
    print(f"time per query: full {1000 * t_full / n:.2f} ms, restricted {1000 * t_restr / n:.2f} ms "
          f"({t_full / max(t_restr, 1e-12):.1f}x)")


if __name__ == "__main__":
    main()
