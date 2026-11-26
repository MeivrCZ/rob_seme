"""The analytic ring-bar distance (SoftScene(analytic_ring=True)) against the exact meshes.

Near the task: random puzzle placements over the work area (build_reachability_map's
cell_model), the ring threaded on random wire points with random orientations (every
IK solution within the joint limits minus 5.5 deg), and the same with random joint
offsets (+-6 deg) for poses close to touching. For each:

* the exact distance between the ring proper (the hoop mesh without the handle's
  sector) and the puzzle mesh (FCL, with the closest points);
* the bound (SoftScene.ring_bar_distance) must never exceed it: soundness; cases where
  the closest exact point is on the puzzle's foot do not count (the foot stays a box);
* how conservative it is (bound - exact), and the time per clearance query with the
  ring as boxes and analytic.

    uv run python Prokop/experiments/validate_analytic_ring.py [N_PLACEMENTS]
"""
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

PROKOP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROKOP), str(PROKOP / "tools"), str(PROKOP.parent / "Natka")]
from ctu_crs import CRS93  # noqa: E402

from build_reachability_map import cell_model  # noqa: E402
from placement import MODELS, TaskScene, load_urdf  # noqa: E402
from solver.hoop_pose import flange_target, ring_in_flange  # noqa: E402
from solver.soft_collision import SoftScene, _data  # noqa: E402
from solver.wire import checkpoint_arclengths, wire_frame_at  # noqa: E402

JOINT_MARGIN = np.deg2rad(5.5)
DIAG = False


def ring_mesh() -> trimesh.Trimesh:
    """The ring proper in the hoop_center frame (faces within 34.95 mm of the axis)."""
    hoop = load_urdf(MODELS / "hoop.urdf")
    T = np.linalg.inv(hoop.get_transform("hoop_center", "hoop_link"))
    m = hoop.scene.dump(concatenate=True)
    m.apply_transform(T)
    rho = np.hypot(m.vertices[:, 0], m.vertices[:, 1])
    keep = np.all(rho[m.faces] <= 0.03495, axis=1)
    return m.submesh([np.flatnonzero(keep)], append=True)


def main() -> None:
    n_place = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rng = np.random.default_rng(0)
    kin = CRS93(tty_dev=None)
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN
    to_ring = ring_in_flange()
    ring = ring_mesh()
    puzzle = TaskScene().puzzle.scene.dump(concatenate=True)
    foot = next(b for b in _data()["puzzle"] if b["name"] == "puzzle_foot")
    foot_top = float(np.array(foot["T"])[2, 3] + abs(np.array(foot["T"])[2, :3]) @ np.array(foot["half"]))
    s_end = checkpoint_arclengths(0, 0.015)[-1]
    rows, t_box, t_ana, n_q = [], 0.0, 0.0, 0
    for _ in range(n_place):
        model = cell_model(float(rng.uniform(0.38, 0.68)), float(rng.uniform(-np.pi, np.pi)))
        box_scene = SoftScene(model, hoop_puzzle_margin=0.0, restricted=True)
        ana_scene = SoftScene(model, hoop_puzzle_margin=0.0, restricted=True, analytic_ring=True)
        P = model.puzzle_T(0.0)
        def is_foot(x, P=P):
            """In the foot's region: below 1 mm (the foot, and the stem's bottom inside it,
            below the centreline's end at the pad; both inside the foot box, which stays a
            box; FCL's contact points are approximate, some land a little above z = 0)."""
            p = (np.linalg.inv(P) @ np.append(x, 1.0))[:3]
            return p[2] < 0.001
        cm = trimesh.collision.CollisionManager()
        cm.add_object("puzzle", puzzle, transform=P)
        for _ in range(5):
            F = P @ wire_frame_at(float(rng.uniform(0.0, s_end)), 0.015)
            tilt = float(rng.choice(np.deg2rad([0, 10, 20, 30, 40])))
            target = flange_target(F, float(rng.uniform(-np.pi, np.pi)), -1, tilt,
                                   float(rng.uniform(0, 2 * np.pi)))
            for q in kin.ik(target):
                if not (np.all(q >= lo) and np.all(q <= hi)):
                    continue
                for qq in (q, np.clip(q + rng.uniform(-0.1, 0.1, 6), lo, hi)):
                    W = kin.fk(qq) @ to_ring
                    exact, data = cm.min_distance_single(ring, transform=W, return_data=True)
                    if exact <= 0.0:
                        exact = -1e-9  # touching or inside: the bound must be <= 0 too
                        _, cdata = cm.in_collision_single(ring, transform=W, return_data=True)
                        on_foot = any(is_foot(c.point) for c in cdata)  # the foot box catches it
                    else:
                        on_foot = is_foot(data.point("puzzle"))
                    bound = ana_scene.ring_bar_distance(qq, 0.0)
                    t0 = time.perf_counter()
                    box_scene.clearance(qq, 0.0)
                    t1 = time.perf_counter()
                    ana_scene.clearance(qq, 0.0)
                    t2 = time.perf_counter()
                    t_box += t1 - t0
                    t_ana += t2 - t1
                    n_q += 1
                    rows.append((bound, exact, on_foot))
                    if bound > exact + 1e-6 and not on_foot and DIAG:
                        if exact <= 0:  # touching: where? the deepest-contact points
                            _, cdata = cm.in_collision_single(ring, transform=W,
                                                              return_data=True)
                            pts = [c.point for c in cdata]
                        else:
                            pts = [data.point("puzzle")]
                        from solver.soft_collision import bar_centreline
                        cl = bar_centreline(0.0005)
                        loc = [np.round(1000 * (np.linalg.inv(P) @ np.append(x, 1))[:3], 1)
                               for x in pts[:3]]
                        off = [round(float(np.linalg.norm(cl - l / 1000, axis=1).min()) * 1000, 1)
                               for l in loc]
                        loc = [f"{l.tolist()} ({o} mm from the centreline)" for l, o in zip(loc, off)]
                        print(f"  unsound: bound {1000 * bound:+.1f} mm, exact "
                              f"{1000 * exact:+.1f} mm; puzzle-frame contact/closest "
                              f"points {loc}")
    r = np.array([(b, e) for b, e, f in rows if not f])
    unsound = r[r[:, 0] > r[:, 1] + 1e-6]
    near = r[r[:, 1] < 0.02]
    print(f"{len(rows)} ring poses near the task ({n_place} placements); "
          f"{sum(f for *_, f in rows)} with the closest exact point on the foot (not counted)")
    print(f"exact ring-puzzle distance: {np.sum(r[:, 1] <= 0)} touching, "
          f"{len(near)} within 20 mm")
    print(f"UNSOUND (bound > exact): {len(unsound)}"
          + (f", worst by {1000 * (unsound[:, 0] - unsound[:, 1]).max():.2f} mm" if len(unsound) else ""))
    gap = (r[:, 1] - r[:, 0])[r[:, 1] > 0]
    gap_near = (near[:, 1] - near[:, 0])[near[:, 1] > 0]
    print(f"how conservative (exact - bound): median {1000 * np.median(gap):.2f} mm overall, "
          f"{1000 * np.median(gap_near):.2f} mm within 20 mm (max {1000 * gap_near.max():.2f})")
    print(f"clearance query: ring as boxes {1000 * t_box / n_q:.2f} ms, analytic "
          f"{1000 * t_ana / n_q:.2f} ms ({t_box / max(t_ana, 1e-12):.1f}x)")


if __name__ == "__main__":
    main()
