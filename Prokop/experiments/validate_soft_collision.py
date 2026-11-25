"""Validates the soft collision model (solver.soft_collision) against the exact meshes.

1. Forward kinematics: the model's own link chain vs the URDF.
2. Coverage: every vertex of every mesh lies inside its boxes (arm parts:
   the union; turntable: the intersection; base, puzzle, cell: the union).
3. Distances on random states (placement in the work area, turntable angle,
   q within the limits): the soft distance never exceeds the exact one (per
   group: table, cell, task), and every exact contact (self-collision too)
   is a soft contact. Reports how much smaller the soft distances are.
4. Rings around the bar: of the IK candidates at the checkpoints that are
   exactly collision free, how many the soft model (with and without its
   margins) still accepts.
5. The hierarchical query (SoftScene.clearance) is a lower bound too: it
   never reports more than the exact distance (counted as "hierarchy" in 3;
   it may differ slightly from the leaf pairs' minimum where it does not
   refine, since an enclosing box has its own separating axes).
6. Speed of one clearance query (random q, and rings around the bar).

    uv run python Prokop/experiments/validate_soft_collision.py [--states N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))

from cell import WORK_AREA_XY, cell_meshes  # noqa: E402
from collision import SceneCollider, arm_parts  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from placement import TaskScene, collision_meshes, load_urdf, MODELS  # noqa: E402

from solver.contracts import TurntableModel  # noqa: E402
from solver.hoop_pose import candidates, orientation_grid  # noqa: E402
from solver.soft_collision import Boxes, SoftScene, _data, link_frames  # noqa: E402
from solver.wire import wire_path  # noqa: E402

TOL = 1e-6  # m


def facing_model(axis_xy: np.ndarray) -> tuple[TurntableModel, float]:
    """Static model (no spin) with the puzzle's +x (its open end) facing the robot, and theta."""
    model = TurntableModel(axis_xy=np.asarray(axis_xy, float), base_offset=np.zeros(2),
                           base_yaw=0.0, direction=1, omega=0.0)
    theta = 0.0
    for _ in range(3):  # the puzzle origin may sit off the axis: iterate
        T = model.puzzle_T(theta)
        theta += np.arctan2(-T[1, 3], -T[0, 3]) - np.arctan2(T[1, 0], T[0, 0])
    return model, float(np.arctan2(np.sin(theta), np.cos(theta)))


def inside(points: np.ndarray, items: list[dict], T: np.ndarray = np.eye(4)) -> np.ndarray:
    """Which points lie inside at least one of the boxes (posed by T)."""
    boxes = Boxes.of(items, T)
    local = np.einsum("nki,pnk->pni", boxes.R, points[:, None, :] - boxes.c[None])
    return np.any(np.all(np.abs(local) <= boxes.h[None] + TOL, axis=2), axis=1)


def check_fk(kin, rng) -> None:
    urdf = load_urdf(MODELS / "crs93.urdf")
    err = 0.0
    for _ in range(50):
        q = rng.uniform(kin.q_min, kin.q_max)
        urdf.update_cfg(q)
        frames = link_frames(q)
        for link in frames:
            err = max(err, np.abs(frames[link] - urdf.get_transform(link, urdf.base_link)).max())
    print(f"1. forward kinematics: max |soft - URDF| = {err:.1e}")


def check_coverage() -> bool:
    data = _data()
    ok = True
    print("2. coverage (mesh vertices outside their boxes):")
    _, parts = arm_parts()
    by_part = {p["part"]: p["boxes"] for p in data["arm"]}
    for link, name, mesh, offset in parts:
        v = mesh.vertices @ offset[:3, :3].T + offset[:3, 3]  # into the link frame
        out = int((~inside(v, by_part[name])).sum())
        ok &= out == 0
        print(f"   {name:10s} {len(by_part[name]):3d} boxes: {out} of {len(v)} outside")
    task = TaskScene()
    tt = np.vstack([m.vertices for m in collision_meshes(task.turntable)])
    out_tt = int(sum((~inside(tt, [b])).sum() for b in data["turntable"]))
    base = np.vstack([m.vertices for m in collision_meshes(task.base)])
    puz = np.vstack([m.vertices for m in collision_meshes(task.puzzle)])
    out_b, out_p = int((~inside(base, data["base"])).sum()), int((~inside(puz, data["puzzle"])).sum())
    cell = np.vstack([m.vertices for m in cell_meshes().values()])
    out_c = int((~inside(cell, data["cell"])).sum())
    print(f"   turntable (both boxes) {out_tt}, base {out_b}, puzzle {out_p} of {len(puz)}, "
          f"cell {out_c}")
    if out_p:
        bad = puz[~inside(puz, data["puzzle"])]
        print("   puzzle vertices outside (mm):", np.round(bad[:8] * 1000, 1).tolist())
    return ok and out_tt == 0 and out_b == 0 and out_p == 0 and out_c == 0


def check_distances(kin, rng, states: int) -> None:
    task = TaskScene()
    exact = SceneCollider(task)
    viol = {"table": 0, "cell": 0, "task": 0, "self": 0, "hierarchy": 0}
    diff = {"table": [], "cell": [], "task": []}
    contacts = {"exact": 0, "soft": 0}
    for _ in range(states):
        p = task.random_placement(rng, area=WORK_AREA_XY)
        theta = rng.uniform(-np.pi, np.pi)
        q = rng.uniform(kin.q_min, kin.q_max)
        model = TurntableModel(axis_xy=np.asarray(p.turntable_xy, float),
                               base_offset=np.asarray(p.base_xy, float),
                               base_yaw=float(p.base_yaw), direction=1, omega=0.0)
        scene0 = SoftScene(model, 0.0, 0.0, 0.0)
        soft = scene0.group_distances(q, theta)
        c = scene0.clearance(q, theta)
        exact.place(q, p, theta)
        e_table = exact.placed_table()[1]
        e = {"table": e_table, "cell": exact.placed_group("cell")[1],
             "task": exact.placed_group("task")[1]}
        for g in e:
            if soft[g] > e[g] + TOL:
                viol[g] += 1
            diff[g].append(e[g] - soft[g])
        if c.distance > min(e.values()) + TOL:
            viol["hierarchy"] += 1
        if exact.placed_self_hits():
            contacts["exact"] += 1
            if soft["self"] > 0:
                viol["self"] += 1
            if c.self_gap > 0:
                viol["hierarchy"] += 1
        if soft["self"] <= 0:
            contacts["soft"] += 1
    print(f"3. {states} random states: soft distance > exact (must be 0): {viol}")
    for g, d in diff.items():
        d = np.array(d) * 1000
        print(f"   {g:6s} exact - soft [mm]: median {np.median(d):5.1f}, 90 % {np.percentile(d, 90):5.1f}, "
              f"max {d.max():5.1f}")
    print(f"   self-collision: exact contacts {contacts['exact']}, soft contacts {contacts['soft']}")


def check_rings(kin) -> list[tuple]:
    exact = SceneCollider(TaskScene())
    grid = orientation_grid(np.deg2rad(30))
    print("4. rings around the bar (IK candidates at tier-2 checkpoints, exactly free):")
    for axis in ((0.487, 0.0), (0.567, 0.1)):
        model, theta = facing_model(np.array(axis))
        from placement import Placement
        p = Placement(np.asarray(model.axis_xy), np.zeros(2), 0.0)
        soft, soft0 = SoftScene(model), SoftScene(model, 0.0, 0.0, 0.0)
        free = acc = acc0 = 0
        threaded = []
        path = wire_path(2)
        for i in range(len(path.frames)):
            qs, _ = candidates(kin, model.checkpoint_T(path, i, theta), grid, -1)
            for q in qs:
                if exact.check(q, p, theta)[0]:
                    continue
                free += 1
                threaded.append((soft, q, theta))
                acc += soft.clearance(q, theta).free
                acc0 += soft0.clearance(q, theta).free
        print(f"   axis {axis}: {free} exactly free; soft accepts {acc} ({100 * acc / free:.0f} %) "
              f"with margins, {acc0} ({100 * acc0 / free:.0f} %) without")
    return threaded


def check_speed(kin, rng, threaded: list[tuple]) -> None:
    model, theta = facing_model(np.array([0.487, 0.0]))
    scene = SoftScene(model)
    qs = rng.uniform(kin.q_min, kin.q_max, size=(200, 6))
    scene.clearance(qs[0], theta)
    t = time.perf_counter()
    for q in qs:
        scene.clearance(q, theta)
    random_ms = 1000 * (time.perf_counter() - t) / len(qs)
    sample = threaded[:: max(1, len(threaded) // 300)]
    t = time.perf_counter()
    for sc, q, th in sample:
        sc.clearance(q, th)
    ring_ms = 1000 * (time.perf_counter() - t) / len(sample)
    print(f"6. one clearance query: {random_ms:.2f} ms (random q), {ring_ms:.2f} ms (ring around "
          f"the bar)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", type=int, default=1000)
    args = parser.parse_args()
    kin, rng = CRS93(tty_dev=None), np.random.default_rng(0)
    check_fk(kin, rng)
    check_coverage()
    check_distances(kin, rng, args.states)
    threaded = check_rings(kin)
    check_speed(kin, rng, threaded)


if __name__ == "__main__":
    main()
