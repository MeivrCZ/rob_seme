"""Builds the reachability map of puzzle D's checkpoints (user, 2026-11-26).

Whether a checkpoint can be reached depends only on where the puzzle stands relative
to the robot: turning everything about the robot's base axis changes only joint 1
(which has +-170 deg of room). So per checkpoint the map is 2D:

    rho = the horizontal distance of the puzzle frame's origin from the robot's base axis
    psi = the puzzle's heading (its x axis) relative to the line from the robot to it

Reachable = the ring centre on the checkpoint, side B, some orientation of a reduced
grid (tilt 0 / 20 / 40 deg, the tilt direction every 45 deg, the roll every 30 deg),
the IK within the joint limits minus JOINT_MARGIN, and free in the soft model with the
puzzle at that pose (HOOP_MARGIN). The turntable and the base stand around the puzzle as
for a base centred on the axis (only the arm against the turntable is approximate).
Static: the puzzle's pose decides it. At most MAX_CLEARANCE_CHECKS collision checks per
cell (in-room poses that all collide beyond that: counted unreachable).

    uv run python Prokop/tools/build_reachability_map.py [--workers 6]

Writes solver/data/reachability_map.npz (loaded by solver.reachability).
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROKOP), str(PROKOP.parent / "Natka")]

RHO = np.round(np.arange(0.30, 0.75 + 1e-9, 0.01), 4)  # m
PSI = np.deg2rad(np.arange(-180.0, 180.0, 10.0))  # rad
ENTRY, BEND_POINTS = 0.015, 3
HOOP_MARGIN = 0.006  # m
JOINT_MARGIN = np.deg2rad(5.5)
TILTS = np.deg2rad([0.0, 20.0, 40.0])
DIRECTION_STEP = np.deg2rad(45.0)
ROLL_STEP = np.deg2rad(30.0)
MAX_CLEARANCE_CHECKS = 60
OUT = PROKOP / "solver" / "data" / "reachability_map.npz"


def cell_model(rho: float, psi: float):
    """A turntable model whose puzzle origin is at (rho, 0) with heading psi (at angle 0)."""
    from solver.contracts import TurntableModel
    zero = np.zeros(2)
    m0 = TurntableModel(axis_xy=zero, base_offset=zero, base_yaw=0.0, direction=1, omega=0.0)
    P = m0.puzzle_T(0.0)
    yaw = psi - np.arctan2(P[1, 0], P[0, 0])
    m1 = TurntableModel(axis_xy=zero, base_offset=zero, base_yaw=yaw, direction=1, omega=0.0)
    axis = np.array([rho, 0.0]) - m1.puzzle_T(0.0)[:2, 3]
    return TurntableModel(axis_xy=axis, base_offset=zero, base_yaw=yaw, direction=1,
                          omega=0.0)


def row(i_rho: int) -> np.ndarray:
    """reachable[k, i_psi] for one rho."""
    from ctu_crs import CRS93
    from solver.hoop_pose import flange_target
    from solver.soft_collision import SoftScene
    from solver.wire import wire_path
    kin = CRS93(tty_dev=None)
    lo, hi = kin.q_min + JOINT_MARGIN, kin.q_max - JOINT_MARGIN
    path = wire_path(0, ENTRY, BEND_POINTS)
    orients = [(roll, tilt, d) for tilt in TILTS
               for d in ([0.0] if tilt == 0 else np.arange(0.0, 2 * np.pi, DIRECTION_STEP))
               for roll in np.arange(-np.pi, np.pi, ROLL_STEP)]
    out = np.zeros((len(path.frames), len(PSI)), bool)
    for j, psi in enumerate(PSI):
        model = cell_model(float(RHO[i_rho]), float(psi))
        scene = SoftScene(model, hoop_puzzle_margin=HOOP_MARGIN)
        for k in range(len(path.frames)):
            F = model.checkpoint_T(path, k, 0.0)
            checks = 0
            for roll, tilt, d in orients:
                for q in kin.ik(flange_target(F, float(roll), -1, float(tilt), float(d))):
                    if not (np.all(q >= lo) and np.all(q <= hi)):
                        continue
                    checks += 1
                    if scene.clearance(q, 0.0).free:
                        out[k, j] = True
                        break
                    if checks >= MAX_CLEARANCE_CHECKS:
                        break
                if out[k, j] or checks >= MAX_CLEARANCE_CHECKS:
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.monotonic()
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(row, range(len(RHO)))):
            rows.append(r)
            print(f"rho {RHO[i]:.2f} m: {100 * r.all(axis=0).mean():5.1f} % of the headings "
                  f"reach every checkpoint ({time.monotonic() - t0:.0f} s)", flush=True)
    reachable = np.stack(rows, axis=1)  # (k, rho, psi)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, rho=RHO, psi=PSI, reachable=reachable,
                        hoop_margin=HOOP_MARGIN, joint_margin=JOINT_MARGIN, entry=ENTRY,
                        bend_points=BEND_POINTS)
    print(f"written {OUT} ({time.monotonic() - t0:.0f} s): {reachable.shape[0]} checkpoints x "
          f"{len(RHO)} distances x {len(PSI)} headings; every checkpoint reachable in "
          f"{100 * reachable.all(axis=0).mean():.1f} % of the cells")


if __name__ == "__main__":
    main()
