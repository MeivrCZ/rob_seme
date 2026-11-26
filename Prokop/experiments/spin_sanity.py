"""Sanity check of the spinning planner on one placement (true model, no simulator).

plan_spinning (backwards in time from the end angles) on the true model, then
an independent replay of the plan against the EXACT meshes with the puzzle
turning: every 0.5 deg of joint motion, at the angle of that moment. Run from
the repo root (several placements in parallel = several processes):

    uv run python Prokop/experiments/spin_sanity.py AXIS_X AXIS_Y OFF_X OFF_Y [YAW_DEG [DIR [SPEED]]]
"""
import sys
import time

sys.path[:0] = ["Prokop", "Prokop/experiments"]
import numpy as np  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from placement import Placement, TaskScene  # noqa: E402
from collision import SceneCollider  # noqa: E402
from static_plan import _q_at  # noqa: E402

from solver import planner  # noqa: E402
from solver.contracts import OMEGA, TurntableModel  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402


def replay_spinning(model, theta_start, timing, qs, arrivals, step=np.deg2rad(0.5)):
    """Contacts (move index, pairs) and the least exact clearance [m] of a spinning plan."""
    exact = SceneCollider(TaskScene())
    p = Placement(np.asarray(model.axis_xy, float), np.asarray(model.base_offset, float),
                  model.base_yaw)
    _, moves = timing.schedule(qs[0], list(qs[1:]))
    least, hits = np.inf, []
    for k in range(1, len(qs)):
        n = max(2, int(np.ceil(2 * np.abs(qs[k] - qs[k - 1]).max() / step)) + 1)
        for t in np.linspace(arrivals[k - 1], arrivals[k], n):
            theta = theta_start + model.direction * model.omega * t
            h, d = exact.check(_q_at(moves, qs[0], t), p, theta)
            least = min(least, d)
            if h:
                hits.append((k, h))
                break
    return hits, least


if __name__ == "__main__":
    ax, ay, ox, oy = (float(v) for v in sys.argv[1:5])
    yaw = np.deg2rad(float(sys.argv[5])) if len(sys.argv) > 5 else 0.0
    direction = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    speed = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
    name = f"axis ({ax:.3f}, {ay:+.3f}) offset ({1000 * ox:+.0f}, {1000 * oy:+.0f}) mm " \
           f"yaw {np.rad2deg(yaw):+.0f} dir {direction:+d}"
    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, speed)
    model = TurntableModel(axis_xy=np.array([ax, ay]), base_offset=np.array([ox, oy]),
                           base_yaw=yaw, direction=direction, omega=OMEGA)
    t0 = time.monotonic()

    def progress(theta_end, r):
        print(f"{name}: end angle {np.rad2deg(theta_end):+.0f} deg: "
              f"{'collision free' if r.collision_free else r.reason} ({r.seconds:.0f} s)",
              flush=True)

    theta, r = planner.plan_spinning(kin, timing, model, progress=progress)
    if r is None:
        print(f"RESULT {name}: NO PLAN ({time.monotonic() - t0:.0f} s)", flush=True)
        sys.exit()
    hits, least = replay_spinning(model, theta, timing, r.qs, r.arrivals)
    print(f"RESULT {name}: plan in {time.monotonic() - t0:.0f} s, {len(r.qs)} waypoints, "
          f"traversal {r.arrivals[-1]:.1f} s, tilts "
          f"{np.round(np.rad2deg(r.orientations[:, 1])).astype(int).tolist()}, soft "
          f"{1000 * r.min_clearance:.1f} mm; exact replay: "
          f"{'no contact' if not hits else f'CONTACT {hits}'}, least exact "
          f"{1000 * least:.1f} mm; final pose "
          f"{'stays free while the puzzle turns' if r.end_waits else 'HIT by the turning puzzle'}",
          flush=True)
