"""Direct checkpoint-to-checkpoint moves against the clock: at each checkpoint, which
tilts still have IK, the jump, clearance, and whether the move is free."""
import sys
from dataclasses import replace
sys.path[:0] = ["Prokop"]
import numpy as np
from ctu_crs import CRS93
from solver import planner
from solver.contracts import TurntableModel, OMEGA
from solver.hoop_pose import flange_target, orientation_grid
from solver.soft_collision import SoftScene
from solver.timing import RobotTiming
from solver.wire import wire_path

if __name__ == "__main__":
    kin = CRS93(tty_dev=None); timing = RobotTiming(kin, 0.5)
    model = TurntableModel(axis_xy=np.array([0.5954, 0.1826]), base_offset=np.array([0.028, -0.035]),
                           base_yaw=0.0, direction=1, omega=OMEGA)
    offset = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    theta = planner.start_angles(model, count=1)[0] + np.deg2rad(offset)
    scene = SoftScene(model)
    static = planner.plan_rrt(kin, timing, replace(model, omega=0.0), theta, 3, scene=scene)
    path = wire_path(0, static.entry, 3)
    angle = lambda t: theta + model.omega * t
    grid = orientation_grid()
    q, t = static.qs[0], 0.0
    print(f"offset {offset:+.0f} deg: start angle {np.rad2deg(theta):.1f} deg; static {'ok' if static.collision_free else 'FAILS'}, tilts "
          f"{np.round(np.rad2deg(static.orientations[:, 1])).astype(int).tolist()}", flush=True)
    for k in range(1, len(path.frames)):
        best = None
        tilt_ok = {}
        for o in grid:  # every orientation: IK at the arrival time, closest to q
            roll, tilt, d = (float(x) for x in o)
            g, ta = q, t
            for _ in range(3):
                ta = t + timing.move_duration(q, g)
                sols = [x for x in kin.ik(flange_target(model.checkpoint_T(path, k, angle(ta)),
                                                        roll, static.face, tilt, d)) if kin.in_limits(x)]
                if not sols:
                    g = None
                    break
                g = min(sols, key=lambda x: float(np.abs(x - q).max()))
            if g is None:
                continue
            ta = t + timing.move_duration(q, g)
            if not scene.clearance(g, angle(ta)).free:
                continue
            tilt_ok[round(np.rad2deg(tilt))] = tilt_ok.get(round(np.rad2deg(tilt)), 0) + 1
            if timing.move_duration(q, g) > 1.2:
                continue
            chk = planner.check_moves(scene, lambda x: angle(t + x), model.omega, timing,
                                      np.array([q, g]), np.array([-1, 0]))[0][0]
            if not chk.hits and (best is None or ta < best[1]):
                best = (g, ta, o, chk.clearance)
        print(f"checkpoint {k}: free poses per tilt at arrival {dict(sorted(tilt_ok.items()))}; "
              + (f"direct move (<= 1.2 s) to tilt {np.rad2deg(best[2][1]):.0f} deg, arrives t "
                 f"{best[1]:.2f} s (puzzle turned {np.rad2deg(model.omega * best[1]):.0f} deg), jump "
                 f"{np.rad2deg(np.abs(best[0] - q).max()):.1f} deg, move least {1000*best[3]:.1f} mm"
                 if best else "NO free direct move (<= 1.2 s)"), flush=True)
        if best is None:
            print(f"SUMMARY offset {offset:+.0f}: stuck before checkpoint {k}")
            break
    else:
        print(f"SUMMARY offset {offset:+.0f}: ALL checkpoints reached, end at t {t:.2f} s")
        q, t = best[0], best[1]
