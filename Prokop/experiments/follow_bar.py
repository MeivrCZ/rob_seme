"""Can the ring follow the turning top bar? Step along the wire, solve each step
at its arrival time, report clearance, joint jumps and whether each move is free."""
import sys
from dataclasses import replace
sys.path[:0] = ["Prokop"]
import numpy as np
from ctu_crs import CRS93
from solver import planner
from solver.contracts import TurntableModel, OMEGA
from solver.hoop_pose import flange_target
from solver.soft_collision import SoftScene
from solver.timing import RobotTiming
from solver.wire import checkpoint_arclengths, wire_frame_at

if __name__ == "__main__":
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    kin = CRS93(tty_dev=None); timing = RobotTiming(kin, speed)
    model = TurntableModel(axis_xy=np.array([0.5954, 0.1826]), base_offset=np.array([0.028, -0.035]),
                           base_yaw=0.0, direction=1, omega=OMEGA)
    theta = planner.start_angles(model, count=1)[0]
    scene = SoftScene(model)
    static = planner.plan_rrt(kin, timing, replace(model, omega=0.0), theta, 3, scene=scene)
    face, entry = static.face, static.entry
    s_cp = checkpoint_arclengths(0, entry, 3)
    o1, o2 = static.orientations[1], static.orientations[2]
    print(f"speed {speed}; static plan side {planner.FACE_NAMES[face]}, entry {1000*entry:.0f} mm; "
          f"tilt at cp1 {np.rad2deg(o1[1]):.0f}, cp2 {np.rad2deg(o2[1]):.0f} deg; top bar "
          f"{1000*(s_cp[2]-s_cp[1]):.0f} mm", flush=True)

    for label, mdl in (("FROZEN", replace(model, omega=0.0)), ("TURNING", model)):
        angle = lambda t: theta + mdl.direction * mdl.omega * t

        def solve(s, orient, q_near, t_near):
            roll, tilt, d = (float(x) for x in orient)
            q, t = q_near, t_near
            for _ in range(3):
                t = t_near + timing.move_duration(q_near, q)
                frame = mdl.puzzle_T(angle(t)) @ wire_frame_at(s, entry)
                sols = [x for x in kin.ik(flange_target(frame, roll, face, tilt, d)) if kin.in_limits(x)]
                if not sols:
                    return None, t, 0
                q = min(sols, key=lambda x: float(np.abs(x - q_near).max()))
            return q, t_near + timing.move_duration(q_near, q), len(sols)

        # checkpoint 1 in space-time, from the waiting pose at t = 0
        q0 = static.qs[0]
        q, t, _ = solve(s_cp[1], o1, q0, 0.0)
        print(f"\n{label}: checkpoint 1 at t {t:.2f} s (from the waiting pose, jump "
              f"{np.rad2deg(np.abs(q - q0).max()):.1f} deg)", flush=True)
        steps = 30
        for i in range(1, steps + 1):
            w = i / steps
            s = s_cp[1] + w * (s_cp[2] - s_cp[1])
            orient = o1 + w * np.angle(np.exp(1j * (o2 - o1)))
            q_new, t_new, nsol = solve(s, orient, q, t)
            if q_new is None:
                print(f"  {100*w:3.0f} %: NO IK solution", flush=True)
                break
            jump = np.rad2deg(np.abs(q_new - q).max())
            c = scene.clearance(q_new, angle(t_new))
            chk = planner.check_moves(scene, lambda x: angle(t + x), mdl.omega, timing,
                                      np.array([q, q_new]), np.array([-1, 0]))[0][0]
            flag = "" if not chk.hits else f"  MOVE BLOCKED {chk.hits} (least {1000*chk.clearance:.1f} mm)"
            print(f"  {100*w:3.0f} %: t {t_new:5.2f} s, jump {jump:5.1f} deg (joint "
                  f"{int(np.argmax(np.abs(q_new - q))) + 1}), {nsol} IK sols, clearance "
                  f"{1000*c.distance:5.1f} mm {c.pair[0]}-{c.pair[1]}, self {1000*c.self_gap:.1f} mm, "
                  f"move least {1000*chk.clearance:.1f} mm, {chk.samples} samples{flag}", flush=True)
            q, t = q_new, t_new
