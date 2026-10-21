"""Camera parameters of the CRS93 station for the simulation: resolution, time per
grab_image and the image delay (age of the image when grab_image returns).

Run from the repo root on the station computer:
    uv run Prokop/tools/camera_survey.py              # timing only, no motion
    uv run Prokop/tools/camera_survey.py --delay      # + delay test, moves joint 1

The delay test needs the robot homed (by you, this script never homes) and in
q_home. It moves the arm at low speed to a survey pose with the hoop ~50 cm above
the work area, in the middle of the camera view (pose and the joint path to it
are checked against the table and the cell model first; keep the work area
free), swings joint 1 by +15 deg and back at the lowest speed while frames are
grabbed, and returns to q_home. Output: Prokop/camera_survey/<date-time>/ with
summary.json, q_home.png, survey.png and the delay CSVs.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROKOP))
GRABS = 30  # grab_image calls for the timing statistics
DELAY_MOVE_DEG = 15.0  # joint 1 step of the delay test
DELAY_SPEED = 0.0  # relative speed of the delay test (the lowest: ~5 deg/s on j1)
APPROACH_SPEED = 0.1  # relative speed to and from the survey pose
MIN_CLEARANCE = 0.08  # m, survey pose and the paths to the cell and the table
HOOP_HEIGHT = (0.40, 0.60)  # m above the table, hoop centre in the survey pose
IMAGE_MARGIN = 0.2  # fraction of the image size kept free around the hoop centre
HOLD = 0.5  # s, recording at the turning point of the move
MAX_DELAY = 1.0  # s, delays searched in [0, MAX_DELAY]


def grab_timing(robot, out: Path) -> dict:
    """Resolution and time per grab_image (the first call also opens the camera)."""
    import cv2
    durations, img = [], None
    for _ in range(GRABS):
        t = time.monotonic()
        img = robot.grab_image()
        durations.append(time.monotonic() - t)
    cv2.imwrite(str(out / "q_home.png"), img)
    d = np.array(durations[1:])
    print(f"camera {img.shape[1]} x {img.shape[0]}, grab_image median "
          f"{1000 * np.median(d):.0f} ms, max {1000 * d.max():.0f} ms")
    return {"width": int(img.shape[1]), "height": int(img.shape[0]),
            "channels": int(img.shape[2]), "first_grab_s": durations[0],
            "grab_s_median": float(np.median(d)), "grab_s_max": float(d.max())}


def project(p: np.ndarray) -> np.ndarray:
    """Pixel (u, v) of the point @p (robot frame) in the modelled camera."""
    from cell import CAMERA_K, CAMERA_T
    pc = np.linalg.solve(CAMERA_T, [*p, 1.0])[:3]
    uvw = CAMERA_K @ pc
    return uvw[:2] / uvw[2]


def survey_pose(robot, checker=None) -> tuple[np.ndarray, dict]:
    """Start configuration of the joint 1 swing: the flange pointing down over the
    work area, turned by joint 1 half a swing to the right, so that the swing is
    centred in the image. The hoop centre stays HOOP_HEIGHT above the table and
    inside the camera view (IMAGE_MARGIN) at both ends of the swing; clear of the
    cell and the table by MIN_CLEARANCE in the pose, on the path from q_home and
    during the swing. Of those, the one needing the smallest joint motion from
    q_home."""
    from cell import CAMERA_RESOLUTION, TABLE_Z
    from collision import ArmChecker
    checker = checker or ArmChecker()
    size = np.array(CAMERA_RESOLUTION, dtype=float)
    swing = np.zeros(6)
    swing[0] = np.deg2rad(DELAY_MOVE_DEG)
    candidates = []
    for x in np.arange(0.30, 0.56, 0.05):
        for z in np.arange(0.35, 0.80, 0.05):
            for yaw in np.arange(8) * np.pi / 4:
                T = np.eye(4)
                c, s = np.cos(yaw), np.sin(yaw)
                T[:3, :3] = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]])  # z down
                T[:3, 3] = x, 0.0, z
                for q in robot.ik(T):
                    q = q - swing / 2
                    if not (robot.in_limits(q) and robot.in_limits(q + swing)):
                        continue
                    ok = True
                    for qq in (q, q + swing):
                        h = checker.hoop_center(qq)
                        uv = project(h)
                        ok &= HOOP_HEIGHT[0] <= h[2] - TABLE_Z <= HOOP_HEIGHT[1]
                        ok &= bool(np.all(uv > IMAGE_MARGIN * size)
                                   and np.all(uv < (1 - IMAGE_MARGIN) * size))
                    if ok:
                        candidates.append(q)
    candidates.sort(key=lambda q: np.abs(q - robot.q_home).max())
    for q in candidates:
        checks = {"pose": checker.clearance(q),
                  "path_from_q_home": checker.path_clearance(robot.q_home, q),
                  "swing": checker.path_clearance(q, q + swing)}
        if all(d >= MIN_CLEARANCE for d, _ in checks.values()):
            h = checker.hoop_center(q)
            return q, {"q_deg": np.rad2deg(q).round(2).tolist(),
                       "hoop_center_m": h.round(3).tolist(),
                       "hoop_pixel": project(h).round(0).tolist(),
                       "clearance_m": {k: [round(d, 3), w]
                                       for k, (d, w) in checks.items()}}
    raise RuntimeError("no survey pose found")


def record_out_and_back(robot, q0: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Joint 1 from @q0 (default q_home) to +DELAY_MOVE_DEG and back while
    alternating grab_image and get_q. Returns frames (t_return, image change) and
    states (t, q1)."""
    ref = robot.grab_image().astype(np.int16)
    q0 = (robot.q_home if q0 is None else q0).copy()
    q1 = q0.copy()
    q1[0] += np.deg2rad(DELAY_MOVE_DEG)
    frames, states = [], []
    t0 = time.monotonic()

    def sample() -> None:
        img = robot.grab_image()
        t_img = time.monotonic()
        frames.append([t_img - t0, np.abs(img.astype(np.int16) - ref).mean()])
        t = time.monotonic()
        q = robot.get_q()
        states.append([(t + time.monotonic()) / 2 - t0, q[0]])

    robot.set_speed_relative(DELAY_SPEED)
    for target in (q1, q0):
        robot.move_to_q(target)
        sample()
        while robot.in_motion():
            sample()
        t_hold = time.monotonic()
        while time.monotonic() - t_hold < HOLD:
            sample()
    return np.array(frames), np.array(states)


def estimate_delay(frames: np.ndarray, states: np.ndarray) -> dict:
    """Delay d at which the image change, plotted against the joint displacement
    d seconds before the frame returned, is the same on the way out and back.

    The image change grows with the displacement in some unknown way, the same in
    both directions; with a wrong d the out and back branches separate into a
    loop. d is the minimum of the mismatch over a grid (step 2 ms)."""
    t_q, disp = states[:, 0], np.abs(states[:, 1] - states[0, 1])
    t_img, change = frames[:, 0], frames[:, 1]
    change = (change - change.min()) / max(np.ptp(change), 1e-9)
    t_peak = t_q[np.argmax(disp)]
    level = np.linspace(0.1, 0.9, 41) * disp.max()
    delays = np.arange(0.0, MAX_DELAY, 0.002)
    losses = []
    for d in delays:
        s = np.interp(t_img - d, t_q, disp)
        loss = np.inf
        out = (t_img - d < t_peak) & (t_img - d > t_q[0])
        back = t_img - d > t_peak
        if out.sum() > 3 and back.sum() > 3:
            a = np.interp(level, *_sorted(s[out], change[out]))
            b = np.interp(level, *_sorted(s[back], change[back]))
            loss = float(np.mean((a - b) ** 2))
        losses.append(loss)
    losses = np.array(losses)
    i = int(np.argmin(losses))
    return {"delay_s": float(delays[i]), "mismatch": float(losses[i]),
            "mismatch_at_0": float(losses[0])}


def _sorted(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.argsort(x)
    return x[k], y[k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--delay", action="store_true",
                        help="also measure the image delay (moves the arm to a "
                             "survey pose, joint 1 +15 deg and back, then home)")
    args = parser.parse_args()

    from ctu_crs import CRS93
    robot = CRS93()
    out = PROKOP / "camera_survey" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    summary: dict = {"started": datetime.now().isoformat(timespec="seconds")}
    try:
        summary["timing"] = grab_timing(robot, out)
        if args.delay:
            robot.initialize(home=False)
            if not np.allclose(robot.get_q(), robot.q_home, atol=np.deg2rad(2)):
                raise SystemExit(f"the robot is not in q_home (q [deg] = "
                                 f"{np.rad2deg(robot.get_q()).round(1)})")
            import cv2
            q_s, info = survey_pose(robot)
            summary["survey_pose"] = info
            print("survey pose:", json.dumps(info, indent=2))
            input(f"The arm moves at low speed to the survey pose above (hoop over "
                  f"the work area, keep it free), joint 1 swings +{DELAY_MOVE_DEG:.0f} "
                  "deg and back, then the arm returns to q_home. Enter to start ")
            robot.set_speed_relative(APPROACH_SPEED)
            robot.move_to_q(q_s)
            robot.wait_for_motion_stop()
            time.sleep(1.0)  # settle, auto exposure
            cv2.imwrite(str(out / "survey.png"), robot.grab_image())
            frames, states = record_out_and_back(robot, q_s)
            robot.set_speed_relative(APPROACH_SPEED)
            robot.move_to_q(robot.q_home)
            robot.wait_for_motion_stop()
            np.savetxt(out / "delay_frames.csv", frames, delimiter=",",
                       header="t_grab_return,image_change", comments="")
            np.savetxt(out / "delay_states.csv", states, delimiter=",",
                       header="t_get_q,q1", comments="")
            summary["delay"] = estimate_delay(frames, states)
            print(f"image delay {1000 * summary['delay']['delay_s']:.0f} ms")
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Results in {out}  -> zip this folder and send it")
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
