"""Camera parameters of the CRS93 station for the simulation: resolution, time per
grab_image and the image delay (age of the image when grab_image returns).

Run from the repo root on the station computer:
    uv run Prokop/tools/camera_survey.py              # timing only, no motion
    uv run Prokop/tools/camera_survey.py --delay      # + delay test, moves joint 1

The delay test needs the robot homed (by you, this script never homes) and in
q_home with the arm visible in the camera: joint 1 moves +10 deg and back at low
speed while frames are grabbed. Output: Prokop/camera_survey/<date-time>/ with
summary.json, q_home.png and delay.csv. Zip the folder and send it.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parent.parent
GRABS = 30  # grab_image calls for the timing statistics
DELAY_MOVE_DEG = 10.0  # joint 1 step of the delay test
DELAY_SPEED = 0.25  # relative speed of the delay test
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


def record_out_and_back(robot) -> tuple[np.ndarray, np.ndarray]:
    """Joint 1 to +DELAY_MOVE_DEG and back while alternating grab_image and get_q.
    Returns frames (t_return, image change) and states (t, q1)."""
    ref = robot.grab_image().astype(np.int16)
    q0 = robot.q_home.copy()
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
                        help="also measure the image delay (moves joint 1 +-10 deg)")
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
            input(f"Joint 1 moves +{DELAY_MOVE_DEG:.0f} deg and back, the arm must be "
                  "visible in the camera. Enter to start ")
            frames, states = record_out_and_back(robot)
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
