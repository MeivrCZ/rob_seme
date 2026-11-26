"""Offline analysis of a simulated execution (run_solver.py's execution.pkl).

    uv run python Prokop/experiments/execution_replay.py Prokop/runs/<time>/execution.pkl

1. The camera tracker, frame by frame: every image the tracker grabbed during
   phase 3 is rendered again (the same scene at the image's time: the arm's
   true configuration and the true turntable angle), detected and measured
   like the tracker does (ArucoBlockDetector + measure_base_pose), compared
   with the truth, and fed in order through Vojtech's CameraPhaseTracker
   (its _observe, as his tests do): which measurements it accepted and how
   far its angle then was from the truth.
2. The steps' timing: for every step, how long the command waited on the
   simulator link, when the motion really started against the executor's
   prediction, and how long it really took against the timing model.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))
from run_solver import simulated_calibration  # noqa: E402  (puts Vojtech's dirs on the path)
from advanced_aruco import AdvancedArucoFinder, ArucoResult  # noqa: E402
from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from camera import SceneRenderer  # noqa: E402
from camera_phase_tracker import CameraPhaseTracker  # noqa: E402
from placement import Placement, TaskScene  # noqa: E402
from turntable_orbit import MARKER_HEIGHT_ABOVE_TABLE_M, measure_base_pose  # noqa: E402

from solver.contracts import TurntableModel  # noqa: E402


def wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def main() -> None:
    d = pickle.loads(Path(sys.argv[1]).read_bytes())
    t_ref, theta_ref = d["truth_angle"]
    pl = d["placement"]
    placement = Placement(pl["turntable_xy"], pl["base_xy"], pl["base_yaw"])
    truth = TurntableModel(axis_xy=pl["turntable_xy"], base_offset=pl["base_xy"],
                           base_yaw=pl["base_yaw"], direction=d["direction"],
                           omega=d["omega_true"], theta_ref=theta_ref, t_ref=t_ref)
    fitted = TurntableModel(**d["model"])
    track_t = np.array([t for t, _ in d["track"]])
    track_q = np.array([q for _, q in d["track"]])

    def q_at(t: float) -> np.ndarray:
        return np.array([np.interp(t, track_t, track_q[:, j]) for j in range(6)])

    def error(theta_fitted: float, t: float) -> float:
        """Fitted-convention angle minus the truth, by base orientation [rad]."""
        return wrap(theta_fitted + fitted.base_yaw - truth.theta_at(t) - truth.base_yaw)

    t0 = d["t_attempt"]
    start = d["t_start"]
    print(f"run: {d['reason'] or 'traversal done'}; traversal started at "
          f"t+{(start - t0) if start else float('nan'):.2f} s (t = 0: the executor's start)")

    # 1. The camera tracker, frame by frame.
    calibration = simulated_calibration()
    detector = ArucoBlockDetector()
    marker_z = calibration.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
    renderer = SceneRenderer(TaskScene())
    tracker = CameraPhaseTracker(AdvancedArucoFinder(detector, calibration, fitted),
                                 lambda: None, d["camera_delay"])
    grabs = [(a, b) for name, a, b in d["calls"] if name == "grab_image"]
    server = d.get("grabs", [])  # the simulator's own timings of each grab (if recorded)

    def true_time(a, b):
        """The simulator's record of the grab answered within the call [a, b]."""
        # The reply sent last before the call returned (the previous call's reply can be
        # sent just before this call starts).
        rec = [g for g in server if a <= g[5] <= b + 0.01]
        return max(rec, key=lambda g: g[5]) if rec else None

    print(f"\n1. the camera tracker: {len(grabs)} images (t = the tracker's stamp - executor "
          f"start; 'stamp late' = stamp - the true image time)")
    print("   t [s]   markers  xy err   yaw err   tracker   its error   arm      stamp late")
    late_rows = []
    for a, b in grabs:
        t_img = b - d["camera_delay"]  # what the tracker stamps
        rec = true_time(a, b)
        t_true = rec[0] if rec else t_img  # what the image really shows
        if rec:
            late_rows.append((t_img - t0, t_img - rec[0], rec[1] - rec[0], rec[2] - rec[1],
                              rec[4] - rec[3], b - rec[5]))
        q = q_at(t_true)
        image = renderer.render(q, placement, truth.theta_at(t_true))
        pose = detector.detect(image)
        arm = "moving" if start and t_img >= start else "waiting"
        if pose is None:
            print(f"  {t_img - t0:6.2f}   none                                            {arm}")
            continue
        m = measure_base_pose(pose, t_img, calibration, marker_z)
        xy_t, yaw_t = truth.measurement_at(t_img)  # the truth at the stamp
        before = tracker._estimate.t if tracker._estimate is not None else None
        tracker._observe(ArucoResult(t_img, m.xy.copy(), m.yaw, pose, t_img))
        est = tracker._estimate
        accepted = est is not None and est.t == t_img and est.t != before
        err = f"{np.rad2deg(error(est.theta, est.t)):+6.2f} deg" if accepted else ""
        print(f"  {t_img - t0:6.2f}   {len(pose.visible_ids)} {str(sorted(pose.visible_ids)):7s}"
              f" {1000 * np.linalg.norm(m.xy - xy_t):5.1f} mm {np.rad2deg(wrap(m.yaw - yaw_t)):+6.2f}"
              f" deg  {'accepted' if accepted else 'rejected':8s}  {err:10s}  {arm:7s}"
              + (f"  {1000 * (t_img - t_true):+5.0f} ms" if rec else ""))
    if late_rows:
        rows = np.array(late_rows)
        print("\n   where a late stamp comes from (ms): stamp late = monitor wait + render +"
              " ... ; per frame with the stamp > 20 ms late:")
        print("   t [s]   stamp late  monitor wait  render  return overrun  transfer")
        for r in rows[np.abs(rows[:, 1]) > 0.02]:
            print(f"  {r[0]:6.2f}   {1000 * r[1]:8.0f}   {1000 * r[2]:10.0f}   {1000 * r[3]:6.0f}"
                  f"   {1000 * r[4]:12.0f}   {1000 * r[5]:7.0f}")
        print(f"   all {len(rows)} frames: stamp late median {1000 * np.median(rows[:, 1]):.0f} ms, "
              f"max {1000 * rows[:, 1].max():.0f} ms; render median {1000 * np.median(rows[:, 3]):.0f}"
              f" ms; transfer median {1000 * np.median(rows[:, 5]):.0f} ms")

    # 2. The steps' timing.
    moves = [(a, b) for name, a, b in d["calls"] if name == "move_to_q"]
    print("\n2. the steps (times in s; 'pred' = the executor's timing model)")
    print("   step  link wait  start: pred   real   late   duration: pred   real   arrival late")
    q_prev, queue_end = np.asarray(d["plan"]["q_start"]), None
    for s in d["steps"]:
        call = min((m for m in moves if m[0] >= s["t_send"] - 0.005),
                   key=lambda m: m[0] - s["t_send"], default=None)
        wait = (call[1] - call[0]) if call else float("nan")
        pred_start = s["t_send"] + d["command_delay"]
        if queue_end is not None:
            pred_start = max(pred_start, queue_end)
        moving = np.abs(track_q - q_prev).max(axis=1) > np.deg2rad(0.05)
        after = (track_t >= s["t_send"] - 0.5) & moving
        real_start = track_t[after][0] if after.any() else np.nan
        there = (track_t >= s["t_send"]) & (np.abs(track_q - s["q"]).max(axis=1) < np.deg2rad(0.3))
        real_arrival = track_t[there][0] if there.any() else np.nan
        print(f"   {s['index']:4d}  {1000 * wait:6.0f} ms   {pred_start - t0:6.2f} {real_start - t0:6.2f} "
              f"{real_start - pred_start:+6.2f}   {s['t_arrival'] - pred_start:13.2f} "
              f"{real_arrival - real_start:6.2f}   {real_arrival - s['t_arrival']:+6.2f}")
        q_prev, queue_end = np.asarray(s["q"]), s["t_arrival"]


if __name__ == "__main__":
    main()
