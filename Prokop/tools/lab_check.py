"""Checks at the CRS93 station that the solver's environment parameters are right.

What the solver assumes and nobody has measured on the real station yet (the
October experiments E1-E9 in docs/EXPERIMENTS.md measured the cell, the turntable
period, the motion profile, the delays, blending and the camera timing):

    dims        no robot   the model's dimensions, to check with a caliper (+-1 mm
                           is what the margins assume)
    calibrate   MOVES      the camera calibration (homography) from the hoop over a
                           grid in the EMPTY work area -> the JSON for run_solver --real
    timing      MOVES      a chain of small moves near q_home at speed 0.5, sent the
                           way the executor sends them (0.15 s before the previous move
                           ends): predicted vs real arrivals; again with a camera thread
                           grabbing (does a grab hold up the robot's commands?)
    turntable   no motion  perception on the spinning turntable: the orbit fit (axis,
                           offset, period vs 28.0 s, residuals), then 30 s of
                           Vojtech's camera tracker against the fit (drift, lost markers)
    fov         no motion  the camera's view of the work area: move the base around,
                           every second which markers are seen and where

Usage, from the repo root (on the station PC; --sim: the same against the standalone
simulator, to try the script first):

    uv run Prokop/tools/lab_check.py dims
    uv run Prokop/tools/lab_check.py calibrate [--sim]
    uv run Prokop/tools/lab_check.py timing [--sim]
    uv run Prokop/tools/lab_check.py turntable --calibration FILE [--sim]
    uv run Prokop/tools/lab_check.py fov [--calibration FILE] [--sim]

Safety (real robot): the script NEVER homes (homing is not collision checked): home
the robot by hand first and leave it in q_home; the script refuses to start
otherwise. Every motion is checked against the soft model of the table and the cell
first and asks for Enter before it starts. Keep a hand on the emergency stop.

Results: Prokop/experiments/lab/<time>_<check>/ (summary.json, log.txt, images); the
calibration also as Prokop/data/calibration/homography_<time>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
REPO = PROKOP.parent
for d in (PROKOP, PROKOP / "tools", REPO / "Vojtěch" / "ArucoFinder", REPO / "Vojtěch" / "Homografy",
          REPO / "Natka"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
# Vojtech's hoop detector (rob_crs93_v4) at the END: that folder has its own module
# "homography" too, the Homografy one must win.
sys.path.append(str(REPO / "Vojtěch" / "rob_crs93_v4"))

OUT = PROKOP / "experiments" / "lab"
CALIBRATION_DIR = PROKOP / "data" / "calibration"
D = np.deg2rad
HOME_TOLERANCE = D(2.0)
CAL_SPEED = 0.2  # the calibration's moves (low)
TIMING_SPEED = 0.5  # the solver's speed
GUARD = 0.15  # s, the executor sends a move this long before the previous one ends
# The calibration grid: the hoop's centre over the work area's middle (the corners may
# be out of reach with the ring held flat), at Vojtech's Mainframe heights.
CAL_X = (0.39, 0.58)
CAL_Y = (-0.15, 0.15)
CAL_HEIGHTS = (0.075, 0.13, 0.19)  # m above the table
CAL_N = 3
# The timing chain (offsets from q_home [deg], joints 1, 4 and 6 only: the arm stays up
# as in tools/log_motion.py): direction changes between the moves, so the robot stops
# at each, as on a planned traversal.
TIMING_CHAIN = ([12, 0, 0, 25, 0, 0], [12, 0, 0, 25, 0, 40], [-8, 0, 0, 25, 0, 40],
                [-8, 0, 0, 0, 0, 10], [0, 0, 0, 0, 0, 0])


class Log:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.file = (folder / "log.txt").open("w", encoding="utf-8")

    def __call__(self, *parts) -> None:
        text = " ".join(str(p) for p in parts)
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()


def confirm(args, what: str) -> None:
    """On the real robot: Enter to go on, anything else (or Ctrl+C) aborts."""
    if args.sim:
        return
    answer = input(f"\n>>> {what}\n    Enter to start, anything else to abort: ")
    if answer.strip():
        raise SystemExit("aborted")


def connect(args, log: Log):
    """The robot: the real one (at q_home, never homed here) or the simulator's."""
    from simulation import CAMERA_DELAY
    if args.sim:
        from sim_process import SimProcess
        sim = SimProcess(port=args.port)
        robot = sim.robot
        robot.initialize(home=False)
        log(f"simulator: http://localhost:{args.port}")
        return robot, sim
    from ctu_crs import CRS93
    robot = CRS93()
    robot.initialize(home=False)  # never home from a script
    robot.camera_delay = CAMERA_DELAY  # ctu_crs has none (E8: 0.11 s)
    q = robot.get_q()
    off = float(np.abs(q - robot.q_home).max())
    if off > HOME_TOLERANCE:
        raise SystemExit(f"the robot is {np.rad2deg(off):.1f} deg from q_home: home it by hand "
                         f"first (this script never homes)")
    log("real robot at q_home")
    return robot, None


def free_scene():
    """The soft model with the task far away: only the table and the cell count (the
    calibration and the timing moves are made with the work area empty / the arm up)."""
    from solver.contracts import TurntableModel
    from solver.soft_collision import SoftScene
    return SoftScene(TurntableModel(axis_xy=np.array([5.0, 5.0]), base_offset=np.zeros(2),
                                    base_yaw=0.0, direction=1, omega=0.0))


def path_free(scene, timing, qs) -> tuple[bool, float]:
    from solver.planner import check_moves
    checks, _ = check_moves(scene, lambda t: 0.0, 0.0, timing, np.array(qs),
                            np.zeros(len(qs), int))
    return not any(c.hits for c in checks), min(c.clearance for c in checks)


# --- dims ---

def cmd_dims(args, log: Log) -> dict:
    """The model's dimensions to check with a caliper / tape."""
    from ctu_crs import CRS93
    from cell import TABLE_Z
    from solver.hoop_pose import ring_in_flange
    from solver import wire
    from turntable_orbit import MARKER_HEIGHT_ABOVE_TABLE_M
    from solver.soft_collision import RING_CENTRE_Z
    kin = CRS93(tty_dev=None)
    ring = kin.fk(kin.q_home) @ ring_in_flange()
    mid = ring[:3, 3] + RING_CENTRE_Z * ring[:3, 2]  # hoop_center is on a face: the middle
    rows = [
        ("puzzle bar: side of the square cross-section", 2 * wire.BAR_HALF, "caliper, anywhere on the bar"),
        ("puzzle: top bar centreline above the pad (base top)", wire.TOP_Z - wire.PAD_TOP_Z,
         "from the pad to the top bar's centre (top edge - 5 mm)"),
        ("puzzle: open end from the stem's axis (horizontal)", wire.OPEN_END_X,
         "stem centre to the top bar's open end face"),
        ("puzzle: bend radius (centreline)", wire.ARC_RADIUS, "the bend's inner radius + 5 mm"),
        ("puzzle: jog top above the pad", wire.JOG_TOP_Z - wire.PAD_TOP_Z, "where the vertical run starts to bend"),
        ("puzzle: jog bottom above the pad", wire.JOG_BOTTOM[2] - wire.PAD_TOP_Z, "where the stem starts"),
        ("hoop: hole diameter", 2 * wire.RING_INNER_RADIUS, "caliper, inner"),
        ("hoop: ring outer diameter", 2 * wire.RING_OUTER_RADIUS, "caliper, outer"),
        ("hoop: ring thickness", wire.RING_THICKNESS, "caliper"),
        ("base: marker plane above the table", MARKER_HEIGHT_ABOVE_TABLE_M,
         "table top to the ArUco stickers' surface"),
        ("ring centre (mid-thickness) above the table at q_home", float(mid[2] - TABLE_Z),
         "robot at q_home: tape from the table to the middle of the ring"),
        ("ring centre in front of the robot axis at q_home", float(mid[0]),
         "robot at q_home: horizontal distance robot axis -> ring centre"),
    ]
    log("Model dimensions to check (the margins assume each right to its 1 mm rounding, "
        "sd 1/sqrt(12) mm: pass within +-1 mm):\n")
    log(f"  {'what':55s} {'model':>9s}   how")
    for what, value, how in rows:
        log(f"  {what:55s} {1000 * value:7.1f} mm   {how}")
    log("\nWrite the measured values next to these; differences > 1 mm: tell Prokop (the "
        "soft model / wire.py constants would have to change).")
    return {"rows": [{"what": w, "model_mm": 1000 * v, "how": h} for w, v, h in rows]}


# --- calibrate ---

def cmd_calibrate(args, log: Log) -> dict:
    """The homography from the hoop's centre over a grid: the ring held flat (its axis
    vertical, the camera looks at it from above), the centre from the forward kinematics
    of the measured joints, the pixel from the ring's detection."""
    import cv2
    from cell import TABLE_Z
    from homography import Homography
    from solver.hoop_pose import ring_in_flange
    from solver.timing import RobotTiming
    robot, sim = connect(args, log)
    try:
        if sim is not None:  # the simulator always has the task: park it out of the way
            from placement import Placement
            sim.set_placement(Placement(np.array([-0.40, 0.0]), np.zeros(2), 0.0))
        else:
            confirm(args, "Remove the turntable, the base and the puzzle from the work area "
                          "(the hoop goes down to 75 mm above the table).")
        timing = RobotTiming(robot, CAL_SPEED)
        scene = free_scene()
        to_ring = ring_in_flange()
        flange_of_ring = np.linalg.inv(to_ring)
        hom = Homography(x_range_m=CAL_X, y_range_m=CAL_Y, z_levels_m=CAL_HEIGHTS,
                         table_z_m=TABLE_Z, nx=CAL_N, ny=CAL_N)
        points = np.asarray(hom.points)
        # A flat-ring pose per point (axis up or down, any roll), the one closest to the
        # previous one, each move checked against the table and the cell.
        qs, q_prev = [], robot.q_home
        for p in points:
            best = None
            for up in (1.0, -1.0):
                for roll in np.arange(-np.pi, np.pi, D(15)):
                    c, s_ = np.cos(roll), np.sin(roll)
                    T = np.eye(4)
                    T[:3, :3] = np.diag([1.0, up, up]) @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
                    T[:3, 3] = p
                    for q in robot.ik(T @ flange_of_ring):
                        if not robot.in_limits(q) or not scene.clearance(q, 0.0).free:
                            continue
                        d = float(np.abs(q - q_prev).max())
                        if best is None or d < best[0]:
                            best = (d, q)
            if best is None:
                log(f"  point {np.round(1000 * p).astype(int).tolist()} mm: no free flat-ring "
                    f"pose (skipped)")
                qs.append(None)
                continue
            qs.append(best[1])
            q_prev = best[1]
        route = [robot.q_home] + [q for q in qs if q is not None] + [robot.q_home]
        ok, least = path_free(scene, timing, route)
        log(f"{sum(q is not None for q in qs)} of {len(points)} grid points reachable; the route "
            f"q_home -> points -> q_home is {'free' if ok else 'NOT free'} of the table and the "
            f"cell (least soft clearance {1000 * least:.0f} mm)")
        if not ok:
            raise SystemExit("the calibration route collides in the soft model: not moving")
        confirm(args, f"The hoop visits {len(route) - 2} points at speed {CAL_SPEED} and "
                      f"returns to q_home.")
        robot.set_speed_relative(CAL_SPEED)
        from hoop_detection import detect_hoop_circle  # Vojtech's (rob_crs93_v4)
        from analyze_survey import find_ring
        from camera import Camera
        cam = Camera()
        recorded = 0
        for i, q in enumerate(qs):
            if q is None:
                continue
            robot.move_to_q(q)
            robot.wait_for_motion_stop()
            time.sleep(0.5)  # the arm settles; the image is ~0.11 s old when it returns
            q_meas = robot.get_q()
            xyz = (robot.fk(q_meas) @ to_ring)[:3, 3]
            image = robot.grab_image()
            if image is None:
                log(f"  point {i:2d}: the camera returned no image (skipped)")
                continue
            cv2.imwrite(str(log.folder / f"point_{i:02d}.png"), image)
            uv, how = None, ""
            try:
                det = detect_hoop_circle(image)
                if det.center is not None:
                    uv, how = np.asarray(det.center, float), "hole detector"
            except Exception as error:  # noqa: BLE001 - report, try the other detector
                how = f"hole detector failed ({type(error).__name__})"
            if uv is None:  # the fallback: less accurate (in the simulator 2-13 px off)
                try:
                    uv_fit, _ = find_ring(image, cam.project(xyz))
                    if args.ring_fallback:
                        uv, how = uv_fit, "blue-ring fit"
                    else:
                        how += (f"; the blue-ring fit found it at ({uv_fit[0]:.1f}, "
                                f"{uv_fit[1]:.1f}) px, not used (--ring-fallback to use it)")
                except Exception as error:  # noqa: BLE001
                    how += f"; ring fit failed ({type(error).__name__})"
            if uv is None:
                log(f"  point {i:2d}: ring NOT found by the hole detector ({how})")
                continue
            hom.record(i, (float(uv[0]), float(uv[1])), tuple(float(x) for x in xyz))
            recorded += 1
            log(f"  point {i:2d}: ring at ({uv[0]:.1f}, {uv[1]:.1f}) px ({how}), centre "
                f"{np.round(1000 * xyz, 1).tolist()} mm; the camera model predicts "
                f"{np.round(cam.project(xyz), 1).tolist()} px")
        robot.move_to_q(robot.q_home)
        robot.wait_for_motion_stop()
        if recorded < 8:
            raise SystemExit(f"only {recorded} points recorded: not enough for a calibration")
        hom.calculate(validation_fraction=0.25)
        fit = dict(hom.fit_info)
        val = hom.validate()  # on the held-out points (independent)
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIBRATION_DIR / f"homography_{datetime.now():%Y%m%d_%H%M%S}{'_sim' if args.sim else ''}.json"
        hom.save(path)
        log(f"\ncalibration from {recorded} points: {fit.get('inlier_count')} inliers, rejected "
            f"(RANSAC) {fit.get('rejected_ids')}; on {fit.get('validation_count')} held-out "
            f"points: {val.get('pixel_rmse', float('nan')):.2f} px RMS, "
            f"{val.get('xy_rmse_mm', float('nan')):.2f} mm RMS in the table plane\n"
            f"saved: {path}\nuse it: uv run Prokop/run_solver.py --real --calibration {path}")
        return {"recorded": recorded, "fit": fit, "validation": val, "path": str(path)}
    finally:
        if sim is not None:
            sim.stop()


# --- timing ---

def run_chain(robot, timing, camera_thread: bool, log: Log) -> list[dict]:
    """The chain sent like the executor (each move GUARD before the previous predicted
    end); get_q polled in the same thread; optionally a camera thread grabbing."""
    stop = threading.Event()
    grabs = []

    def grabber():
        while not stop.is_set():
            t0 = time.monotonic()
            robot.grab_image()
            grabs.append(time.monotonic() - t0)
    thread = None
    if camera_thread:
        robot.grab_image()  # the first grab opens the camera (seconds on the real one)
        thread = threading.Thread(target=grabber, daemon=True)
        thread.start()
        time.sleep(1.0)
    targets = [robot.q_home + D(np.array(o, float)) for o in TIMING_CHAIN]
    log_q, calls = [], []  # (t at the middle of the get_q round trip, q); send times

    def sample():
        a = time.monotonic()
        q = robot.get_q()
        log_q.append((0.5 * (a + time.monotonic()), q))
    q_start = robot.get_q()
    prev_end = None
    for q in targets:
        if prev_end is not None:  # poll until it is time to send (as the executor does)
            while time.monotonic() < prev_end - timing.command_delay - GUARD:
                sample()
        t_send = time.monotonic()
        robot.move_to_q(q)
        calls.append((t_send, time.monotonic() - t_send))
        prev_end = timing.schedule(q_start, targets[:len(calls)],
                                   call_times=[c[0] for c in calls])[0][-1]
    while robot.in_motion() or time.monotonic() < prev_end + 0.5:
        sample()
    stop.set()
    if thread is not None:
        thread.join(timeout=2.0)
    arrivals, moves = timing.schedule(q_start, targets, call_times=[c[0] for c in calls])
    ts = np.array([t for t, _ in log_q])
    qq = np.array([q for _, q in log_q])
    period = float(np.median(np.diff(ts))) if len(ts) > 1 else float("nan")
    # Every sample inside a move: its progress along the move -> the time the model
    # predicts for it; the median of (sample time - predicted time) = how late the
    # real move is (the arrival error).
    rows = []
    for k, m in enumerate(moves):
        dq = m.q_to - m.q_from
        s_ = (qq - m.q_from) @ dq / float(dq @ dq)
        win = (ts > m.t0 - 0.3) & (ts < m.t_end + 0.3) & (s_ > 0.03) & (s_ < 0.97)
        late = [t - m.time_at(float(x)) for t, x in zip(ts[win], s_[win])]
        rows.append({"move": k + 1, "move_to_q_s": calls[k][1] if k < len(calls) else 0.0,
                     "predicted_end": m.t_end,
                     "error_s": float(np.median(late)) if late else float("nan"),
                     "error_spread_s": float(np.std(late)) if late else float("nan"),
                     "samples": len(late), "sample_period_s": period})
    if grabs:
        rows.append({"grab_s_median": float(np.median(grabs)), "grabs": len(grabs)})
    return rows


def cmd_timing(args, log: Log) -> dict:
    from solver.timing import RobotTiming
    robot, sim = connect(args, log)
    try:
        timing = RobotTiming(robot, TIMING_SPEED)
        scene = free_scene()
        chain = [robot.q_home] + [robot.q_home + D(np.array(o, float)) for o in TIMING_CHAIN]
        ok, least = path_free(scene, timing, chain)
        log(f"the chain ({len(TIMING_CHAIN)} moves, joints 1/4/6, at most 12 deg on joint 1) is "
            f"{'free' if ok else 'NOT free'} of the table and the cell ({1000 * least:.0f} mm)")
        if not ok:
            raise SystemExit("the timing chain collides in the soft model: not moving")
        out = {}
        for camera_thread in (False, True):
            label = "with a camera thread grabbing" if camera_thread else "without the camera"
            confirm(args, f"Timing chain {label}: {len(TIMING_CHAIN)} small moves near q_home "
                          f"at speed {TIMING_SPEED}, back to q_home.")
            robot.set_speed_relative(TIMING_SPEED)
            rows = run_chain(robot, timing, camera_thread, log)
            moves = [r for r in rows if "move" in r]
            log(f"\n{label} (get_q every {1000 * moves[0]['sample_period_s']:.0f} ms; the "
                f"lateness fitted over each whole move):")
            for r in moves:
                log(f"  move {r['move']}: move_to_q {1000 * r['move_to_q_s']:4.0f} ms, arrived "
                    f"{1000 * r['error_s']:+5.0f} ms against the prediction (sd "
                    f"{1000 * r['error_spread_s']:.0f} ms over {r['samples']} samples)")
            for r in rows:
                if "grabs" in r:
                    log(f"  camera: {r['grabs']} grabs, {1000 * r['grab_s_median']:.0f} ms each")
            errs = np.array([r["error_s"] for r in moves])
            log(f"  -> arrival error median {1000 * np.nanmedian(errs):+.0f} ms, worst "
                f"{1000 * np.nanmax(np.abs(errs)):.0f} ms (the solver assumes ~0; the "
                f"simulator: +10 ms, sd 2 ms, without the camera)")
            out["camera" if camera_thread else "plain"] = rows
            time.sleep(1.0)
        return out
    finally:
        if sim is not None:
            sim.stop()


# --- turntable ---

def cmd_turntable(args, log: Log) -> dict:
    from aruco_block_detector import ArucoBlockDetector
    from homography import Homography
    from turntable_orbit import MARKER_HEIGHT_ABOVE_TABLE_M, capture_series, fit_orbit, measure_base
    from advanced_aruco import AdvancedArucoFinder
    from camera_phase_tracker import CameraPhaseTracker
    robot, sim = connect(args, log)
    try:
        if sim is not None:
            from cell import WORK_AREA_XY
            from placement import TaskScene
            from solver.contracts import OMEGA
            sim.set_placement(TaskScene().random_placement(np.random.default_rng(3), WORK_AREA_XY))
            sim.set_spin(OMEGA)
            calibration = __import__("run_solver").simulated_calibration()
        else:
            if args.calibration is None:
                raise SystemExit("--calibration FILE needed (from: lab_check.py calibrate)")
            calibration = Homography.load(args.calibration)
            confirm(args, "The turntable spinning, the base with the puzzle on it, in the "
                          "camera's view. Nothing moves the robot.")
        detector = ArucoBlockDetector()
        marker_z = calibration.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
        found = []
        for t, image in capture_series(robot.grab_image, args.images, 1.0, robot.camera_delay):
            m = measure_base(image, t, detector, calibration, marker_z)
            log(f"  image at {t:.2f}: " + ("no markers" if m is None else
                f"{m.n_markers} marker(s), base ({1000 * m.xy[0]:.1f}, {1000 * m.xy[1]:.1f}) mm, "
                f"yaw {np.rad2deg(m.yaw):+.1f} deg"))
            if m is not None:
                found.append(m)
        if len(found) < 4:
            raise SystemExit(f"only {len(found)} detections: the base is not in view")
        fit = fit_orbit(found)
        model = fit.model
        log(f"\norbit fit from {len(found)} images: axis ({1000 * model.axis_xy[0]:.1f}, "
            f"{1000 * model.axis_xy[1]:.1f}) mm, offset {1000 * fit.radius_m:.1f} mm, period "
            f"{fit.period_s:.2f} s (E3 stopwatch: 28.00 s), direction {model.direction:+d}; "
            f"residuals {1000 * fit.xy_rmse_m:.2f} mm, {np.rad2deg(fit.yaw_rmse_rad):.2f} deg")
        log(f"\ncamera tracker for {args.track:.0f} s against the fit (Vojtech's "
            f"CameraPhaseTracker; the drift = how far the fit alone would be off):")
        cam = CameraPhaseTracker(AdvancedArucoFinder(detector, calibration, model),
                                 robot.grab_image, robot.camera_delay)
        drift, lost, ages = [], 0, []
        with cam:
            if not cam.wait_ready(timeout=15.0):
                raise SystemExit("the camera tracker did not confirm the rotation in 15 s")
            t_end = time.monotonic() + args.track
            was_lost = False
            while time.monotonic() < t_end:
                now = time.monotonic()
                e = cam.estimate()
                age = now - e.t
                ages.append(age)
                if age > 1.0 and not was_lost:
                    lost += 1
                was_lost = age > 1.0
                d = np.arctan2(np.sin(e.predict(now)[0] - model.theta_at(now)),
                               np.cos(e.predict(now)[0] - model.theta_at(now)))
                drift.append((now, float(d)))
                time.sleep(0.5)
        d = np.rad2deg([x for _, x in drift])
        log(f"  tracker - fit: {d[0]:+.2f} deg at the start, {d[-1]:+.2f} deg after "
            f"{args.track:.0f} s (the fit's speed error, about {abs(d[-1] - d[0]) / args.track:.3f}"
            f" deg/s); markers lost {lost} time(s); the latest measurement at most "
            f"{max(ages):.2f} s old")
        return {"detections": len(found), "axis_mm": (1000 * model.axis_xy).tolist(),
                "offset_mm": 1000 * fit.radius_m, "period_s": fit.period_s,
                "direction": model.direction, "xy_rmse_mm": 1000 * fit.xy_rmse_m,
                "yaw_rmse_deg": float(np.rad2deg(fit.yaw_rmse_rad)),
                "tracker_minus_fit_deg": [float(x) for x in d], "lost": lost}
    finally:
        if sim is not None:
            sim.stop()


# --- fov ---

def cmd_fov(args, log: Log) -> dict:
    from aruco_block_detector import ArucoBlockDetector
    from homography import Homography
    from turntable_orbit import MARKER_HEIGHT_ABOVE_TABLE_M, measure_base
    robot, sim = connect(args, log)
    try:
        calibration = None
        if sim is not None:
            from cell import WORK_AREA_XY
            from placement import TaskScene
            sim.set_placement(TaskScene().random_placement(np.random.default_rng(0), WORK_AREA_XY))
            calibration = __import__("run_solver").simulated_calibration()
        elif args.calibration is not None:
            calibration = Homography.load(args.calibration)
        detector = ArucoBlockDetector()
        log("every second: the markers seen (move the base to the work area's corners and "
            "edges; Ctrl+C to stop)")
        rows = []
        n = args.images if args.sim else 10 ** 9
        try:
            for _ in range(n):
                t = time.monotonic()
                image = robot.grab_image()
                if image is None:
                    log(f"  {datetime.now():%H:%M:%S}: the camera returned no image")
                    time.sleep(1.0)
                    continue
                pose = detector.detect(image)
                text = "no markers"
                row = {"t": t, "markers": 0}
                if pose is not None:
                    row["markers"] = len(pose.visible_ids)
                    text = f"markers {sorted(pose.visible_ids)}"
                    if calibration is not None:
                        m = measure_base(image, t - robot.camera_delay, detector, calibration,
                                         calibration.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M)
                        if m is not None:
                            row.update(x_mm=1000 * m.xy[0], y_mm=1000 * m.xy[1])
                            text += f", base at ({1000 * m.xy[0]:.0f}, {1000 * m.xy[1]:.0f}) mm"
                log(f"  {datetime.now():%H:%M:%S}: {text}")
                rows.append(row)
                time.sleep(max(0.0, 1.0 - (time.monotonic() - t)))
        except KeyboardInterrupt:
            pass
        return {"samples": rows}
    finally:
        if sim is not None:
            sim.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=("dims", "calibrate", "timing", "turntable", "fov"))
    ap.add_argument("--sim", action="store_true", help="against the standalone simulator")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--calibration", type=Path, help="homography JSON (turntable, fov)")
    ap.add_argument("--images", type=int, default=10, help="turntable: images for the fit; "
                    "fov --sim: how many")
    ap.add_argument("--track", type=float, default=30.0, help="turntable: tracker seconds")
    ap.add_argument("--ring-fallback", action="store_true",
                    help="calibrate: also use the (less accurate) blue-ring fit where the hole "
                         "detector finds nothing")
    args = ap.parse_args()
    folder = OUT / f"{datetime.now():%Y%m%d_%H%M%S}_{args.check}{'_sim' if args.sim else ''}"
    log = Log(folder)
    where = "no robot" if args.check == "dims" else "simulator" if args.sim else "REAL ROBOT"
    log(f"lab check '{args.check}' ({where}); results in {folder}")
    result = {"dims": cmd_dims, "calibrate": cmd_calibrate, "timing": cmd_timing,
              "turntable": cmd_turntable, "fov": cmd_fov}[args.check](args, log)
    (folder / "summary.json").write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
