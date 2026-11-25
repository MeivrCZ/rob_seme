"""Interaktivní konzole pro simulovaný CRS93."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CAPTURES = ROOT / "captures"
MAX_HOOP_HEIGHT_M = 0.04

for directory in (REPO, REPO / "Prokop", ROOT.parent / "ArucoFinder", ROOT.parent / "Homografy"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from advanced_aruco import AdvancedArucoFinder, TimedFrame  # noqa: E402
from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from camera_phase_tracker import CameraPhaseTracker  # noqa: E402
from cell import TABLE_Z, WORK_AREA_XY  # noqa: E402
from homography import Homography  # noqa: E402
from Natka.crs_puzzle_d_hoop_path import ARC_SEGMENTS, puzzle_path_contract  # noqa: E402
from Natka.hoop_targets import hoop_ring_pose_from_flange  # noqa: E402
from placement import HOOP_MOUNT, MODELS, Placement, load_urdf  # noqa: E402
from simulation import CRS93Sim  # noqa: E402
from solver.contracts import BaseMeasurement, TurntableModel  # noqa: E402
from solver.executor import Executor  # noqa: E402
from solver.puzzle_plan import PuzzleCollisionChecker, plan_puzzle, plan_withdrawal  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from turntable_orbit import (  # noqa: E402
    MARKER_HEIGHT_ABOVE_TABLE_M, OrbitFit, capture_series, draw_orbit, fit_orbit, measure_base,
)


HELP = """Příkazy
  robot status                       poloha kloubů a příruby
  robot jog q0..q5 DELTA_DEG          posun jednoho kloubu ve stupních
  robot goto deg|rad Q0 Q1 Q2 Q3 Q4 Q5   absolutní poloha kloubů
  robot wait                         počká na dokončení pohybu
  robot soft-home                    vrátí robot do q_home
  camera snap                        pořídí a uloží snímek PNG
  aruco detect                       změří polohu držáku nebo ji odhadne při zakrytí značek
  homo run                           projede polohy obruče a spočítá matici 3×4
  homo status                        zobrazí aktivní matici a zdroj kalibrace
  homo orbit [N [INTERVAL_S]]         odhadne dráhu držáku (výchozí 10 snímků po 1 s)
  homo predict [SEKUND]               předpoví polohu držáku, výchozí je teď
  puzzle place X Y YAW_DEG          umístí vystředěnou základnu (XY v metrech)
  puzzle static [RYCHLOST | demo]    naplánuje a projede D na stojící točně
  puzzle spin [RYCHLOST | demo]      změří rotaci, naplánuje a projede D za pohybu
  help                               zobrazí příkazy
  quit                               ukončí simulaci
"""


@dataclass
class CameraCalibration:
    aruco: AdvancedArucoFinder
    path: Path | None = None


def capture_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def save_png(path: Path, image: np.ndarray) -> None:
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"Snímek se nepodařilo uložit do {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())


def capture(robot: CRS93Sim) -> tuple[np.ndarray, Path, float]:
    image = robot.grab_image()
    t_capture = time.monotonic() - robot.camera_delay
    path = CAPTURES / f"{capture_stamp()}.png"
    save_png(path, image)
    print(f"Snímek {image.shape[1]} × {image.shape[0]} px uložen do {path}")
    return image, path, t_capture


def simulated_homography(robot: CRS93Sim) -> Homography:
    """Fit the initial calibration from the simulator camera's projection."""
    model = Homography(
        x_range_m=tuple(WORK_AREA_XY[0]),
        y_range_m=tuple(WORK_AREA_XY[1]),
        z_levels_m=(0.03, 0.10, 0.20),
        table_z_m=TABLE_Z,
        nx=4,
        ny=4,
    )
    for index, xyz in enumerate(model.points):
        model.record(index, tuple(robot.camera.project(xyz)), tuple(xyz))
    model.calculate(validation_fraction=0.0, ransac=False)
    return model


def calibrate_from_hoop(robot: CRS93Sim) -> tuple[Homography, Path]:
    robot_model = load_urdf(MODELS / "crs93.urdf")
    hoop_model = load_urdf(MODELS / "hoop.urdf")
    flange_to_center = (
        robot_model.get_transform("adapter_front", "flange")
        @ HOOP_MOUNT
        @ hoop_model.get_transform("hoop_center", hoop_model.base_link)
    )
    model = Homography(
        x_range_m=(0.415, 0.535),
        y_range_m=(-0.12, 0.12),
        z_levels_m=(0.075, 0.13, 0.19),
        table_z_m=TABLE_Z,
        nx=3, ny=3,
        min_circle_diameter_px=180,
        max_circle_diameter_px=450,
        hoop_center_in_tcp_m=tuple(flange_to_center[:3, 3]),
    )
    # Příruba míří dolů, obruč zůstává vodorovně a směřuje od příruby podél +Y.
    model.set_fixed_orientation(np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                                source="fixed simulator calibration orientation")

    robot.wait_for_motion_stop()
    start_q = robot.get_q()
    previous_q = start_q
    targets = []
    for index, pose in enumerate(model.poses):
        reachable = [q for q in robot.ik(pose) if robot.in_limits(q)]
        if not reachable:
            raise RuntimeError(f"Kalibrační bod {index:02d} je mimo dosah robota.")
        previous_q = min(reachable, key=lambda q: np.linalg.norm(q - previous_q))
        targets.append(previous_q)

    directory = CAPTURES / f"homography_{capture_stamp()}"
    print(f"Kalibrační průjezd má {len(targets)} poloh. Snímky se ukládají do {directory}")
    for index, target in enumerate(targets):
        robot.move_to_q(target)
        robot.wait_for_motion_stop()
        xyz = (robot.fk(robot.get_q()) @ flange_to_center)[:3, 3]
        frame = robot.grab_image()
        save_png(directory / f"point_{index:02d}.png", frame)
        try:
            uv = model.capture_circle(frame)
        except RuntimeError as error:
            print(f"  Bod {index:02d} přeskočen. {error}")
            continue
        model.record(index, uv, tuple(xyz))
        print(f"  Bod {index:02d} | XYZ {np.round(xyz, 4)} m | UV {np.round(uv, 2)} px")

    robot.move_to_q(start_q)
    robot.wait_for_motion_stop()
    model.calculate()
    validation = model.validate()
    path = directory / "homography.json"
    model.save(path)
    print(f"Kalibrace uložena do {path}")
    print(f"Naměřeno {len(model.samples)}/{len(targets)} bodů | "
          f"{model.fit_info['inlier_count']} bodů použitých při výpočtu | "
          f"RMSE {validation['pixel_rmse']:.3f} px / {validation['xy_rmse_mm']:.3f} mm "
          f"({validation['count']} bodů, {validation['subset']})")
    print("Projekční matice P [3×4]\n", model.P, sep="")
    return model, path


def detect_markers(robot: CRS93Sim, finder: AdvancedArucoFinder) -> None:
    image, raw_path, t_capture = capture(robot)
    result = finder.advAruco(TimedFrame(image, t_capture))
    pose = result.pose if result is not None else None
    annotated_path = raw_path.with_name(f"{raw_path.stem}_aruco.png")
    save_png(annotated_path, finder.detector.draw(image, pose))
    print(f"Vyznačený snímek uložen do {annotated_path}")
    if result is None:
        print("Na snímku nebyly nalezeny značky základny (ID 1 a 2).")
        return

    if pose is None:
        elapsed = result.t_capture - result.t_last_valid
        print(f"Odhad z modelu točny | poslední detekce před {elapsed:.2f} s")
    else:
        marker_ids = ", ".join(str(marker_id) for marker_id in pose.visible_ids)
        print(f"ArUco ID {marker_ids} | střed v obraze {pose.center_px} px | "
              f"úhel v obraze {pose.angle_deg:+.2f}°")
    x, y = result.xy
    print(f"Základna v souřadnicích robota XY ({x:.4f}, {y:.4f}) m | "
          f"natočení {np.rad2deg(result.yaw):+.2f}°")


def measure_turntable(robot: CRS93Sim, detector: ArucoBlockDetector, homography: Homography,
                      count: int, interval_s: float) -> tuple[OrbitFit, BaseMeasurement]:
    directory = CAPTURES / f"orbit_{capture_stamp()}"
    marker_z = homography.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
    measurements = []
    frames = []
    reference_image = None
    print(f"Měřím dráhu držáku. Snímky se ukládají do {directory}")
    for index, (t_capture, image) in enumerate(
        capture_series(robot.grab_image, count, interval_s, robot.camera_delay), start=1
    ):
        name = f"frame_{index:02d}.png"
        save_png(directory / name, image)
        sample = measure_base(image, t_capture, detector, homography, marker_z)
        record = {"image": name, "t_capture": t_capture, "n_markers": 0}
        if sample is None:
            print(f"  {index:02d}/{count} | značky nenalezeny")
        else:
            measurements.append(sample)
            reference_image = image
            record.update(xy=sample.xy.tolist(), yaw=sample.yaw, n_markers=sample.n_markers)
            print(f"  {index:02d}/{count} | {sample.n_markers} značky | XY {sample.xy.round(4)} m")
        frames.append(record)

    fit = fit_orbit(measurements)
    model = fit.model
    save_png(directory / "orbit.png", draw_orbit(reference_image, fit, measurements, homography, marker_z))
    data = {
        "model": {
            "axis_xy": model.axis_xy.tolist(), "base_offset": model.base_offset.tolist(),
            "base_yaw": model.base_yaw, "direction": model.direction, "omega": model.omega,
            "theta_ref": model.theta_ref, "t_ref": model.t_ref,
        },
        "radius_m": fit.radius_m, "signed_omega_rad_s": fit.signed_omega, "period_s": fit.period_s,
        "xy_rmse_m": fit.xy_rmse_m, "yaw_rmse_rad": fit.yaw_rmse_rad, "omega_sd_rad_s": fit.omega_sd_rad_s,
        "marker_z_m": marker_z, "camera_delay_s": robot.camera_delay, "frames": frames,
    }
    path = directory / "orbit.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Osa XY {model.axis_xy.round(4)} m | poloměr {1000 * fit.radius_m:.1f} mm | "
          f"rychlost {np.rad2deg(fit.signed_omega):+.2f}°/s | perioda {fit.period_s:.2f} s")
    print(f"RMSE polohy {1000 * fit.xy_rmse_m:.2f} mm | model {path}")
    return fit, measurements[-1]


def run_puzzle(robot: CRS93Sim, calibration: CameraCalibration, *, spinning: bool,
               speed: float, demo: bool = False) -> None:
    if not 0.0 < speed <= 1.0:
        raise ValueError("Relativní rychlost robota musí být v rozsahu (0, 1].")
    viewer = robot.viewer
    if viewer.turntable_spinning != spinning:
        action = "zapni" if spinning else "vypni"
        raise RuntimeError(f"Před průjezdem {action} Spin v panelu Turntable.")
    if demo:
        robot.wait_for_motion_stop()
        if not np.allclose(robot.get_q(), robot.q_home, atol=1e-4, rtol=0):
            raise RuntimeError("Pro ukázkovou polohu začni v q_home.")
        viewer.set_placement(Placement(np.array([0.55, 0.0]), np.zeros(2), np.pi / 2))

    scene_changed = Event()
    mark_changed = scene_changed.set
    viewer.on_change.append(mark_changed)
    try:
        robot.wait_for_motion_stop()
        start_q = robot.get_q()
        finder = calibration.aruco
        measurement_started = time.monotonic()
        if spinning:
            fit, sample = measure_turntable(robot, finder.detector, finder.homography, 10, 1.0)
            model = fit.model
        else:
            image = robot.grab_image()
            timestamp = time.monotonic() - robot.camera_delay
            marker_z = finder.homography.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
            sample = measure_base(image, timestamp, finder.detector, finder.homography, marker_z)
            if sample is None:
                raise RuntimeError("Kamera nevidí značky základny (ID 1 a 2).")
            # The simulated fixture supplies the fixed axis. The base XY and yaw
            # both come from the image, including a base mounted off centre.
            axis = viewer.placement.turntable_xy.copy()
            model = TurntableModel(axis, sample.xy - axis, sample.yaw, 1, omega=0.0,
                                   t_ref=sample.t_capture)

        if scene_changed.is_set():
            raise RuntimeError("Scéna se během měření změnila. Spusť průjezd znovu.")
        active_finder = AdvancedArucoFinder(
            finder.detector, finder.homography, model, last_measurement=sample,
        )
        calibration.aruco = active_finder if spinning else AdvancedArucoFinder(finder.detector, finder.homography)
        measurement_s = time.monotonic() - measurement_started

        planning_started = time.monotonic()
        path = puzzle_path_contract()
        marker_z = finder.homography.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
        depth_step = path.final_index
        if spinning:
            # Finish on the final straight stem once the requested height is
            # reached. The extra points farther down the stem are useful in
            # the stationary demonstration.
            world_frames = model.puzzle_T(model.theta_ref) @ path.frames[:path.final_index + 1]
            heights = world_frames[:, 2, 3] - marker_z
            stem = np.all(np.isclose(path.positions[:, :2], path.positions[path.final_index, :2]), axis=1)
            goals = np.flatnonzero(stem[:len(heights)] & (heights > 0.0) & (heights < MAX_HOOP_HEIGHT_M))
            depth = int(goals[0]) if goals.size else path.final_index
            # Retain the entry, both tangencies and the lower bend. Fewer
            # stops on the smooth arc leave more time to clear the turning
            # puzzle. The full spaces between targets are collision-checked.
            inward = [i for i in range(depth + 1) if not 2 < i < 2 + ARC_SEGMENTS or i % 2 == 0]
            depth_step = len(inward) - 1
            indices = inward + inward[-2::-1]
            path = replace(path, frames=path.frames[indices], clearance=path.clearance[indices],
                           roll_limits=path.roll_limits[indices] if path.roll_limits is not None else None,
                           final_index=len(indices) - 1)
        # This base yaw worked for the centred demo at 28 s/rev. Recompute
        # and recheck the entire route from the current camera measurements.
        preferred_phase = None
        if demo and spinning and model.direction == 1 and np.isclose(fit.period_s, 28.0, atol=0.3):
            preferred_phase = np.deg2rad(202.5) - model.base_yaw
        plan = plan_puzzle(robot, model, path, speed, preferred_phase=preferred_phase,
                           return_to_waiting=spinning)
        planning_s = time.monotonic() - planning_started
        print(f"Měření {measurement_s:.1f} s | plánování {planning_s:.1f} s.")
        if not plan.feasible:
            raise RuntimeError(f"Průjezd D není dosažitelný ({plan.reason}).")
        if scene_changed.is_set() or not np.allclose(robot.get_q(), start_q, atol=1e-4, rtol=0):
            raise RuntimeError("Poloha robota nebo scéna se změnila. Spusť průjezd znovu.")
        print(f"Plán má {len(plan.steps)} bodů | startovní fáze {np.rad2deg(plan.theta_start):.1f}° "
              f"| nejmenší odstup {1000 * plan.min_margin:.2f} mm.")
        waiting_xyz = hoop_ring_pose_from_flange(robot.fk(plan.q_start))[:3, 3]
        print(f"Čekací poloha středu obruče XYZ {waiting_xyz.round(3)} m.")

        timing = RobotTiming(robot, plan.speed)
        checker = PuzzleCollisionChecker(model, command_delay=timing.command_delay)

        tracker_context = (CameraPhaseTracker(active_finder, robot.grab_image, robot.camera_delay,
                                              speed_sd=fit.omega_sd_rad_s)
                           if spinning else nullcontext(None))
        execution_started = time.monotonic()
        result = None
        with tracker_context as tracker:
            if tracker is not None and not tracker.wait_ready(timeout=4.0):
                raise RuntimeError("Kamera nepotvrdila fázi točny. Robot zůstává v klidu.")
            if scene_changed.is_set():
                raise RuntimeError("Scéna se před nájezdem změnila. Spusť průjezd znovu.")
            try:
                result = Executor(robot, model, plan, tracker, timing, validate_move=checker.validate_move,
                                  scene_changed=scene_changed.is_set).run()
            finally:
                if spinning and (result is None or not result.success):
                    # An interrupted move can leave the hoop threaded. On
                    # success the plan leaves it in the cleared waiting pose.
                    viewer.stop_spin()
                    calibration.aruco = AdvancedArucoFinder(finder.detector, finder.homography)

        height = None
        withdrawal = None
        if len(result.steps) > depth_step and result.steps[depth_step].t_stopped is not None:
            reached = result.steps[depth_step].q if spinning else robot.get_q()
            z = hoop_ring_pose_from_flange(robot.fk(reached))[2, 3]
            height = float(z - marker_z)
        if result.success:
            if height is None or not 0.0 < height < MAX_HOOP_HEIGHT_M:
                result.success = False
                result.reason = "obruč nedosáhla požadované výšky pod 40 mm nad základnou"
            elif not spinning:
                print(f"Obruč dosáhla {1000 * height:.1f} mm nad základnou. Plánuji výjezd.")
                if not scene_changed.is_set():
                    exit_plan = plan_withdrawal(robot, model, path, speed, [step.q for step in result.steps])
                    withdrawal = Executor(robot, model, exit_plan, None, timing,
                                          validate_move=checker.validate_move,
                                          scene_changed=scene_changed.is_set).run()
                    if not withdrawal.success:
                        result.success = False
                        result.reason = f"výjezd nedokončen ({withdrawal.reason})"
                else:
                    result.success = False
                    result.reason = "scéna se před výjezdem změnila"

        log_path = CAPTURES / f"puzzle_{capture_stamp()}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            **asdict(result), "hoop_height_m": height,
            "depth_step": depth_step,
            "measurement_s": measurement_s, "planning_s": planning_s,
            "before_first_step_s": (result.t_start - execution_started if result.t_start is not None else None),
            "withdrawal": asdict(withdrawal) if withdrawal is not None else None,
        }
        log_path.write_text(json.dumps(report, default=lambda array: array.tolist(), indent=2), encoding="utf-8")
        print(f"Záznam průjezdu uložen do {log_path}")
        if not result.success:
            raise RuntimeError(f"Průjezd D přerušen ({result.reason}).")
        print(f"Průjezd D dokončen včetně výjezdu. U základny byl střed obruče ve výšce {1000 * height:.1f} mm.")
    finally:
        viewer.on_change.remove(mark_changed)


def dispatch(robot: CRS93Sim, calibration: CameraCalibration, words: list[str]) -> bool:
    finder = calibration.aruco
    match words:
        case []:
            pass
        case ["help"]:
            print(HELP, end="")
        case ["quit" | "exit"]:
            return False
        case ["robot", "status"]:
            q = robot.get_q()
            print("Klouby [deg]", np.round(np.rad2deg(q), 3))
            print("Příruba XYZ [m]", np.round(robot.fk(q)[:3, 3], 4))
            print("Stav", "v pohybu" if robot.in_motion() else "v klidu")
        case ["robot", "jog", joint, degrees]:
            if joint not in ("q0", "q1", "q2", "q3", "q4", "q5"):
                raise ValueError("Kloub musí být q0 až q5.")
            delta = float(degrees)
            if not np.isfinite(delta):
                raise ValueError("Posun musí být konečné číslo.")
            # Po předchozích povelech odečti skutečnou dosaženou polohu.
            robot.wait_for_motion_stop()
            q = robot.get_q().copy()
            q[int(joint[1])] += np.deg2rad(delta)
            robot.move_to_q(q)
            print(f"Zadán posun {joint} o {delta:+g}°.")
        case ["robot", "goto", ("deg" | "rad") as unit, *values] if len(values) == 6:
            q = np.array(values, dtype=float)
            if not np.isfinite(q).all():
                raise ValueError("Polohy kloubů musí být konečná čísla.")
            robot.move_to_q(np.deg2rad(q) if unit == "deg" else q)
            print("Zadán absolutní pohyb kloubů.")
        case ["robot", "wait"]:
            robot.wait_for_motion_stop()
            print("Pohyb dokončen.")
        case ["robot", "soft-home"]:
            robot.soft_home()
            print("Robot je v q_home.")
        case ["camera", "snap"]:
            capture(robot)
        case ["aruco", "detect"]:
            detect_markers(robot, finder)
        case ["homo", "run"]:
            viewer = robot.viewer
            if viewer.turntable_spinning:
                raise RuntimeError("Před kalibrací vypni Spin v panelu Turntable.")
            robot.wait_for_motion_stop()
            placement = viewer.placement
            # Park the fixture behind the robot while the hoop calibrates the
            # front work area. Restore the selected placement in this session.
            viewer.set_placement(Placement(np.array([-0.55, 0.0]), np.zeros(2), 0.0))
            try:
                homography, path = calibrate_from_hoop(robot)
            finally:
                viewer.set_placement(placement)
            calibration.aruco = AdvancedArucoFinder(finder.detector, homography)
            calibration.path = path
        case ["homo", "status"]:
            print("Zdroj kalibrace", calibration.path or "syntetická kalibrace simulátoru")
            print("Projekční matice P [3×4]\n", finder.homography.P, sep="")
        case ["homo", "orbit", *options] if len(options) <= 2:
            count = int(options[0]) if options else 10
            interval = float(options[1]) if len(options) == 2 else 1.0
            fit, last_sample = measure_turntable(robot, finder.detector, finder.homography, count, interval)
            calibration.aruco = AdvancedArucoFinder(
                finder.detector, finder.homography, fit.model, last_measurement=last_sample,
            )
        case ["homo", "predict", *options] if len(options) <= 1:
            if finder.turntable is None:
                raise RuntimeError("Nejdřív změř dráhu držáku příkazem homo orbit.")
            seconds = float(options[0]) if options else 0.0
            if not np.isfinite(seconds):
                raise ValueError("Čas předpovědi musí být konečný.")
            xy, yaw = finder.turntable.measurement_at(time.monotonic() + seconds)
            print(f"Předpověď pro t {seconds:+g} s | střed XY {xy.round(4)} m | natočení {np.rad2deg(yaw):+.2f}°")
        case ["puzzle", "place", x, y, yaw]:
            xy = np.array([float(x), float(y)])
            angle = float(yaw)
            if not np.isfinite([*xy, angle]).all() or not np.all(
                (WORK_AREA_XY[:, 0] <= xy) & (xy <= WORK_AREA_XY[:, 1])
            ):
                raise ValueError("XY musí ležet v pracovním poli kamery a úhel musí být konečný.")
            viewer = robot.viewer
            if viewer.turntable_spinning:
                raise RuntimeError("Před přemístěním vypni Spin v panelu Turntable.")
            robot.wait_for_motion_stop()
            base_yaw = np.deg2rad(angle) - viewer.turntable_angle(viewer.clock())
            viewer.set_placement(Placement(xy, np.zeros(2), base_yaw))
            calibration.aruco = AdvancedArucoFinder(finder.detector, finder.homography)
            print(f"Základna XY {xy.round(4)} m | natočení {angle:g}°.")
        case ["puzzle", ("static" | "spin") as mode, *options] if len(options) <= 1:
            spinning = mode == "spin"
            demo = options == ["demo"]
            speed = float(options[0]) if options and not demo else (0.8 if spinning else 0.4)
            run_puzzle(robot, calibration, spinning=spinning, speed=speed, demo=demo)
        case _:
            print("Neznámý nebo neúplný příkaz. Zadej help.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="port Viseru (výchozí 8080)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port musí být v rozsahu 1 až 65535")

    # Viser, animace a čas simulace běží ve vláknech Prokopova simulátoru.
    robot = CRS93Sim(port=args.port, collisions=True, keep_alive=False)
    try:
        robot.initialize(home=False)
        finder = AdvancedArucoFinder(ArucoBlockDetector(), simulated_homography(robot))
        calibration = CameraCalibration(finder)
        print(f"Mainframe připraven. Viser běží na http://localhost:{robot.viewer.server.get_port()}")
        print("Seznam příkazů zobrazíš příkazem help.")
        while True:
            words = input("crs93-sim> ").lower().split()
            try:
                if not dispatch(robot, calibration, words):
                    break
            except (ValueError, RuntimeError, OSError, AssertionError, cv2.error,
                    np.linalg.LinAlgError) as error:
                print(f"Chyba: {error}")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
