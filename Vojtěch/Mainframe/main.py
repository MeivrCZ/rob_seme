"""Interaktivní konzole pro simulovaný CRS93."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CAPTURES = ROOT / "captures"

for directory in (REPO / "Prokop", ROOT.parent / "ArucoFinder", ROOT.parent / "Homografy"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from cell import TABLE_Z, WORK_AREA_XY  # noqa: E402
from homography import Homography  # noqa: E402
from placement import HOOP_MOUNT, MODELS, load_urdf  # noqa: E402
from simulation import CRS93Sim  # noqa: E402
from solver.contracts import TurntableModel  # noqa: E402
from turntable_orbit import (  # noqa: E402
    MARKER_HEIGHT_ABOVE_TABLE_M, capture_series, draw_orbit, fit_orbit, measure_base, measure_base_pose,
)


HELP = """Příkazy
  robot status                       poloha kloubů a příruby
  robot jog q0..q5 DELTA_DEG          posun jednoho kloubu ve stupních
  robot goto deg|rad Q0 Q1 Q2 Q3 Q4 Q5   absolutní poloha kloubů
  robot wait                         počká na dokončení pohybu
  robot soft-home                    vrátí robot do q_home
  camera snap                        pořídí a uloží snímek PNG
  aruco detect                       vyfotí scénu, najde značky a uloží výsledek
  homo run                           projede polohy obruče a spočítá matici 3×4
  homo status                        zobrazí aktivní matici a zdroj kalibrace
  homo orbit [N [INTERVAL_S]]         odhadne dráhu držáku (výchozí 10 snímků po 1 s)
  homo predict [SEKUND]               předpoví polohu držáku, výchozí je teď
  help                               zobrazí příkazy
  quit                               ukončí simulaci
"""


@dataclass
class CameraCalibration:
    model: Homography
    path: Path | None = None
    turntable: TurntableModel | None = None


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


def detect_markers(robot: CRS93Sim, detector: ArucoBlockDetector,
                   homography: Homography) -> None:
    image, raw_path, t_capture = capture(robot)
    pose = detector.detect(image)
    annotated_path = raw_path.with_name(f"{raw_path.stem}_aruco.png")
    save_png(annotated_path, detector.draw(image, pose))
    print(f"Vyznačený snímek uložen do {annotated_path}")
    if pose is None:
        print("Na snímku nebyly nalezeny značky základny (ID 1 a 2).")
        return

    marker_z = homography.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
    sample = measure_base_pose(pose, t_capture, homography, marker_z)
    marker_ids = ", ".join(str(marker_id) for marker_id in pose.visible_ids)
    print(f"ArUco ID {marker_ids} | střed v obraze {pose.center_px} px | "
          f"úhel v obraze {pose.angle_deg:+.2f}°")
    x, y = sample.xy
    print(f"Základna v souřadnicích robota XY ({x:.4f}, {y:.4f}) m | "
          f"natočení {np.rad2deg(sample.yaw):+.2f}°")


def measure_turntable(robot: CRS93Sim, detector: ArucoBlockDetector, homography: Homography,
                      count: int, interval_s: float) -> TurntableModel:
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
        "xy_rmse_m": fit.xy_rmse_m, "yaw_rmse_rad": fit.yaw_rmse_rad,
        "marker_z_m": marker_z, "camera_delay_s": robot.camera_delay, "frames": frames,
    }
    path = directory / "orbit.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Osa XY {model.axis_xy.round(4)} m | poloměr {1000 * fit.radius_m:.1f} mm | "
          f"rychlost {np.rad2deg(fit.signed_omega):+.2f}°/s | perioda {fit.period_s:.2f} s")
    print(f"RMSE polohy {1000 * fit.xy_rmse_m:.2f} mm | model {path}")
    return model


def dispatch(robot: CRS93Sim, detector: ArucoBlockDetector,
             calibration: CameraCalibration, words: list[str]) -> bool:
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
            detect_markers(robot, detector, calibration.model)
        case ["homo", "run"]:
            calibration.model, calibration.path = calibrate_from_hoop(robot)
        case ["homo", "status"]:
            print("Zdroj kalibrace", calibration.path or "syntetická kalibrace simulátoru")
            print("Projekční matice P [3×4]\n", calibration.model.P, sep="")
        case ["homo", "orbit", *options] if len(options) <= 2:
            count = int(options[0]) if options else 10
            interval = float(options[1]) if len(options) == 2 else 1.0
            calibration.turntable = measure_turntable(robot, detector, calibration.model, count, interval)
        case ["homo", "predict", *options] if len(options) <= 1:
            if calibration.turntable is None:
                raise RuntimeError("Nejdřív změř dráhu držáku příkazem homo orbit.")
            seconds = float(options[0]) if options else 0.0
            if not np.isfinite(seconds):
                raise ValueError("Čas předpovědi musí být konečný.")
            xy, yaw = calibration.turntable.measurement_at(time.monotonic() + seconds)
            print(f"Předpověď pro t {seconds:+g} s | střed XY {xy.round(4)} m | natočení {np.rad2deg(yaw):+.2f}°")
        case _:
            print("Neznámý nebo neúplný příkaz. Zadej help.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="port Viseru (výchozí 8080)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port musí být v rozsahu 1 až 65535")

    detector = ArucoBlockDetector()
    # Viser, animace a čas simulace běží ve vláknech Prokopova simulátoru.
    robot = CRS93Sim(port=args.port, collisions=False, keep_alive=False)
    try:
        robot.initialize(home=False)
        calibration = CameraCalibration(simulated_homography(robot))
        print(f"Mainframe připraven. Viser běží na http://localhost:{robot.viewer.server.get_port()}")
        print("Seznam příkazů zobrazíš příkazem help.")
        while True:
            words = input("crs93-sim> ").lower().split()
            try:
                if not dispatch(robot, detector, calibration, words):
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
