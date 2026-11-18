"""Interaktivní konzole pro simulovaný CRS93."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CAPTURES = ROOT / "captures"

for directory in (REPO / "Prokop", REPO / "Natka", ROOT.parent / "ArucoFinder",
                  ROOT.parent / "Homografy"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from cell import TABLE_Z  # noqa: E402
from homography import Homography  # noqa: E402
from placement import HOOP_MOUNT, MODELS, load_urdf  # noqa: E402
from sim_find_and_solve import marker_base_pose, simulated_homography  # noqa: E402
from simulation import CRS93Sim  # noqa: E402


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
  help                               zobrazí příkazy
  quit                               ukončí simulaci
"""


@dataclass
class CameraCalibration:
    model: Homography
    path: Path | None = None


def capture_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def save_png(path: Path, image: np.ndarray) -> None:
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"Snímek se nepodařilo uložit do {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())


def capture(robot: CRS93Sim) -> tuple[np.ndarray, Path]:
    image = robot.grab_image()
    path = CAPTURES / f"{capture_stamp()}.png"
    save_png(path, image)
    print(f"Snímek {image.shape[1]} × {image.shape[0]} px uložen do {path}")
    return image, path


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
        z_levels_m=(0.225, 0.30, 0.39),
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
    image, raw_path = capture(robot)
    pose = detector.detect(image)
    annotated_path = raw_path.with_name(f"{raw_path.stem}_aruco.png")
    save_png(annotated_path, detector.draw(image, pose))
    print(f"Vyznačený snímek uložen do {annotated_path}")
    if pose is None:
        print("Na snímku nebyly nalezeny značky základny (ID 1 a 2).")
        return

    xy, yaw = marker_base_pose(robot, detector, homography, image)
    marker_ids = ", ".join(str(marker_id) for marker_id in pose.visible_ids)
    print(f"ArUco ID {marker_ids} | střed v obraze {pose.center_px} px | "
          f"úhel v obraze {pose.angle_deg:+.2f}°")
    print(f"Základna v souřadnicích robota XY ({xy[0]:.4f}, {xy[1]:.4f}) m | "
          f"natočení {np.rad2deg(yaw):+.2f}°")


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
