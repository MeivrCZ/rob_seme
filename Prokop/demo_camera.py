"""Showcase of the simulated camera: the robot holds the hoop in the camera view
in a few poses, photographs it, then photographs the spinning turntable.

Run from the repo root:
    uv run Prokop/demo_camera.py                  # images -> Prokop/demo_output/
    uv run Prokop/demo_camera.py --keep           # keep the viewer open afterwards

Open http://localhost:8080 to watch; the viewer's "Camera" panel shows every
image as it is taken. Saved: one PNG per photo (1920 x 1200, as grab_image
returns it) and contact_sheet.jpg with all of them.

The robot is used only through the CRS93 interface (move_to_q, grab_image, ...);
placing and spinning the turntable (robot.viewer) exists in the simulation only.
Every pose and move is checked against the cell and the table first
(collision.ArmChecker), and the robot is never homed.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROKOP = Path(__file__).resolve().parent
sys.path.insert(0, str(PROKOP / "tools"))

from camera_survey import survey_pose  # noqa: E402
from cell import CAMERA_RESOLUTION  # noqa: E402
from collision import ArmChecker  # noqa: E402
from placement import Placement  # noqa: E402
from simulation import CRS93Sim  # noqa: E402
from view_models import SPIN_PERIOD  # noqa: E402

D = np.deg2rad
SPEED = 0.3  # relative joint speed of the moves
MIN_CLEARANCE = 0.08  # m, poses and moves to the cell and the table
IMAGE_MARGIN = 0.1  # fraction of the image kept free around the hoop centre
# Turntable in the camera view, base slightly off-centre and turned.
PLACEMENT = Placement(np.array([0.50, 0.06]), np.array([0.01, -0.02]), 0.4)
# Joint offsets [deg] from the survey pose (flange down, hoop in view), in order.
POSES = {
    "survey_pose": [0, 0, 0, 0, 0, 0],
    "joint1_+7.5deg": [7.5, 0, 0, 0, 0, 0],
    "joint1_+15deg": [15, 0, 0, 0, 0, 0],
    "wrist_turned_+30deg": [7.5, 0, 0, 0, 0, 30],
    "wrist_turned_-30deg": [7.5, 0, 0, 0, 0, -30],
    "lower_10deg_joint3": [7.5, 0, 10, 0, -10, 0],
}
SPIN_SHOTS = 3  # photos of the spinning turntable
SPIN_GAP = 3.0  # s between them (~39 deg of turn at 28 s/rev)


def checked_move(robot, checker: ArmChecker, q: np.ndarray) -> bool:
    """Moves to @q if the pose and the straight joint-space move are clear.

    Args:
        robot: The robot.
        checker: Collision checker.
        q: Target configuration [rad].

    Returns:
        Whether the robot moved.
    """
    if not robot.in_limits(q):
        print("  skipped: joint limits")
        return False
    d, what = checker.path_clearance(robot.get_q(), q)
    if d < MIN_CLEARANCE:
        print(f"  skipped: only {100 * d:.1f} cm from {what}")
        return False
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    return True


def hoop_in_view(robot, checker: ArmChecker, q: np.ndarray) -> bool:
    """Whether the hoop centre projects into the image (with IMAGE_MARGIN)."""
    uv = robot.camera.project(checker.hoop_center(q))
    size = np.array(CAMERA_RESOLUTION, dtype=float)
    return bool(np.all(uv > IMAGE_MARGIN * size)
                and np.all(uv < (1 - IMAGE_MARGIN) * size))


def contact_sheet(images: list[tuple[str, np.ndarray]], columns: int = 3) -> np.ndarray:
    """Thumbnails of all photos with their names, in a grid.

    Args:
        images: (name, BGR image) pairs.
        columns: Thumbnails per row.

    Returns:
        The sheet, BGR.
    """
    w, h = 480, 300
    rows = -(-len(images) // columns)
    sheet = np.full((rows * (h + 30), columns * w, 3), 255, np.uint8)
    for i, (name, image) in enumerate(images):
        r, c = divmod(i, columns)
        y, x = r * (h + 30), c * w
        sheet[y:y + h, x:x + w] = cv2.resize(image, (w, h))
        cv2.putText(sheet, f"{i + 1}. {name}", (x + 8, y + h + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, default=PROKOP / "demo_output",
                        help="output folder (a timestamped subfolder is made)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--keep", action="store_true",
                        help="keep the viewer open at the end (Ctrl+C to quit)")
    args = parser.parse_args()
    out = args.out / datetime.now().strftime("camera_%Y%m%d-%H%M%S")
    out.mkdir(parents=True)

    robot = CRS93Sim(port=args.port, keep_alive=args.keep)
    robot.viewer.set_placement(PLACEMENT)
    robot.initialize(home=False)  # standing in q_home
    robot.set_speed_relative(SPEED)
    checker = ArmChecker()
    q_survey, info = survey_pose(robot, checker)
    print(f"survey pose {info['q_deg']} deg, hoop {info['hoop_center_m']} m")

    photos = []

    def photo(name: str) -> None:
        image = robot.grab_image()
        path = out / f"{len(photos) + 1:02d}_{name}.png"
        cv2.imwrite(str(path), image)
        photos.append((name, image))
        print(f"  saved {path.name}")

    for name, offset in POSES.items():
        q = q_survey + D(offset)
        print(f"{name}:")
        if not hoop_in_view(robot, checker, q):
            print("  skipped: the hoop would leave the camera view")
            continue
        if checked_move(robot, checker, q):
            photo(name)

    print(f"spinning turntable ({SPIN_PERIOD:.0f} s/rev):")
    robot.viewer.set_spin(2 * np.pi / SPIN_PERIOD)
    for k in range(SPIN_SHOTS):
        photo(f"spinning_{k + 1}")
        robot._sleep(SPIN_GAP)  # sim time (time_scale), like time.sleep on the robot
    robot.viewer.set_spin(0.0)

    print("back to q_home:")
    if not checked_move(robot, checker, robot.q_home):
        print("  stays here (the move back would not be clear)")
    cv2.imwrite(str(out / "contact_sheet.jpg"), contact_sheet(photos))
    print(f"{len(photos)} photos and contact_sheet.jpg in {out}")
    robot.close()


if __name__ == "__main__":
    main()
