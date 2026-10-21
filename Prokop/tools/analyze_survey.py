"""Compares the hoop in a camera survey image with the camera model (experiment E8).

Finds the blue ring of the hoop in survey.png, compares its centre and size
with the projection of the hoop placed by the robot in the survey pose, and
converts the offset into a correction of the camera position.

Usage:
    From the repo root::

        uv run Prokop/tools/analyze_survey.py [experiments/camera_survey/<date-time>]

    Default: the newest survey with a survey.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROKOP))
sys.path.insert(0, str(PROKOP / "tools"))
SURVEYS = PROKOP / "experiments" / "camera_survey"
# Blue core of the hoop in HSV (OpenCV ranges) and its mean diameter [m]
# (inner 62.4 mm, outer 67.2 mm).
BLUE_LOW, BLUE_HIGH = (95, 80, 30), (130, 255, 200)
RING_DIAMETER = 0.0648
SEARCH_RADIUS = 400  # px around the predicted centre where the ring is searched
MIN_PIXELS = 2000  # a blue region at least this large can be the ring


def find_ring(image: np.ndarray, near: np.ndarray) -> tuple[np.ndarray, float]:
    """Fits an ellipse to the blue ring of the hoop.

    The ring is the connected blue region whose centre lies nearest to @near
    (within SEARCH_RADIUS); other blue regions (the stickers, the triangle on the
    handle) are separate or smaller.

    Args:
        image: BGR image.
        near: Pixel (u, v) near the ring centre (the model prediction).

    Returns:
        The ring centre (u, v) and its mean diameter [px].

    Raises:
        ValueError: If no blue region of at least MIN_PIXELS is near @near.
    """
    import cv2
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), BLUE_LOW, BLUE_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    best, best_dist = None, SEARCH_RADIUS
    for i in range(1, n):
        dist = np.hypot(*(centres[i] - near))
        if stats[i, cv2.CC_STAT_AREA] >= MIN_PIXELS and dist < best_dist:
            best, best_dist = i, dist
    if best is None:
        raise ValueError("ring not found")
    v, u = np.nonzero(labels == best)
    (cu, cv), (w, h), _ = cv2.fitEllipse(np.column_stack([u, v]).astype(np.float32))
    return np.array([cu, cv]), (w + h) / 2


def analyze(folder: Path) -> dict:
    """Compares the ring in a survey with the camera model.

    Args:
        folder: A camera_survey output folder with summary.json and survey.png.

    Returns:
        predicted_px and measured_px ring centres, offset_px, offset_m in the
        robot frame (x, y) at the hoop's distance, distance_model_m (lens to
        hoop) and distance_from_size_m (from the ring size and the focal length).
    """
    import cv2
    from camera_survey import project
    from cell import CAMERA_K, CAMERA_T
    from collision import ArmChecker
    summary = json.loads((folder / "summary.json").read_text())
    q = np.deg2rad(summary["survey_pose"]["q_deg"])
    hoop = ArmChecker().hoop_center(q)
    predicted = project(hoop)
    measured, diameter = find_ring(cv2.imread(str(folder / "survey.png")), predicted)
    offset = measured - predicted
    distance = CAMERA_T[2, 3] - hoop[2]
    f = CAMERA_K[0, 0]
    # image right = robot +y, image down = robot +x (cell.CAMERA_T)
    offset_m = CAMERA_T[:3, :3] @ np.array([*offset, 0.0]) * distance / f
    return {"predicted_px": predicted.round(1).tolist(),
            "measured_px": measured.round(1).tolist(),
            "offset_px": offset.round(1).tolist(),
            "offset_m": offset_m[:2].round(4).tolist(),
            "distance_model_m": round(float(distance), 3),
            "distance_from_size_m": round(float(f * RING_DIAMETER / diameter), 3)}


def main() -> None:
    """Analyzes the given (or newest) survey and prints the comparison."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", type=Path,
                        help="camera_survey output folder (default: the newest)")
    args = parser.parse_args()
    folder = args.folder or max(p.parent for p in SURVEYS.glob("*/survey.png"))
    r = analyze(folder)
    print(f"survey {folder.name}: ring centre {r['measured_px']} px, model "
          f"{r['predicted_px']} px, offset {r['offset_px']} px = "
          f"{r['offset_m']} m (x, y) at the hoop")
    print(f"lens-hoop distance: model {r['distance_model_m']} m, from the ring size "
          f"{r['distance_from_size_m']} m")
    print("A camera further out by dx shows the hoop dx nearer to the robot: "
          "the camera correction is -offset_m.")


if __name__ == "__main__":
    main()
