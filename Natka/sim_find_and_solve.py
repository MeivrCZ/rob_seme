"""Find puzzle D from the simulated camera and simulate the hoop path.

Run from the repository root::

    uv run --project rob_seme python rob_seme/Natka/sim_find_and_solve.py --keep

The scene starts at a random viable position in the camera-visible work area,
but the path planner receives only the base pose estimated from the simulated
camera image. The simulated camera model provides synthetic calibration
correspondences to Vojtech's
Homography implementation; a real camera setup should load a measured
calibration instead. Before each joint move, a fresh camera image updates the
puzzle pose and the move is checked against the simulator's known scene meshes.
This check does not detect unmodeled obstacles from the color image.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROKOP = ROOT / "Prokop"
NATKA = ROOT / "Natka"
VOJTECH = ROOT / "Vojtěch"
for module_path in (
    PROKOP,
    NATKA,
    VOJTECH / "ArucoFinder",
    VOJTECH / "Homografy",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from cell import TABLE_Z, WORK_AREA_XY  # noqa: E402
from crs_puzzle_d_hoop_path import profile_points  # noqa: E402
from homography import Homography  # noqa: E402
from placement import (  # noqa: E402
    BASE_FOAM,
    MIN_CLEARANCE,
    STABILITY_MARGIN,
    Placement,
    TaskScene,
    in_cage,
)
from simulate_crs_puzzle_d_hoop_path import (  # noqa: E402
    IncompletePathPlanning,
    MAX_JOINT_STEP_DEG,
    MAX_RECOVERY_TILT_DEG,
    PathPlanningState,
    flange_targets,
    solve_path,
    validate_path,
)
from simulation import CRS93Sim  # noqa: E402


MARKER_HEIGHT_ABOVE_BASE_M = 0.005
CALIBRATION_HEIGHTS_ABOVE_TABLE_M = (0.03, 0.10, 0.20)
CALIBRATION_GRID_SIZE = 4
MAX_CAMERA_POSITION_CHANGE_M = 0.005
MAX_CAMERA_YAW_CHANGE_DEG = 2.0
MAX_WAYPOINT_REPLANS = 3


def random_viable_placement(
    task: TaskScene,
    rng: np.random.Generator,
    max_tries: int = 1000,
) -> Placement:
    """Choose a random camera-visible viable pose with the base centered."""
    for _ in range(max_tries):
        placement = Placement(
            rng.uniform(WORK_AREA_XY[:, 0], WORK_AREA_XY[:, 1]),
            np.zeros(2),
            float(rng.uniform(0.0, 2.0 * math.pi)),
        )
        if (
            in_cage(placement.turntable_xy, task.turntable_radius)
            and task.stability_margin(placement) >= STABILITY_MARGIN
            and task.clearance_to_robot(placement) >= MIN_CLEARANCE
        ):
            return placement
    raise RuntimeError(
        f"No random viable camera-visible placement found in {max_tries} tries."
    )


def simulated_homography(robot: CRS93Sim) -> Homography:
    """Fit Vojtech's 3D projection model using the simulated camera geometry."""
    homography = Homography(
        x_range_m=tuple(WORK_AREA_XY[0]),
        y_range_m=tuple(WORK_AREA_XY[1]),
        z_levels_m=CALIBRATION_HEIGHTS_ABOVE_TABLE_M,
        table_z_m=TABLE_Z,
        nx=CALIBRATION_GRID_SIZE,
        ny=CALIBRATION_GRID_SIZE,
    )
    for index, point in enumerate(homography.points):
        pixel = robot.camera.project(point)
        homography.record(
            index,
            (float(pixel[0]), float(pixel[1])),
            tuple(float(value) for value in point),
        )
    homography.calculate(validation_fraction=0.0, ransac=False)
    return homography


def estimate_base_placement(
    robot: CRS93Sim,
    detector: ArucoBlockDetector,
    homography: Homography,
    image: np.ndarray | None = None,
    display: bool = True,
) -> tuple[Placement, tuple[int, int], float]:
    """Estimate the base center and yaw from one camera image."""
    if image is None:
        image = robot.grab_image()
    pose = detector.detect(image)
    if pose is None:
        raise RuntimeError("The camera image contains no recognized base ArUco markers.")

    task = robot.viewer.task
    marker_transform = task.base.get_transform("guide_1", task.base.base_link)
    marker_height = (
        TABLE_Z
        + task.T_top[2, 3]
        + BASE_FOAM
        + marker_transform[2, 3]
    )
    center_u, center_v = pose.center_px
    center_x, center_y = homography.pixel_to_robot(
        float(center_u),
        float(center_v),
        float(marker_height),
    )

    image_angle = math.radians(pose.angle_deg)
    angle_probe = (
        center_u + 100.0 * math.cos(image_angle),
        center_v + 100.0 * math.sin(image_angle),
    )
    probe_x, probe_y = homography.pixel_to_robot(
        angle_probe[0],
        angle_probe[1],
        float(marker_height),
    )
    base_yaw = math.atan2(probe_y - center_y, probe_x - center_x)
    estimated_base_xy = np.array([center_x, center_y], dtype=float)

    # This simulation fixture puts the base at the turntable center. Encoding
    # the detected base pose as a centered placement lets existing path and
    # collision helpers use the camera estimate without reading the fixture pose.
    turntable_xy = estimated_base_xy - task.T_top[:2, 3]
    estimate = Placement(turntable_xy, np.zeros(2), base_yaw)

    if display:
        annotated = detector.draw(image, pose)
        cv2.putText(
            annotated,
            f"Estimated base: ({center_x:.3f}, {center_y:.3f}) m, "
            f"yaw {math.degrees(base_yaw):.1f} deg",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 140, 20),
            2,
            cv2.LINE_AA,
        )
        robot.viewer.server.gui.add_image(
            cv2.cvtColor(annotated[::4, ::4], cv2.COLOR_BGR2RGB),
            label="ArUco pose estimate",
            format="jpeg",
        )
        print(
            f"Camera detected ArUco IDs {pose.visible_ids}: "
            f"base center ({center_x:.4f}, {center_y:.4f}) m, "
            f"yaw {math.degrees(base_yaw):.2f} degrees."
        )
    return estimate, pose.center_px, base_yaw


def check_next_move_against_camera(
    robot: CRS93Sim,
    q_next: np.ndarray,
    detector: ArucoBlockDetector,
    homography: Homography,
) -> tuple[Placement, float]:
    """Re-localize the puzzle and reject modeled collisions along the next move."""
    image = robot.grab_image()
    placement, _, _ = estimate_base_placement(
        robot,
        detector,
        homography,
        image=image,
        display=False,
    )
    robot.viewer.set_placement(placement)

    q_current = robot.get_q()
    turntable_angle = robot.viewer.turntable_angle(robot._now())
    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    sample_count = max(
        1,
        math.ceil(float(np.max(np.abs(q_next - q_current))) / max_joint_step),
    )
    minimum_clearance = math.inf
    for sample in range(sample_count + 1):
        fraction = sample / sample_count
        q_sample = q_current + fraction * (q_next - q_current)
        hits, clearance = robot._collider.check(
            q_sample,
            placement,
            turntable_angle,
        )
        if hits:
            pairs = ", ".join(f"{first}-{second}" for first, second in hits)
            raise RuntimeError(
                "Camera-based pre-move check rejected the next move: "
                f"collision with {pairs} at path sample "
                f"{sample}/{sample_count}; move was not issued."
            )
        minimum_clearance = min(minimum_clearance, clearance)
    return placement, minimum_clearance


def camera_pose_changed(
    planned: Placement,
    observed: Placement,
) -> bool:
    """Whether a fresh camera pose requires rebuilding the next waypoint target."""
    position_change = float(
        np.linalg.norm(planned.turntable_xy - observed.turntable_xy)
    )
    yaw_change = math.degrees(
        math.atan2(
            math.sin(observed.base_yaw - planned.base_yaw),
            math.cos(observed.base_yaw - planned.base_yaw),
        )
    )
    return (
        position_change > MAX_CAMERA_POSITION_CHANGE_M
        or abs(yaw_change) > MAX_CAMERA_YAW_CHANGE_DEG
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="Viser viewer port")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.2,
        help="relative joint speed in the range (0, 1] (default: 0.2)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the simulation viewer open after the path finishes",
    )
    parser.add_argument(
        "--check-collisions",
        action="store_true",
        help="enable CRS93Sim's continuous collision monitor",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="optional random seed for repeatable viable puzzle placement",
    )
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed must be greater than zero and at most one")

    robot = CRS93Sim(
        port=args.port,
        keep_alive=args.keep,
        collisions=args.check_collisions,
    )
    robot.initialize(home=False)
    robot.set_speed_relative(args.speed)
    placement_rng = np.random.default_rng(args.seed)
    initial_placement = random_viable_placement(
        robot.viewer.task,
        placement_rng,
    )
    robot.viewer.set_placement(initial_placement)
    print(
        "Puzzle placed at a random viable camera-visible position: "
        f"({initial_placement.turntable_xy[0]:.3f}, "
        f"{initial_placement.turntable_xy[1]:.3f}) m with a "
        f"{math.degrees(initial_placement.base_yaw):.1f}-degree base yaw."
    )

    homography = simulated_homography(robot)
    print(
        "Simulated homography fitted from "
        f"{homography.fit_info['training_count']} 3D/2D samples."
    )
    detector = ArucoBlockDetector()
    points = profile_points()
    planning_state = PathPlanningState()
    last_finished_waypoint = -1
    planning_stopped = False

    while last_finished_waypoint < len(points) - 1:
        waypoint = last_finished_waypoint + 1
        waypoint_finished = False
        retry_reason = "camera pose changed"
        for attempt in range(1, MAX_WAYPOINT_REPLANS + 1):
            image = robot.grab_image()
            planned_placement, _, _ = estimate_base_placement(
                robot,
                detector,
                homography,
                image=image,
                display=last_finished_waypoint == -1,
            )
            robot.viewer.set_placement(planned_placement)
            targets = flange_targets(robot, points)
            q_start = robot.get_q()
            attempt_state = PathPlanningState(
                roll_degrees=planning_state.roll_degrees,
                tilt_operations=planning_state.tilt_operations.copy(),
                previous_sign=planning_state.previous_sign,
            )

            plan_complete = True
            try:
                waypoint_path = solve_path(
                    robot,
                    targets,
                    points,
                    start_index=waypoint,
                    end_index=waypoint + 1,
                    q_start=q_start,
                    state=attempt_state,
                )
            except IncompletePathPlanning as error:
                if not error.path:
                    print(
                        f"Planning stopped before waypoint {waypoint}; "
                        "the last completed waypoint remains "
                        f"{last_finished_waypoint}."
                    )
                    planning_stopped = True
                    break
                waypoint_path = error.path
                plan_complete = False
                retry_reason = "the planner found only a partial waypoint path"
                print(
                    f"Waypoint {waypoint} planning found only a valid prefix; "
                    "the waypoint will remain pending."
                )

            try:
                center_deviation, normal_error = validate_path(
                    robot,
                    waypoint_path,
                    points,
                    max_normal_error_deg=MAX_RECOVERY_TILT_DEG,
                )
            except RuntimeError as error:
                print(
                    f"Waypoint {waypoint} preflight warning; continuing "
                    f"visualization: {error}"
                )
            else:
                print(
                    f"Waypoint {waypoint} preflight passed: max center "
                    f"deviation {center_deviation:.1f} mm, hoop-plane error "
                    f"{normal_error:.1f} degrees."
                )

            replan = False
            for q in waypoint_path:
                observed_placement, clearance = check_next_move_against_camera(
                    robot,
                    q,
                    detector,
                    homography,
                )
                if camera_pose_changed(planned_placement, observed_placement):
                    print(
                        f"Camera pose changed before waypoint {waypoint}; "
                        "discarding the stale target and replanning."
                    )
                    retry_reason = "the camera pose kept changing"
                    replan = True
                    break

                print(
                    "Camera-based pre-move check passed: "
                    f"minimum modeled clearance {clearance * 1000.0:.1f} mm."
                )
                robot.move_to_q(q)
                robot.wait_for_motion_stop()

            if replan:
                continue
            if len(waypoint_path) == 0:
                planning_stopped = True
                break
            if not plan_complete:
                continue

            planning_state = attempt_state
            waypoint_finished = True
            last_finished_waypoint = waypoint
            print(f"Finished waypoint {last_finished_waypoint}.")
            break

        if planning_stopped:
            break
        if not waypoint_finished:
            print(
                f"Could not complete waypoint {waypoint} after "
                f"{MAX_WAYPOINT_REPLANS} replans because {retry_reason}; "
                "stopping without marking it complete."
            )
            planning_stopped = True
            break

    if not planning_stopped and last_finished_waypoint == len(points) - 1:
        print("Puzzle D hoop path complete.")
    else:
        print(
            "Simulation stopped with last completed waypoint "
            f"{last_finished_waypoint}; the remaining path was not completed."
        )


if __name__ == "__main__":
    main()
