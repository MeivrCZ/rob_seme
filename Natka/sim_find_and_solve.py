"""Find puzzle D from the simulated camera and simulate the hoop path.

Run from the repository root::

    uv run --project rob_seme python rob_seme/Natka/sim_find_and_solve.py --keep

The scene starts at a viable work-area placement selected by checking that
waypoint 0 is reachable, but the path planner receives only the base pose
estimated from the simulated camera image. The simulated camera model provides
synthetic calibration correspondences to Vojtech's
Homography implementation; a real camera setup should load a measured
calibration instead. Before each joint move, a fresh camera image updates the
puzzle pose and the move is checked against the simulator's known scene meshes.
This check does not detect unmodeled obstacles from the color image. After
waypoint 0, the planner waits up to 28 seconds for a safe continuation while
the puzzle keeps spinning.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Literal

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
from crs_puzzle_d_hoop_path import (  # noqa: E402
    HOOP_RING_TO_MOUNT_DISTANCE,
    PROFILE_START,
    profile_points,
)
from homography import Homography  # noqa: E402
from placement import (  # noqa: E402
    BASE_FOAM,
    Placement,
    collision_meshes,
)
from simulate_crs_puzzle_d_hoop_path import (  # noqa: E402
    IncompletePathPlanning,
    HoopOrbitPlanningError,
    MAX_JOINT_STEP_DEG,
    MAX_RECOVERY_TILT_DEG,
    PathPlanningState,
    flange_to_hoop_root,
    flange_targets,
    rotation_z,
    solve_path,
    validate_path,
    waypoint_zero_reachable_placement,
)
from simulation import CRS93Sim  # noqa: E402


MARKER_HEIGHT_ABOVE_BASE_M = 0.005
CALIBRATION_HEIGHTS_ABOVE_TABLE_M = (0.03, 0.10, 0.20)
CALIBRATION_GRID_SIZE = 4
TURNTABLE_SPIN_PERIOD_S = 26.0
MAX_CAMERA_POSITION_CHANGE_M = 0.005
MAX_CAMERA_YAW_CHANGE_DEG = 2.0
MAX_WAYPOINT_REPLANS = 3
WAYPOINT_1_WAIT_AFTER_WAYPOINT_0_S = 28.0
WAYPOINT_RETRY_INTERVAL_S = 0.25
HORIZONTAL_FINISH_HEIGHT_M = 0.04

SpinDirection = Literal["clockwise", "counter-clockwise", "stationary"]


class CameraLocalizationError(RuntimeError):
    """Raised when the ArUco camera image cannot localize the puzzle."""


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
        raise CameraLocalizationError(
            "The camera image contains no recognized base ArUco markers."
        )

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
    observed_base_yaw = math.atan2(probe_y - center_y, probe_x - center_x)
    image_time = robot._now() - robot.camera_delay
    base_yaw = observed_base_yaw - robot.viewer.turntable_angle(image_time)
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


def marker_base_pose(
    robot: CRS93Sim,
    detector: ArucoBlockDetector,
    homography: Homography,
    image: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate the base center and absolute yaw from one ArUco image."""
    pose = detector.detect(image)
    if pose is None:
        raise CameraLocalizationError(
            "The camera image contains no recognized base ArUco markers."
        )

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
    absolute_yaw = math.atan2(probe_y - center_y, probe_x - center_x)
    return np.array([center_x, center_y], dtype=float), absolute_yaw


def determine_spin_direction(
    robot: CRS93Sim,
    detector: ArucoBlockDetector,
    homography: Homography,
) -> SpinDirection:
    """Determine turntable direction by comparing two timed ArUco poses."""
    first_image = robot.grab_image()
    first_time = robot._now() - robot.camera_delay
    _, first_yaw = marker_base_pose(
        robot,
        detector,
        homography,
        first_image,
    )

    second_image = robot.grab_image()
    second_time = robot._now() - robot.camera_delay
    _, second_yaw = marker_base_pose(
        robot,
        detector,
        homography,
        second_image,
    )

    elapsed = second_time - first_time
    if elapsed <= 0.0:
        raise CameraLocalizationError(
            "Camera timestamps did not advance while detecting turntable direction."
        )
    yaw_change = math.atan2(
        math.sin(second_yaw - first_yaw),
        math.cos(second_yaw - first_yaw),
    )
    if abs(yaw_change) < math.radians(0.5):
        return "stationary"
    return "counter-clockwise" if yaw_change > 0.0 else "clockwise"


def estimate_path_start_position(
    robot: CRS93Sim,
    base_center_xy: np.ndarray,
    absolute_base_yaw: float,
) -> np.ndarray:
    """Transform Puzzle D's first path point from the marker pose to robot XYZ."""
    task = robot.viewer.task
    c, s = math.cos(absolute_base_yaw), math.sin(absolute_base_yaw)
    T_robot_base = np.eye(4)
    T_robot_base[:3, :3] = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    T_robot_base[:3, 3] = (
        base_center_xy[0],
        base_center_xy[1],
        TABLE_Z + task.T_top[2, 3] + BASE_FOAM,
    )
    path_start = np.append(np.asarray(PROFILE_START, dtype=float) / 1000.0, 1.0)
    return (T_robot_base @ task.T_mount @ path_start)[:3]


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


def horizontal_hoop_target(
    robot: CRS93Sim,
    q: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Build a target with a horizontal hoop plane and measure base clearance."""
    placement = robot.viewer.placement
    task = robot.viewer.task
    turntable_angle = robot.viewer.turntable_angle(robot._now())
    T_base_base = (
        placement.turntable_T()
        @ rotation_z(turntable_angle)
        @ task.T_top
        @ placement.base_T_top()
    )
    base_top_z = max(
        float(mesh.bounds[1, 2])
        for mesh in collision_meshes(task.base, T_base_base)
    )

    T_flange_root = flange_to_hoop_root(robot)
    T_base_root = robot.fk(q) @ T_flange_root
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0],
        dtype=float,
    )
    ring_center = (
        T_base_root[:3, 3]
        + T_base_root[:3, :3] @ center_from_root
    )

    ring_x = T_base_root[:3, 0].copy()
    ring_x[2] = 0.0
    if np.linalg.norm(ring_x) < 1e-8:
        ring_x = T_base_root[:3, 1].copy()
        ring_x[2] = 0.0
    ring_x /= np.linalg.norm(ring_x)
    ring_y = np.cross(np.array([0.0, 0.0, 1.0]), ring_x)
    T_base_root[:3, :3] = np.column_stack(
        (ring_x, ring_y, np.array([0.0, 0.0, 1.0]))
    )
    T_base_root[:3, 3] = (
        ring_center - T_base_root[:3, :3] @ center_from_root
    )
    target = T_base_root @ np.linalg.inv(T_flange_root)
    return target, float(ring_center[2]), base_top_z


def try_horizontal_finish(
    robot: CRS93Sim,
    detector: ArucoBlockDetector,
    homography: Homography,
) -> None:
    """At shutdown, level the hoop if it is low enough and a safe IK exists."""
    for _ in range(MAX_WAYPOINT_REPLANS):
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
        target, center_z, base_top_z = horizontal_hoop_target(robot, q_current)
        height_above_base = center_z - base_top_z
        if height_above_base > HORIZONTAL_FINISH_HEIGHT_M:
            print(
                "Skipping final hoop leveling: hoop center is "
                f"{height_above_base * 100.0:.1f} cm above the base top "
                f"(limit {HORIZONTAL_FINISH_HEIGHT_M * 100.0:.0f} cm)."
            )
            return

        candidates = sorted(
            (q for q in robot.ik(target) if robot.in_limits(q)),
            key=lambda q: float(np.linalg.norm(q - q_current)),
        )
        if not candidates:
            print("Final hoop leveling skipped: no in-limit horizontal IK solution.")
            return

        for q_next in candidates:
            try:
                observed_placement, clearance = check_next_move_against_camera(
                    robot,
                    q_next,
                    detector,
                    homography,
                )
            except HoopOrbitPlanningError as error:
                print(
                    "Final hoop leveling candidate rejected by collision "
                    f"check: {error}"
                )
                continue
            if camera_pose_changed(placement, observed_placement):
                print(
                    "Camera pose changed before final hoop leveling; "
                    "recomputing the horizontal target."
                )
                break

            robot.move_to_q(q_next)
            robot.wait_for_motion_stop()
            print(
                "Final hoop leveling completed: hoop plane is horizontal; "
                f"minimum modeled clearance {clearance * 1000.0:.1f} mm."
            )
            return
        else:
            print(
                "Final hoop leveling skipped: no horizontal IK solution "
                "had a collision-free path."
            )
            return

    print("Final hoop leveling skipped: camera pose did not stabilize.")


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
    initial_placement = waypoint_zero_reachable_placement(
        robot,
        turntable_xy=np.array([0.4, 0.0]),
    )
    robot.viewer.set_placement(initial_placement)
    robot.viewer.set_spin(2.0 * math.pi / TURNTABLE_SPIN_PERIOD_S)
    print(
        "Puzzle placed where waypoint 0 is reachable: "
        f"({initial_placement.turntable_xy[0]:.3f}, "
        f"{initial_placement.turntable_xy[1]:.3f}) m with a "
        f"{math.degrees(initial_placement.base_yaw):.1f}-degree base yaw, "
        f"spinning at {TURNTABLE_SPIN_PERIOD_S:.0f} s/rev."
    )

    homography = simulated_homography(robot)
    print(
        "Simulated homography fitted from "
        f"{homography.fit_info['training_count']} 3D/2D samples."
    )
    detector = ArucoBlockDetector()
    spin_direction = determine_spin_direction(robot, detector, homography)
    if spin_direction == "stationary":
        raise CameraLocalizationError(
            "The ArUco pose did not rotate; cannot determine turntable direction."
        )
    spin_sign = 1.0 if spin_direction == "counter-clockwise" else -1.0
    robot.viewer.set_spin(
        spin_sign * 2.0 * math.pi / TURNTABLE_SPIN_PERIOD_S
    )
    print(f"ArUco markers show the puzzle is spinning {spin_direction}.")

    start_image = robot.grab_image()
    initial_camera_placement, _, _ = estimate_base_placement(
        robot,
        detector,
        homography,
        image=start_image,
        display=True,
    )
    start_center_xy, absolute_base_yaw = marker_base_pose(
        robot,
        detector,
        homography,
        start_image,
    )
    robot.viewer.set_placement(initial_camera_placement)
    path_start_position = estimate_path_start_position(
        robot,
        start_center_xy,
        absolute_base_yaw,
    )
    print(
        "Camera-estimated Puzzle D path start in robot coordinates: "
        f"({path_start_position[0]:.3f}, {path_start_position[1]:.3f}, "
        f"{path_start_position[2]:.3f}) m."
    )

    points = profile_points()
    planning_state = PathPlanningState()
    last_finished_waypoint = -1
    planning_stopped = False

    while last_finished_waypoint < len(points) - 1:
        waypoint = last_finished_waypoint + 1
        waypoint_finished = False
        retry_reason = "camera pose changed"
        wait_for_spin = last_finished_waypoint == 0 and waypoint == 1
        retry_deadline = (
            time.monotonic() + WAYPOINT_1_WAIT_AFTER_WAYPOINT_0_S
            if wait_for_spin
            else None
        )
        attempt = 0
        while (
            attempt < MAX_WAYPOINT_REPLANS
            or (
                retry_deadline is not None
                and time.monotonic() < retry_deadline
            )
        ):
            attempt += 1
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
                    retry_reason = "no safe continuation was found"
                    if retry_deadline is not None:
                        remaining = retry_deadline - time.monotonic()
                        if remaining > 0.0:
                            print(
                                "No safe move to waypoint 1 yet; the robot "
                                "will wait at waypoint 0 and retry while the "
                                "base continues spinning."
                            )
                            time.sleep(min(WAYPOINT_RETRY_INTERVAL_S, remaining))
                        continue
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
                if retry_deadline is not None:
                    continue
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
            if retry_deadline is not None:
                print(
                    f"Waited up to {WAYPOINT_1_WAIT_AFTER_WAYPOINT_0_S:.0f} "
                    f"seconds at waypoint 0 but could not find a safe move "
                    f"to waypoint 1 ({retry_reason}); waypoint 1 remains "
                    "incomplete."
                )
            else:
                print(
                    f"Could not complete waypoint {waypoint} after "
                    f"{MAX_WAYPOINT_REPLANS} replans because {retry_reason}; "
                    "stopping without marking it complete."
                )
            planning_stopped = True
            break

    if not planning_stopped and last_finished_waypoint == len(points) - 1:
        try:
            try_horizontal_finish(robot, detector, homography)
        except CameraLocalizationError as error:
            print(f"Final hoop leveling skipped: {error}")
        except HoopOrbitPlanningError as error:
            print(f"Final hoop leveling skipped: {error}")

    if not planning_stopped and last_finished_waypoint == len(points) - 1:
        print("Puzzle D hoop path complete.")
    else:
        print(
            "Simulation stopped with last completed waypoint "
            f"{last_finished_waypoint}; the remaining path was not completed."
        )


if __name__ == "__main__":
    main()
