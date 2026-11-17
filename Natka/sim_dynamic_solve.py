"""Receding-horizon CRS93 simulation for a spinning Puzzle D.

Run from the repository root::

    uv run --project rob_seme python rob_seme/Natka/sim_dynamic_solve.py --keep

The script tracks the spinning ArUco base, predicts its pose at the expected
arrival time of each short move, checks the move against the predicted moving
scene, and replans from the current robot configuration.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATKA = ROOT / "Natka"
PROKOP = ROOT / "Prokop"
for module_path in (NATKA, PROKOP):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from crs_puzzle_d_hoop_path import profile_points  # noqa: E402
from placement import Placement  # noqa: E402
from sim_find_and_solve import (  # noqa: E402
    MAX_CAMERA_POSITION_CHANGE_M,
    MAX_CAMERA_YAW_CHANGE_DEG,
    MAX_WAYPOINT_REPLANS,
    TURNTABLE_SPIN_PERIOD_S,
    WAYPOINT_1_WAIT_AFTER_WAYPOINT_0_S,
    WAYPOINT_RETRY_INTERVAL_S,
    CameraLocalizationError,
    marker_base_pose,
    simulated_homography,
)
import simulate_crs_puzzle_d_hoop_path as planner  # noqa: E402
from simulate_crs_puzzle_d_hoop_path import (  # noqa: E402
    HoopOrbitPlanningError,
    IncompletePathPlanning,
    MAX_JOINT_STEP_DEG,
    MAX_RECOVERY_TILT_DEG,
    PathPlanningState,
    StartingWaypointUnreachable,
    flange_targets,
    solve_path,
    validate_path,
)
from simulation import CRS93Sim  # noqa: E402
from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from homography import Homography  # noqa: E402


SPIN_SMOOTHING = 0.5
MAX_ABS_SPIN_RATE_RAD_S = 2.0
PLANNING_PREDICTION_ITERATIONS = 3
PLANNING_OVERHEAD_S = 0.15
PRE_MOVE_CHECK_OVERHEAD_S = 0.30
DYNAMIC_MIN_CLEARANCE_M = 0.005
ARUCO_DETECTION_TIMEOUT_S = 13.0
ARUCO_RETRY_INTERVAL_S = 0.25


@dataclass(frozen=True)
class TimedPuzzlePose:
    center_xy: np.ndarray
    yaw: float
    timestamp: float


class PuzzlePoseTracker:
    """Estimate Puzzle D's planar pose and angular velocity from ArUco images."""

    def __init__(
        self,
        robot: CRS93Sim,
        detector: ArucoBlockDetector,
        homography: Homography,
    ) -> None:
        self.robot = robot
        self.detector = detector
        self.homography = homography
        self.pose: TimedPuzzlePose | None = None
        self.omega = 0.0
        self._has_rate_estimate = False

    def capture(self) -> TimedPuzzlePose:
        deadline = time.monotonic() + ARUCO_DETECTION_TIMEOUT_S
        while True:
            image = self.robot.grab_image()
            timestamp = self.robot._now() - self.robot.camera_delay
            try:
                center_xy, yaw = marker_base_pose(
                    self.robot,
                    self.detector,
                    self.homography,
                    image,
                )
            except CameraLocalizationError:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise CameraLocalizationError(
                        "ArUco markers were not detectable within "
                        f"{ARUCO_DETECTION_TIMEOUT_S:.0f} seconds."
                    ) from None
                print(
                    "ArUco markers not detected; retrying camera capture "
                    f"for up to {remaining:.1f} more seconds."
                )
                time.sleep(min(ARUCO_RETRY_INTERVAL_S, remaining))
                continue
            break
        observed = TimedPuzzlePose(center_xy, yaw, timestamp)

        if self.pose is not None:
            elapsed = observed.timestamp - self.pose.timestamp
            if elapsed <= 0.0:
                raise CameraLocalizationError(
                    "Camera timestamps did not advance while tracking Puzzle D."
                )
            yaw_delta = wrap_angle(observed.yaw - self.pose.yaw)
            measured_omega = yaw_delta / elapsed
            if abs(measured_omega) > MAX_ABS_SPIN_RATE_RAD_S:
                raise CameraLocalizationError(
                    "ArUco rotation estimate exceeded the supported angular "
                    f"speed ({measured_omega:.2f} rad/s)."
                )
            if self._has_rate_estimate:
                self.omega = (
                    SPIN_SMOOTHING * measured_omega
                    + (1.0 - SPIN_SMOOTHING) * self.omega
                )
            else:
                self.omega = measured_omega
                self._has_rate_estimate = True
        self.pose = observed
        return observed

    def predict(self, timestamp: float) -> TimedPuzzlePose:
        if self.pose is None:
            raise CameraLocalizationError("Puzzle pose has not been observed yet.")
        return TimedPuzzlePose(
            self.pose.center_xy.copy(),
            wrap_angle(self.pose.yaw + self.omega * (timestamp - self.pose.timestamp)),
            timestamp,
        )

    def placement_at(
        self,
        pose: TimedPuzzlePose,
        reference_time: float,
    ) -> Placement:
        task = self.robot.viewer.task
        turntable_xy = pose.center_xy - task.T_top[:2, 3]
        base_yaw = wrap_angle(
            pose.yaw - self.robot.viewer.turntable_angle(reference_time)
        )
        return Placement(turntable_xy, np.zeros(2), base_yaw)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_changed(
    first: TimedPuzzlePose,
    second: TimedPuzzlePose,
) -> bool:
    return (
        float(np.linalg.norm(first.center_xy - second.center_xy))
        > MAX_CAMERA_POSITION_CHANGE_M
        or abs(math.degrees(wrap_angle(second.yaw - first.yaw)))
        > MAX_CAMERA_YAW_CHANGE_DEG
    )


def path_duration(robot: CRS93Sim, path: list[np.ndarray]) -> float:
    """Estimate total move time from the simulator's coordinated profiles."""
    duration = 0.0
    for q_start, q_end in zip(path, path[1:]):
        duration += robot._profile(q_end - q_start)[0]
        duration += robot.command_delay
    return duration


def plan_waypoint(
    robot: CRS93Sim,
    tracker: PuzzlePoseTracker,
    points: list[tuple[float, float, float]],
    waypoint: int,
    q_start: np.ndarray,
    state: PathPlanningState,
) -> tuple[
    list[np.ndarray],
    PathPlanningState,
    TimedPuzzlePose,
    float,
    tuple[float, float] | None,
]:
    """Plan a single waypoint against an iteratively predicted arrival pose."""
    now = robot._now()
    arrival = now + PLANNING_OVERHEAD_S
    final_path: list[np.ndarray] = []
    final_state = PathPlanningState(
        state.roll_degrees,
        state.tilt_operations.copy(),
        state.previous_sign,
    )
    final_arrival_pose = tracker.predict(arrival)
    validation: tuple[float, float] | None = None

    for _ in range(PLANNING_PREDICTION_ITERATIONS):
        arrival_pose = tracker.predict(arrival)
        reference_time = robot._now()
        robot.viewer.set_placement(
            tracker.placement_at(arrival_pose, reference_time)
        )
        original_puzzle_transform = planner.puzzle_transform
        attempt_state = PathPlanningState(
            state.roll_degrees,
            state.tilt_operations.copy(),
            state.previous_sign,
        )
        try:
            frozen_puzzle_transform = original_puzzle_transform(robot)
            planner.puzzle_transform = (
                lambda _robot: frozen_puzzle_transform.copy()
            )
            targets = flange_targets(robot, points)
            path = solve_path(
                robot,
                targets,
                points,
                start_index=waypoint,
                end_index=waypoint + 1,
                q_start=q_start,
                state=attempt_state,
            )
            try:
                validation = validate_path(
                    robot,
                    path,
                    points,
                    max_normal_error_deg=MAX_RECOVERY_TILT_DEG,
                )
            except RuntimeError as error:
                print(f"Waypoint {waypoint} preflight warning: {error}")
        finally:
            planner.puzzle_transform = original_puzzle_transform
            current_pose = tracker.predict(robot._now())
            robot.viewer.set_placement(
                tracker.placement_at(current_pose, robot._now())
            )

        final_path = path
        final_state = attempt_state
        final_arrival_pose = arrival_pose
        travel_time = path_duration(robot, [q_start, *path])
        next_arrival = (
            robot._now()
            + travel_time
            + PLANNING_OVERHEAD_S
            + len(path) * PRE_MOVE_CHECK_OVERHEAD_S
        )
        if abs(next_arrival - arrival) < 0.1:
            arrival = next_arrival
            break
        arrival = next_arrival

    return final_path, final_state, final_arrival_pose, arrival, validation


def check_dynamic_move(
    robot: CRS93Sim,
    tracker: PuzzlePoseTracker,
    planned_arrival_pose: TimedPuzzlePose,
    q_start: np.ndarray,
    q_target: np.ndarray,
) -> tuple[bool, float]:
    """Check a joint move against predicted puzzle poses throughout its motion."""
    tracker.capture()
    move_start = robot._now() + robot.command_delay
    move_duration = robot._profile(q_target - q_start)[0]
    move_end = move_start + move_duration
    if pose_changed(
        planned_arrival_pose,
        tracker.predict(planned_arrival_pose.timestamp),
    ):
        return False, math.inf

    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    joint_samples = max(
        1,
        math.ceil(float(np.max(np.abs(q_target - q_start))) / max_joint_step),
    )
    time_samples = max(1, math.ceil(move_duration / 0.05))
    sample_count = max(joint_samples, time_samples)
    minimum_clearance = math.inf

    for sample in range(sample_count + 1):
        fraction = sample / sample_count
        q = q_start + fraction * (q_target - q_start)
        timestamp = move_start + fraction * move_duration
        predicted_pose = tracker.predict(timestamp)
        placement = tracker.placement_at(predicted_pose, timestamp)
        hits, clearance = robot._collider.check(
            q,
            placement,
            robot.viewer.turntable_angle(timestamp),
        )
        if hits:
            pairs = ", ".join(f"{first}-{second}" for first, second in hits)
            raise HoopOrbitPlanningError(
                "Predicted moving-scene collision with "
                f"{pairs} at move sample {sample}/{sample_count}."
            )
        if clearance < DYNAMIC_MIN_CLEARANCE_M:
            raise HoopOrbitPlanningError(
                "Predicted moving-scene clearance "
                f"{clearance * 1000.0:.1f} mm is below the "
                f"{DYNAMIC_MIN_CLEARANCE_M * 1000.0:.1f} mm safety margin."
            )
        minimum_clearance = min(minimum_clearance, clearance)

    return True, minimum_clearance


def run_dynamic_solve(
    robot: CRS93Sim,
    detector: ArucoBlockDetector,
    homography: Homography,
) -> None:
    points = profile_points()
    tracker = PuzzlePoseTracker(robot, detector, homography)
    tracker.capture()
    tracker.capture()
    if abs(tracker.omega) < math.radians(0.5):
        raise CameraLocalizationError(
            "The ArUco base appears stationary; dynamic solving needs measurable spin."
        )
    direction = "counter-clockwise" if tracker.omega > 0 else "clockwise"
    robot.viewer.set_spin(
        math.copysign(2.0 * math.pi / TURNTABLE_SPIN_PERIOD_S, tracker.omega)
    )
    print(
        f"Tracking ArUco rotation {direction} at "
        f"{tracker.omega:.3f} rad/s."
    )

    latest_pose = tracker.capture()
    robot.viewer.set_placement(
        tracker.placement_at(latest_pose, robot._now())
    )
    planning_state = PathPlanningState()
    last_finished_waypoint = -1

    while last_finished_waypoint < len(points) - 1:
        waypoint = last_finished_waypoint + 1
        q_start = robot.get_q()
        waiting_for_spin = last_finished_waypoint == 0 and waypoint == 1
        retry_deadline = (
            time.monotonic() + WAYPOINT_1_WAIT_AFTER_WAYPOINT_0_S
            if waiting_for_spin
            else None
        )
        attempts = 0
        completed = False

        while attempts < MAX_WAYPOINT_REPLANS or (
            retry_deadline is not None and time.monotonic() < retry_deadline
        ):
            attempts += 1
            if tracker.pose is None:
                tracker.capture()
            try:
                (
                    path,
                    next_state,
                    planned_arrival_pose,
                    arrival,
                    validation,
                ) = plan_waypoint(
                    robot,
                    tracker,
                    points,
                    waypoint,
                    q_start,
                    planning_state,
                )
            except (IncompletePathPlanning, StartingWaypointUnreachable):
                if retry_deadline is None:
                    raise
                remaining = retry_deadline - time.monotonic()
                if remaining > 0.0:
                    print(
                        "No safe predicted continuation to waypoint 1 yet; "
                        "waiting at waypoint 0 for the spinning puzzle."
                    )
                    time.sleep(min(WAYPOINT_RETRY_INTERVAL_S, remaining))
                tracker.capture()
                q_start = robot.get_q()
                continue

            if not path:
                raise RuntimeError(f"Planner returned no move for waypoint {waypoint}.")

            if validation is not None:
                center_error, normal_error = validation
                print(
                    f"Waypoint {waypoint} preflight: center deviation "
                    f"{center_error:.1f} mm, plane error {normal_error:.1f} deg."
                )

            stale = False
            for q_target in path:
                q_current = robot.get_q()
                try:
                    safe, clearance = check_dynamic_move(
                        robot,
                        tracker,
                        planned_arrival_pose,
                        q_current,
                        q_target,
                    )
                except HoopOrbitPlanningError:
                    if retry_deadline is None:
                        raise
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise
                    print(
                        "Predicted collision on the move to waypoint 1; "
                        "waiting at waypoint 0 for another safe rotation phase."
                    )
                    time.sleep(min(WAYPOINT_RETRY_INTERVAL_S, remaining))
                    stale = True
                    break
                if not safe:
                    print(
                        f"Prediction changed before waypoint {waypoint}; "
                        "replanning from the current pose."
                    )
                    stale = True
                    break

                print(
                    f"Predicted moving-scene check passed "
                    f"({clearance * 1000.0:.1f} mm clearance); "
                    f"arrival in about {max(0.0, arrival - robot._now()):.2f} s."
                )
                robot.move_to_q(q_target)
                robot.wait_for_motion_stop()
                q_start = robot.get_q()
                planning_state = next_state

            if stale:
                tracker.capture()
                q_start = robot.get_q()
                continue

            last_finished_waypoint = waypoint
            completed = True
            print(f"Finished waypoint {last_finished_waypoint}.")
            break

        if not completed:
            raise RuntimeError(
                f"Could not find a safe dynamic continuation to waypoint "
                f"{waypoint} after {attempts} planning attempts."
            )

    print("Dynamic Puzzle D path complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="Viser viewer port")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--check-collisions", action="store_true")
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
    placement = planner.waypoint_zero_reachable_placement(
        robot,
        turntable_xy=np.array([0.4, 0.0]),
    )
    robot.viewer.set_placement(placement)
    robot.viewer.set_spin(2.0 * math.pi / TURNTABLE_SPIN_PERIOD_S)
    print(
        "Waypoint-0-reachable placement: "
        f"({placement.turntable_xy[0]:.3f}, {placement.turntable_xy[1]:.3f}) m; "
        f"spin period {TURNTABLE_SPIN_PERIOD_S:.0f} s/rev."
    )

    homography = simulated_homography(robot)
    detector = ArucoBlockDetector()
    run_dynamic_solve(robot, detector, homography)


if __name__ == "__main__":
    main()
