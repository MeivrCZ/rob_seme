"""Move a real CRS93 through Puzzle D while predicting the puzzle's rotation.

The script uses the real robot's camera for an initial pose/spin estimate and
then predicts the rotating puzzle pose between waypoint observations. It
requires a measured 3D camera calibration and a typed confirmation before
sending any motion commands.

Example::

    uv run --project rob_seme python rob_seme/Natka/dynamic_solve.py \
        --calibration /path/to/homography_calibration.json
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATKA = ROOT / "Natka"
PROKOP = ROOT / "Prokop"
VOJTECH = ROOT / "Vojtěch"
for module_path in (
    NATKA,
    PROKOP,
    VOJTECH / "ArucoFinder",
    VOJTECH / "Homografy",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from cell import TABLE_Z  # noqa: E402
from collision import SceneCollider  # noqa: E402
from crs_puzzle_d_hoop_path import profile_points  # noqa: E402
from homography import Homography  # noqa: E402
from placement import (  # noqa: E402
    BASE_FOAM,
    HOOP_MOUNT,
    MODELS,
    TASK_PUZZLE,
    Placement,
    collision_meshes,
    load_urdf,
)
import simulate_crs_puzzle_d_hoop_path as planner  # noqa: E402
from simulate_crs_puzzle_d_hoop_path import (  # noqa: E402
    HoopOrbitPlanningError,
    IncompletePathPlanning,
    MAX_CENTER_DEVIATION_MM,
    MAX_JOINT_STEP_DEG,
    MAX_NORMAL_ERROR_DEG,
    MAX_RECOVERY_TILT_DEG,
    PathPlanningState,
    StartingWaypointUnreachable,
    flange_targets,
    solve_path,
)


DEFAULT_SPIN_PERIOD_S = 26.0
DEFAULT_SPEED = 0.1
SPIN_SAMPLE_INTERVAL_S = 0.75
SPIN_SMOOTHING = 0.25
MAX_ABS_SPIN_RATE_RAD_S = 2.0
WAYPOINT_0_WAIT_S = 60.0
ALIGNMENT_SEARCH_STEP_S = 0.025
PLANNING_LEAD_S = 0.5
COMMAND_OVERHEAD_S = 0.15
DYNAMIC_MIN_CLEARANCE_M = 0.005
MOTION_DURATION_SCALE = 1.15
CAMERA_RETRY_INTERVAL_S = 0.25
MAX_CAMERA_RETRIES_S = 13.0


@dataclass(frozen=True)
class TimedPuzzlePose:
    center_xy: np.ndarray
    yaw: float
    timestamp: float


class HardwareScene:
    """Load the known task meshes and their base/turntable transforms."""

    def __init__(self) -> None:
        self.turntable = load_urdf(MODELS / "turntable.urdf")
        self.base = load_urdf(MODELS / "base.urdf")
        self.puzzle = load_urdf(MODELS / f"{TASK_PUZZLE}.urdf")
        self.T_top = self.turntable.get_transform(
            "top", self.turntable.base_link
        )
        self.T_mount = self.base.get_transform(
            "puzzle_mount", self.base.base_link
        )


class HardwareViewerAdapter:
    """Provide the planner with predicted real-world task poses, not a viewer."""

    def __init__(self, task: HardwareScene, robot_urdf) -> None:
        self.task = task
        self.robot_urdf = robot_urdf
        self.placement: Placement | None = None

    def set_placement(self, placement: Placement) -> None:
        self.placement = placement

    @staticmethod
    def turntable_angle(_timestamp: float) -> float:
        # The estimated absolute base yaw is stored directly in Placement.
        return 0.0


class HardwarePlannerAdapter:
    """Expose the real CRS93 kinematics and offline collision preflight to planner."""

    def __init__(self, hardware, duration_scale: float) -> None:
        self.hardware = hardware
        self.q_home = np.asarray(hardware.q_home, dtype=float)
        self.viewer = HardwareViewerAdapter(
            HardwareScene(),
            load_urdf(MODELS / "crs93.urdf"),
        )
        self._collider = SceneCollider(self.viewer.task)
        self.command_delay = COMMAND_OVERHEAD_S
        self.duration_scale = duration_scale
        self._speed_fraction = 1.0
        self._v_max = np.ones(6)
        self._a_max = np.ones(6)

    @property
    def speed_fraction(self) -> float:
        return self._speed_fraction

    @speed_fraction.setter
    def speed_fraction(self, value: float) -> None:
        self._speed_fraction = value

    def set_speed_relative(self, fraction: float) -> None:
        self._speed_fraction = fraction
        self.hardware.set_speed_relative(fraction)
        irc_per_rad = np.abs(self.hardware._deg_to_irc) * 180.0 / math.pi
        speed_irc = self.hardware._min_speed_irc256_per_ms + (
            fraction
            * (
                self.hardware._max_speed_irc256_per_ms
                - self.hardware._min_speed_irc256_per_ms
            )
        )
        self._v_max = speed_irc * 1000.0 / 256.0 / irc_per_rad
        self._a_max = (
            np.asarray(self.hardware._default_acceleration_irc_per_ms)
            * 1e6
            / 256.0
            / irc_per_rad
        )

    def _now(self) -> float:
        return time.monotonic()

    def _profile(self, dq: np.ndarray) -> tuple[float, float]:
        distance = np.abs(np.asarray(dq, dtype=float))
        ramp = self._v_max / self._a_max
        durations = np.where(
            distance > self._v_max * ramp,
            distance / self._v_max + ramp,
            2.0 * np.sqrt(distance / self._a_max),
        )
        index = int(np.argmax(durations))
        raw_duration = float(durations[index])
        return (
            raw_duration * self.duration_scale,
            min(float(ramp[index]) * self.duration_scale, raw_duration / 2.0),
        )

    def in_limits(self, q: np.ndarray) -> bool:
        return bool(self.hardware.in_limits(q))

    def ik(self, target: np.ndarray) -> list[np.ndarray]:
        return self.hardware.ik(target)

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.hardware.fk(q)

    def get_q(self) -> np.ndarray:
        return np.asarray(self.hardware.get_q(), dtype=float)

    def __getattr__(self, name: str):
        return getattr(self.hardware, name)


class PuzzlePoseTracker:
    """Track base center/yaw observations and predict using measured spin."""

    def __init__(
        self,
        robot,
        planner_robot: HardwarePlannerAdapter,
        detector: ArucoBlockDetector,
        homography: Homography,
        expected_omega: float,
    ) -> None:
        self.robot = robot
        self.planner_robot = planner_robot
        self.detector = detector
        self.homography = homography
        self.expected_omega = expected_omega
        self.pose: TimedPuzzlePose | None = None
        self.omega: float | None = None
        self.base_marker_height = self._marker_height()

    def _marker_height(self) -> float:
        task = self.planner_robot.viewer.task
        marker_transform = task.base.get_transform(
            "guide_1", task.base.base_link
        )
        return (
            TABLE_Z
            + task.T_top[2, 3]
            + BASE_FOAM
            + marker_transform[2, 3]
        )

    def observe(self) -> TimedPuzzlePose:
        deadline = time.monotonic() + MAX_CAMERA_RETRIES_S
        while True:
            timestamp = time.monotonic()
            image = self.robot.grab_image()
            detected = self.detector.detect(image)
            if detected is not None:
                center_u, center_v = detected.center_px
                center_x, center_y = self.homography.pixel_to_robot(
                    float(center_u),
                    float(center_v),
                    self.base_marker_height,
                )
                angle = math.radians(detected.angle_deg)
                probe_x, probe_y = self.homography.pixel_to_robot(
                    center_u + 100.0 * math.cos(angle),
                    center_v + 100.0 * math.sin(angle),
                    self.base_marker_height,
                )
                yaw = math.atan2(probe_y - center_y, probe_x - center_x)
                observed = TimedPuzzlePose(
                    np.array([center_x, center_y], dtype=float),
                    yaw,
                    timestamp,
                )
                self._update(observed)
                return observed

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    "Puzzle ArUco markers were not detectable within "
                    f"{MAX_CAMERA_RETRIES_S:.0f} seconds."
                )
            print(
                "Puzzle markers not detected; waiting for a camera frame "
                f"({remaining:.1f} seconds remain)."
            )
            time.sleep(min(CAMERA_RETRY_INTERVAL_S, remaining))

    def _update(self, observed: TimedPuzzlePose) -> None:
        if self.pose is not None:
            elapsed = observed.timestamp - self.pose.timestamp
            if elapsed <= 0:
                raise RuntimeError("Camera observation timestamps did not advance.")
            yaw_delta = observed.yaw - self.pose.yaw
            if self.omega is not None:
                expected_delta = self.omega * elapsed
                yaw_delta += 2.0 * math.pi * round(
                    (expected_delta - yaw_delta) / (2.0 * math.pi)
                )
            else:
                yaw_delta = wrap_angle(yaw_delta)
            measured = yaw_delta / elapsed
            if abs(measured) > MAX_ABS_SPIN_RATE_RAD_S:
                raise RuntimeError(
                    f"Measured puzzle spin {measured:.2f} rad/s exceeds "
                    "the supported limit."
                )
            if self.omega is None:
                self.omega = measured
            else:
                self.omega = (
                    SPIN_SMOOTHING * measured
                    + (1.0 - SPIN_SMOOTHING) * self.omega
                )
        self.pose = observed

    def predict(self, timestamp: float) -> TimedPuzzlePose:
        if self.pose is None or self.omega is None:
            raise RuntimeError("Puzzle pose and spin direction are not initialized.")
        return TimedPuzzlePose(
            self.pose.center_xy.copy(),
            wrap_angle(self.pose.yaw + self.omega * (timestamp - self.pose.timestamp)),
            timestamp,
        )

    def placement_at(self, pose: TimedPuzzlePose) -> Placement:
        task = self.planner_robot.viewer.task
        turntable_xy = pose.center_xy - task.T_top[:2, 3]
        return Placement(turntable_xy, np.zeros(2), pose.yaw)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def marker_spin_estimate(
    tracker: PuzzlePoseTracker,
    expected_period_s: float,
) -> float:
    first = tracker.observe()
    observations = [first]
    for _ in range(2):
        time.sleep(SPIN_SAMPLE_INTERVAL_S)
        observations.append(tracker.observe())
    elapsed = observations[-1].timestamp - observations[0].timestamp
    measured = wrap_angle(observations[-1].yaw - observations[0].yaw) / elapsed
    expected_magnitude = 2.0 * math.pi / expected_period_s
    if abs(measured) < math.radians(0.5):
        raise RuntimeError(
            "The ArUco base did not visibly rotate during spin calibration."
        )
    if abs(abs(measured) - expected_magnitude) > expected_magnitude * 0.75:
        raise RuntimeError(
            f"Measured spin rate {measured:.3f} rad/s is inconsistent with "
            f"the configured {expected_period_s:.1f} s/revolution."
        )
    tracker.omega = math.copysign(expected_magnitude, measured)
    tracker.pose = observations[-1]
    return tracker.omega


def puzzle_transform_at(
    robot: HardwarePlannerAdapter,
    tracker: PuzzlePoseTracker,
    timestamp: float,
) -> np.ndarray:
    placement = tracker.placement_at(tracker.predict(timestamp))
    task = robot.viewer.task
    return (
        placement.turntable_T()
        @ task.T_top
        @ placement.base_T_top()
        @ task.T_mount
    )


def moving_route_error(
    robot: HardwarePlannerAdapter,
    tracker: PuzzlePoseTracker,
    q: np.ndarray,
    timestamp: float,
    points: list[tuple[float, float, float]],
) -> tuple[float, float]:
    puzzle_to_robot = puzzle_transform_at(robot, tracker, timestamp)
    robot_to_puzzle = np.linalg.inv(puzzle_to_robot)
    flange_to_root = planner.flange_to_hoop_root(robot)
    robot_to_root = robot.fk(q) @ flange_to_root
    center_from_root = np.array(
        [0.0, -planner.HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    center_robot = (
        robot_to_root[:3, 3]
        + robot_to_root[:3, :3] @ center_from_root
    )
    center_puzzle = (robot_to_puzzle @ np.append(center_robot, 1.0))[:3]
    route = np.asarray(points, dtype=float) / 1000.0
    tangents = np.unwrap(
        np.radians([planner.tangent_angle(points, i) for i in range(len(points))])
    )
    closest = (math.inf, 0, 0.0)
    for index, (start, end) in enumerate(zip(route, route[1:])):
        segment = end - start
        fraction = float(
            np.clip(
                np.dot(center_puzzle - start, segment) / np.dot(segment, segment),
                0.0,
                1.0,
            )
        )
        distance = float(np.linalg.norm(center_puzzle - (start + fraction * segment)))
        if distance < closest[0]:
            closest = distance, index, fraction

    distance, segment_index, segment_fraction = closest
    route_angle = tangents[segment_index] + segment_fraction * (
        tangents[segment_index + 1] - tangents[segment_index]
    )
    waypoint_index = (
        segment_index if segment_fraction < 0.5 else segment_index + 1
    )
    if waypoint_index == planner.BASE_PARALLEL_WAYPOINT:
        route_normal = robot_to_puzzle[:3, :3] @ np.array([0.0, 0.0, -1.0])
    else:
        route_normal = np.array([math.sin(route_angle), 0.0, math.cos(route_angle)])
    normal = (robot_to_puzzle @ robot_to_root)[:3, 2]
    normal_error = math.degrees(
        math.acos(float(np.clip(abs(np.dot(normal, route_normal)), 0.0, 1.0)))
    )
    return distance, normal_error


def waypoint_zero_alignment_time(
    robot: HardwarePlannerAdapter,
    tracker: PuzzlePoseTracker,
    q: np.ndarray,
    points: list[tuple[float, float, float]],
    deadline: float,
    earliest_time: float,
) -> float:
    """Calculate the next phase where the stationary hoop aligns to waypoint 0."""
    start = max(
        time.monotonic() + PLANNING_LEAD_S,
        earliest_time,
    )
    if start >= deadline:
        raise RuntimeError("The waypoint-0 alignment wait has expired.")

    flange_to_root = planner.flange_to_hoop_root(robot)
    robot_to_root = robot.fk(q) @ flange_to_root
    window_start = None
    window_end = None
    timestamp = start
    while timestamp <= deadline:
        puzzle_to_robot = puzzle_transform_at(robot, tracker, timestamp)
        robot_to_puzzle = np.linalg.inv(puzzle_to_robot)
        center_from_root = np.array(
            [0.0, -planner.HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
        )
        center_robot = (
            robot_to_root[:3, 3]
            + robot_to_root[:3, :3] @ center_from_root
        )
        center_puzzle = (
            robot_to_puzzle @ np.append(center_robot, 1.0)
        )[:3]
        target = np.asarray(points[0], dtype=float) / 1000.0
        distance = float(np.linalg.norm(center_puzzle - target))
        route_normal = np.array(
            [
                math.sin(math.radians(planner.tangent_angle(points, 0))),
                0.0,
                math.cos(math.radians(planner.tangent_angle(points, 0))),
            ]
        )
        normal = (robot_to_puzzle @ robot_to_root)[:3, 2]
        angle = math.degrees(
            math.acos(float(np.clip(abs(np.dot(normal, route_normal)), 0.0, 1.0)))
        )
        aligned = (
            distance <= MAX_CENTER_DEVIATION_MM / 1000.0
            and angle <= MAX_NORMAL_ERROR_DEG
        )
        if aligned:
            if window_start is None:
                window_start = timestamp
            window_end = timestamp
        elif window_start is not None:
            break
        timestamp += ALIGNMENT_SEARCH_STEP_S

    if window_start is None or window_end is None:
        raise RuntimeError(
            f"No predicted waypoint-0 alignment window was found within "
            f"{WAYPOINT_0_WAIT_S:.0f} seconds."
        )
    return (window_start + window_end) / 2.0


def plan_waypoint(
    robot: HardwarePlannerAdapter,
    tracker: PuzzlePoseTracker,
    points: list[tuple[float, float, float]],
    waypoint: int,
    q_start: np.ndarray,
    state: PathPlanningState,
    departure_time: float,
) -> tuple[list[np.ndarray], PathPlanningState, float]:
    """Plan a waypoint against the puzzle pose predicted at its arrival."""
    arrival = max(time.monotonic(), departure_time) + PLANNING_LEAD_S
    result_path: list[np.ndarray] = []
    result_state = PathPlanningState(
        state.roll_degrees,
        state.tilt_operations.copy(),
        state.previous_sign,
    )
    for _ in range(3):
        pose = tracker.predict(arrival)
        robot.viewer.set_placement(tracker.placement_at(pose))
        local_state = PathPlanningState(
            state.roll_degrees,
            state.tilt_operations.copy(),
            state.previous_sign,
        )
        result_path = solve_path(
            robot,
            flange_targets(robot, points),
            points,
            start_index=waypoint,
            end_index=waypoint + 1,
            q_start=q_start,
            state=local_state,
            check_static_route=False,
        )
        result_state = local_state
        duration = sum(
            robot._profile(q_end - q_begin)[0] + robot.command_delay
            for q_begin, q_end in zip([q_start, *result_path], result_path)
        )
        updated_arrival = max(
            time.monotonic(),
            departure_time,
        ) + duration + PLANNING_LEAD_S
        if abs(updated_arrival - arrival) < 0.1:
            arrival = updated_arrival
            break
        arrival = updated_arrival
    return result_path, result_state, arrival


def preflight_move(
    robot: HardwarePlannerAdapter,
    tracker: PuzzlePoseTracker,
    q_start: np.ndarray,
    q_target: np.ndarray,
    start_time: float,
    points: list[tuple[float, float, float]],
    waypoint: int,
    is_waypoint_end: bool,
) -> float:
    duration, ramp = robot._profile(q_target - q_start)
    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    sample_count = max(
        1,
        math.ceil(float(np.max(np.abs(q_target - q_start))) / max_joint_step),
        math.ceil(duration / 0.05),
    )
    minimum_clearance = math.inf
    for sample in range(sample_count + 1):
        elapsed = duration * sample / sample_count
        if ramp > 0 and elapsed < ramp:
            progress = 0.5 * elapsed**2 / (ramp * (duration - ramp))
        elif ramp > 0 and elapsed > duration - ramp:
            progress = 1.0 - 0.5 * (duration - elapsed) ** 2 / (
                ramp * (duration - ramp)
            )
        elif duration > ramp:
            progress = (elapsed - ramp / 2.0) / (duration - ramp)
        else:
            progress = elapsed / duration
        q = q_start + np.clip(progress, 0.0, 1.0) * (q_target - q_start)
        timestamp = start_time + elapsed
        placement = tracker.placement_at(tracker.predict(timestamp))
        hits, clearance = robot._collider.check(q, placement, 0.0)
        if hits:
            pairs = ", ".join(f"{first}-{second}" for first, second in hits)
            raise HoopOrbitPlanningError(
                f"Predicted real-robot path collides with {pairs} at sample "
                f"{sample}/{sample_count}."
            )
        if clearance < DYNAMIC_MIN_CLEARANCE_M:
            raise HoopOrbitPlanningError(
                f"Predicted clearance is only {clearance * 1000:.1f} mm "
                f"(minimum {DYNAMIC_MIN_CLEARANCE_M * 1000:.1f} mm)."
            )
        if is_waypoint_end and sample == sample_count:
            center_error, normal_error = moving_route_error(
                robot,
                tracker,
                q,
                timestamp,
                points,
            )
            if center_error > MAX_CENTER_DEVIATION_MM / 1000.0:
                raise HoopOrbitPlanningError(
                    f"Moving waypoint {waypoint} is "
                    f"{center_error * 1000:.1f} mm from the route center."
                )
            if normal_error > MAX_RECOVERY_TILT_DEG:
                raise HoopOrbitPlanningError(
                    f"Moving waypoint {waypoint} hoop-plane error is "
                    f"{normal_error:.1f} degrees."
                )
        minimum_clearance = min(minimum_clearance, clearance)
    return minimum_clearance


def execute_waypoint_path(
    robot: HardwarePlannerAdapter,
    tracker: PuzzlePoseTracker,
    path: list[np.ndarray],
    waypoint: int,
    points: list[tuple[float, float, float]],
    scheduled_start: float,
) -> float:
    """Preflight the predicted path, then execute it one move at a time."""
    q_start = robot.get_q()
    cursor = scheduled_start
    minimum_clearance = math.inf
    for index, target in enumerate(path):
        minimum_clearance = min(
            minimum_clearance,
            preflight_move(
                robot,
                tracker,
                q_start,
                target,
                cursor + robot.command_delay,
                points,
                waypoint,
                index == len(path) - 1,
            ),
        )
        cursor += robot._profile(target - q_start)[0] + robot.command_delay
        q_start = target

    delay = scheduled_start - time.monotonic()
    if delay > 0:
        time.sleep(delay)

    for target in path:
        robot.hardware.move_to_q(target)
        robot.hardware.wait_for_motion_stop()
    return minimum_clearance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=VOJTECH / "Homografy" / "homography_calibration.json",
        help="measured 3D calibration saved by Vojtech's Homografy module",
    )
    parser.add_argument(
        "--spin-period",
        type=float,
        default=DEFAULT_SPIN_PERIOD_S,
        help="expected turntable period in seconds per revolution",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="relative CRS93 speed in (0, 1] (default: 0.1)",
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=MOTION_DURATION_SCALE,
        help="conservative multiplier for estimated real move durations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="connect and plan waypoint 0 without moving the real robot",
    )
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed must be greater than zero and at most one")
    if args.spin_period <= 0.0:
        parser.error("--spin-period must be positive")
    if args.duration_scale < 1.0:
        parser.error("--duration-scale must be at least one")
    if not args.calibration.is_file():
        parser.error(
            f"Measured camera calibration not found: {args.calibration}. "
            "Calibrate the real camera with Vojtech's Homografy module first."
        )

    homography = Homography.load(args.calibration)
    if homography.P is None:
        parser.error(f"Calibration has no fitted projection matrix: {args.calibration}")
    detector = ArucoBlockDetector()

    from ctu_crs import CRS93

    hardware = CRS93()
    try:
        hardware.initialize(home=False)
        planner_robot = HardwarePlannerAdapter(hardware, args.duration_scale)
        planner_robot.set_speed_relative(args.speed)
        current_q = planner_robot.get_q()
        if not np.allclose(current_q, planner_robot.q_home, atol=math.radians(2.0)):
            raise RuntimeError(
                "For a checked approach to waypoint 0, first place the real robot "
                "within 2 degrees of q_home. This script will not home the arm."
            )

        tracker = PuzzlePoseTracker(
            hardware,
            planner_robot,
            detector,
            homography,
            2.0 * math.pi / args.spin_period,
        )
        omega = marker_spin_estimate(tracker, args.spin_period)
        spin_name = "counter-clockwise" if omega > 0 else "clockwise"
        print(
            f"Detected {spin_name} puzzle rotation; predicting "
            f"{args.spin_period:.1f} seconds per revolution."
        )

        points = profile_points()
        state = PathPlanningState()
        q_start = planner_robot.get_q()
        initial_departure = time.monotonic() + PLANNING_LEAD_S
        path, next_state, arrival = plan_waypoint(
            planner_robot,
            tracker,
            points,
            0,
            q_start,
            state,
            initial_departure,
        )
        if not path:
            raise RuntimeError("Planner returned no move for waypoint 0.")
        if args.dry_run:
            print(
                f"Dry run: waypoint 0 has {len(path)} planned joint target(s), "
                f"with predicted arrival in {arrival - time.monotonic():.1f} s."
            )
            return

        confirmation = input(
            "Real CRS93 motion is about to begin. Clear the workspace and "
            "verify the emergency stop is accessible. Type MOVE to continue: "
        )
        if confirmation.strip() != "MOVE":
            print("Motion cancelled; no robot move was sent.")
            return

        clearance = execute_waypoint_path(
            planner_robot,
            tracker,
            path,
            0,
            points,
            initial_departure,
        )
        state = next_state
        print(
            "Finished waypoint 0; predicted minimum clearance "
            f"{clearance * 1000:.1f} mm."
        )
        tracker.observe()

        deadline = time.monotonic() + WAYPOINT_0_WAIT_S
        earliest = time.monotonic() + PLANNING_LEAD_S
        waypoint = 1
        while waypoint < len(points):
            departure = (
                waypoint_zero_alignment_time(
                    planner_robot,
                    tracker,
                    planner_robot.get_q(),
                    points,
                    deadline,
                    earliest,
                )
                if waypoint == 1
                else time.monotonic() + PLANNING_LEAD_S
            )
            q_start = planner_robot.get_q()
            try:
                path, next_state, arrival = plan_waypoint(
                    planner_robot,
                    tracker,
                    points,
                    waypoint,
                    q_start,
                    state,
                    departure,
                )
                clearance = execute_waypoint_path(
                    planner_robot,
                    tracker,
                    path,
                    waypoint,
                    points,
                    departure,
                )
            except (
                IncompletePathPlanning,
                StartingWaypointUnreachable,
                HoopOrbitPlanningError,
            ) as error:
                if waypoint != 1 or time.monotonic() >= deadline:
                    raise
                print(
                    f"Predicted phase was not usable for waypoint 1 ({error}); "
                    "searching for the next calculated alignment window."
                )
                earliest = departure + ALIGNMENT_SEARCH_STEP_S
                continue

            state = next_state
            print(
                f"Finished waypoint {waypoint}; predicted arrival in "
                f"{max(0.0, arrival - time.monotonic()):.1f} s and minimum "
                f"clearance {clearance * 1000:.1f} mm."
            )
            tracker.observe()
            waypoint += 1

    finally:
        hardware.close()


if __name__ == "__main__":
    main()
