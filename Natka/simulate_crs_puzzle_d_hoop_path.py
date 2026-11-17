"""Simulate the CRS93 passing its hoop through puzzle D.

Run from the ``rob_seme`` project directory::

    uv run Natka/simulate_crs_puzzle_d_hoop_path.py

Open the printed viewer URL to watch. Add ``--keep`` to leave the viewer open
after the path finishes (Ctrl+C to quit). Collision monitoring is off by
default because the saved centreline path has not been validated against the
full hoop mount mesh; use ``--check-collisions`` to enable the simulator's
continuous collision monitor. Hoop-orbit candidates are always preflighted
against the scene meshes. Use ``--circle-hoop`` to run the fixed-center hoop
orbit instead of the puzzle path; ``--circle-sweep`` sets its angle in degrees.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

PROKOP = Path(__file__).resolve().parents[1] / "Prokop"
if str(PROKOP) not in sys.path:
    sys.path.insert(0, str(PROKOP))

from crs_puzzle_d_hoop_path import (  # noqa: E402
    HOOP_INNER_DIAMETER,
    HOOP_RING_TO_MOUNT_DISTANCE,
    profile_points,
    tangent_angle,
)
from placement import (  # noqa: E402
    HOOP_MOUNT,
    MIN_CLEARANCE,
    STABILITY_MARGIN,
    WORK_AREA_XY,
    Placement,
    in_cage,
)
from simulation import CRS93Sim  # noqa: E402


MAX_CENTER_DEVIATION_MM = HOOP_INNER_DIAMETER / 2
MAX_RECOVERY_CENTER_OFFSET_MM = MAX_CENTER_DEVIATION_MM
MAX_NORMAL_ERROR_DEG = 10.0
MAX_RECOVERY_TILT_DEG = 70.0
MAX_JOINT_STEP_DEG = 0.5
CIRCLE_CENTER_TOLERANCE_MM = 1.0
BASE_PARALLEL_WAYPOINT = 23


class HoopOrbitPlanningError(RuntimeError):
    """Raised when an orbit cannot satisfy its kinematic and motion limits."""


class IncompletePathPlanning(RuntimeError):
    """Raised when planning stops after producing a valid route prefix."""

    def __init__(self, waypoint: int, path: Sequence[np.ndarray]) -> None:
        super().__init__(f"Path planning stopped at waypoint {waypoint}.")
        self.waypoint = waypoint
        self.path = list(path)


class StartingWaypointUnreachable(RuntimeError):
    """Raised when no reachable hoop-center position is found around waypoint 0."""


@dataclass
class PathPlanningState:
    """Orientation recovery state retained between incremental planning calls."""

    roll_degrees: float = 0.0
    tilt_operations: list[tuple[int, float]] = field(default_factory=list)
    previous_sign: int = 0


def farthest_viable_placement(robot: CRS93Sim) -> Placement:
    """Choose the farthest valid work-area corner, breaking ties by clearance."""
    task = robot.viewer.task
    candidates = []
    for x in WORK_AREA_XY[0]:
        for y in WORK_AREA_XY[1]:
            placement = Placement(
                np.array([x, y], dtype=float),
                np.zeros(2),
                math.pi,
            )
            if not in_cage(placement.turntable_xy, task.turntable_radius):
                continue
            if task.stability_margin(placement) < STABILITY_MARGIN:
                continue
            clearance = task.clearance_to_robot(placement)
            if clearance < MIN_CLEARANCE:
                continue
            candidates.append(
                (
                    float(np.linalg.norm(placement.turntable_xy)),
                    clearance,
                    placement,
                )
            )
    if not candidates:
        raise RuntimeError("No viable work-area corner placement was found.")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def waypoint_zero_reachable_placement(
    robot: CRS93Sim,
    turntable_xy: np.ndarray | None = None,
) -> Placement:
    """Find a viable placement with reachable waypoint 0.

    If ``turntable_xy`` is provided, only that camera-visible location is
    considered; otherwise, the work area is searched.
    """
    task = robot.viewer.task
    points = profile_points()
    if turntable_xy is not None:
        requested_xy = np.asarray(turntable_xy, dtype=float)
        if requested_xy.shape != (2,):
            raise ValueError("turntable_xy must contain exactly two coordinates.")
        if not np.all(
            (requested_xy >= WORK_AREA_XY[:, 0])
            & (requested_xy <= WORK_AREA_XY[:, 1])
        ):
            raise ValueError(
                f"Requested placement {requested_xy.tolist()} is outside "
                "the camera-visible work area."
            )
        turntable_positions = [requested_xy]
    else:
        interval = 0.025
        x_values = np.linspace(
            WORK_AREA_XY[0, 0],
            WORK_AREA_XY[0, 1],
            math.ceil(np.ptp(WORK_AREA_XY[0]) / interval) + 1,
        )
        y_values = np.linspace(
            WORK_AREA_XY[1, 0],
            WORK_AREA_XY[1, 1],
            math.ceil(np.ptp(WORK_AREA_XY[1]) / interval) + 1,
        )
        turntable_positions = sorted(
            (np.array([x, y], dtype=float) for x in x_values for y in y_values),
            key=lambda xy: (xy[0], abs(xy[1]), xy[1]),
        )
    base_yaws = (math.pi, 0.0, math.pi / 2.0, -math.pi / 2.0)
    waypoint_zero_reachable = 0
    clearance_rejected = 0

    for turntable_xy in turntable_positions:
        for base_yaw in base_yaws:
            placement = Placement(turntable_xy, np.zeros(2), base_yaw)
            if (
                not in_cage(placement.turntable_xy, task.turntable_radius)
                or task.stability_margin(placement) < STABILITY_MARGIN
            ):
                continue

            robot.viewer.set_placement(placement)
            try:
                solve_path(
                    robot,
                    flange_targets(robot, points),
                    points,
                    start_index=0,
                    end_index=1,
                )
            except StartingWaypointUnreachable:
                continue

            waypoint_zero_reachable += 1
            clearance = task.clearance_to_robot(placement)
            if clearance < MIN_CLEARANCE:
                clearance_rejected += 1
                continue

            print(
                "Selected a viable placement with reachable waypoint 0: "
                f"xy=({turntable_xy[0]:.3f}, {turntable_xy[1]:.3f}) m, "
                f"base yaw={math.degrees(base_yaw):.0f} degrees, "
                f"minimum clearance={clearance * 1000.0:.1f} mm."
            )
            return placement

    requested_location = (
        f" at {turntable_positions[0].tolist()}"
        if turntable_xy is not None
        else ""
    )
    raise RuntimeError(
        "No viable camera-visible placement with a reachable waypoint 0 was "
        f"found{requested_location} after checking "
        f"{len(turntable_positions) * len(base_yaws)} position and orientation "
        f"combinations ({waypoint_zero_reachable} passed waypoint-0 IK, "
        f"{clearance_rejected} failed the clearance requirement)."
    )


def rotation_x(angle: float) -> np.ndarray:
    """Return a homogeneous rotation about X."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, c, -s, 0.0],
         [0.0, s, c, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )


def rotation_y(angle: float) -> np.ndarray:
    """Return a homogeneous rotation about Y."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-s, 0.0, c, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )


def rotation_z(angle: float) -> np.ndarray:
    """Return a homogeneous rotation about Z."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, -s, 0.0, 0.0],
         [s, c, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )


def puzzle_transform(robot: CRS93Sim) -> np.ndarray:
    """Return puzzle D's current pose in the CRS93 base frame."""
    viewer = robot.viewer
    placement = viewer.placement
    angle = viewer.turntable_angle(robot._now())
    return (
        placement.turntable_T()
        @ rotation_z(angle)
        @ viewer.task.T_top
        @ placement.base_T_top()
        @ viewer.task.T_mount
    )


def flange_to_hoop_root(robot: CRS93Sim) -> np.ndarray:
    """Return the fixed flange-to-hoop-root transform used by the scene model."""
    viewer = robot.viewer
    viewer.robot_urdf.update_cfg(robot.q_home)
    T_base_flange = robot.fk(robot.q_home)
    T_base_adapter = viewer.robot_urdf.get_transform(
        "adapter_front", viewer.robot_urdf.base_link
    )
    return (
        np.linalg.inv(T_base_flange)
        @ T_base_adapter
        @ HOOP_MOUNT
    )


def flange_targets(
    robot: CRS93Sim,
    points: Sequence[tuple[float, float, float]],
) -> list[list[np.ndarray]]:
    """Make the hoop's face perpendicular to the trajectory at every waypoint.

    The hoop model's ring normal is its local Z axis. Orient that axis along the
    path tangent, then position the hoop root so the model's ring centre lands
    on the requested point. The opposite normal is also offered to IK by
    flipping about the local mount axis, preserving the ring center.
    """
    T_base_puzzle = puzzle_transform(robot)
    T_flange_root = flange_to_hoop_root(robot)
    T_root_flange = np.linalg.inv(T_flange_root)
    ring_center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    targets = []

    for index, (x, y, z) in enumerate(points):
        T_puzzle_root = rotation_y(math.radians(tangent_angle(points, index)))
        if index == BASE_PARALLEL_WAYPOINT:
            T_base_root_rotation = (
                T_base_puzzle[:3, :3] @ T_puzzle_root[:3, :3]
            )
            ring_normal_base = np.array([0.0, 0.0, -1.0])
            ring_x_base = T_base_root_rotation[:, 0]
            ring_x_base -= np.dot(ring_x_base, ring_normal_base) * ring_normal_base
            ring_x_base /= np.linalg.norm(ring_x_base)
            ring_y_base = np.cross(ring_normal_base, ring_x_base)
            T_base_root_rotation = np.column_stack(
                (ring_x_base, ring_y_base, ring_normal_base)
            )
            T_puzzle_root[:3, :3] = (
                T_base_puzzle[:3, :3].T @ T_base_root_rotation
            )
        ring_center = np.array([x, y, z]) / 1000.0
        waypoint_targets = []
        for axis_sign in (1.0, -1.0):
            T_oriented_root = T_puzzle_root.copy()
            if axis_sign < 0:
                T_oriented_root = T_oriented_root @ rotation_y(math.pi)
            T_oriented_root[:3, 3] = (
                ring_center - T_oriented_root[:3, :3] @ ring_center_from_root
            )
            waypoint_targets.append(
                T_base_puzzle @ T_oriented_root @ T_root_flange
            )
        targets.append(waypoint_targets)

    return targets


def rolled_targets(
    robot: CRS93Sim,
    targets: Sequence[np.ndarray],
    roll_degrees: float,
    tilt_transform: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Apply a hoop-axis roll while keeping each target ring center fixed."""
    T_flange_root = flange_to_hoop_root(robot)
    T_root_flange = np.linalg.inv(T_flange_root    )
    roll = rotation_z(math.radians(roll_degrees))
    tilt = np.eye(4) if tilt_transform is None else tilt_transform
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    rolled = []
    for target in targets:
        T_base_root = target @ T_flange_root
        center_base = (
            T_base_root[:3, 3]
            + T_base_root[:3, :3] @ center_from_root
        )
        T_base_root = T_base_root @ roll @ tilt
        T_base_root[:3, 3] = (
            center_base - T_base_root[:3, :3] @ center_from_root
        )
        rolled.append(T_base_root @ T_root_flange)
    return rolled


def segment_within_route_limits(
    robot: CRS93Sim,
    q_start: np.ndarray,
    q_end: np.ndarray,
    points: Sequence[tuple[float, float, float]],
    max_normal_error_deg: float = MAX_NORMAL_ERROR_DEG,
) -> bool:
    """Check that an interpolated joint segment stays in the route tube."""
    T_base_puzzle = puzzle_transform(robot)
    T_puzzle_base = np.linalg.inv(T_base_puzzle)
    T_flange_root = flange_to_hoop_root(robot)
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    route = np.asarray(points, dtype=float) / 1000.0
    tangent_angles = np.unwrap(
        np.radians([tangent_angle(points, index) for index in range(len(points))])
    )
    q_delta = q_end - q_start
    sample_count = max(
        1,
        math.ceil(
            float(np.max(np.abs(q_delta))) / math.radians(MAX_JOINT_STEP_DEG)
        ),
    )

    for sample in range(sample_count + 1):
        fraction = sample / sample_count
        T_base_root = robot.fk(q_start + fraction * q_delta) @ T_flange_root
        center_base = (
            T_base_root[:3, 3]
            + T_base_root[:3, :3] @ center_from_root
        )
        center_puzzle = (T_puzzle_base @ np.append(center_base, 1.0))[:3]

        closest_distance = math.inf
        closest_segment = 0
        closest_fraction = 0.0
        for index, (start, end) in enumerate(zip(route, route[1:])):
            segment = end - start
            segment_fraction = float(
                np.clip(
                    np.dot(center_puzzle - start, segment)
                    / np.dot(segment, segment),
                    0.0,
                    1.0,
                )
            )
            distance = float(
                np.linalg.norm(center_puzzle - (start + segment_fraction * segment))
            )
            if distance < closest_distance:
                closest_distance = distance
                closest_segment = index
                closest_fraction = segment_fraction

        if closest_distance > MAX_CENTER_DEVIATION_MM / 1000.0:
            return False

        route_angle = (
            tangent_angles[closest_segment]
            + closest_fraction
            * (tangent_angles[closest_segment + 1] - tangent_angles[closest_segment])
        )
        route_tangent = np.array(
            [math.sin(route_angle), 0.0, math.cos(route_angle)]
        )
        normal_puzzle = (T_puzzle_base @ T_base_root)[:3, 2]
        normal_error = math.degrees(
            math.acos(
                float(np.clip(abs(np.dot(normal_puzzle, route_tangent)), 0.0, 1.0))
            )
        )
        if normal_error > max_normal_error_deg:
            return False

    return True


def flange_target_at(
    robot: CRS93Sim,
    point: np.ndarray,
    tangent: float,
    roll_degrees: float,
    sign: int,
    tilt_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Build a flange target at a route point with a fixed-center hoop roll."""
    T_base_puzzle = puzzle_transform(robot)
    T_root_flange = np.linalg.inv(flange_to_hoop_root(robot))
    T_puzzle_root = rotation_y(tangent)
    if sign:
        T_puzzle_root = T_puzzle_root @ rotation_x(math.pi)
    T_puzzle_root = T_puzzle_root @ rotation_z(math.radians(roll_degrees))
    if tilt_transform is not None:
        T_puzzle_root = T_puzzle_root @ tilt_transform
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    center = point / 1000.0
    T_puzzle_root[:3, 3] = (
        center - T_puzzle_root[:3, :3] @ center_from_root
    )
    return T_base_puzzle @ T_puzzle_root @ T_root_flange


def cartesian_continuation(
    robot: CRS93Sim,
    q_start: np.ndarray,
    points: Sequence[tuple[float, float, float]],
    index: int,
    roll_degrees: float,
    sign: int,
    tilt_transform: np.ndarray | None = None,
    endpoint_offset_mm: np.ndarray | None = None,
) -> list[np.ndarray] | None:
    """Find a seeded-IK continuation, optionally returning to the route center."""
    T_base_root = robot.fk(q_start) @ flange_to_hoop_root(robot)
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    center_base = (
        T_base_root[:3, 3]
        + T_base_root[:3, :3] @ center_from_root
    )
    start = (np.linalg.inv(puzzle_transform(robot)) @ np.append(center_base, 1.0))[:3]
    start *= 1000.0
    end = np.asarray(points[index], dtype=float)
    if endpoint_offset_mm is not None:
        end = end + endpoint_offset_mm
    start_angle = math.radians(tangent_angle(points, index - 1))
    end_angle = math.radians(tangent_angle(points, index))
    end_angle += 2.0 * math.pi * round((start_angle - end_angle) / (2.0 * math.pi))
    sample_count = max(2, math.ceil(float(np.linalg.norm(end - start))))
    path = [q_start]
    q_previous = q_start

    for sample in range(1, sample_count + 1):
        fraction = sample / sample_count
        point = start + fraction * (end - start)
        tangent = start_angle + fraction * (end_angle - start_angle)
        target = flange_target_at(
            robot,
            point,
            tangent,
            roll_degrees,
            sign,
            tilt_transform,
        )
        solutions = [
            q for q in robot.ik(target) if robot.in_limits(q)
        ]
        solutions = [
            q
            for q in solutions
            if segment_within_route_limits(
                robot,
                q_previous,
                q,
                points,
                (
                    MAX_RECOVERY_TILT_DEG
                    if tilt_transform is not None
                    else MAX_NORMAL_ERROR_DEG
                ),
            )
        ]
        if not solutions:
            return None
        q_previous = min(
            solutions,
            key=lambda q: np.linalg.norm(q - q_previous),
        )
        path.append(q_previous)

    return path


def recovery_center_offsets(
    points: Sequence[tuple[float, float, float]],
    index: int,
) -> list[np.ndarray]:
    """Sample center offsets up to the recovery limit, normal to the route."""
    start = np.asarray(points[index - 1], dtype=float)
    end = np.asarray(points[index], dtype=float)
    tangent = end - start
    tangent /= np.linalg.norm(tangent)
    across = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangent, across)
    normal /= np.linalg.norm(normal)
    offsets = [np.zeros(3)]
    radii = list(np.arange(2.0, MAX_RECOVERY_CENTER_OFFSET_MM, 2.0))
    radii.append(MAX_RECOVERY_CENTER_OFFSET_MM)
    for radius in radii:
        for angle in range(0, 360, 15):
            theta = math.radians(angle)
            offsets.append(
                radius * (math.cos(theta) * across + math.sin(theta) * normal)
            )
    return offsets


def restore_route_alignment(
    robot: CRS93Sim,
    q_start: np.ndarray,
    path: list[np.ndarray],
    tilt_operations: list[tuple[int, float]],
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Try to restore the route-normal alignment at a fixed ring center."""
    if not tilt_operations:
        return q_start, []
    restoration_path = [q_start]
    q_previous = q_start
    try:
        for axis, angle in reversed(tilt_operations):
            segment = plan_hoop_tilt(robot, q_previous, -angle, axis)
            restoration_path.extend(segment[1:])
            q_previous = segment[-1]
    except HoopOrbitPlanningError:
        return q_start, tilt_operations
    path.extend(restoration_path[1:])
    return q_previous, []


def compose_tilt_transform(
    operations: Sequence[tuple[int, float]],
) -> np.ndarray | None:
    """Return the accumulated local-axis tilt used during recovery."""
    if not operations:
        return None
    transform = np.eye(4)
    for axis, angle in operations:
        rotation = rotation_x if axis == 0 else rotation_y
        transform = transform @ rotation(math.radians(angle))
    return transform


def starting_waypoint_ring(
    points: Sequence[tuple[float, float, float]],
) -> list[np.ndarray]:
    """Sample waypoint-0 center positions around its base-local nominal point.

    The nominal point is transformed by the current camera-estimated puzzle
    pose. Offsets span concentric rings in the plane normal to the entry path,
    within the hoop's available centerline clearance.
    """
    start = np.asarray(points[0], dtype=float)
    tangent = np.asarray(points[1], dtype=float) - start
    tangent /= np.linalg.norm(tangent)
    across = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangent, across)
    normal /= np.linalg.norm(normal)

    offsets = [np.zeros(3)]
    radii = list(np.arange(2.0, MAX_CENTER_DEVIATION_MM, 2.0))
    if not np.isclose(MAX_CENTER_DEVIATION_MM % 2.0, 0.0):
        radii.append(MAX_CENTER_DEVIATION_MM)
    for radius in radii:
        for angle in range(0, 360, 15):
            theta = math.radians(angle)
            offsets.append(
                radius
                * (math.cos(theta) * across + math.sin(theta) * normal)
            )
    return offsets


def solve_path(
    robot: CRS93Sim,
    targets: Sequence[Sequence[np.ndarray]],
    points: Sequence[tuple[float, float, float]],
    *,
    start_index: int = 0,
    end_index: int | None = None,
    q_start: np.ndarray | None = None,
    state: PathPlanningState | None = None,
) -> list[np.ndarray]:
    """Plan a route, rolling the hoop in place when a segment is infeasible."""
    end_index = len(targets) if end_index is None else end_index
    if not 0 <= start_index <= end_index <= len(targets):
        raise ValueError("Waypoint planning range is outside the target sequence.")

    planning_state = state if state is not None else PathPlanningState()
    q_previous = np.asarray(
        robot.q_home if q_start is None else q_start,
        dtype=float,
    )
    path = []
    roll_degrees = planning_state.roll_degrees
    tilt_operations = planning_state.tilt_operations.copy()
    previous_sign = planning_state.previous_sign
    circle_step_degrees = 5.0
    max_circle_steps = int(360.0 / circle_step_degrees)

    for index in range(start_index, end_index):
        alternatives = targets[index]
        if index == 0:
            previous_sign = 1
            start_offsets = starting_waypoint_ring(points)
            T_base_puzzle = puzzle_transform(robot)
            selected_start = None
            for center_offset in start_offsets:
                shifted_target = alternatives[previous_sign].copy()
                shifted_target[:3, 3] += (
                    T_base_puzzle[:3, :3] @ (center_offset / 1000.0)
                )
                candidates = []
                for candidate_roll in range(0, 360, 15):
                    target = rolled_targets(
                        robot,
                        [shifted_target],
                        candidate_roll,
                    )[0]
                    for q in robot.ik(target):
                        if robot.in_limits(q):
                            candidates.append(
                                (
                                    float(np.linalg.norm(q - q_previous)),
                                    candidate_roll,
                                    q,
                                )
                            )
                if candidates:
                    selected_start = (
                        float(np.linalg.norm(center_offset)),
                        center_offset,
                        min(candidates, key=lambda item: item[0]),
                    )
                    break

            if selected_start is None:
                raise StartingWaypointUnreachable(
                    "The exact flipped hoop face is unreachable at waypoint 0 "
                    "on the sampled starting-position rings within the "
                    f"{MAX_CENTER_DEVIATION_MM:.1f} mm route tube."
                )
            center_offset_mm, _, (_, roll_degrees, q_previous) = selected_start
            print(
                "Waypoint 0: selected the exact flipped hoop face "
                f"with a {roll_degrees:.0f}-degree in-plane roll and "
                f"{center_offset_mm:.1f} mm center offset."
            )
            path.append(q_previous)
            continue

        rolled = rolled_targets(
            robot,
            alternatives,
            roll_degrees,
            compose_tilt_transform(tilt_operations),
        )
        candidates = [
            (sign, q)
            for sign, target in enumerate(rolled)
            for q in robot.ik(target)
            if robot.in_limits(q)
            and segment_within_route_limits(
                robot,
                q_previous,
                q,
                points,
                MAX_RECOVERY_TILT_DEG if tilt_operations else MAX_NORMAL_ERROR_DEG,
            )
        ]
        if candidates:
            same_sign = [item for item in candidates if item[0] == previous_sign]
            previous_sign, q_previous = min(
                same_sign or candidates,
                key=lambda item: np.linalg.norm(item[1] - q_previous),
            )
            path.append(q_previous)
            q_previous, tilt_operations = restore_route_alignment(
                robot,
                q_previous,
                path,
                tilt_operations,
            )
            continue

        continuation = cartesian_continuation(
            robot,
            q_previous,
            points,
            index,
            roll_degrees,
            previous_sign,
            compose_tilt_transform(tilt_operations),
        )
        if continuation is not None:
            path.extend(continuation[1:])
            q_previous = continuation[-1]
            q_previous, tilt_operations = restore_route_alignment(
                robot,
                q_previous,
                path,
                tilt_operations,
            )
            continue

        orbit_candidates: list[
            tuple[float, float, float, list[np.ndarray], int, list[np.ndarray]]
        ] = []
        for direction in (1.0, -1.0):
            orbit_q = q_previous
            orbit_path: list[np.ndarray] = []
            sweep = 0.0
            minimum_clearance = math.inf
            for _ in range(max_circle_steps):
                try:
                    orbit_segment, segment_clearance = _plan_hoop_orbit(
                        robot,
                        orbit_q,
                        direction * circle_step_degrees,
                        steps_per_revolution=72,
                    )
                except HoopOrbitPlanningError:
                    break
                orbit_q = orbit_segment[-1]
                orbit_path.extend(orbit_segment[1:])
                minimum_clearance = min(minimum_clearance, segment_clearance)
                sweep += direction * circle_step_degrees
                candidate_roll = roll_degrees + sweep
                step_candidates = []
                for sign in (previous_sign, 1 - previous_sign):
                    continuation = cartesian_continuation(
                        robot,
                        orbit_q,
                        points,
                        index,
                        candidate_roll,
                        sign,
                        compose_tilt_transform(tilt_operations),
                    )
                    if continuation is not None:
                        step_candidates.append((sign, continuation))
                if step_candidates:
                    sign, continuation = min(
                        step_candidates,
                        key=lambda item: np.linalg.norm(item[1][-1] - orbit_q),
                    )
                    orbit_candidates.append(
                        (
                            minimum_clearance,
                            abs(sweep),
                            sweep,
                            orbit_path.copy(),
                            sign,
                            continuation,
                        )
                    )

        found_continuation = bool(orbit_candidates)
        if found_continuation:
            (
                minimum_clearance,
                _,
                sweep,
                orbit_path,
                previous_sign,
                continuation,
            ) = max(
                orbit_candidates,
                key=lambda candidate: (candidate[0], -candidate[1]),
            )
            path.extend(orbit_path)
            path.extend(continuation[1:])
            q_previous = continuation[-1]
            roll_degrees += sweep
            q_previous, tilt_operations = restore_route_alignment(
                robot,
                q_previous,
                path,
                tilt_operations,
            )
            direction_name = "counter-clockwise" if sweep > 0 else "clockwise"
            print(
                f"Waypoint {index}: selected a {direction_name} in-place hoop "
                f"rotation of {abs(sweep):.0f} degrees after checking both "
                f"directions through their collision-free arcs "
                f"({minimum_clearance * 1000.0:.1f} mm minimum clearance)."
            )

        if not found_continuation:
            for tilt_axis in (0, 1):
                for tilt_degrees in range(
                    int(circle_step_degrees),
                    int(MAX_RECOVERY_TILT_DEG) + 1,
                    int(circle_step_degrees),
                ):
                    for signed_tilt in (float(tilt_degrees), -float(tilt_degrees)):
                        try:
                            tilt_path = plan_hoop_tilt(
                                robot,
                                q_previous,
                                signed_tilt,
                                tilt_axis,
                            )
                        except HoopOrbitPlanningError:
                            continue

                        for sign in (previous_sign, 1 - previous_sign):
                            continuation = cartesian_continuation(
                                robot,
                                tilt_path[-1],
                                points,
                                index,
                                roll_degrees,
                                sign,
                                compose_tilt_transform(
                                    [*tilt_operations, (tilt_axis, signed_tilt)]
                                ),
                            )
                            if continuation is None:
                                continue

                            previous_sign = sign
                            q_previous = continuation[-1]
                            tilt_operations.append((tilt_axis, signed_tilt))
                            path.extend(tilt_path[1:])
                            path.extend(continuation[1:])
                            q_previous, tilt_operations = restore_route_alignment(
                                robot,
                                q_previous,
                                path,
                                tilt_operations,
                            )
                            found_continuation = True
                            axis_name = "X" if tilt_axis == 0 else "Y"
                            print(
                                f"Waypoint {index}: planned a {signed_tilt:.0f}-degree "
                                f"temporary hoop tilt about local {axis_name}."
                            )
                            break
                        if found_continuation:
                            break
                    if found_continuation:
                        break
                if found_continuation:
                    break

        if not found_continuation:
            tilt_transform = compose_tilt_transform(tilt_operations)
            for center_offset in recovery_center_offsets(points, index):
                for sign in (previous_sign, 1 - previous_sign):
                    continuation = cartesian_continuation(
                        robot,
                        q_previous,
                        points,
                        index,
                        roll_degrees,
                        sign,
                        tilt_transform,
                        center_offset,
                    )
                    if continuation is None:
                        continue

                    previous_sign = sign
                    q_previous = continuation[-1]
                    path.extend(continuation[1:])
                    q_previous, tilt_operations = restore_route_alignment(
                        robot,
                        q_previous,
                        path,
                        tilt_operations,
                    )
                    found_continuation = True
                    print(
                        f"Waypoint {index}: planned a route-tube offset of "
                        f"{np.linalg.norm(center_offset):.0f} mm; the next "
                        "continuation will return toward the saved centerline."
                    )
                    break
                if found_continuation:
                    break

        if not found_continuation:
            raise IncompletePathPlanning(index, path)

    planning_state.roll_degrees = roll_degrees
    planning_state.tilt_operations = tilt_operations
    planning_state.previous_sign = previous_sign
    return path


def validate_path(
    robot: CRS93Sim,
    joint_path: Sequence[np.ndarray],
    points: Sequence[tuple[float, float, float]],
    max_normal_error_deg: float = MAX_NORMAL_ERROR_DEG,
) -> tuple[float, float]:
    """Reject paths whose interpolated hoop center or face leaves its limits."""
    T_base_puzzle = puzzle_transform(robot)
    T_puzzle_base = np.linalg.inv(T_base_puzzle)
    T_flange_root = flange_to_hoop_root(robot)
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    route = np.asarray(points, dtype=float) / 1000.0
    tangent_angles = np.unwrap(
        np.radians([tangent_angle(points, index) for index in range(len(points))])
    )
    max_center_deviation = 0.0
    max_normal_error = 0.0

    def check_configuration(q: np.ndarray) -> None:
        nonlocal max_center_deviation, max_normal_error
        T_base_root = robot.fk(q) @ T_flange_root
        center_base = (
            T_base_root[:3, 3]
            + T_base_root[:3, :3] @ center_from_root
        )
        center_puzzle = (T_puzzle_base @ np.append(center_base, 1.0))[:3]

        closest_distance = math.inf
        closest_segment = 0
        closest_fraction = 0.0
        for index, (start, end) in enumerate(zip(route, route[1:])):
            segment = end - start
            fraction = float(
                np.clip(
                    np.dot(center_puzzle - start, segment)
                    / np.dot(segment, segment),
                    0.0,
                    1.0,
                )
            )
            distance = float(
                np.linalg.norm(center_puzzle - (start + fraction * segment))
            )
            if distance < closest_distance:
                closest_distance = distance
                closest_segment = index
                closest_fraction = fraction

        T_puzzle_root = T_puzzle_base @ T_base_root
        normal_puzzle = T_puzzle_root[:3, 2]
        angle_start = tangent_angles[closest_segment]
        angle_end = tangent_angles[closest_segment + 1]
        route_angle = angle_start + closest_fraction * (angle_end - angle_start)
        route_tangent = np.array([math.sin(route_angle), 0.0, math.cos(route_angle)])
        normal_error = math.degrees(
            math.acos(
                float(np.clip(abs(np.dot(normal_puzzle, route_tangent)), 0.0, 1.0))
            )
        )
        max_center_deviation = max(max_center_deviation, closest_distance)
        max_normal_error = max(max_normal_error, normal_error)

    if not joint_path:
        raise ValueError("The hoop path has no joint configurations to validate.")

    check_configuration(joint_path[0])
    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    for q_start, q_end in zip(joint_path, joint_path[1:]):
        q_delta = q_end - q_start
        sample_count = max(1, math.ceil(float(np.max(np.abs(q_delta))) / max_joint_step))
        for sample in range(1, sample_count + 1):
            fraction = sample / sample_count
            check_configuration(q_start + fraction * q_delta)

    center_deviation_mm = max_center_deviation * 1000.0
    if center_deviation_mm > MAX_CENTER_DEVIATION_MM:
        raise RuntimeError(
            "Path preflight failed: hoop center deviates "
            f"{center_deviation_mm:.1f} mm from the saved route "
            f"(limit {MAX_CENTER_DEVIATION_MM:.1f} mm)."
        )
    if max_normal_error > max_normal_error_deg:
        raise RuntimeError(
            "Path preflight failed: hoop-plane normal differs from the route "
            f"tangent by {max_normal_error:.1f} degrees "
            f"(limit {max_normal_error_deg:.1f} degrees)."
        )

    return center_deviation_mm, max_normal_error


def check_path_collision_clearance(
    robot: CRS93Sim,
    joint_path: Sequence[np.ndarray],
) -> float:
    """Check the robot's sampled path against the simulator's collision meshes."""
    placement = robot.viewer.placement
    turntable_angle = robot.viewer.turntable_angle(robot._now())
    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    minimum_clearance = math.inf

    def check_configuration(q: np.ndarray, sample: int, total: int) -> None:
        nonlocal minimum_clearance
        hits, clearance = robot._collider.check(q, placement, turntable_angle)
        if hits:
            pairs = ", ".join(f"{first}-{second}" for first, second in hits)
            raise HoopOrbitPlanningError(
                f"The hoop orbit would collide with {pairs} at sample "
                f"{sample}/{total}."
            )
        minimum_clearance = min(minimum_clearance, clearance)

    if not joint_path:
        raise ValueError("The collision-check path must contain a configuration.")
    total_samples = 1
    for q_start, q_end in zip(joint_path, joint_path[1:]):
        q_delta = q_end - q_start
        total_samples += max(
            1,
            math.ceil(float(np.max(np.abs(q_delta))) / max_joint_step),
        )

    sample_index = 1
    check_configuration(joint_path[0], sample_index, total_samples)
    for q_start, q_end in zip(joint_path, joint_path[1:]):
        q_delta = q_end - q_start
        sample_count = max(
            1,
            math.ceil(float(np.max(np.abs(q_delta))) / max_joint_step),
        )
        for sample in range(1, sample_count + 1):
            sample_index += 1
            check_configuration(
                q_start + (sample / sample_count) * q_delta,
                sample_index,
                total_samples,
            )
    return minimum_clearance


def _plan_hoop_orbit(
    robot: CRS93Sim,
    q_start: np.ndarray,
    sweep_degrees: float,
    steps_per_revolution: int = 72,
) -> tuple[list[np.ndarray], float]:
    """Plan a fixed-center orbit; positive angles are CCW along local +Z."""
    if not 0.0 < abs(sweep_degrees) <= 360.0:
        raise ValueError("sweep_degrees must be nonzero and at most 360 in magnitude.")
    if steps_per_revolution < 12:
        raise ValueError("steps_per_revolution must be at least 12.")
    T_flange_root = flange_to_hoop_root(robot)
    T_root_flange = np.linalg.inv(T_flange_root)
    T_base_root_start = robot.fk(q_start) @ T_flange_root
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    center_base = (
        T_base_root_start[:3, 3]
        + T_base_root_start[:3, :3] @ center_from_root
    )
    T_base_center = T_base_root_start.copy()
    T_base_center[:3, 3] = center_base
    T_center_root_start = np.linalg.inv(T_base_center) @ T_base_root_start

    joint_path = [q_start]
    q_previous = q_start
    sample_count = max(
        1,
        math.ceil(abs(sweep_degrees) * steps_per_revolution / 360.0),
    )
    for sample in range(1, sample_count + 1):
        angle = math.radians(sweep_degrees * sample / sample_count)
        T_base_root = (
            T_base_center
            @ rotation_z(angle)
            @ T_center_root_start
        )
        flange_target = T_base_root @ T_root_flange
        solutions = [
            q for q in robot.ik(flange_target) if robot.in_limits(q)
        ]
        if not solutions:
            raise HoopOrbitPlanningError(
                f"The hoop orbit is unreachable at sample {sample}/{sample_count}."
            )
        q_previous = min(
            solutions,
            key=lambda q: np.linalg.norm(q - q_previous),
        )
        joint_path.append(q_previous)

    max_center_error = 0.0
    max_normal_error = 0.0
    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    normal_start = T_base_root_start[:3, 2]
    for q_segment_start, q_segment_end in zip(joint_path, joint_path[1:]):
        q_delta = q_segment_end - q_segment_start
        samples = max(
            1,
            math.ceil(float(np.max(np.abs(q_delta))) / max_joint_step),
        )
        for sample in range(1, samples + 1):
            q = q_segment_start + (sample / samples) * q_delta
            T_base_root = robot.fk(q) @ T_flange_root
            center = (
                T_base_root[:3, 3]
                + T_base_root[:3, :3] @ center_from_root
            )
            max_center_error = max(
                max_center_error,
                float(np.linalg.norm(center - center_base)),
            )
            normal_error = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            abs(np.dot(T_base_root[:3, 2], normal_start)),
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            max_normal_error = max(max_normal_error, normal_error)

    if max_center_error * 1000.0 > CIRCLE_CENTER_TOLERANCE_MM:
        raise HoopOrbitPlanningError(
            "Hoop orbit preflight failed: ring center moves "
            f"{max_center_error * 1000.0:.2f} mm (limit "
            f"{CIRCLE_CENTER_TOLERANCE_MM:.1f} mm)."
        )
    if max_normal_error > MAX_NORMAL_ERROR_DEG:
        raise HoopOrbitPlanningError(
            "Hoop orbit preflight failed: ring normal changes by "
            f"{max_normal_error:.2f} degrees (limit "
            f"{MAX_NORMAL_ERROR_DEG:.1f} degrees)."
        )
    minimum_clearance = check_path_collision_clearance(robot, joint_path)
    return joint_path, minimum_clearance


def plan_hoop_orbit(
    robot: CRS93Sim,
    q_start: np.ndarray,
    sweep_degrees: float,
    steps_per_revolution: int = 72,
) -> list[np.ndarray]:
    """Plan a collision-free hoop-axis rotation with a fixed ring center."""
    joint_path, _ = _plan_hoop_orbit(
        robot,
        q_start,
        sweep_degrees,
        steps_per_revolution,
    )
    return joint_path


def plan_hoop_tilt(
    robot: CRS93Sim,
    q_start: np.ndarray,
    tilt_degrees: float,
    tilt_axis: int = 0,
) -> list[np.ndarray]:
    """Tilt the hoop about a ring-plane axis while fixing its center."""
    if not 0.0 < abs(tilt_degrees) <= MAX_RECOVERY_TILT_DEG:
        raise ValueError(
            "tilt_degrees must be nonzero and within the recovery tilt limit."
        )
    if tilt_axis not in (0, 1):
        raise ValueError("tilt_axis must be 0 (local X) or 1 (local Y).")
    tilt_rotation = rotation_x if tilt_axis == 0 else rotation_y
    T_flange_root = flange_to_hoop_root(robot)
    T_root_flange = np.linalg.inv(T_flange_root)
    T_base_root_start = robot.fk(q_start) @ T_flange_root
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    center_base = (
        T_base_root_start[:3, 3]
        + T_base_root_start[:3, :3] @ center_from_root
    )
    T_base_center = T_base_root_start.copy()
    T_base_center[:3, 3] = center_base
    T_center_root_start = np.linalg.inv(T_base_center) @ T_base_root_start

    sample_count = max(1, math.ceil(abs(tilt_degrees) / 5.0))
    joint_path = [q_start]
    q_previous = q_start
    for sample in range(1, sample_count + 1):
        angle = math.radians(tilt_degrees * sample / sample_count)
        T_base_root = (
            T_base_center
            @ tilt_rotation(angle)
            @ T_center_root_start
        )
        flange_target = T_base_root @ T_root_flange
        solutions = [
            q for q in robot.ik(flange_target) if robot.in_limits(q)
        ]
        if not solutions:
            raise HoopOrbitPlanningError(
                f"The hoop tilt is unreachable at sample {sample}/{sample_count}."
            )
        q_previous = min(
            solutions,
            key=lambda q: np.linalg.norm(q - q_previous),
        )
        joint_path.append(q_previous)

    max_center_error = 0.0
    max_normal_error = 0.0
    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    normal_start = T_base_root_start[:3, 2]
    for q_segment_start, q_segment_end in zip(joint_path, joint_path[1:]):
        q_delta = q_segment_end - q_segment_start
        samples = max(
            1,
            math.ceil(float(np.max(np.abs(q_delta))) / max_joint_step),
        )
        for sample in range(1, samples + 1):
            q = q_segment_start + (sample / samples) * q_delta
            T_base_root = robot.fk(q) @ T_flange_root
            center = (
                T_base_root[:3, 3]
                + T_base_root[:3, :3] @ center_from_root
            )
            max_center_error = max(
                max_center_error,
                float(np.linalg.norm(center - center_base)),
            )
            normal_error = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            abs(np.dot(T_base_root[:3, 2], normal_start)),
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            max_normal_error = max(max_normal_error, normal_error)

    if max_center_error * 1000.0 > CIRCLE_CENTER_TOLERANCE_MM:
        raise HoopOrbitPlanningError(
            "Temporary hoop tilt moves the ring center by "
            f"{max_center_error * 1000.0:.2f} mm."
        )
    if max_normal_error > MAX_RECOVERY_TILT_DEG:
        raise HoopOrbitPlanningError(
            "Temporary hoop tilt exceeds the configured normal-angle limit."
        )
    return joint_path


def circle_hoop_in_ring_plane(
    robot: CRS93Sim,
    start_flange_target: np.ndarray,
    sweep_degrees: float = 180.0,
    steps_per_revolution: int = 72,
) -> tuple[float, float]:
    """Move collision-free in-plane, choosing the direction with best clearance."""
    if not 0.0 < sweep_degrees <= 360.0:
        raise ValueError("sweep_degrees must be greater than zero and at most 360.")
    if steps_per_revolution < 12:
        raise ValueError("steps_per_revolution must be at least 12.")

    start_solutions = [
        q for q in robot.ik(start_flange_target) if robot.in_limits(q)
    ]
    if not start_solutions:
        raise RuntimeError("The hoop orbit start pose is unreachable.")
    q_start = min(
        start_solutions,
        key=lambda q: np.linalg.norm(q - robot.get_q()),
    )
    orbit_options: list[tuple[float, float, list[np.ndarray]]] = []
    rejected_directions: list[str] = []
    for direction in (1.0, -1.0):
        direction_name = "counter-clockwise" if direction > 0 else "clockwise"
        try:
            path, clearance = _plan_hoop_orbit(
                robot,
                q_start,
                direction * sweep_degrees,
                steps_per_revolution,
            )
        except HoopOrbitPlanningError as error:
            rejected_directions.append(f"{direction_name}: {error}")
            continue
        orbit_options.append((clearance, direction, path))

    if not orbit_options:
        reasons = " ".join(rejected_directions)
        raise HoopOrbitPlanningError(
            f"No collision-free direction is available for the hoop orbit. {reasons}"
        )

    minimum_clearance, direction, joint_path = max(
        orbit_options,
        key=lambda option: option[0],
    )
    direction_name = "counter-clockwise" if direction > 0 else "clockwise"
    print(
        f"Selected {direction_name} hoop rotation "
        f"({minimum_clearance * 1000.0:.1f} mm minimum clearance)."
    )

    T_flange_root = flange_to_hoop_root(robot)
    center_from_root = np.array(
        [0.0, -HOOP_RING_TO_MOUNT_DISTANCE / 1000.0, 0.0]
    )
    T_base_root_start = robot.fk(q_start) @ T_flange_root
    center_start = (
        T_base_root_start[:3, 3]
        + T_base_root_start[:3, :3] @ center_from_root
    )
    center_error = 0.0
    normal_error = 0.0
    max_joint_step = math.radians(MAX_JOINT_STEP_DEG)
    for q_segment_start, q_segment_end in zip(joint_path, joint_path[1:]):
        q_delta = q_segment_end - q_segment_start
        samples = max(
            1,
            math.ceil(float(np.max(np.abs(q_delta))) / max_joint_step),
        )
        for sample in range(1, samples + 1):
            q = q_segment_start + (sample / samples) * q_delta
            T_base_root = robot.fk(q) @ T_flange_root
            center = (
                T_base_root[:3, 3]
                + T_base_root[:3, :3] @ center_from_root
            )
            center_error = max(
                center_error,
                float(np.linalg.norm(center - center_start)),
            )
            normal_error = max(
                normal_error,
                math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                abs(
                                    np.dot(
                                        T_base_root[:3, 2],
                                        T_base_root_start[:3, 2],
                                    )
                                ),
                                0.0,
                                1.0,
                            )
                        )
                    )
                ),
            )

    robot.move_to_q(q_start)
    robot.wait_for_motion_stop()
    for q in joint_path[1:]:
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

    return center_error * 1000.0, normal_error


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
        help="enable CRS93Sim's collision monitor (may stop on hoop/puzzle contact)",
    )
    parser.add_argument(
        "--circle-hoop",
        action="store_true",
        help="move the hoop through a circular arc at the route entry",
    )
    parser.add_argument(
        "--circle-sweep",
        type=float,
        default=180.0,
        help="hoop orbit sweep in degrees (default: 180; max: 360)",
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
    placement = waypoint_zero_reachable_placement(
        robot,
        turntable_xy=np.array([0.4, 0.0]),
    )
    robot.viewer.set_placement(placement)
    print(
        "Placed the turntable, base and puzzle at "
        f"xy=({placement.turntable_xy[0]:.3f}, "
        f"{placement.turntable_xy[1]:.3f}) m; base centered on turntable, "
        f"base yaw={math.degrees(placement.base_yaw):.0f} degrees."
    )
    if not args.check_collisions:
        print(
            "The continuous collision monitor is off; hoop-orbit candidates "
            "are still preflighted against the scene meshes."
        )

    points = profile_points()
    if args.circle_hoop:
        start_target = flange_targets(robot, [points[0]])[0][0]
        try:
            center_error, normal_error = circle_hoop_in_ring_plane(
                robot,
                start_target,
                sweep_degrees=args.circle_sweep,
            )
        except HoopOrbitPlanningError as error:
            print(f"Hoop orbit skipped: {error}")
            return
        print(
            f"Hoop orbit through {args.circle_sweep:.1f} degrees complete: "
            f"ring-center drift {center_error:.2f} mm, "
            f"ring-normal change {normal_error:.2f} degrees."
        )
        return

    targets = flange_targets(robot, points)
    incomplete_at: int | None = None
    try:
        joint_path = solve_path(robot, targets, points)
    except IncompletePathPlanning as error:
        joint_path = error.path
        incomplete_at = error.waypoint

    try:
        center_deviation, normal_error = validate_path(
            robot,
            joint_path,
            points,
            max_normal_error_deg=MAX_RECOVERY_TILT_DEG,
        )
    except RuntimeError as error:
        print(f"Path preflight warning; continuing visualization: {error}")
    else:
        print(
            "Path preflight passed: "
            f"max center deviation {center_deviation:.1f} mm, "
            f"max hoop-plane error {normal_error:.1f} degrees."
        )
    print(
        "The initial repositioning from home to the route entry is not included "
        "in this preflight."
    )
    if incomplete_at is None:
        print(f"Following {len(joint_path)} hoop waypoints through puzzle D.")
    else:
        print(
            f"Following the planned simulation prefix ({len(joint_path)} "
            f"configurations); planning stopped at waypoint {incomplete_at}."
        )

    for q in joint_path:
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

    if incomplete_at is None:
        print("Puzzle D hoop path complete.")
    else:
        print("Simulation prefix complete; the full puzzle D path was not planned.")


if __name__ == "__main__":
    main()
