"""Describe and simulate a CRS hoop path along puzzle D in RoboDK.

The path is relative to a RoboDK frame aligned with D.step: X is across the
profile, Y is the 10 mm part thickness, and Z is vertical. The hoop drawing
specifies a 62.4 mm inner ring diameter and 135 mm from ring center to mount.
Define the RoboDK tool with its TCP at the ring center and its ring axis along
the tool's local Z; include the 135 mm mount offset in that tool definition.

Run without arguments to print the waypoints. Use --simulate with RoboDK
item names to run the path in the RoboDK station simulator. This does not
connect to or move a physical robot.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence


# Puzzle D centerline, in millimeters, measured from its STEP/drawing profile.
PROFILE_START = (100.0, 0.0, 155.0)  # 15 mm past the open end for entry
TOP_END = (85.0, 0.0, 155.0)
ARC_CENTER_X = 35.0
ARC_CENTER_Z = 105.0
ARC_RADIUS = 50.0
ARC_SEGMENTS = 18
BASE_STOP_Z = 20.0  # Keep the hoop clear of the mounting foot.
HOOP_INNER_DIAMETER = 62.4
HOOP_OUTER_DIAMETER = 67.2
HOOP_THICKNESS = 6.6
HOOP_RING_TO_MOUNT_DISTANCE = 135.0


def profile_points() -> list[tuple[float, float, float]]:
    """Return ring-center points from D's open end down toward its base."""
    points = [PROFILE_START, TOP_END, (ARC_CENTER_X, 0.0, ARC_CENTER_Z + ARC_RADIUS)]

    # Follow the R50 quarter bend from the top tangent to the vertical tangent.
    for index in range(1, ARC_SEGMENTS + 1):
        angle = math.pi / 2 + (math.pi / 2) * index / ARC_SEGMENTS
        points.append(
            (
                ARC_CENTER_X + ARC_RADIUS * math.cos(angle),
                0.0,
                ARC_CENTER_Z + ARC_RADIUS * math.sin(angle),
            )
        )

    # The lower 45-degree jog and stem follow the dimensions on drawing page 4.
    points.extend([(-15.0, 0.0, 45.0), (0.0, 0.0, 30.0), (0.0, 0.0, BASE_STOP_Z)])
    return points


def tangent_angle(points: Sequence[tuple[float, float, float]], index: int) -> float:
    """Return the RoboDK Y rotation that points tool Z along the path tangent."""
    start = points[max(0, index - 1)]
    end = points[min(len(points) - 1, index + 1)]
    delta_x = end[0] - start[0]
    delta_z = end[2] - start[2]
    return math.degrees(math.atan2(delta_x, delta_z))


def print_path(points: Sequence[tuple[float, float, float]]) -> None:
    print("Index     X mm     Y mm     Z mm   Tool-Z angle deg")
    for index, (x, y, z) in enumerate(points):
        angle = tangent_angle(points, index)
        print(f"{index:5d} {x:9.2f} {y:8.2f} {z:8.2f} {angle:16.2f}")


def simulate(
    points: Sequence[tuple[float, float, float]],
    robot_name: str,
    frame_name: str,
    tool_name: str,
    speed_mm_s: float,
) -> None:
    try:
        from robodk.robolink import ITEM_TYPE_FRAME, ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, RUNMODE_SIMULATE, Robolink
        from robodk.robomath import roty, transl
    except ImportError as error:
        raise SystemExit(
            "RoboDK's Python API is required for simulation. Run this script "
            "with RoboDK's Python environment or install its API package."
        ) from error

    station = Robolink()
    robot = station.Item(robot_name, ITEM_TYPE_ROBOT)
    frame = station.Item(frame_name, ITEM_TYPE_FRAME)
    tool = station.Item(tool_name, ITEM_TYPE_TOOL)
    for name, item in ((robot_name, robot), (frame_name, frame), (tool_name, tool)):
        if not item.Valid():
            raise SystemExit(f"RoboDK station item not found: {name}")

    station.setRunMode(RUNMODE_SIMULATE)
    robot.setPoseFrame(frame)
    robot.setPoseTool(tool)
    robot.setSpeed(speed_mm_s)

    def target(index: int):
        x, y, z = points[index]
        angle = math.radians(tangent_angle(points, index))
        return transl(x, y, z) * roty(angle)

    robot.MoveJ(target(0))
    for index in range(1, len(points)):
        robot.MoveL(target(index))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run the path in the RoboDK station simulator",
    )
    parser.add_argument("--robot", help="RoboDK CRS robot item name")
    parser.add_argument("--frame", help="RoboDK frame aligned to D.step")
    parser.add_argument("--tool", help="RoboDK hoop tool with TCP at ring center")
    parser.add_argument(
        "--speed",
        type=float,
        default=20.0,
        help="simulated linear speed in mm/s (default: 20)",
    )
    args = parser.parse_args()

    points = profile_points()
    print_path(points)
    if args.simulate:
        missing = [
            option
            for option, value in (
                ("--robot", args.robot),
                ("--frame", args.frame),
                ("--tool", args.tool),
            )
            if not value
        ]
        if missing:
            parser.error("--simulate requires " + ", ".join(missing))
        if args.speed <= 0:
            parser.error("--speed must be greater than zero")
        simulate(points, args.robot, args.frame, args.tool, args.speed)


if __name__ == "__main__":
    main()