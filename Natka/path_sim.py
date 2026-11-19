"""Simulate the Puzzle D hoop path in a RoboDK station.

The station frame must be aligned with the puzzle-D STEP/URDF frame. Configure
the tool TCP at the hoop ring centre, with its ring axis along local Z; include
the 135 mm ring-to-mount offset in the tool definition. This script uses the
RoboDK station simulator and does not connect to or move a physical robot.

Example:
    uv run Natka/path_sim.py --robot CRS93 --frame PuzzleD --tool Hoop
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

from crs_puzzle_d_hoop_path import profile_points, tangent_angle


def simulate(
    points: Sequence[tuple[float, float, float]],
    robot_name: str,
    frame_name: str,
    tool_name: str,
    speed_mm_s: float,
) -> None:
    """Follow the checkpoints in RoboDK using linear moves."""
    try:
        from robodk.robolink import (
            ITEM_TYPE_FRAME,
            ITEM_TYPE_ROBOT,
            ITEM_TYPE_TOOL,
            RUNMODE_SIMULATE,
            Robolink,
        )
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
    parser.add_argument("--robot", required=True, help="RoboDK CRS robot item name")
    parser.add_argument("--frame", required=True, help="RoboDK frame aligned to D.step")
    parser.add_argument("--tool", required=True,
                        help="RoboDK hoop tool with TCP at ring centre")
    parser.add_argument(
        "--speed",
        type=float,
        default=20.0,
        help="simulated linear speed in mm/s (default: 20)",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")

    simulate(profile_points(), args.robot, args.frame, args.tool, args.speed)


if __name__ == "__main__":
    main()
