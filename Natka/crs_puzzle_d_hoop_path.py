"""Describe and validate a CRS hoop path along puzzle D.

The path is relative to a frame aligned with D.step: X is across the profile,
Y is the 10 mm part thickness, and Z is vertical. The hoop drawing specifies a
62.4 mm inner ring diameter and 135 mm from ring center to mount.

Run without arguments to print the checkpoints. Use --check-mesh to validate
them against the Puzzle D collision mesh. RoboDK path simulation is in
path_sim.py.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh


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

# puzzle_D.urdf is the source of truth for the frame and collision mesh.
MODELS_DIR = Path(__file__).resolve().parents[1] / "Prokop" / "models"
PUZZLE_URDF = MODELS_DIR / "puzzle_D.urdf"

# All the points are in the reference coordinate system
# might need to be transformed to the robot coordinate system before sending to the robot

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
    points.extend(
        [
            (-15.0, 0.0, 45.0),
            (0.0, 0.0, 30.0),
            (0.0, 0.0, 25.0),
            (0.0, 0.0, BASE_STOP_Z),
        ]
    )
    return points


def load_puzzle_mesh(urdf_path: str | Path = PUZZLE_URDF) -> trimesh.Trimesh:
    """Load puzzle_D.urdf's collision mesh in the URDF (puzzle) frame.

    The URDF origin is the STEP/Drawing origin and the STL is in millimetres,
    with a scale of 0.001.  This function follows both values instead of
    hard-coding transformed coordinates.
    """
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    collision = root.find(".//collision/geometry/mesh")
    if collision is None:
        raise ValueError(f"No collision mesh in {urdf_path}")
    filename = collision.get("filename")
    if not filename:
        raise ValueError(f"No mesh filename in {urdf_path}")
    mesh_path = urdf_path.parent / filename
    mesh = trimesh.load(mesh_path, force="mesh")

    scale = collision.get("scale", "1 1 1")
    sx, sy, sz = (float(v) for v in scale.split())
    mesh.apply_scale([sx, sy, sz])
    return mesh


def checkpoint_frame(position_mm: tuple[float, float, float],
                     tangent_xyz: tuple[float, float, float]) -> np.ndarray:
    """Create one D4/D5 checkpoint frame, with SI metre coordinates."""
    z = np.asarray(tangent_xyz, dtype=float)
    z /= np.linalg.norm(z)
    x = np.array([0.0, 1.0, 0.0])
    x -= np.dot(x, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = np.asarray(position_mm, dtype=float) * 0.001
    return T


def checkpoint_frames(points: Sequence[tuple[float, float, float]] | None = None
                      ) -> np.ndarray:
    """Ring-centre checkpoint frames in the puzzle frame, shape (N, 4, 4)."""
    if points is None:
        points = profile_points()
    frames = []
    for index, point in enumerate(points):
        start = points[max(0, index - 1)]
        end = points[min(len(points) - 1, index + 1)]
        tangent = (np.asarray(end) - np.asarray(start))
        frames.append(checkpoint_frame(point, tuple(tangent)))
    return np.asarray(frames)


def make_ring_mesh(inner_diameter: float = HOOP_INNER_DIAMETER,
                   outer_diameter: float = HOOP_OUTER_DIAMETER,
                   thickness: float = HOOP_THICKNESS,
                   major_segments: int = 96,
                   minor_segments: int = 12) -> trimesh.Trimesh:
    """Make the dimensional hoop ring in mm, with local z along the ring axis."""
    inner_r = inner_diameter * 0.5
    outer_r = outer_diameter * 0.5
    major_r = (inner_r + outer_r) * 0.5
    tube_r = (outer_r - inner_r) * 0.5

    # Rectangular axial faces are represented by an annular tube swept around z.
    # This is an exact enough CAD surrogate for clearance and collision tests.
    vertices = []
    faces = []
    for i in range(major_segments):
        a = 2.0 * math.pi * i / major_segments
        ca, sa = math.cos(a), math.sin(a)
        for j in range(minor_segments):
            b = 2.0 * math.pi * j / minor_segments
            cb, sb = math.cos(b), math.sin(b)
            radius = major_r + tube_r * cb
            z = (thickness * 0.5) * sb
            vertices.append([radius * ca, radius * sa, z])
    for i in range(major_segments):
        ni = (i + 1) % major_segments
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            a = i * minor_segments + j
            b = ni * minor_segments + j
            c = ni * minor_segments + nj
            d = i * minor_segments + nj
            faces.extend([[a, b, d], [b, c, d]])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces),
                           process=True)


def ring_clearance_mm(point_mm: tuple[float, float, float],
                      tangent_xyz: tuple[float, float, float],
                      puzzle: trimesh.Trimesh) -> float:
    """Minimum ring/puzzle distance at a checkpoint (zero means collision)."""
    ring = make_ring_mesh()
    ring.apply_scale(0.001)
    ring.apply_transform(checkpoint_frame(point_mm, tangent_xyz))
    ring_manager = trimesh.collision.CollisionManager()
    puzzle_manager = trimesh.collision.CollisionManager()
    ring_manager.add_object("ring", ring)
    puzzle_manager.add_object("puzzle", puzzle)
    if ring_manager.in_collision_other(puzzle_manager):
        return 0.0
    return float(ring_manager.min_distance_other(puzzle_manager) * 1000.0)


def path_clearances(points: Sequence[tuple[float, float, float]] | None = None,
                    puzzle: trimesh.Trimesh | None = None) -> np.ndarray:
    """Clearances for every ring checkpoint, in metres for PuzzlePath."""
    if points is None:
        points = profile_points()
    if puzzle is None:
        puzzle = load_puzzle_mesh()

    result = []
    ring_template = make_ring_mesh()
    p_manager = trimesh.collision.CollisionManager()
    p_manager.add_object("puzzle", puzzle)
    for index, point in enumerate(points):
        start = points[max(0, index - 1)]
        end = points[min(len(points) - 1, index + 1)]
        tangent = tuple(np.asarray(end) - np.asarray(start))
        ring = ring_template.copy()
        ring.apply_scale(0.001)
        ring.apply_transform(checkpoint_frame(point, tangent))
        r_manager = trimesh.collision.CollisionManager()
        r_manager.add_object("ring", ring)
        if r_manager.in_collision_other(p_manager):
            result.append(0.0)
        else:
            result.append(r_manager.min_distance_other(p_manager))
    return np.asarray(result)


def puzzle_path_contract(points: Sequence[tuple[float, float, float]] | None = None,
                         puzzle: trimesh.Trimesh | None = None):
    """Build the shared-contract PuzzlePath without copying its conventions."""
    import sys
    prokop_dir = MODELS_DIR.parent
    if str(prokop_dir) not in sys.path:
        sys.path.insert(0, str(prokop_dir))
    from solver.contracts import PuzzlePath

    points = profile_points() if points is None else list(points)
    final_index = len(points) - 1
    frames = checkpoint_frames(points)
    clearances = path_clearances(points, puzzle)
    return PuzzlePath(frames=frames, clearance=clearances,
                      roll_limits=None, final_index=final_index)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=str(PUZZLE_URDF),
                        help="puzzle URDF used for collision-mesh recreation")
    parser.add_argument("--check-mesh", action="store_true",
                        help="check every ring checkpoint against the puzzle mesh")
    parser.add_argument("--json", action="store_true",
                        help="print checkpoint frames and clearances as JSON")
    args = parser.parse_args()

    points = profile_points()
    print_path(points)
    if args.check_mesh or args.json:
        puzzle = load_puzzle_mesh(args.urdf)
        frames = checkpoint_frames(points)
        clearances = path_clearances(points, puzzle)
        print("\nPuzzle-frame checkpoints (metres):")
        for i, (frame, clearance) in enumerate(zip(frames, clearances)):
            xyz = frame[:3, 3]
            print(f"{i:5d}  xyz={xyz[0]: .6f} {xyz[1]: .6f} {xyz[2]: .6f}"
                  f"  clearance={clearance * 1000:7.2f} mm")
        if np.any(clearances <= 0.0):
            print("WARNING: ring intersection at checkpoint(s): "
                  + ", ".join(str(i) for i, d in enumerate(clearances) if d <= 0.0))
        if args.json:
            import json
            print(json.dumps({
                "urdf": str(Path(args.urdf)),
                "frames": frames.tolist(),
                "clearance_m": clearances.tolist(),
                "final_index": len(points) - 1,
            }, indent=2))


if __name__ == "__main__":
    main()