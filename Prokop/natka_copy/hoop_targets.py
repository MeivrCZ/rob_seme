"""P2.2: construct hoop flange targets from Puzzle D checkpoints.

Uses the SE3 conventions from the robotics lab homework: homogeneous transforms
are 4x4 matrices, and composition follows T_world_A @ T_A_B.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROKOP_DIR = PROJECT_ROOT / "Prokop"
if str(PROKOP_DIR) not in sys.path:
    sys.path.insert(0, str(PROKOP_DIR))

from placement import HOOP_MOUNT, MODELS, load_urdf


def se3_compose(T_left: np.ndarray, T_right: np.ndarray) -> np.ndarray:
    """Compose homogeneous transforms using the lab SE3 convention."""
    return _validate_transform(T_left) @ _validate_transform(T_right)


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Invert an SE3 transform without inverting an arbitrary 4x4 matrix."""
    transform = _validate_transform(T)
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def _validate_transform(T: np.ndarray) -> np.ndarray:
    transform = np.asarray(T, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("a transform must have shape (4, 4)")
    if not np.all(np.isfinite(transform)):
        raise ValueError("a transform must contain only finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("the bottom row of a homogeneous transform must be [0, 0, 0, 1]")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7):
        raise ValueError("the transform rotation block must be orthonormal")
    if abs(np.linalg.det(transform[:3, :3]) - 1.0) > 1e-6:
        raise ValueError("the transform rotation block must have determinant +1")
    return transform


def so3_rx(angle: float) -> np.ndarray:
    """SO3 rotation around the local X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def so3_rz(angle: float) -> np.ndarray:
    """SO3 rotation around the local Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def checkpoint_ring_target(checkpoint_T: np.ndarray, roll: float = 0.0,
                           tilt: float = 0.0) -> np.ndarray:
    """Construct a desired hoop ring-centre frame for one checkpoint.

    The ring frame origin is the ring centre. Its z axis is the checkpoint
    tangent, after tilt around checkpoint x. Roll is applied around that ring
    axis and cannot move the ring centre.
    """
    checkpoint = _validate_transform(checkpoint_T)
    if not np.isfinite(roll) or not np.isfinite(tilt):
        raise ValueError("roll and tilt must be finite")

    ring = np.eye(4)
    # PROKOP COPY: side B of the hoop leads (contracts C13): the
    # ring frame is the checkpoint frame turned by pi about its x, then tilt, roll.
    ring[:3, :3] = checkpoint[:3, :3] @ so3_rx(np.pi) @ so3_rx(tilt) @ so3_rz(roll)
    ring[:3, 3] = checkpoint[:3, 3]
    return ring


# PROKOP COPY: cached; the URDFs were parsed on every call (~86 % of the planner's time).
@functools.cache
def _fixed_mount_frames() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load flange-adapter, hoop-mount, and hoop-link-ring frames."""
    robot = load_urdf(MODELS / "crs93.urdf")
    hoop = load_urdf(MODELS / "hoop.urdf")
    flange_to_adapter = robot.get_transform("adapter_front", "flange")
    hoop_link_to_ring = hoop.get_transform("hoop_center", hoop.base_link)
    hoop_mount = np.asarray(HOOP_MOUNT, dtype=float)
    return (_validate_transform(flange_to_adapter),
            _validate_transform(hoop_mount),
            _validate_transform(hoop_link_to_ring))


def hoop_mount_target(checkpoint_T: np.ndarray, roll: float = 0.0,
                      tilt: float = 0.0) -> np.ndarray:
    """Construct the adapter-front pose for a hoop ring checkpoint."""
    ring_target = checkpoint_ring_target(checkpoint_T, roll=roll, tilt=tilt)
    _, hoop_mount, hoop_link_to_ring = _fixed_mount_frames()
    adapter_to_ring = se3_compose(hoop_mount, hoop_link_to_ring)
    return se3_compose(ring_target, se3_inverse(adapter_to_ring))


def flange_target(checkpoint_T: np.ndarray, roll: float = 0.0,
                  tilt: float = 0.0) -> np.ndarray:
    """Construct the flange pose whose mounted hoop reaches a checkpoint.

    The result is a 4x4 flange transform in the robot base frame and can be
    passed directly to ctu_crs.CRS93.ik.
    """
    adapter_target = hoop_mount_target(checkpoint_T, roll=roll, tilt=tilt)
    flange_to_adapter, _, _ = _fixed_mount_frames()
    return se3_compose(adapter_target, se3_inverse(flange_to_adapter))


def hoop_ring_pose_from_flange(flange_T: np.ndarray) -> np.ndarray:
    """Get the mounted hoop ring-centre frame for a flange pose."""
    flange = _validate_transform(flange_T)
    flange_to_adapter, hoop_mount, hoop_link_to_ring = _fixed_mount_frames()
    adapter = se3_compose(flange, flange_to_adapter)
    return se3_compose(se3_compose(adapter, hoop_mount), hoop_link_to_ring)
