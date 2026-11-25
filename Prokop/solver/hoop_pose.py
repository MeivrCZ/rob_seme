"""Hoop targets: where the flange must be to put the ring centre on a checkpoint.

A checkpoint fixes only the ring centre (hoop.urdf's hoop_center). Its
orientation is free on a grid (orientation_grid): the ring axis' tilt away from
the direction of travel (in any direction) and the roll about the axis, for
one leading side of the hoop: face -1 = side B (axis against the travel, as
contract C13), face +1 = side A. candidates() gives every IK solution within
the joint limits over such a grid.

Example:
    ::

        F = flange_target(checkpoint, roll=0.3, face=-1, tilt=0.5, direction=1.0)
        qs, orients = candidates(kin, checkpoint, orientation_grid(), face=-1)
"""

from __future__ import annotations

import functools

import numpy as np

from placement import HOOP_MOUNT, MODELS, load_urdf

ROLL_STEP = np.deg2rad(15.0)  # rad, hoop roll grid
# Tilt of the ring axis away from the direction of travel, and the directions
# it may tilt in. A tilted ring's opening seen along the wire narrows with
# cos(tilt); past ~75 deg the 10 mm bar no longer fits (the collision check
# decides). The two faces stay separate: turning over would cut the bar.
TILTS = np.deg2rad([0.0, 15.0, 30.0, 45.0, 60.0])
TILT_DIRECTION_STEP = np.deg2rad(45.0)
FACES = (-1, 1)  # leading side of the hoop: -1 = side B (C13), +1 = side A
FACE_NAMES = {-1: "B", 1: "A"}


@functools.cache
def ring_in_flange() -> np.ndarray:
    """Ring centre frame (hoop.urdf's hoop_center) in the flange frame."""
    robot = load_urdf(MODELS / "crs93.urdf")
    hoop = load_urdf(MODELS / "hoop.urdf")
    return (robot.get_transform("adapter_front", "flange") @ HOOP_MOUNT
            @ hoop.get_transform("hoop_center", "hoop_link"))


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def orientation_grid(roll_step: float = ROLL_STEP, tilts: np.ndarray = TILTS,
                     direction_step: float = TILT_DIRECTION_STEP) -> np.ndarray:
    """Ring orientations relative to a checkpoint: rows (roll, tilt, tilt direction) [rad]."""
    rows = []
    for tilt in tilts:
        directions = [0.0] if tilt == 0 else np.arange(0.0, 2 * np.pi, direction_step)
        for direction in directions:
            for roll in np.arange(-np.pi, np.pi, roll_step):
                rows.append((roll, tilt, direction))
    return np.array(rows)


def flange_target(checkpoint: np.ndarray, roll: float, face: int = -1, tilt: float = 0.0,
                  direction: float = 0.0) -> np.ndarray:
    """Flange pose that puts the ring centre on a checkpoint (robot frame).

    Args:
        checkpoint: Checkpoint frame (z = direction of travel).
        roll: Rotation of the ring about its own axis [rad].
        face: -1 side B leads (hoop -z along travel, C13); +1 side A leads.
        tilt: Angle between the ring axis and the (signed) direction of travel [rad].
        direction: Direction in the ring plane the axis tilts towards [rad].
    """
    flip = _rx(np.pi) if face == -1 else np.eye(4)
    tilted = _rz(direction) @ _rx(tilt) @ _rz(-direction)
    return checkpoint @ flip @ tilted @ _rz(roll) @ np.linalg.inv(ring_in_flange())


def candidates(kin, checkpoint: np.ndarray, orientations: np.ndarray, face: int = -1
               ) -> tuple[np.ndarray, np.ndarray]:
    """All IK solutions within the joint limits over an orientation grid.

    Returns:
        Configurations (M, 6) [rad] and their orientations (M, 3) (orientation_grid rows).
    """
    qs, os_ = [], []
    for o in orientations:
        roll, tilt, direction = (float(x) for x in o)
        for q in kin.ik(flange_target(checkpoint, roll, face, tilt, direction)):
            if kin.in_limits(q):
                qs.append(np.asarray(q, dtype=float))
                os_.append(o)
    return np.array(qs).reshape(-1, 6), np.array(os_).reshape(-1, 3)
