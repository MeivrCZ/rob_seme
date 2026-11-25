"""Puzzle D's wire and its checkpoints at a chosen granularity (Prokop's own P2.1).

The wire is a 10 x 10 mm square bar. Its centreline lies in the puzzle frame's
x-z plane (measured on models/meshes/puzzle_D.stl, the same as the drawing):

* the top bar at z = 155 mm, from the open end at x = 85 mm to x = 35 mm
  (the path starts ENTRY_OUTSIDE before the open end, outside the bar);
* the R50 quarter bend, centre (35, 105) mm, down to vertical at x = -15 mm;
* the vertical run down to z = 45 mm;
* the 45 deg jog to x = 0 at z = 30 mm;
* the stem: the path ends END_BELOW_JOG below the jog (the whole jog is
  passed). There the ring must be lower than SUCCESS_HEIGHT above the pad
  (the base's top, PAD_TOP_Z), the assignment's success condition; the
  planner checks it for the final pose (ring_top_above_pad).

Granularity: tier 0 has a checkpoint only at the segment ends (so the bend is
one move, its start and end); every next tier adds a checkpoint between each
pair of neighbours, half way along the wire. A segment has 2**tier + 1
checkpoints, the path len(SEGMENTS) * 2**tier + 1.

Checkpoint frames follow the contracts' convention: z along the direction of
travel, x along the puzzle frame's +y, y = z cross x. Where two straight
segments meet at an angle (the jog), z is the bisector of the two directions.

Example:
    ::

        path = wire_path(tier=2)        # contracts.PuzzlePath, puzzle frame
        segment_of(tier=2)              # segment of the move into each checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver.contracts import PuzzlePath

BAR_HALF = 0.005  # m, half the side of the square bar
RING_INNER_RADIUS = 0.030  # m, radius of the hoop's hole (CAD: models/meshes/hoop.stl)
# Radial clearance with the ring perpendicular to the wire: the bar's corner is
# its farthest point from the centreline.
RING_CLEARANCE = RING_INNER_RADIUS - BAR_HALF * np.sqrt(2)
ENTRY_OUTSIDE = 0.015  # m, the path starts this far before the open end
OPEN_END_X = 0.085  # m
TOP_Z = 0.155  # m
ARC_CENTRE = np.array([0.035, 0.0, 0.105])  # m
ARC_RADIUS = 0.050  # m
JOG_TOP_Z = 0.045  # m
JOG_BOTTOM = np.array([0.0, 0.0, 0.030])  # m
END_BELOW_JOG = 0.005  # m, the path ends this far into the stem
END_Z = JOG_BOTTOM[2] - END_BELOW_JOG  # m
PAD_TOP_Z = 0.0006  # m, top of the base (the pad with the markers), puzzle frame (CAD)
SUCCESS_HEIGHT = 0.04  # m, the ring must end lower than this above the pad
RING_OUTER_RADIUS = 0.0348  # m (CAD: models/meshes/hoop.stl)
RING_THICKNESS = 0.0066  # m, the ring lies on the -z side of hoop_center


@dataclass(frozen=True)
class Line:
    """Straight segment from a to b [m], puzzle frame."""

    a: np.ndarray
    b: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    def point(self, u: float) -> np.ndarray:
        """Point at the fraction u in [0, 1] of the length."""
        return self.a + u * (self.b - self.a)

    def tangent(self, u: float) -> np.ndarray:
        """Unit direction of travel (the same everywhere)."""
        d = self.b - self.a
        return d / np.linalg.norm(d)


@dataclass(frozen=True)
class Arc:
    """Circular arc in the x-z plane: centre, radius, angles from +x towards +z [rad]."""

    centre: np.ndarray
    radius: float
    angle_from: float
    angle_to: float

    @property
    def length(self) -> float:
        return float(abs(self.angle_to - self.angle_from) * self.radius)

    def _angle(self, u: float) -> float:
        return self.angle_from + u * (self.angle_to - self.angle_from)

    def point(self, u: float) -> np.ndarray:
        a = self._angle(u)
        return self.centre + self.radius * np.array([np.cos(a), 0.0, np.sin(a)])

    def tangent(self, u: float) -> np.ndarray:
        a = self._angle(u)
        sign = np.sign(self.angle_to - self.angle_from)
        return sign * np.array([-np.sin(a), 0.0, np.cos(a)])


SEGMENTS: tuple[Line | Arc, ...] = (
    Line(np.array([OPEN_END_X + ENTRY_OUTSIDE, 0.0, TOP_Z]),
         np.array([ARC_CENTRE[0], 0.0, TOP_Z])),                       # top bar
    Arc(ARC_CENTRE, ARC_RADIUS, np.pi / 2, np.pi),                     # the bend
    Line(np.array([ARC_CENTRE[0] - ARC_RADIUS, 0.0, ARC_CENTRE[2]]),
         np.array([ARC_CENTRE[0] - ARC_RADIUS, 0.0, JOG_TOP_Z])),      # vertical run
    Line(np.array([ARC_CENTRE[0] - ARC_RADIUS, 0.0, JOG_TOP_Z]), JOG_BOTTOM),  # jog
    Line(JOG_BOTTOM, np.array([0.0, 0.0, END_Z])),                     # stem
)
SEGMENT_NAMES = ("top bar", "bend", "vertical run", "jog", "stem")


def checkpoint_frame(position: np.ndarray, travel: np.ndarray) -> np.ndarray:
    """Checkpoint frame (contracts' convention) at a point, travelling along @travel."""
    z = travel / np.linalg.norm(travel)
    x = np.array([0.0, 1.0, 0.0])  # the path lies in the x-z plane
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, position
    return T


def wire_points(tier: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Checkpoint positions and directions of travel at a granularity tier.

    Args:
        tier: 0 = only the segment ends; each tier halves every interval.

    Returns:
        Positions [m] (N, 3), unit directions of travel (N, 3), and the
        segment index of each interval ending at checkpoint k (N,; -1 at 0).
    """
    assert tier >= 0, "tier must be >= 0"
    n = 2 ** tier
    points, travel, segment = [], [], []
    for k, seg in enumerate(SEGMENTS):
        for j in range(0 if k == 0 else 1, n + 1):
            u = j / n
            t = seg.tangent(u)
            if j == n and k + 1 < len(SEGMENTS):  # a joint: bisect when there is a corner
                t = t + SEGMENTS[k + 1].tangent(0.0)
            points.append(seg.point(u))
            travel.append(t / np.linalg.norm(t))
            segment.append(-1 if (k == 0 and j == 0) else k)
    return np.array(points), np.array(travel), np.array(segment)


def wire_path(tier: int = 0) -> PuzzlePath:
    """The PuzzlePath of puzzle D at a granularity tier (see the module doc)."""
    points, travel, _ = wire_points(tier)
    frames = np.array([checkpoint_frame(p, t) for p, t in zip(points, travel)])
    return PuzzlePath(frames=frames, clearance=np.full(len(frames), RING_CLEARANCE),
                      roll_limits=None, final_index=len(frames) - 1)


def segment_of(tier: int) -> np.ndarray:
    """Segment index of the move into each checkpoint (-1 for checkpoint 0)."""
    return wire_points(tier)[2]


def ring_top_above_pad(ring: np.ndarray) -> float:
    """Height of the ring's highest point above the pad [m].

    Args:
        ring: Ring-centre frame (hoop_center) in the puzzle frame; the ring is
            the annulus around its z axis, from z = -RING_THICKNESS to 0.
    """
    n = ring[:3, 2]
    top = (ring[2, 3] + max(0.0, -RING_THICKNESS * n[2])
           + RING_OUTER_RADIUS * np.sqrt(max(0.0, 1.0 - n[2] ** 2)))
    return float(top - PAD_TOP_Z)
