"""Shared contracts of the solver: the data passed between its parts, and fakes.

Every part of the pipeline (docs/SOLVER_PLAN.md) exchanges only these types.
Changing a field breaks the others' code: agree on changes first and announce
them in Prokop/NOTIFY.md.

Conventions (as everywhere in Prokop/):

* Units SI (m, rad, s). Frame: the ctu_crs robot base frame (x forward, y left,
  z up, table at z = cell.TABLE_Z).
* Time: time.monotonic() seconds on the robot computer. A camera measurement
  carries the time the image was TAKEN: grab_image return time - camera delay.
* Turntable angle theta: rotation of the turntable top about its vertical axis,
  counter-clockwise seen from above positive (in the simulation the same as
  SceneViewer.turntable_angle). theta(t) = theta_ref + direction * omega * (t - t_ref).
* Puzzle frame: the puzzle URDF origin (= the D.step origin), seated in the
  base's groove. The hoop path of puzzle D lies in its x-z plane.
* Checkpoint frame: origin on the ring centre's path, z along the direction of
  travel, x along the puzzle frame's +y (perpendicular to the path plane),
  y = z cross x. The hoop roll is measured about z from this x.

The fakes (example_model, fake_measurements, example_path, example_plan) let
every part be developed and tested before the parts it depends on exist.

Example:
    ::

        import sys; sys.path.insert(0, "<repo>/Prokop")
        from solver.contracts import BaseMeasurement, TurntableModel, Plan
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SPIN_PERIOD = 28.0  # s per revolution at the slowest speed (docs/MEASUREMENTS.md)
OMEGA = 2 * np.pi / SPIN_PERIOD  # rad/s, magnitude of the turntable speed
CAMERA_DELAY = 0.11  # s, age of the image when grab_image returns


def rot_z(angle: float) -> np.ndarray:
    """Rotation about z as a 4x4 transform.

    Args:
        angle: Angle [rad].

    Returns:
        The homogeneous transform.
    """
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def translation(x: float, y: float, z: float) -> np.ndarray:
    """Translation as a 4x4 transform.

    Args:
        x: Offset x [m].
        y: Offset y [m].
        z: Offset z [m].

    Returns:
        The homogeneous transform.
    """
    T = np.eye(4)
    T[:3, 3] = x, y, z
    return T


@functools.lru_cache(maxsize=1)
def _fixed_frames() -> dict[str, np.ndarray]:
    """Fixed transforms of the task models (from their URDFs) and constants.

    Returns:
        table_z [m]; T_top: turntable top in the turntable frame; base_z: base
        frame above the top (the foam); T_mount: puzzle in the base frame;
        marker_z: marker surface above the base frame.
    """
    from cell import TABLE_Z
    from placement import BASE_FOAM, MODELS, load_urdf
    turntable = load_urdf(MODELS / "turntable.urdf")
    base = load_urdf(MODELS / "base.urdf")
    return {
        "table_z": np.array(TABLE_Z),
        "T_top": turntable.get_transform("top", turntable.base_link),
        "base_z": np.array(BASE_FOAM),
        "T_mount": base.get_transform("puzzle_mount", base.base_link),
        "marker_z": np.array(0.0056),  # top of the 0.6 mm marker relief
    }


# --- Phase 1: measurements and the turntable model ---

@dataclass
class BaseMeasurement:
    """One camera measurement of the ArUco base (P1.2 -> P1.3, P3.1).

    Attributes:
        t_capture: Time the image was taken [s, time.monotonic()].
        xy: Base frame origin (the centre of the base), robot frame [m], shape (2,).
        yaw: Rotation of the base frame's x axis about z, robot frame [rad].
        n_markers: Markers detected, 1 or 2 (1: weaker, especially in yaw).
        quality: Optional measure of the fit (e.g. reprojection error [px]);
            None if not available.
    """

    t_capture: float
    xy: np.ndarray
    yaw: float
    n_markers: int
    quality: float | None = None


@dataclass
class TurntableModel:
    """The turntable with the base and the puzzle as a one-joint robot (P1.3 -> P2, P3).

    Chain: robot frame -> turntable axis (axis_xy, on the table) -> revolute
    joint theta about z -> turntable top -> base (base_offset, base_yaw, on
    its foam) -> puzzle (the base's puzzle_mount).

    Attributes:
        axis_xy: Turntable axis on the table, robot frame [m], shape (2,).
        base_offset: Base centre in the turntable top frame [m], shape (2,).
        base_yaw: Base rotation in the turntable top frame [rad].
        direction: +1 counter-clockwise from above, -1 clockwise.
        omega: Speed magnitude [rad/s] (OMEGA unless re-estimated).
        theta_ref: Turntable angle at t_ref [rad].
        t_ref: Reference time [s, time.monotonic()].
        covariance: Optional covariance of the estimated parameters, in the
            order (axis x, axis y, offset x, offset y, base_yaw, theta_ref).
    """

    axis_xy: np.ndarray
    base_offset: np.ndarray
    base_yaw: float
    direction: int
    omega: float = OMEGA
    theta_ref: float = 0.0
    t_ref: float = 0.0
    covariance: np.ndarray | None = None

    def theta_at(self, t: float) -> float:
        """Turntable angle at a time.

        Args:
            t: Time [s, time.monotonic()].

        Returns:
            theta [rad] (not wrapped).
        """
        return self.theta_ref + self.direction * self.omega * (t - self.t_ref)

    def top_T(self, theta: float) -> np.ndarray:
        """Pose of the turntable top (its centre, top surface) at angle theta."""
        f = _fixed_frames()
        return (translation(*self.axis_xy, float(f["table_z"])) @ rot_z(theta)
                @ f["T_top"])

    def base_T(self, theta: float) -> np.ndarray:
        """Pose of the base frame (centre, bottom of the plate) at angle theta."""
        f = _fixed_frames()
        return (self.top_T(theta) @ translation(*self.base_offset, float(f["base_z"]))
                @ rot_z(self.base_yaw))

    def puzzle_T(self, theta: float) -> np.ndarray:
        """Pose of the puzzle frame at angle theta.

        Reference implementation; the turntable chain of task P1.1 must give
        the same result.

        Args:
            theta: Turntable angle [rad].

        Returns:
            4x4 transform, robot frame.
        """
        return self.base_T(theta) @ _fixed_frames()["T_mount"]

    def checkpoint_T(self, path: PuzzlePath, i: int, theta: float) -> np.ndarray:
        """Pose of checkpoint i of a path at angle theta.

        Args:
            path: The puzzle path.
            i: Checkpoint index.
            theta: Turntable angle [rad].

        Returns:
            4x4 transform, robot frame.
        """
        return self.puzzle_T(theta) @ path.frames[i]

    def measurement_at(self, t: float) -> tuple[np.ndarray, float]:
        """What a perfect camera would measure at a time.

        Args:
            t: Capture time [s].

        Returns:
            Base centre xy [m] and base yaw [rad], robot frame.
        """
        T = self.base_T(self.theta_at(t))
        return T[:2, 3].copy(), float(np.arctan2(T[1, 0], T[0, 0]))


# --- Phase 2: path, timing, plan ---

@dataclass
class PuzzlePath:
    """The ring centre's path through the puzzle, in the puzzle frame (P2.1 -> P2).

    Attributes:
        frames: Checkpoint frames in the puzzle frame, shape (N, 4, 4)
            (convention in the module doc), in the order of travel.
        clearance: Radial clearance between the ring and the puzzle at each
            checkpoint with the ring perpendicular to the path [m], shape (N,).
        roll_limits: Allowed roll range per checkpoint [rad], shape (N, 2);
            None if the roll is free everywhere.
        final_index: Index of the checkpoint where the task counts as done
            (hoop low enough above the base).
    """

    frames: np.ndarray
    clearance: np.ndarray
    roll_limits: np.ndarray | None = None
    final_index: int = -1

    @property
    def positions(self) -> np.ndarray:
        """Ring centre positions in the puzzle frame [m], shape (N, 3)."""
        return self.frames[:, :3, 3]

    @property
    def tangents(self) -> np.ndarray:
        """Directions of travel in the puzzle frame, shape (N, 3)."""
        return self.frames[:, :3, 2]


class RobotTiming(Protocol):
    """Timing of the robot's moves (P3.2 -> P2.6, P3.3)."""

    def move_duration(self, q0: np.ndarray, q1: np.ndarray, speed: float) -> float:
        """Duration [s] of one coordinated move q0 -> q1 at a relative speed."""

    def blends(self, dq_prev: np.ndarray, dq_next: np.ndarray) -> bool:
        """Whether the next queued move blends into the previous one (no stop)."""


@dataclass
class PhaseEstimate:
    """The tracker's estimate of the turntable's rotation (P3.1 -> P3.3).

    Attributes:
        t: Time of the estimate [s].
        theta: Turntable angle at t [rad].
        omega: Signed angular speed [rad/s] (sign = direction).
        covariance: 2x2 covariance of (theta, omega).
    """

    t: float
    theta: float
    omega: float
    covariance: np.ndarray

    def predict(self, t: float) -> tuple[float, float]:
        """Predicted angle and its standard deviation at a time.

        Args:
            t: Time [s].

        Returns:
            theta [rad] and its standard deviation [rad].
        """
        dt = t - self.t
        F = np.array([1.0, dt])
        return self.theta + self.omega * dt, float(np.sqrt(F @ self.covariance @ F))


@dataclass
class PlanStep:
    """One checkpoint of a plan.

    Attributes:
        index: Checkpoint index in the PuzzlePath.
        q: Planned configuration [rad], shape (6,).
        roll: Hoop roll at the checkpoint [rad] (convention in the module doc).
        t_rel: Planned arrival time after the start of the traversal [s].
        margin: Worst safety margin on the way to this checkpoint
            (dimensionless, > 0 safe; definition in P2.4).
    """

    index: int
    q: np.ndarray
    roll: float
    t_rel: float
    margin: float


@dataclass
class Plan:
    """The planner's result (P2.6, P2.7 -> P3.3).

    Attributes:
        feasible: Whether a plan was found; False means "report unreachable".
        reason: Why not (empty if feasible).
        speed: Relative robot speed the plan is timed for.
        direction: Spin direction it was planned for (+1 / -1).
        theta_start: Turntable angle at which the traversal starts [rad].
        q_start: Configuration at the start of the traversal (pre-entry) [rad].
        steps: The checkpoints in order.
        approach: Configurations from q_home to q_start (collision-free path).
        min_margin: Worst margin of the whole plan.
    """

    feasible: bool
    reason: str = ""
    speed: float = 0.0
    direction: int = 1
    theta_start: float = 0.0
    q_start: np.ndarray = field(default_factory=lambda: np.zeros(6))
    steps: list[PlanStep] = field(default_factory=list)
    approach: list[np.ndarray] = field(default_factory=list)
    min_margin: float = 0.0


# --- Fakes ---

def example_model(direction: int = 1, t_ref: float = 0.0) -> TurntableModel:
    """A typical turntable model: in the work area, base slightly off the axis.

    Args:
        direction: Spin direction (+1 / -1).
        t_ref: Reference time [s].

    Returns:
        The model.
    """
    return TurntableModel(axis_xy=np.array([0.45, 0.0]),
                          base_offset=np.array([0.02, -0.01]), base_yaw=0.3,
                          direction=direction, theta_ref=0.0, t_ref=t_ref)


def fake_measurements(model: TurntableModel, times: np.ndarray,
                      sigma_xy: float = 0.002, sigma_yaw: float = np.deg2rad(1.0),
                      p_one_marker: float = 0.1, p_missing: float = 0.0,
                      rng: np.random.Generator | None = None
                      ) -> list[BaseMeasurement]:
    """Synthetic camera measurements of a model.

    Args:
        model: The true model.
        times: Capture times [s].
        sigma_xy: Noise of xy [m] (two markers; doubled with one).
        sigma_yaw: Noise of yaw [rad] (two markers; tripled with one).
        p_one_marker: Probability that only one marker is seen.
        p_missing: Probability that no marker is seen (no measurement).
        rng: Random generator (default seed 0).

    Returns:
        The measurements, in time order (missing ones left out).
    """
    rng = rng or np.random.default_rng(0)
    out = []
    for t in times:
        if rng.uniform() < p_missing:
            continue
        one = rng.uniform() < p_one_marker
        xy, yaw = model.measurement_at(t)
        out.append(BaseMeasurement(
            t_capture=float(t),
            xy=xy + rng.normal(0.0, sigma_xy * (2 if one else 1), 2),
            yaw=float(yaw + rng.normal(0.0, sigma_yaw * (3 if one else 1))),
            n_markers=1 if one else 2))
    return out


def checkpoint_frame(position: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Builds a checkpoint frame (convention in the module doc).

    Args:
        position: Ring centre in the puzzle frame [m].
        tangent: Direction of travel in the puzzle's x-z plane.

    Returns:
        The 4x4 frame in the puzzle frame.
    """
    z = np.asarray(tangent, dtype=float) / np.linalg.norm(tangent)
    x = np.array([0.0, 1.0, 0.0])
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = position
    return T


def example_path() -> PuzzlePath:
    """A simplified puzzle-D-like path: straight in, a quarter bend, straight down.

    Not the real puzzle D (task P2.1 provides that); the shape and the numbers
    are only plausible, for developing the planner.

    Returns:
        The path, 15 mm checkpoint spacing on the straight parts.
    """
    pts, tans = [], []
    for x in np.arange(0.100, 0.035, -0.015):  # straight in along -x at z 155 mm
        pts.append([x, 0.0, 0.155])
        tans.append([-1.0, 0.0, 0.0])
    for a in np.linspace(np.pi / 2, np.pi, 7)[1:]:  # R50 bend down
        pts.append([0.035 + 0.05 * np.cos(a), 0.0, 0.105 + 0.05 * np.sin(a)])
        tans.append([-np.sin(a), 0.0, np.cos(a)])
    for z in np.arange(0.090, 0.015, -0.015):  # straight down
        pts.append([-0.015, 0.0, z])
        tans.append([0.0, 0.0, -1.0])
    frames = np.array([checkpoint_frame(np.array(p), np.array(t))
                       for p, t in zip(pts, tans)])
    return PuzzlePath(frames=frames, clearance=np.full(len(frames), 0.026),
                      roll_limits=None, final_index=len(frames) - 1)


def example_plan() -> Plan:
    """A hand-made feasible plan (small joint moves near q_home), for the executor.

    Not related to any real path; only the structure and the timing are
    meaningful.

    Returns:
        The plan.
    """
    q_home = np.deg2rad([0.0, 0.0, -45.0, 0.0, -45.0, 0.0])
    steps = [PlanStep(index=i, q=q_home + np.deg2rad([2.0 * i, -1.0 * i, 0, 0, 0, 0]),
                      roll=0.0, t_rel=0.4 * i, margin=1.0) for i in range(1, 6)]
    return Plan(feasible=True, speed=0.2, direction=1, theta_start=0.0,
                q_start=q_home.copy(), steps=steps, approach=[q_home.copy()],
                min_margin=1.0)
