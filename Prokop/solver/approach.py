"""Task P2.7: collision-free approach and retreat paths (straight task-space lines).

Moves between q_home and the plan's pre-entry configuration q_start. The puzzle
keeps spinning while the robot approaches, so the arm is checked against the
volume the turntable, base and puzzle sweep in a full turn (from the estimated
TurntableModel), besides the cell, the table and itself. (RRT was replaced by
the straight line on 2026-11-22: the work area lies right below q_home.)

Straight line in task space (plan_line): the ring centre moves on a straight
line and its orientation turns at a constant rate about one axis, position and
rotation interpolated separately (not the SE3 screw motion, whose position
follows a helix when the rotation changes):

    p(s) = p0 + s (p1 - p0),    R(s) = R0 exp(s log(R0^T R1)),    s in [0, 1].

For now s only places the samples (one move_to_q each); it can later carry
the timing.

    ::

        line = plan_line(robot, robot.q_home, plan.q_start, checker)
        if line.path is None:
            print(line.reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import trimesh
from cell import TABLE_Z, cell_meshes
from collision import SceneCollider, arm_parts
from placement import Placement, TaskScene

from hoop_targets import hoop_ring_pose_from_flange
from plan_ik import all_solutions, select_continuous_branch, so3_log

from solver.contracts import TurntableModel

MARGIN = 0.02  # m, required clearance of the arm on the approach
CHECK_STEP = np.deg2rad(1.0)  # rad, largest joint change between checked configurations
LINE_STEP = 0.02  # m, largest ring-centre move between samples of a line
LINE_ANGLE = np.deg2rad(5.0)  # rad, largest rotation between samples of a line
LINE_MAX_JUMP = np.deg2rad(20.0)  # rad, largest joint change between samples


class ApproachChecker:
    """Clearance of the arm (with the hoop) for the approach.

    Obstacles: the cell, the table, the volume swept by the spinning task, and
    the arm itself.

    Attributes:
        margin: Required clearance [m].
    """

    def __init__(self, model: TurntableModel, margin: float = MARGIN,
                 extra_obstacles: dict[str, trimesh.Trimesh] | None = None,
                 task: TaskScene | None = None) -> None:
        """Builds the obstacles.

        Args:
            model: The estimated turntable model (its swept volume is an obstacle).
            margin: Required clearance [m].
            extra_obstacles: More static obstacles by name, robot frame (tests).
            task: The task models; loaded if None.
        """
        self.margin = margin
        task = task or TaskScene()
        self._self = SceneCollider(task)
        self.urdf, self._parts = arm_parts()
        self._arm = trimesh.collision.CollisionManager()
        for _, name, mesh, _ in self._parts:
            self._arm.add_object(name, mesh)
        self._obstacles = trimesh.collision.CollisionManager()
        placement = Placement(np.asarray(model.axis_xy, float),
                              np.asarray(model.base_offset, float), model.base_yaw)
        obstacles = {**cell_meshes(), "task_swept": task.swept_task_mesh(placement),
                     **(extra_obstacles or {})}
        for name, mesh in obstacles.items():
            self._obstacles.add_object(name, mesh)
        self._hull = {name: mesh.convex_hull.vertices for link, name, mesh, _ in self._parts
                      if link != self.urdf.base_link}
        self._free: dict[tuple, bool] = {}

    def clearance(self, q: np.ndarray) -> tuple[float, str]:
        """Clearance of the arm in a configuration.

        Args:
            q: Joint configuration [rad].

        Returns:
            Distance [m] to the nearest obstacle (0 if touching or in
            self-collision) and its name ("table", "self", "task_swept", a cell
            mesh name or an extra obstacle).
        """
        if self._self.self_hits(q):
            return 0.0, "self"
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        poses = {}
        for link, name, _, offset in self._parts:
            poses[name] = self.urdf.get_transform(link, self.urdf.base_link) @ offset
            self._arm.set_transform(name, poses[name])
        d, (_, what) = self._arm.min_distance_other(self._obstacles, return_names=True)
        for name, points in self._hull.items():
            z = trimesh.transform_points(points, poses[name])[:, 2].min() - TABLE_Z
            if z < d:
                d, what = z, "table"
        return max(0.0, float(d)), what

    def is_free(self, q: np.ndarray) -> bool:
        """Whether configuration @q keeps the required clearance (cached)."""
        key = tuple(np.round(np.asarray(q, float), 9))
        if key not in self._free:
            self._free[key] = self.clearance(q)[0] >= self.margin
        return self._free[key]

    def segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether the straight joint-space segment a -> b keeps the clearance.

        Checked every CHECK_STEP of the largest joint change, endpoints included,
        in bisection order (the middle first), so that a blocked segment fails
        after few checks.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.abs(b - a).max() / CHECK_STEP)))
        return all(self.is_free(a + k / n * (b - a)) for k in _bisection_order(n))


def _bisection_order(n: int) -> list[int]:
    """Indices 0..n ordered coarse to fine: ends, middle, quarters, ...

    Args:
        n: Number of steps of the segment.

    Returns:
        Every index of 0..n once.
    """
    order, seen, step = [0, n], {0, n}, n
    while step > 1:
        step = (step + 1) // 2
        for k in range(step, n, step):
            if k not in seen:
                seen.add(k)
                order.append(k)
    order += [k for k in range(n + 1) if k not in seen]
    return order


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (lab 1 exp map, Rodrigues).

    Args:
        w: Rotation vector [rad], shape (3,).

    Returns:
        3x3 rotation matrix.
    """
    w = np.asarray(w, dtype=float)
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def line_poses(T0: np.ndarray, T1: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Poses on the straight task-space line from @T0 to @T1.

    Position and rotation separately: p(s) = p0 + s (p1 - p0) and
    R(s) = R0 exp(s log(R0^T R1)).

    Args:
        T0: Start pose (4x4).
        T1: End pose (4x4).
        s: Line parameters in [0, 1], shape (M,).

    Returns:
        The poses, shape (M, 4, 4); s = 0 gives T0, s = 1 gives T1.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    R0, p0 = T0[:3, :3], T0[:3, 3]
    w = so3_log(R0.T @ T1[:3, :3])
    out = np.tile(np.eye(4), (len(s), 1, 1))
    for k, sk in enumerate(s):
        out[k, :3, :3] = R0 @ so3_exp(sk * w)
        out[k, :3, 3] = p0 + sk * (T1[:3, 3] - p0)
    return out


def line_samples(T0: np.ndarray, T1: np.ndarray, max_step: float = LINE_STEP,
                 max_angle: float = LINE_ANGLE) -> np.ndarray:
    """Evenly spaced line parameters, fine enough for both limits.

    Args:
        T0: Start pose (4x4).
        T1: End pose (4x4).
        max_step: Largest position change between samples [m].
        max_angle: Largest rotation between samples [rad].

    Returns:
        s from 0 to 1 (both included), shape (n + 1,), n >= 1.
    """
    dist = np.linalg.norm(T1[:3, 3] - T0[:3, 3])
    angle = np.linalg.norm(so3_log(T0[:3, :3].T @ T1[:3, :3]))
    n = max(1, int(np.ceil(max(dist / max_step, angle / max_angle))))
    return np.linspace(0.0, 1.0, n + 1)


@cache
def ring_in_flange() -> np.ndarray:
    """Pose of the ring centre in the flange frame (Natka's hoop chain, cached)."""
    return hoop_ring_pose_from_flange(np.eye(4))


@dataclass
class LinePath:
    """Result of plan_line.

    Attributes:
        path: One configuration per sample, from q_from to q_to (both
            included), or None if the line cannot be followed.
        s: The line parameters of the samples.
        reason: Why the line failed ("" if it did not).
        max_deviation: Largest distance [m] of the tool point from the line at
            the joint-space midpoints between samples (the robot moves straight
            in joint space between them).
    """

    path: list[np.ndarray] | None
    s: np.ndarray
    reason: str = ""
    max_deviation: float = 0.0


def plan_line(robot, q_from: np.ndarray, q_to: np.ndarray,
              checker: ApproachChecker | None = None, s: np.ndarray | None = None,
              tool: np.ndarray | None = None, max_jump: float = LINE_MAX_JUMP
              ) -> LinePath:
    """Joint configurations that move the tool on a straight task-space line.

    The tool frame (by default the ring centre) moves from its pose at @q_from
    to its pose at @q_to on line_poses. At every sample the IK solution nearest
    to the previous configuration is taken (Natka's plan_ik: analytical IK incl.
    the +-2 pi wrist variants, refined numerically near the wrist singularity),
    so the line stays on one IK branch. It fails if a
    sample is unreachable, needs a joint jump larger than @max_jump (a branch
    switch or a singularity), ends on a different branch than @q_to, or (with a
    checker) a segment between samples is not free.

    Args:
        robot: CRS93 / CRS93Sim (fk, joint limits, DH parameters).
        q_from: Start configuration [rad].
        q_to: End configuration [rad].
        checker: If given, every joint-space segment between samples is checked.
        s: Line parameters (increasing, from 0 to 1); default line_samples.
        tool: Tool frame in the flange frame; default the ring centre.
        max_jump: Largest accepted change of any joint between samples [rad].

    Returns:
        The LinePath.
    """
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    tool = ring_in_flange() if tool is None else np.asarray(tool, float)
    tool_inv = np.linalg.inv(tool)
    T0, T1 = robot.fk(q_from) @ tool, robot.fk(q_to) @ tool
    s = line_samples(T0, T1) if s is None else np.asarray(s, float)
    targets = line_poses(T0, T1, s) @ tool_inv  # flange poses
    path = [q_from]
    for k in range(1, len(targets)):
        q = select_continuous_branch(all_solutions(robot, targets[k], q_previous=path[-1]),
                                     path[-1])
        if q is None:
            return LinePath(None, s, f"no IK solution within the limits at s = {s[k]:.3f}")
        jump = np.abs(q - path[-1]).max()
        if jump > max_jump:
            return LinePath(None, s, f"joint jump of {np.rad2deg(jump):.1f} deg at "
                            f"s = {s[k]:.3f} (branch switch or singularity)")
        path.append(q)
    if np.abs(path[-1] - q_to).max() > 1e-3:
        return LinePath(None, s, "q_to is on another IK branch than the line from "
                        "q_from reaches")
    path[-1] = q_to
    if checker is not None:
        for k, (a, b) in enumerate(zip(path, path[1:])):
            if not checker.segment_free(a, b):
                what = checker.clearance(a + 0.5 * (b - a))[1]
                return LinePath(None, s, f"segment {k} (s = {s[k]:.3f} to {s[k + 1]:.3f}) "
                                f"not free (nearest: {what})")
    p0, p1 = T0[:3, 3], T1[:3, 3]
    u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-12)
    dev = 0.0
    for a, b in zip(path, path[1:]):
        p = (robot.fk(a + 0.5 * (b - a)) @ tool)[:3, 3] - p0
        dev = max(dev, float(np.linalg.norm(p - (p @ u) * u)))
    return LinePath(path, s, max_deviation=dev)
