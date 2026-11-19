"""IK helpers for planning CRS93 flange poses.

The analytical IK is provided by ``ctu_crs.CRS93.ik``. This module expands
wrist-angle wraps within the robot's joint limits, refines solutions near the
wrist singularity when a previous configuration is available, and selects a
continuous solution along a target sequence.
"""

from __future__ import annotations

from itertools import product
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

WRIST_JOINTS = (3, 4, 5)
TWO_PI = 2.0 * np.pi
SINGULARITY_TOL = np.deg2rad(3.0)


class CRS93Kinematics(Protocol):
    """The kinematic interface used by these IK helpers."""

    q_min: ArrayLike
    q_max: ArrayLike

    def ik(self, pose: np.ndarray) -> list[np.ndarray]: ...

    def fk(self, q: ArrayLike) -> np.ndarray: ...

    def in_limits(self, q: ArrayLike) -> bool: ...


def _validate_pose(pose: ArrayLike) -> np.ndarray:
    T = np.asarray(pose, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("pose must be a finite 4x4 homogeneous transform")
    return T


def so3_log(R: np.ndarray) -> np.ndarray:
    """Return the rotation vector for a rotation matrix."""
    cosine = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    if angle < 1e-9:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / 2.0
    if np.pi - angle < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        index = int(np.argmax(axis))
        for other in range(3):
            if other != index:
                axis[other] *= np.sign(R[index, other] + R[other, index])
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            axis[index] = 1.0
            norm = 1.0
        return angle * axis / norm
    vector = np.array([R[2, 1] - R[1, 2],
                       R[0, 2] - R[2, 0],
                       R[1, 0] - R[0, 1]])
    return angle * vector / (2.0 * np.sin(angle))


def pose_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Six-dimensional pose error: translation [m], then rotation [rad]."""
    return np.concatenate((
        desired[:3, 3] - current[:3, 3],
        so3_log(desired[:3, :3] @ current[:3, :3].T),
    ))


def jacobian_finite_difference(
    robot: CRS93Kinematics,
    q: ArrayLike,
    delta: float = 1e-6,
) -> np.ndarray:
    """Finite-difference CRS93 flange Jacobian in the robot frame."""
    configuration = np.asarray(q, dtype=float)
    if configuration.ndim != 1 or not np.all(np.isfinite(configuration)):
        raise ValueError("q must be a finite one-dimensional configuration")
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    current = robot.fk(configuration)
    jacobian = np.zeros((6, configuration.size))
    for joint in range(configuration.size):
        perturbed = configuration.copy()
        perturbed[joint] += delta
        jacobian[:, joint] = pose_error(current, robot.fk(perturbed)) / delta
    return jacobian


def ik_numerical(
    robot: CRS93Kinematics,
    target: ArrayLike,
    q_init: ArrayLike,
    max_iterations: int = 100,
    acceptable_err: float = 1e-7,
    damping: float = 1e-4,
) -> tuple[np.ndarray, bool]:
    """Refine a nearby IK solution by damped least squares and line search."""
    desired = _validate_pose(target)
    q = np.asarray(q_init, dtype=float).copy()
    if q.ndim != 1 or not np.all(np.isfinite(q)):
        raise ValueError("q_init must be a finite one-dimensional configuration")
    if max_iterations < 1 or acceptable_err <= 0.0 or damping < 0.0:
        raise ValueError("iteration count, tolerance, or damping is invalid")

    for _ in range(max_iterations):
        current = robot.fk(q)
        error = pose_error(current, desired)
        if np.linalg.norm(error) <= acceptable_err:
            return q, True

        jacobian = jacobian_finite_difference(robot, q)
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(6), error
        )
        old_error = np.linalg.norm(error)
        scale = 1.0
        while scale >= 1e-6:
            candidate = q + scale * step
            candidate_error = np.linalg.norm(pose_error(robot.fk(candidate), desired))
            if candidate_error < old_error:
                q = candidate
                break
            scale *= 0.5
        else:
            return q, False

    return q, bool(np.linalg.norm(pose_error(robot.fk(q), desired)) <= acceptable_err)


def _wrist_variants(
    robot: CRS93Kinematics,
    q: ArrayLike,
) -> list[np.ndarray]:
    """Return all +/-2pi wrist variants lying within the joint limits."""
    solution = np.asarray(q, dtype=float)
    q_min = np.asarray(robot.q_min, dtype=float)
    q_max = np.asarray(robot.q_max, dtype=float)
    if solution.shape != q_min.shape or q_min.shape != q_max.shape:
        raise ValueError("IK result and robot joint-limit shapes do not match")

    offsets: list[range] = []
    for joint in WRIST_JOINTS:
        first = int(np.ceil((q_min[joint] - solution[joint]) / TWO_PI))
        last = int(np.floor((q_max[joint] - solution[joint]) / TWO_PI))
        offsets.append(range(first, last + 1))

    variants = []
    for wraps in product(*offsets):
        candidate = solution.copy()
        candidate[list(WRIST_JOINTS)] += TWO_PI * np.asarray(wraps)
        if robot.in_limits(candidate):
            variants.append(candidate)
    return variants


def _append_unique(solutions: list[np.ndarray], candidate: np.ndarray) -> None:
    if not any(np.allclose(candidate, existing, atol=1e-8, rtol=0.0)
               for existing in solutions):
        solutions.append(candidate)


def _matches_pose(robot: CRS93Kinematics, q: np.ndarray, target: np.ndarray) -> bool:
    error = pose_error(robot.fk(q), target)
    return bool(np.linalg.norm(error[:3]) <= 1e-6
                and np.linalg.norm(error[3:]) <= 1e-6)


def all_solutions(
    robot: CRS93Kinematics,
    target: ArrayLike,
    q_previous: ArrayLike | None = None,
    singularity_tol: float = SINGULARITY_TOL,
) -> list[np.ndarray]:
    """Return valid analytical solutions and wrist-wrap variants.

    Near q5 = 0, the analytical solver fixes q4 to zero because the wrist
    orientation does not uniquely determine q4 and q6. When ``q_previous`` is
    supplied, numerical IK is additionally seeded with each analytical
    position branch and the previous wrist configuration to preserve a
    continuous branch through the singularity.
    """
    desired = _validate_pose(target)
    if singularity_tol < 0.0:
        raise ValueError("singularity_tol must be non-negative")
    previous = None
    if q_previous is not None:
        previous = np.asarray(q_previous, dtype=float)
        if previous.shape != np.asarray(robot.q_min).shape or not np.all(
            np.isfinite(previous)
        ):
            raise ValueError("q_previous must be a finite robot configuration")

    raw_solutions = robot.ik(desired)
    solutions: list[np.ndarray] = []
    for raw in raw_solutions:
        candidate = np.asarray(raw, dtype=float)
        if candidate.shape != np.asarray(robot.q_min).shape:
            raise ValueError("robot.ik returned a configuration with an invalid shape")

        if abs(candidate[4]) <= singularity_tol:
            seed = candidate.copy()
            if previous is not None:
                seed[list(WRIST_JOINTS)] = previous[list(WRIST_JOINTS)]
            refined, converged = ik_numerical(robot, desired, seed)
            if converged:
                for variant in _wrist_variants(robot, refined):
                    _append_unique(solutions, variant)

        if _matches_pose(robot, candidate, desired):
            for variant in _wrist_variants(robot, candidate):
                _append_unique(solutions, variant)

    return sorted(solutions, key=lambda q: tuple(float(v) for v in q))


def select_continuous_branch(
    solutions: list[np.ndarray],
    q_previous: ArrayLike,
) -> np.ndarray | None:
    """Choose the valid solution nearest the previous configuration.

    Wrist wraps must be expanded with ``all_solutions`` first; minimizing
    ordinary joint-space distance then follows the nearest equivalent branch.
    Returns ``None`` if there are no IK solutions.
    """
    previous = np.asarray(q_previous, dtype=float)
    if previous.ndim != 1 or not np.all(np.isfinite(previous)):
        raise ValueError("q_previous must be a finite one-dimensional configuration")
    if not solutions:
        return None

    candidates = [np.asarray(q, dtype=float) for q in solutions]
    if any(q.shape != previous.shape or not np.all(np.isfinite(q))
           for q in candidates):
        raise ValueError("all solutions must match q_previous and be finite")
    return min(candidates, key=lambda q: float(np.linalg.norm(q - previous))).copy()


def continue_ik(
    robot: CRS93Kinematics,
    target: ArrayLike,
    q_previous: ArrayLike,
    singularity_tol: float = SINGULARITY_TOL,
) -> np.ndarray | None:
    """Solve a target and return the solution continuing the previous branch."""
    solutions = all_solutions(
        robot, target, q_previous=q_previous, singularity_tol=singularity_tol
    )
    return select_continuous_branch(solutions, q_previous)
