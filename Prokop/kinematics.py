"""Inverse kinematics helpers on top of ctu_crs."""

from __future__ import annotations

from itertools import product

import numpy as np
from ctu_crs import CRS97


def pose_error(robot: CRS97, q: np.ndarray, T: np.ndarray) -> np.ndarray:
    """6D error (position, rotation vector) of fk(q) w.r.t. @T, in the base frame."""
    F = robot.fk(q)
    dR = T[:3, :3] @ F[:3, :3].T
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    rot = axis * (0.5 if angle < 1e-9 else angle / (2 * np.sin(angle)))
    return np.r_[T[:3, 3] - F[:3, 3], rot]


def refine(robot: CRS97, q: np.ndarray, T: np.ndarray, iters: int = 30,
           damping: float = 1e-6) -> np.ndarray:
    """Damped least squares (numerical Jacobian) polishing of an approximate IK
    solution @q, e.g. the one ctu_crs returns near the wrist singularity (q5 ~ 0),
    where it sets q5 = 0 exactly and misses @T by ~0.1 mm, or the previous
    configuration of a continuous motion."""
    q = q.copy()
    for _ in range(iters):
        e = pose_error(robot, q, T)
        if np.abs(e).max() < 1e-10:
            break
        J = np.column_stack([
            (pose_error(robot, q, T) - pose_error(robot, q + dq, T)) / 1e-7
            for dq in np.eye(6) * 1e-7])
        q += J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
    return q


def ik_solutions(robot: CRS97, T: np.ndarray, tol: float = 1e-6) -> list[np.ndarray]:
    """All joint configurations within the joint limits whose flange pose is @T.

    ctu_crs ik() returns angles in (-pi, pi] and near singularities configurations
    that miss @T slightly; so the solutions are polished (refine), verified by fk
    and extended by +-2 pi variants of joints whose range exceeds a full turn
    (joints 4 and 6)."""
    sols = []
    for q in robot.ik(T):
        if not np.allclose(robot.fk(q), T, atol=tol):
            q = refine(robot, q, T)
            if not np.allclose(robot.fk(q), T, atol=tol):
                continue
        options = [[v for v in (qi - 2 * np.pi, qi, qi + 2 * np.pi) if lo <= v <= hi]
                   for qi, lo, hi in zip(q, robot.q_min, robot.q_max)]
        sols += [np.array(c) for c in product(*options)]
    return sols


def ik_closest(robot: CRS97, T: np.ndarray, q_ref: np.ndarray,
               max_jump: float | None = None, tol: float = 1e-6) -> np.ndarray | None:
    """IK solution for flange pose @T closest to @q_ref (keeps the motion
    continuous), or None if @T is unreachable within the joint limits. With
    @max_jump [rad], solutions moving any joint more than that are refused too
    (e.g. at a joint limit the closest solution may be another arm configuration,
    which the robot would reach by a large sweeping motion)."""
    sols = ik_solutions(robot, T, tol)
    # Local solution from the reference: the continuous one near singularities.
    q = refine(robot, q_ref, T)
    if robot.in_limits(q) and np.allclose(robot.fk(q), T, atol=tol):
        sols.append(q)
    if not sols:
        return None
    best = min(sols, key=lambda q: np.linalg.norm(q - q_ref))
    if max_jump is not None and np.abs(best - q_ref).max() > max_jump:
        return None
    return best
