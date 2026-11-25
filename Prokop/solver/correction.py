"""Task P3.4: correcting planned configurations for the puzzle's actual rotation.

When the turntable is at theta_planned + dtheta at the moment the robot arrives,
the whole planned hoop target is rotated by dtheta about the turntable's vertical
axis: T_corrected = Rz_axis(dtheta) @ fk(q_planned). The hoop roll follows
automatically. The corrected configuration is found by numerical IK seeded with
the planned one, so it stays on the planned branch.

Methods (lab semantics, robotics_labs):

* jacobian_finite_difference: lab 4 (HW04), the 6 x 6 Jacobian of the flange
  pose by finite differences, rows (position, rotation vector) in the robot frame.
* ik_numerical: lab 5 (HW05), iterative IK from an initial guess, here damped
  least squares with that Jacobian, stopped at an acceptable error.
* so3_log: lab 1 (HW01), rotation matrix -> rotation vector (log map).

Example:
    ::

        q, ok = correct_for_rotation(robot, step.q, dtheta, model.axis_xy)
"""

from __future__ import annotations

import numpy as np

MAX_CORRECTION = np.deg2rad(15.0)  # rad, largest accepted turntable correction
MAX_JOINT_CHANGE = np.deg2rad(20.0)  # rad, largest accepted change of any joint


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (lab 1 log map).

    Args:
        R: 3x3 rotation matrix.

    Returns:
        The rotation vector [rad], shape (3,).
    """
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - angle < 1e-6:  # near pi: axis from the diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        k = int(np.argmax(axis))
        axis[np.arange(3) != k] *= np.sign(R[k, np.arange(3) != k] + R[np.arange(3) != k, k])
        return angle * axis / np.linalg.norm(axis)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle * w / (2 * np.sin(angle))


def pose_error(T: np.ndarray, T_desired: np.ndarray) -> np.ndarray:
    """6-D error of a pose w.r.t. the desired one, in the robot frame.

    Args:
        T: Current 4x4 pose.
        T_desired: Desired 4x4 pose.

    Returns:
        (position error [m], rotation vector R_desired R^T [rad]), shape (6,).
    """
    return np.concatenate([T_desired[:3, 3] - T[:3, 3],
                           so3_log(T_desired[:3, :3] @ T[:3, :3].T)])


def jacobian_finite_difference(robot, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Jacobian of the flange pose by finite differences (lab 4 semantics).

    Args:
        robot: CRS93 / CRS93Sim (fk).
        q: Configuration [rad].
        delta: Joint step [rad].

    Returns:
        6x6 Jacobian, rows (linear, angular) in the robot frame.
    """
    T0 = robot.fk(q)
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = np.zeros(len(q))
        dq[i] = delta
        J[:, i] = pose_error(T0, robot.fk(q + dq)) / delta
    return J


def ik_numerical(robot, T_desired: np.ndarray, q_init: np.ndarray,
                 max_iterations: int = 50, acceptable_err: float = 1e-7,
                 damping: float = 1e-4) -> tuple[np.ndarray, bool]:
    """Numerical IK from an initial guess (lab 5 semantics, damped least squares).

    Args:
        robot: CRS93 / CRS93Sim (fk).
        T_desired: Desired flange pose (4x4, robot frame).
        q_init: Initial guess [rad]; the solution stays near it.
        max_iterations: Iterations before giving up.
        acceptable_err: Norm of the 6-D pose error to stop at.
        damping: Damping of the least squares (robust near singularities).

    Returns:
        The configuration [rad] and whether the error got below acceptable_err.
    """
    q = np.asarray(q_init, dtype=float).copy()
    for _ in range(max_iterations):
        e = pose_error(robot.fk(q), T_desired)
        if np.linalg.norm(e) < acceptable_err:
            return q, True
        J = jacobian_finite_difference(robot, q)
        q = q + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
    return q, bool(np.linalg.norm(pose_error(robot.fk(q), T_desired)) < acceptable_err)


def rotation_about_axis(axis_xy: np.ndarray, dtheta: float) -> np.ndarray:
    """Rotation by @dtheta about the vertical axis through @axis_xy.

    Args:
        axis_xy: The turntable axis on the table [m].
        dtheta: Angle [rad], counter-clockwise from above positive.

    Returns:
        4x4 transform, robot frame.
    """
    c, s = np.cos(dtheta), np.sin(dtheta)
    A = np.eye(4)
    A[:2, :2] = [[c, -s], [s, c]]
    A[:2, 3] = np.asarray(axis_xy) - A[:2, :2] @ np.asarray(axis_xy)
    return A


def correct_for_rotation(robot, q_planned: np.ndarray, dtheta: float,
                         axis_xy: np.ndarray, *, max_rotation: float = MAX_CORRECTION
                         ) -> tuple[np.ndarray, bool]:
    """Configuration reaching the planned hoop target rotated with the turntable.

    Args:
        robot: CRS93 / CRS93Sim (fk, in_limits).
        q_planned: Planned configuration [rad].
        dtheta: Actual minus planned turntable angle at arrival [rad].
        axis_xy: The turntable axis [m].
        max_rotation: Maximum permitted phase correction [rad].

    Returns:
        The corrected configuration [rad] and whether it is acceptable: IK
        converged, within the joint limits, |dtheta| <= max_rotation and no
        joint changed by more than MAX_JOINT_CHANGE (a larger change would mean
        another IK branch or a near-singular configuration).
    """
    q_planned = np.asarray(q_planned, dtype=float)
    if dtheta == 0.0:
        return q_planned.copy(), True
    if abs(dtheta) > max_rotation:
        return q_planned.copy(), False
    T = rotation_about_axis(axis_xy, dtheta) @ robot.fk(q_planned)
    q, ok = ik_numerical(robot, T, q_planned)
    ok = ok and bool(robot.in_limits(q)) and bool(
        np.abs(q - q_planned).max() <= MAX_JOINT_CHANGE)
    return q, ok
