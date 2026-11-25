"""Task P3.4: correcting planned configurations for the puzzle's actual rotation.

When the turntable is at theta_planned + dtheta at the moment the robot arrives,
the whole planned hoop target is rotated by dtheta about the turntable's vertical
axis: T_corrected = Rz_axis(dtheta) @ fk(q_planned). The hoop roll follows
automatically. The corrected configuration is found by numerical IK seeded with
the planned one, so it stays on the planned branch.

The numerical IK (lab 5, damped least squares with the lab-4 finite-difference
Jacobian) is Natka's plan_ik.ik_numerical.

Example:
    ::

        q, ok = correct_for_rotation(robot, step.q, dtheta, model.axis_xy)
"""

from __future__ import annotations

import numpy as np
from plan_ik import ik_numerical

MAX_CORRECTION = np.deg2rad(15.0)  # rad, largest accepted turntable correction
MAX_JOINT_CHANGE = np.deg2rad(20.0)  # rad, largest accepted change of any joint


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
                         axis_xy: np.ndarray) -> tuple[np.ndarray, bool]:
    """Configuration reaching the planned hoop target rotated with the turntable.

    Args:
        robot: CRS93 / CRS93Sim (fk, in_limits).
        q_planned: Planned configuration [rad].
        dtheta: Actual minus planned turntable angle at arrival [rad].
        axis_xy: The turntable axis [m].

    Returns:
        The corrected configuration [rad] and whether it is acceptable: IK
        converged, within the joint limits, |dtheta| <= MAX_CORRECTION and no
        joint changed by more than MAX_JOINT_CHANGE (a larger change would mean
        another IK branch or a near-singular configuration).
    """
    q_planned = np.asarray(q_planned, dtype=float)
    if dtheta == 0.0:
        return q_planned.copy(), True
    if abs(dtheta) > MAX_CORRECTION:
        return q_planned.copy(), False
    T = rotation_about_axis(axis_xy, dtheta) @ robot.fk(q_planned)
    q, ok = ik_numerical(robot, T, q_planned)
    ok = ok and bool(robot.in_limits(q)) and bool(
        np.abs(q - q_planned).max() <= MAX_JOINT_CHANGE)
    return q, ok
