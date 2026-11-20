"""Vectorized analytical IK and FK of the CRS93 for many targets at once (for P2.8).

The same equations as ctu_crs's CRSRobot.ik (lab 5, analytical IK with all
solutions): the wrist centre gives joints 1-3 (4 solutions, shoulder left/right
x elbow up/down), the rotation R_36 gives joints 4-6 (2 solutions each, wrist
flip), i.e. up to 8 solutions. Here every step is a numpy operation over a batch
of targets: ~100x faster than calling ctu_crs's ik per target, which matters
for the feasibility map (tens of thousands of targets).

Solutions are returned in (-pi, pi] like ctu_crs; expand_turns() adds the
+-2 pi variants of joints whose range exceeds a full turn (4 and 6 on the CRS93).
Singular special cases of ctu_crs's ik (wrist centre on the base axis, arm fully
stretched, q5 exactly 0 or pi) are not distinguished: those targets come out as
NaN or as one of their (infinitely many) solutions, which is enough for
screening.

Example:
    ::

        Q, valid = batch_ik(robot, T)       # T: (M, 4, 4) -> Q: (M, 8, 6), valid: (M, 8)
"""

from __future__ import annotations

import numpy as np


def _dh(d: float, theta: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """Batched DH transforms (ctu_crs dh_to_se3): Tz(d) Rz(theta) Tx(a) Rx(alpha).

    Args:
        d: Offset along z.
        theta: Joint angles, shape (...,).
        a: Offset along x.
        alpha: Twist.

    Returns:
        Transforms, shape (..., 4, 4).
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros(theta.shape + (4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 2], T[..., 0, 3] = ct, -st * ca, st * sa, a * ct
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st, ct * ca, -ct * sa, a * st
    T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = sa, ca, d
    T[..., 3, 3] = 1.0
    return T


def batch_fk(robot, Q: np.ndarray, n_joints: int = 6) -> np.ndarray:
    """Forward kinematics of many configurations (ctu_crs fk semantics).

    Args:
        robot: CRS93 / CRS93Sim (DH parameters).
        Q: Configurations [rad], shape (..., >= n_joints).
        n_joints: Number of leading joints to chain (3 gives the frame of joint 3).

    Returns:
        Flange (or joint-n) poses, shape (..., 4, 4).
    """
    Q = np.asarray(Q, dtype=float)
    T = np.broadcast_to(np.eye(4), Q.shape[:-1] + (4, 4)).copy()
    for i in range(n_joints):
        T = T @ _dh(robot.dh_d[i], Q[..., i] + robot.dh_offset[i], robot.dh_a[i],
                    robot.dh_alpha[i])
    return T


def batch_ik(robot, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All analytical IK solutions of many flange poses (ctu_crs ik equations).

    Args:
        robot: CRS93 / CRS93Sim (DH parameters).
        T: Flange poses, shape (M, 4, 4).

    Returns:
        Q, shape (M, 8, 6): the solutions [rad] (NaN where none), and valid,
        shape (M, 8): which exist (joint limits NOT applied, see in_limits).
    """
    T = np.asarray(T, dtype=float)
    M = T.shape[0]
    d, a = robot.dh_d, robot.dh_a
    wc = T[:, :3, 3] - d[5] * T[:, :3, 2]  # wrist centre = pose @ [0, 0, -d6]
    x, y, z = wc[:, 0], wc[:, 1], wc[:, 2]
    b = z - d[0]
    c = np.sqrt(b**2 + x**2 + y**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        reach = (c < d[3] + a[1]) & (np.hypot(x, y) > 1e-9)
        cos_a = (a[1] ** 2 + c**2 - d[3] ** 2) / (2 * a[1] * c)
        cos_b = (a[1] ** 2 + d[3] ** 2 - c**2) / (2 * a[1] * d[3])
        acos_a = np.arccos(np.clip(cos_a, -1, 1))
        asin_bc = np.arcsin(np.clip(b / c, -1, 1))
        th2_base = np.pi / 2 - asin_bc + acos_a
        th2_1 = np.arctan2(np.sin(th2_base), np.cos(th2_base))
        th1_pos, th1_neg = np.arctan2(y, x), np.arctan2(-y, -x)
        th3 = np.pi - np.arccos(np.clip(cos_b, -1, 1))
        th2_2 = -np.pi / 2 + asin_bc + acos_a
    q03 = np.stack([np.stack([th1_pos, -th2_1, th3], -1),
                    np.stack([th1_neg, th2_1, -th3], -1),
                    np.stack([th1_pos, th2_2, -th3], -1),
                    np.stack([th1_neg, -th2_2, th3], -1)], axis=1)  # (M, 4, 3)
    R03 = batch_fk(robot, q03, 3)[..., :3, :3]
    P = np.swapaxes(R03, -1, -2) @ T[:, None, :3, :3]  # rot_36, (M, 4, 3, 3)
    th5 = np.arccos(np.clip(P[..., 2, 2], -1, 1))
    sol_a = np.stack([np.arctan2(P[..., 1, 2], P[..., 0, 2]), -th5,
                      np.arctan2(P[..., 2, 1], -P[..., 2, 0])], -1)
    sol_b = np.stack([np.arctan2(-P[..., 1, 2], -P[..., 0, 2]), th5,
                      np.arctan2(-P[..., 2, 1], P[..., 2, 0])], -1)
    Q = np.empty((M, 8, 6))
    Q[:, 0::2, :3] = q03
    Q[:, 1::2, :3] = q03
    Q[:, 0::2, 3:] = sol_a
    Q[:, 1::2, 3:] = sol_b
    valid = np.repeat(reach[:, None], 8, axis=1) & np.all(np.isfinite(Q), axis=-1)
    Q[~valid] = np.nan
    return Q, valid


def in_limits(robot, Q: np.ndarray) -> np.ndarray:
    """Which configurations lie within the joint limits.

    Args:
        robot: CRS93 / CRS93Sim (q_min, q_max).
        Q: Configurations, shape (..., 6); NaN counts as outside.

    Returns:
        Boolean array, shape (...).
    """
    with np.errstate(invalid="ignore"):
        return np.all((Q >= robot.q_min) & (Q <= robot.q_max), axis=-1)


def expand_turns(robot, Q: np.ndarray) -> np.ndarray:
    """Adds the +-2 pi variants of joints whose range exceeds a full turn.

    Args:
        robot: CRS93 / CRS93Sim (q_min, q_max).
        Q: Configurations, shape (M, K, 6).

    Returns:
        Shape (M, K * V, 6) with V = 3 ** (number of such joints); variants
        outside the limits are NaN (filter with in_limits).
    """
    wide = np.flatnonzero(robot.q_max - robot.q_min > 2 * np.pi)
    out = Q
    for j in wide:
        parts = []
        for shift in (0.0, -2 * np.pi, 2 * np.pi):
            v = out.copy()
            v[..., j] += shift
            parts.append(v)
        out = np.concatenate(parts, axis=1)
    out = out.copy()
    out[~in_limits(robot, out)] = np.nan
    return out
