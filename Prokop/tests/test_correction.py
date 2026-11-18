"""solver/correction.py (P3.4): corrections for the turntable's actual rotation."""

import numpy as np
import pytest
from ctu_crs import CRS93
from solver.correction import (MAX_CORRECTION, correct_for_rotation,
                               jacobian_finite_difference, ik_numerical, pose_error,
                               rotation_about_axis, so3_log)

D = np.deg2rad


@pytest.fixture(scope="module")
def crs():
    return CRS93(tty_dev=None)


def rodrigues(w):
    a = np.linalg.norm(w)
    if a == 0:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


@pytest.mark.parametrize("angle", [0.0, 1e-5, 0.3, 2.0, np.pi - 1e-3, np.pi])
def test_so3_log_inverts_rodrigues(angle, rng):
    for _ in range(5):
        axis = rng.normal(size=3)
        w = angle * axis / np.linalg.norm(axis)
        assert np.allclose(rodrigues(so3_log(rodrigues(w))), rodrigues(w), atol=1e-6)


def test_jacobian_against_motion(crs):
    q = D([10, -20, -60, 30, -50, 15])
    J = jacobian_finite_difference(crs, q)
    dq = D([0.1, -0.05, 0.08, 0.2, -0.1, 0.05]) * 1e-2
    assert np.allclose(J @ dq, pose_error(crs.fk(q), crs.fk(q + dq)), atol=1e-8)


def test_ik_numerical_converges_near_the_guess(crs, rng):
    q0 = crs.q_home + D([5, 10, -10, 20, 15, -30])
    for _ in range(5):
        q_true = q0 + rng.uniform(-0.1, 0.1, 6)
        q, ok = ik_numerical(crs, crs.fk(q_true), q0)
        assert ok and np.allclose(crs.fk(q), crs.fk(q_true), atol=1e-6)


def test_rotation_about_axis_keeps_the_axis_fixed():
    A = rotation_about_axis(np.array([0.45, 0.05]), 0.7)
    assert np.allclose(A @ [0.45, 0.05, 0.3, 1], [0.45, 0.05, 0.3, 1])
    assert np.allclose(A[:3, :3] @ [0, 0, 1], [0, 0, 1])


def test_correction_follows_the_turntable(crs):
    axis = np.array([0.45, 0.0])
    q_plan = D([-7.5, -4.04, -91.71, 0.0, -84.26, 0.0])  # flange over the work area
    for dtheta in (D(3), D(-6)):
        q, ok = correct_for_rotation(crs, q_plan, dtheta, axis)
        assert ok
        assert np.allclose(crs.fk(q), rotation_about_axis(axis, dtheta) @ crs.fk(q_plan),
                           atol=1e-6)
        assert np.abs(q - q_plan).max() < D(10)  # same branch, small change


def test_correction_refused_when_too_large(crs):
    q, ok = correct_for_rotation(crs, crs.q_home, MAX_CORRECTION * 1.5, np.array([0.45, 0.0]))
    assert not ok and np.allclose(q, crs.q_home)
    q, ok = correct_for_rotation(crs, crs.q_home, 0.0, np.array([0.45, 0.0]))
    assert ok and np.allclose(q, crs.q_home)
