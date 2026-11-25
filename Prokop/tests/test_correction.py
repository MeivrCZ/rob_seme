"""solver/correction.py (P3.4): corrections for the turntable's actual rotation."""

import numpy as np
import pytest
from ctu_crs import CRS93
from solver.correction import MAX_CORRECTION, correct_for_rotation, rotation_about_axis

D = np.deg2rad


@pytest.fixture(scope="module")
def crs():
    return CRS93(tty_dev=None)


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
