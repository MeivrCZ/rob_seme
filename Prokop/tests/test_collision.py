"""collision.py: arm clearance to the table and the cell."""

import numpy as np
import pytest
from collision import ArmChecker


@pytest.fixture(scope="module")
def checker():
    return ArmChecker()


def test_home_is_clear(checker, robot):
    d, what = checker.clearance(robot.q_home)
    assert d > 0.1 and isinstance(what, str)


def test_leaning_forward_down_nears_the_table(checker, robot):
    q = robot.q_home + np.deg2rad([0, -80, 0, 0, 0, 0])  # -j2 leans forward
    d, what = checker.clearance(q)
    assert what == "table" and d < 0.1


def test_leaning_back_nears_the_back_wall(checker, robot):
    q = robot.q_home + np.deg2rad([0, 80, 60, 0, 0, 0])
    d, what = checker.clearance(q)
    assert what.startswith("cage_wall") and d < 0.15


def test_path_is_min_over_the_line(checker, robot):
    q1 = robot.q_home + np.deg2rad([0, -80, 0, 0, 0, 0])
    d_path, _ = checker.path_clearance(robot.q_home, q1)
    assert d_path <= min(checker.clearance(robot.q_home)[0], checker.clearance(q1)[0])


def test_hoop_center_up_in_home(checker, robot):
    h = checker.hoop_center(robot.q_home)
    F = robot.fk(robot.q_home)[:3, 3]
    assert np.allclose(h - F, [0.010, 0.0, 0.135], atol=0.002)
