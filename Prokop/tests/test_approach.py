"""solver/approach.py (P2.7): approach paths on straight task-space lines."""

import numpy as np
import pytest
import trimesh
from ctu_crs import CRS93
from placement import TaskScene
from solver.approach import (ApproachChecker, _bisection_order, line_poses, line_samples,
                             plan_line, ring_in_flange, so3_exp)
from plan_ik import so3_log
from solver.contracts import example_model

D = np.deg2rad


@pytest.fixture(scope="module")
def crs():
    return CRS93(tty_dev=None)


@pytest.fixture(scope="module")
def task():
    return TaskScene()


def test_bisection_order_covers_every_index():
    for n in (1, 2, 7, 13, 64):
        order = _bisection_order(n)
        assert sorted(order) == list(range(n + 1)) and order[:2] == [0, n]


def test_goal_in_collision_is_refused(crs, task):
    checker = ApproachChecker(example_model(), task=task)
    into_table = crs.q_home + D([0, -80, 0, 0, 0, 0])  # leaning forward, down
    assert not checker.is_free(into_table)
    line = plan_line(crs, crs.q_home, into_table, checker)
    assert line.path is None and "not free" in line.reason


def test_the_spinning_task_is_an_obstacle(crs, task):
    """A configuration with the hoop in the turntable's swept volume is not free."""
    model = example_model()
    checker = ApproachChecker(model, task=task)
    # hoop centre right above the turntable axis, 5 cm above the puzzle top
    from collision import ArmChecker
    arm = ArmChecker()
    best = None
    for q in crs.ik(_flange_down(model.axis_xy, 0.20)):
        if crs.in_limits(q):
            best = q
            break
    assert best is not None
    d, what = checker.clearance(best)
    assert d < checker.margin and what in ("task_swept", "self")
    assert arm.clearance(best)[0] > 0  # only the task makes it unsafe


def _flange_down(xy, z):
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [*xy, z]
    return T


GOAL_DOWN = D([-7.5, -4.04, -91.71, 0.0, -84.26, 0.0])  # hoop down over the work area


def _pose(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def test_so3_exp_inverts_log(rng):
    for _ in range(20):
        w = rng.normal(size=3)
        w *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(w)
        R = so3_exp(w)
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(so3_log(R), w, atol=1e-9)


def test_line_poses_position_straight_rotation_uniform(rng):
    """Position linear in s even with a large rotation (not the SE3 screw motion)."""
    T0 = _pose(so3_exp([0.3, -0.2, 0.1]), [0.3, 0.0, 0.8])
    T1 = _pose(so3_exp([0.3, -0.2, 0.1]) @ so3_exp([0.0, 0.0, 2.5]), [0.4, 0.1, 0.2])
    s = np.linspace(0, 1, 11)
    P = line_poses(T0, T1, s)
    assert np.allclose(P[0], T0) and np.allclose(P[-1], T1)
    assert np.allclose(P[:, :3, 3], T0[:3, 3] + s[:, None] * (T1[:3, 3] - T0[:3, 3]))
    w = so3_log(T0[:3, :3].T @ T1[:3, :3])
    for sk, T in zip(s, P):  # rotation angle linear in s, about one fixed axis
        assert np.allclose(so3_log(T0[:3, :3].T @ T[:3, :3]), sk * w, atol=1e-9)


def test_line_samples_respect_both_limits():
    T0 = np.eye(4)
    T1 = _pose(so3_exp([0, 0, D(30)]), [0.1, 0, 0])
    s = line_samples(T0, T1, max_step=0.02, max_angle=D(5))
    assert s[0] == 0 and s[-1] == 1 and np.all(np.diff(s) > 0)
    assert len(s) - 1 == 6  # 30 deg / 5 deg > 10 cm / 2 cm
    assert len(line_samples(T0, T0)) == 2  # no move: just the two ends


def test_line_from_home_to_the_work_area(crs):
    line = plan_line(crs, crs.q_home, GOAL_DOWN)
    assert line.path is not None, line.reason
    assert np.array_equal(line.path[0], crs.q_home) and np.array_equal(line.path[-1], GOAL_DOWN)
    tool = ring_in_flange()
    expected = line_poses(crs.fk(crs.q_home) @ tool, crs.fk(GOAL_DOWN) @ tool, line.s)
    for q, T in zip(line.path, expected):  # the ring centre is on the line at every sample
        assert np.allclose(crs.fk(q) @ tool, T, atol=1e-6)
        assert crs.in_limits(q)
    assert max(np.abs(b - a).max() for a, b in zip(line.path, line.path[1:])) <= D(20)
    assert line.max_deviation < 0.002


def test_line_to_another_branch_is_refused(crs):
    """The same flange pose on the flipped wrist cannot be reached on one branch."""
    flipped = [q for q in crs.ik(crs.fk(GOAL_DOWN))
               if crs.in_limits(q) and np.abs(q - GOAL_DOWN).max() > 1.0]
    assert flipped
    line = plan_line(crs, crs.q_home, flipped[0])
    assert line.path is None and "branch" in line.reason


def test_blocked_line_reports_the_obstacle(crs, task):
    tool = ring_in_flange()
    mid = 0.5 * ((crs.fk(crs.q_home) @ tool)[:3, 3] + (crs.fk(GOAL_DOWN) @ tool)[:3, 3])
    box = trimesh.creation.box((0.05, 0.05, 0.05))
    box.apply_translation(mid)
    checker = ApproachChecker(example_model(), extra_obstacles={"box": box}, task=task)
    line = plan_line(crs, crs.q_home, GOAL_DOWN, checker)
    assert line.path is None and "box" in line.reason
    free = plan_line(crs, crs.q_home, GOAL_DOWN, ApproachChecker(example_model(), task=task))
    assert free.path is not None, free.reason
