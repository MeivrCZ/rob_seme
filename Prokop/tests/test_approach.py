"""solver/approach.py (P2.7): approach paths with RRT + random shortcut."""

import numpy as np
import pytest
import trimesh
from ctu_crs import CRS93
from placement import TaskScene
from solver.approach import RRT, ApproachChecker, _bisection_order, plan_approach
from solver.contracts import example_model

D = np.deg2rad


@pytest.fixture(scope="module")
def crs():
    return CRS93(tty_dev=None)


@pytest.fixture(scope="module")
def task():
    return TaskScene()


@pytest.fixture(scope="module")
def blocked(task):
    """A box on the arc the flange sweeps when joint 1 turns 80 deg from q_home."""
    a = D(40)
    box = trimesh.creation.box((0.12, 0.12, 0.15))
    box.apply_translation([0.31 * np.cos(a), 0.31 * np.sin(a), 0.88])
    return ApproachChecker(example_model(), extra_obstacles={"box": box}, task=task)


def test_bisection_order_covers_every_index():
    for n in (1, 2, 7, 13, 64):
        order = _bisection_order(n)
        assert sorted(order) == list(range(n + 1)) and order[:2] == [0, n]


def test_straight_when_free(crs, task):
    checker = ApproachChecker(example_model(), task=task)
    goal = D([-7.5, -4.04, -91.71, 0.0, -84.26, 0.0])  # flange down over the work area
    path = plan_approach(crs, checker, crs.q_home, goal)
    assert path is not None and len(path) == 2


def test_detour_around_an_obstacle(crs, blocked):
    goal = crs.q_home + D([80, 0, 0, 0, 0, 0])
    assert not blocked.segment_free(crs.q_home, goal)
    path = plan_approach(crs, blocked, crs.q_home, goal, rng=np.random.default_rng(0))
    assert path is not None and len(path) >= 3
    assert np.allclose(path[0], crs.q_home) and np.allclose(path[-1], goal)
    assert all(blocked.segment_free(a, b) for a, b in zip(path, path[1:]))
    assert all(crs.in_limits(q) for q in path)


def test_shortcut_keeps_the_path_free_and_not_longer(crs, blocked):
    rrt = RRT(crs, blocked, rng=np.random.default_rng(1))
    goal = crs.q_home + D([80, 0, 0, 0, 0, 0])
    path = rrt.plan(crs.q_home, goal, 3000, D(30))
    assert path
    short = rrt.random_shortcut(path, 100)
    length = lambda p: sum(np.linalg.norm(b - a) for a, b in zip(p, p[1:]))  # noqa: E731
    assert length(short) <= length(path) + 1e-9
    assert all(blocked.segment_free(a, b) for a, b in zip(short, short[1:]))


def test_goal_in_collision_is_refused(crs, task):
    checker = ApproachChecker(example_model(), task=task)
    into_table = crs.q_home + D([0, -80, 0, 0, 0, 0])  # leaning forward, down
    assert not checker.is_free(into_table)
    assert plan_approach(crs, checker, crs.q_home, into_table) is None


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
