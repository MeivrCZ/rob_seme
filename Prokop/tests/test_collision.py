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


# --- SceneCollider: the exact-mesh scene of the simulation ---

from collision import EXCLUDED_PAIRS, SceneCollider  # noqa: E402
from placement import Placement  # noqa: E402

PLACEMENT = Placement(np.array([0.50, 0.0]), np.zeros(2), 0.0)
# Flange above the side of the turntable opposite to puzzle D's hook: clear at
# angle 0, the spinning hook hits the adapter after ~150 deg.
IN_HOOK_PATH = np.deg2rad([0.7, -39.4, -98.5, 0.0, -42.2, -179.3])


@pytest.fixture(scope="module")
def collider(task):
    return SceneCollider(task)


@pytest.mark.parametrize("name", ["q_home", "zero"])
def test_scene_clear(collider, robot, name):
    q = robot.q_home if name == "q_home" else np.zeros(6)
    hits, d = collider.check(q, PLACEMENT, 0.0)
    assert hits == [] and d > 0.1


def test_excluded_pairs_not_reported(collider, robot, rng):
    """Neighbours that always overlap and parts mounted face to face never
    show up as self-collisions."""
    for _ in range(50):
        for pair in collider.self_hits(rng.uniform(robot.q_min, robot.q_max)):
            assert frozenset(pair) not in EXCLUDED_PAIRS


def test_table_hit_and_patch(collider):
    q = np.deg2rad([0, 90, 60, 0, 0, 0])  # leaning back into the table
    hits, d = collider.check(q, PLACEMENT, 0.0)
    assert ("hoop", "table") in hits and d == 0.0
    patch, T = collider.body("table")
    assert T[0, 3] < 0  # the contact is behind the robot


def test_base_link_not_checked_against_table(collider, robot):
    """The robot's foot stands in a cut-out of the table: no contact reported."""
    hits, _ = collider.check(robot.q_home, PLACEMENT, 0.0)
    assert not any(b == "table" for _, b in hits)


def test_spinning_puzzle_reaches_the_arm(collider):
    hits, d = collider.check(IN_HOOK_PATH, PLACEMENT, 0.0)
    assert hits == [] and d > 0.02
    hits, _ = collider.check(IN_HOOK_PATH, PLACEMENT, np.deg2rad(160))
    assert ("adapter", "puzzle") in hits


def test_task_poses_follow_the_turntable(collider):
    a = collider.task_poses(PLACEMENT, 0.0)
    b = collider.task_poses(PLACEMENT, np.pi / 2)
    assert np.allclose(a["turntable"], b["turntable"])
    assert not np.allclose(a["puzzle"][:3, :3], b["puzzle"][:3, :3])
