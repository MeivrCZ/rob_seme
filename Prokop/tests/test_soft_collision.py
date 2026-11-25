"""Tests of the soft collision model (solver.soft_collision)."""

import numpy as np
import pytest

from solver.contracts import TurntableModel
from solver.soft_collision import Boxes, SoftScene, box_gaps, link_frames


def _box(centre, half, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Boxes(np.array([centre], float), R[None], np.array([half], float))


def _gap(a, b):
    return float(box_gaps(a, b, np.array([0]), np.array([0]))[0])


def test_face_separated_boxes_give_the_distance():
    assert np.isclose(_gap(_box([0, 0, 0], [1, 1, 1]), _box([5, 0, 0], [1, 2, 3])), 3.0)


def test_overlapping_boxes_give_a_negative_gap():
    assert _gap(_box([0, 0, 0], [1, 1, 1]), _box([1.5, 0.5, 0], [1, 1, 1], yaw=0.3)) < 0


def test_the_gap_is_a_lower_bound_of_the_distance(rng):
    # Points sampled in both boxes are never closer than the gap.
    for _ in range(50):
        a = _box(rng.uniform(-1, 1, 3), rng.uniform(0.1, 1, 3), rng.uniform(-3, 3))
        b = _box(rng.uniform(-3, 3, 3), rng.uniform(0.1, 1, 3), rng.uniform(-3, 3))
        g = _gap(a, b)
        pa = a.c[0] + (rng.uniform(-1, 1, (2000, 3)) * a.h[0]) @ a.R[0].T
        pb = b.c[0] + (rng.uniform(-1, 1, (2000, 3)) * b.h[0]) @ b.R[0].T
        d = np.linalg.norm(pa[:200, None] - pb[None, :200], axis=2).min()
        assert g <= d + 1e-9


def test_link_frames_match_the_urdf(robot, rng):
    from placement import MODELS, load_urdf
    urdf = load_urdf(MODELS / "crs93.urdf")
    for _ in range(10):
        q = rng.uniform(robot.q_min, robot.q_max)
        urdf.update_cfg(q)
        frames = link_frames(q)
        for link, T in frames.items():
            assert np.allclose(T, urdf.get_transform(link, urdf.base_link), atol=1e-12)


@pytest.fixture(scope="module")
def scene():
    model = TurntableModel(axis_xy=np.array([0.48, 0.02]), base_offset=np.array([0.02, -0.01]),
                           base_yaw=0.3, direction=1, omega=0.0)
    return SoftScene(model, 0.0, 0.0, 0.0)


def test_soft_distances_never_exceed_the_exact_ones(scene, robot, rng, task):
    """Leaf pairs and the hierarchical query are both lower bounds of the exact distance."""
    from collision import SceneCollider
    from placement import Placement
    m = scene.model
    p = Placement(np.asarray(m.axis_xy), np.asarray(m.base_offset), m.base_yaw)
    exact = SceneCollider(task)
    for _ in range(30):
        q, theta = rng.uniform(robot.q_min, robot.q_max), rng.uniform(-np.pi, np.pi)
        exact.place(q, p, theta)
        e = min(exact.placed_table()[1], exact.placed_group("cell")[1],
                exact.placed_group("task")[1])
        full = scene.group_distances(q, theta)
        assert min(full["table"], full["cell"], full["task"]) <= e + 1e-9
        assert scene.clearance(q, theta).distance <= e + 1e-9


def test_home_is_free(scene, robot):
    assert scene.clearance(robot.q_home, 0.0).free
