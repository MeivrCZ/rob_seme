"""placement.py: shapes, rules of the random placement, tip-over and clearance."""

import numpy as np
import pytest
import trimesh
from placement import (BASE_FOAM, MIN_CLEARANCE, STABILITY_MARGIN, TABLE_Z,
                       CAGE_XY, WORK_AREA_XY, Placement)


def test_placement_transforms(task):
    p = Placement(np.array([0.45, 0.0]), np.array([0.01, -0.02]), 0.3)
    T, B = p.turntable_T(), p.base_T_top()
    assert T.shape == (4, 4) and B.shape == (4, 4)
    assert T[2, 3] == pytest.approx(TABLE_Z)
    assert B[2, 3] == pytest.approx(BASE_FOAM)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3))
    assert np.allclose(B[:3, :3] @ B[:3, :3].T, np.eye(3))


def test_com_shape(task):
    assert task.com_xy.shape == (2,)
    assert np.abs(task.com_xy).max() < 0.01  # near the base centre


def test_random_placements_follow_rules(task, rng):
    for _ in range(10):
        p = task.random_placement(rng)
        assert p.turntable_xy.shape == (2,) and p.base_xy.shape == (2,)
        assert np.all(p.turntable_xy >= CAGE_XY[:, 0])
        assert np.all(p.turntable_xy <= CAGE_XY[:, 1])
        assert task.stability_margin(p) >= STABILITY_MARGIN
        assert task.clearance_to_robot(p) >= MIN_CLEARANCE


def test_centred_base_is_stable(task):
    p = Placement(np.array([0.45, 0.0]), np.zeros(2), 0.0)
    assert task.stability_margin(p) > 0.05


def test_overhanging_base_tips(task):
    p = Placement(np.array([0.45, 0.0]), np.array([0.12, 0.0]), 0.0)
    assert task.stability_margin(p) < 0


def test_turntable_at_robot_has_no_clearance(task):
    p = Placement(np.zeros(2), np.zeros(2), 0.0)
    assert task.clearance_to_robot(p) == 0.0


def test_swept_volume(task):
    p = Placement(np.array([0.45, 0.0]), np.array([0.05, 0.02]), 0.7)
    m = task.swept_task_mesh(p)
    assert m.is_watertight and m.volume > 0
    # contains the task at any rotation of the turntable
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        c, s = np.cos(a), np.sin(a)
        q = Placement(p.turntable_xy, np.array([[c, -s], [s, c]]) @ p.base_xy,
                      p.base_yaw + a)
        pts = np.vstack([mesh.vertices for mesh in task.task_meshes(q)])
        # signed distance: + inside; points on the surface (turntable) give ~0
        assert trimesh.proximity.signed_distance(m, pts).min() > -1e-6


def test_viable_placements_in_work_area(task, rng):
    for _ in range(10):
        p = task.random_placement(rng, WORK_AREA_XY)
        assert np.all(p.turntable_xy >= WORK_AREA_XY[:, 0])
        assert np.all(p.turntable_xy <= WORK_AREA_XY[:, 1])
        assert task.stability_margin(p) >= STABILITY_MARGIN
        assert task.clearance_to_robot(p) >= MIN_CLEARANCE
