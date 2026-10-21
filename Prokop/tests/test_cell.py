"""cell.py: cage, camera rail and camera geometry."""

import numpy as np
import pytest
import trimesh
from cell import (CAGE_APOTHEM, CAGE_SIDE, CAMERA_T, RAIL_TOP_Z, TABLE_Z,
                  WORK_AREA_XY, cell_meshes, in_cage)
from placement import HOOP_MOUNT, MODELS, collision_meshes, load_urdf


def test_cage_octagon():
    assert CAGE_APOTHEM == pytest.approx(CAGE_SIDE / (2 * np.tan(np.pi / 8)))
    assert in_cage([0.0, 0.0]) and in_cage([CAGE_APOTHEM - 0.01, 0.0])
    assert not in_cage([CAGE_APOTHEM + 0.01, 0.0])
    corner = CAGE_SIDE / 2 / np.sin(np.pi / 8)  # circumradius, at 22.5 deg
    assert in_cage(0.99 * corner * np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)]))
    assert not in_cage([0.7, 0.7])  # beyond the 45 deg wall
    assert not in_cage([0.5, 0.0], margin=0.5)


def test_work_area_inside_cage():
    for x in WORK_AREA_XY[0]:
        for y in WORK_AREA_XY[1]:
            assert in_cage([x, y])


def test_camera_on_axis_looking_down():
    R = CAMERA_T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1)
    assert np.allclose(R[:, 2], [0, 0, -1])
    assert CAMERA_T[1, 3] == 0.0
    assert TABLE_Z < CAMERA_T[2, 3] < RAIL_TOP_Z


def test_cell_meshes_closed():
    meshes = cell_meshes()
    assert len(meshes) == 11  # 8 walls, rail, post, camera
    for m in meshes.values():
        assert m.is_watertight and m.volume > 0


@pytest.mark.parametrize("q", ["home", "zero"])
def test_robot_clear_of_cell(robot, q):
    """The arm with the hoop touches no part of the cell in q_home and upright."""
    urdf, hoop = load_urdf(MODELS / "crs93.urdf"), load_urdf(MODELS / "hoop.urdf")
    urdf.update_cfg(robot.q_home if q == "home" else np.zeros(6))
    T = urdf.get_transform("adapter_front", urdf.base_link) @ HOOP_MOUNT
    arm = trimesh.collision.CollisionManager()
    for i, m in enumerate(collision_meshes(urdf) + collision_meshes(hoop, T)):
        arm.add_object(str(i), m)
    cell = trimesh.collision.CollisionManager()
    for name, m in cell_meshes().items():
        cell.add_object(name, m)
    assert cell.min_distance_other(arm) > 0.05


def test_table_cuts_the_foot():
    """The table is 25 mm below the top of the robot's foot (the square bottom
    50.8 mm of the base mesh), which sits in a cut-out of the table."""
    from cell import FOOT_TOP_Z
    base = load_urdf(MODELS / "crs93.urdf").collision_scene.dump(concatenate=True)
    assert base.bounds[0, 2] == pytest.approx(FOOT_TOP_Z - 0.0508, abs=1e-4)
    assert TABLE_Z == pytest.approx(FOOT_TOP_Z - 0.025)


def test_camera_rail_post_on_cage():
    from cell import CAGE_HEIGHT
    post = cell_meshes()["camera_rail_post"]
    assert post.bounds[0, 2] == pytest.approx(TABLE_Z + CAGE_HEIGHT)
    assert post.bounds[1, 2] == pytest.approx(RAIL_TOP_Z - 0.04)


def test_camera_sees_about_the_work_area():
    """Image footprint on the table ~ the work area (stickers just outside the
    view), image top edge towards the robot."""
    from cell import CAMERA_K, CAMERA_RESOLUTION
    corners = []
    for u, v in [(0, 0), (CAMERA_RESOLUTION[0], CAMERA_RESOLUTION[1])]:
        ray = CAMERA_T[:3, :3] @ np.linalg.solve(CAMERA_K, [u, v, 1.0])
        s = (TABLE_Z - CAMERA_T[2, 3]) / ray[2]
        corners.append(CAMERA_T[:3, 3] + s * ray)
    (x0, y0, _), (x1, y1, _) = corners
    assert x0 < x1  # image top (v = 0) is nearer to the robot
    size = np.array([x1 - x0, y1 - y0])
    assert np.allclose(size, np.ptp(WORK_AREA_XY, axis=1), rtol=0.1)
