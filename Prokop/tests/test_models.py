"""models/: URDFs load, meshes exist, the robot URDF matches ctu_crs."""

import numpy as np
import pytest
import yourdfpy
from placement import MODELS, load_urdf

URDFS = sorted(MODELS.glob("*.urdf"))


@pytest.fixture(scope="module")
def crs93():
    return load_urdf(MODELS / "crs93.urdf")


@pytest.mark.parametrize("path", URDFS, ids=lambda p: p.name)
def test_urdf_loads_with_meshes(path):
    urdf = load_urdf(path)
    for link in urdf.link_map.values():
        for geom in [*link.visuals, *link.collisions]:
            if geom.geometry.mesh is not None:
                assert (path.parent / geom.geometry.mesh.filename).is_file()


def test_robot_joints_match_ctu(crs93, robot):
    names = crs93.actuated_joint_names
    assert len(names) == 6
    lo = np.array([crs93.joint_map[n].limit.lower for n in names])
    hi = np.array([crs93.joint_map[n].limit.upper for n in names])
    assert np.allclose(lo, robot.q_min) and np.allclose(hi, robot.q_max)


def check_flange(crs93, robot, q):
    crs93.update_cfg(q)
    assert np.allclose(crs93.get_transform("flange", crs93.base_link), robot.fk(q),
                       atol=1e-9)


def test_flange_equals_fk_random(crs93, robot, rng):
    for _ in range(100):
        check_flange(crs93, robot, rng.uniform(robot.q_min, robot.q_max))


@pytest.mark.parametrize("name", ["q_min", "q_max", "q_home", "zero"])
def test_flange_equals_fk_special(crs93, robot, name):
    q = np.zeros(6) if name == "zero" else getattr(robot, name)
    check_flange(crs93, robot, q)


def test_robot_collision_meshes_closed(crs93):
    """No holes: every edge is shared by an even number of triangles. (Not
    trimesh's is_watertight, which wants exactly 2: the rebuilt links have a few
    edges where two convex pieces touch, shared by 4.)"""
    scene = crs93.collision_scene
    for name, mesh in scene.geometry.items():
        if name == "crs_adapter.stl":
            continue  # CAD export: stray triangles + touching parts, see below
        _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        assert np.all(counts % 2 == 0), name


def test_adapter_mesh_has_no_open_edges(crs93):
    mesh = crs93.collision_scene.geometry["crs_adapter.stl"]
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    assert np.all(counts >= 2)


def test_adapter_points_up_in_home(crs93, robot):
    """Triangle vertex of the adapter (-y), and with it the hoop, points up in
    q_home (the adapter is turned by 180 deg on the CRS93 station), its front
    (-z) along the flange z axis."""
    crs93.update_cfg(robot.q_home)
    T = crs93.get_transform("adapter_front", crs93.base_link)
    assert np.allclose(-T[:3, 1], [0, 0, 1], atol=1e-9)
    assert np.allclose(-T[:3, 2], robot.fk(robot.q_home)[:3, 2], atol=1e-9)


def test_yourdfpy_version_has_collision_scene():
    assert hasattr(yourdfpy.URDF, "collision_scene")
