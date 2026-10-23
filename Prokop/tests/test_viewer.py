"""view_models.SceneViewer: builds, accepts configurations, turntable spin."""

import numpy as np
import pytest
from conftest import free_port
from view_models import SceneViewer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(scope="module")
def clock():
    return Clock()


@pytest.fixture(scope="module")
def viewer(clock):
    v = SceneViewer(port=free_port(), clock=clock)
    yield v
    v.server.stop()


def test_show_q(viewer, robot):
    for q in (robot.q_home, robot.q_min, robot.q_max, np.zeros(6)):
        viewer.show_q(q)
        T = viewer.robot_urdf.get_transform("flange", viewer.robot_urdf.base_link)
        assert np.allclose(T, robot.fk(q), atol=1e-9)


def test_show_without_q(viewer):
    viewer.show(0.0)


def test_spin(viewer, clock):
    clock.t = 10.0
    viewer.set_spin(2 * np.pi / 16)
    a0 = viewer.turntable_angle(clock.t)
    assert viewer.turntable_angle(clock.t + 4.0) == pytest.approx(a0 + np.pi / 2)
    clock.t += 4.0
    viewer.set_spin(0.0)  # stop: the angle stays
    a1 = viewer.turntable_angle(clock.t)
    assert a1 == pytest.approx((a0 + np.pi / 2) % (2 * np.pi))
    assert viewer.turntable_angle(clock.t + 100.0) == pytest.approx(a1)


def test_spin_clockwise(viewer, clock):
    viewer.set_spin(-2 * np.pi / 16)
    a0 = viewer.turntable_angle(clock.t)
    assert viewer.turntable_angle(clock.t + 4.0) == pytest.approx(a0 - np.pi / 2)
    viewer.set_spin(0.0)


def test_set_placement(viewer):
    from placement import Placement
    viewer.add_turntable_gui(animate=False)
    p = Placement(np.array([0.5, 0.05]), np.array([0.01, -0.02]), 0.4)
    viewer.set_placement(p)
    assert viewer.placement is p
    assert np.allclose(viewer.gizmo.position[:2], p.turntable_xy)
