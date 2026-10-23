"""CRS93Sim's hard collision handling: stop, CollisionError, lock, Release."""

import numpy as np
import pytest
from conftest import free_port
from placement import Placement
from simulation import CollisionError, CRS93Sim
from test_collision import IN_HOOK_PATH, PLACEMENT

INTO_TABLE = np.deg2rad([0, 90, 60, 0, 0, 0])  # ends deep in the table behind


@pytest.fixture
def sim():
    s = CRS93Sim(port=free_port(), time_scale=4.0, keep_alive=False)
    s.viewer.set_placement(PLACEMENT)
    s.initialize(home=False)
    yield s
    s.stop()


def drive_into_table(sim) -> CollisionError:
    sim.set_speed_relative(1.0)
    sim.move_to_q(INTO_TABLE)
    with pytest.raises(CollisionError) as info:
        sim.wait_for_motion_stop()
    return info.value


def test_stops_just_before_contact(sim):
    e = drive_into_table(sim)
    assert ("hoop", "table") in e.hits
    hits, d = sim._collider.check(e.q_stop, sim.viewer.placement, 0.0)
    assert hits == [] and d < 0.01  # stopped at the last collision-free check
    assert np.allclose(e.q_start, sim.q_home, atol=1e-6)
    assert np.allclose(sim.get_q(), e.q_stop)  # get_q works, the robot stays
    assert not sim.in_motion()


def test_locked_until_release(sim):
    e = drive_into_table(sim)
    with pytest.raises(CollisionError):
        sim.move_to_q(sim.q_home)
    with pytest.raises(CollisionError):
        sim.soft_home()
    assert sim.grab_image().shape == (1200, 1920, 3)  # the camera still works
    sim._release()  # the viewer's Release button
    assert np.allclose(sim.get_q(), e.q_start, atol=1e-6)
    target = sim.q_home + np.deg2rad([10, 0, 0, 0, 0, 0])
    sim.move_to_q(target)
    sim.wait_for_motion_stop()
    assert np.allclose(sim.get_q(), target, atol=1e-3)


def test_error_raised_once_for_queries(sim):
    """The first call after the stop raises, whatever it is; later queries
    work."""
    sim.set_speed_relative(1.0)
    sim.move_to_q(INTO_TABLE)
    while sim._collision is None or sim._now() < sim._collision.t_free:
        sim._sleep(0.1)
    with pytest.raises(CollisionError):
        sim.get_q()
    sim.get_q()
    sim.in_motion()


def test_spinning_puzzle_hits_static_arm(sim):
    """The arm stands still in the path of puzzle D's hook; the spinning
    turntable brings the hook to it: robot and turntable stop."""
    sim.set_speed_relative(1.0)
    sim.move_to_q(IN_HOOK_PATH)
    sim.wait_for_motion_stop()
    sim.viewer.set_spin(2 * np.pi / 28.0)
    with pytest.raises(CollisionError) as info:
        for _ in range(400):  # ~20 s sim, the hook arrives after ~12 s
            sim._sleep(0.05)
            sim.get_q()
    assert ("adapter", "puzzle") in info.value.hits
    assert sim.viewer._omega == 0.0  # the turntable stopped too
    assert np.allclose(info.value.q_stop, sim.get_q())


def test_normal_motion_no_false_alarm(sim):
    for dq in ([30, 0, 0, 0, 0, 0], [-30, 10, 0, 0, 0, 0], [0, 0, 0, 90, 0, 90]):
        sim.move_to_q(sim.q_home + np.deg2rad(dq))
    sim.wait_for_motion_stop()
    sim.move_to_q(sim.q_home)
    sim.wait_for_motion_stop()
    assert sim._collision is None


def test_moving_turntable_out_of_the_way_then_release(sim):
    """After the hook hit, moving the turntable away and releasing lets the
    robot continue."""
    sim.set_speed_relative(1.0)
    sim.move_to_q(IN_HOOK_PATH)
    sim.wait_for_motion_stop()
    sim.viewer.set_spin(2 * np.pi / 28.0)
    with pytest.raises(CollisionError):
        for _ in range(400):
            sim._sleep(0.05)
            sim.get_q()
    sim.viewer.set_placement(Placement(np.array([0.50, 0.40]), np.zeros(2), 0.0))
    sim._release()
    sim.move_to_q(sim.q_home)
    sim.wait_for_motion_stop()
    assert sim._collision is None


def test_collisions_off():
    s = CRS93Sim(port=free_port(), time_scale=4.0, keep_alive=False,
                 collisions=False)
    try:
        s.initialize(home=False)
        s.set_speed_relative(1.0)
        s.move_to_q(INTO_TABLE)
        s.wait_for_motion_stop()  # no CollisionError: nothing is checked
        assert np.allclose(s.get_q(), INTO_TABLE, atol=1e-3)
    finally:
        s.stop()
