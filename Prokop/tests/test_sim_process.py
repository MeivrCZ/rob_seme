"""Tests of the standalone simulator process (sim_process.SimProcess)."""

import time

import numpy as np
import pytest
from ctu_crs import CRS93

from sim_process import SimProcess
from simulation import CollisionError

INTO_TABLE = np.deg2rad([0, 90, 60, 0, 0, 0])  # as in test_collision_monitor


@pytest.fixture(scope="module")
def sim():
    from conftest import free_port
    s = SimProcess(port=free_port())
    s.robot.initialize(home=False)
    yield s
    s.stop()


def test_robot_is_a_crs93_with_local_kinematics(sim):
    robot = sim.robot
    assert isinstance(robot, CRS93)
    q = robot.q_home + 0.1
    assert np.allclose(robot.fk(q), CRS93(tty_dev=None).fk(q))
    assert robot.camera_delay > 0
    assert not hasattr(robot, "viewer")  # the scene is not part of the robot interface


def test_moves_images_and_truth(sim):
    robot = sim.robot
    t0 = time.monotonic()
    target = robot.q_home + np.deg2rad([15, 0, 0, 0, 0, 0])
    robot.move_to_q(target)
    robot.wait_for_motion_stop()
    assert np.allclose(robot.get_q(), target, atol=1e-3)
    assert not robot.in_motion()
    assert robot.grab_image().shape == (1200, 1920, 3)
    track = sim.truth_track(t0)
    assert len(track) > 10 and np.isclose(track[-1][1][0], target[0], atol=1e-3)
    sim.set_turntable(0.7, 0.0)
    assert np.isclose(sim.truth_angle(time.monotonic()), 0.7)


def test_collision_raises_collision_error(sim):
    robot = sim.robot
    robot.move_to_q(INTO_TABLE)
    with pytest.raises(CollisionError) as info:
        robot.wait_for_motion_stop()
    assert ("hoop", "table") in info.value.hits
