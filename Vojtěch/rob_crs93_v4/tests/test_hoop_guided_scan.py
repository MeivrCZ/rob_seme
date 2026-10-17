"""Adaptive scan integration tests run ONLY on deterministic fake robot/camera."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from homography_capture import HomographySession
from hoop_detection import HoopDetection
from hoop_guided_scan import GuidedHoopScanner
from test_hoop_auto_scan import FakeRobot, FakeCamera


def create(tmp_path):
    robot, cam = FakeRobot(), FakeCamera()
    robot.q = np.array([.02, 0, .50, 0, 0, 0], float)
    sess = HomographySession(folder=tmp_path)
    sess.target_offset_tcp_m = (.135, 0., 0.)
    robot.set_hoop_offset(sess.target_offset_tcp_m)
    return robot, cam, sess, GuidedHoopScanner(robot, cam, sess)


def detector(robot, *, visible=lambda x, y: True):
    def mock(image):
        T = robot.fk(robot.get_q())
        xy = (T @ np.array([.135, 0., 0., 1.]))[:2]
        u, v = 960 + 3400 * (xy[0] + .115), 600 + 3400 * xy[1]
        if visible(*xy) and 100 < u < 1820 and 100 < v < 1100:
            return HoopDetection((u, v), 70., 'synthetic', 'OK', ((u,v,70.),))
        return HoopDetection(None,None,'synthetic','out of view',())
    return mock


def test_prepare_never_moves_and_plans_reposition_from_wrong_height(tmp_path):
    robot, _, _, scanner = create(tmp_path)
    with patch('hoop_guided_scan.detect_hoop_circle', side_effect=detector(robot)):
        plan = scanner.prepare(.4, 10, 3, 3)
    assert plan.center_z == pytest.approx(.5)
    assert robot.moves == []
    other_plan = scanner.prepare(.3, 10, 3, 3)
    assert other_plan.initial_uv is None
    assert other_plan.seed_q.shape == (6,)
    assert other_plan.center_z == pytest.approx(.4)
    assert not robot.moves


def test_adaptive_run_learns_jacobian_and_solves(tmp_path):
    robot, cam, sess, scanner = create(tmp_path)
    with patch('hoop_guided_scan.detect_hoop_circle', side_effect=detector(robot)):
        scanner.prepare(.4, 10, 3, 3)
        calib = scanner.run()
    assert len(sess.samples) >= 8
    assert len(robot.moves) > 10
    assert calib.rms_px < .01
    assert calib.inliers >= 8
    assert scanner.setup is None
    x, y = sess.pixel_to_world(960, 600)
    assert x == pytest.approx(-.115, abs=.003)
    assert y == pytest.approx(0, abs=.003)


def test_lost_hoop_undoes_only_previous_small_step(tmp_path):
    robot, cam, sess, scanner = create(tmp_path)
    # Leave a narrow stripe of visibility. A step beyond its edge must rollback.
    with patch('hoop_guided_scan.detect_hoop_circle', side_effect=detector(robot, visible=lambda x,y: x < -.110)):
        scanner.prepare(.4, 10, 3, 3)
        first = scanner._observe(sess.target_offset_tcp_m)
        after = scanner._attempt(first, [0.008, 0])
        assert after is None
        np.testing.assert_allclose(robot.get_q(), first.q, atol=1e-9)
        assert scanner.failed_moves == 1
        assert len(robot.moves) == 2
    assert (tmp_path/'guided_lost_raw.png').exists()


def test_invisible_hoop_prepares_but_does_not_move_until_run(tmp_path):
    robot, cam, sess, scanner = create(tmp_path)
    with patch('hoop_guided_scan.detect_hoop_circle', return_value=HoopDetection(None,None,'fake','no ring',())):
        setup = scanner.prepare(.4,10,3,3)
    assert setup.initial_uv is None
    assert robot.moves == []
    with patch('hoop_guided_scan.detect_hoop_circle', return_value=HoopDetection(None,None,'fake','no ring',())):
        with pytest.raises(RuntimeError, match='ZASTAVENY'):
            scanner.run()
    assert robot.moves  # acquisition is permitted only after run
    assert sess.samples == []
    assert scanner.setup is None
    np.testing.assert_allclose(robot.get_q(), setup.seed_q) == []


def test_from_home_autonomously_acquires_seed_before_feedback(tmp_path):
    robot, cam, sess, scanner = create(tmp_path)
    # Artificial high home: target ring Z=0.5, initial Z~0.9.
    robot.q = np.array([0.415, -0.17, .91, 0, 0, 0], float)
    def camera_simulator(image):
        xyz = (robot.fk(robot.get_q()) @ np.array([.135, 0., 0., 1.]))[:3]
        # At home the ring is not visible. Once lowered to the known starting
        # position, the camera shows a ring; displacement maps to 2D pixels.
        if abs(xyz[2] - .5) > .006:
            return HoopDetection(None,None,'synthetic','high above camera',())
        u = 320 + 3000 * (xyz[1] + .17)
        v = 820 + 3000 * (xyz[0] - .28)
        if not (85 < u < 1835 and 85 < v < 1115):
            return HoopDetection(None,None,'synthetic','out of frame',())
        return HoopDetection((u,v),70.,'synthetic','OK',((u,v,70.),))
    with patch('hoop_guided_scan.detect_hoop_circle', side_effect=camera_simulator):
        setup = scanner.prepare(.4, 10, 3, 3)
        assert setup.initial_uv is None
        assert not robot.moves
        assert setup.seed_q[2] == pytest.approx(.5, abs=.03)
        calibration = scanner.run()
    assert len(robot.moves) > 15
    assert len(sess.samples) >= 8
    assert calibration.rms_px < .01
    assert scanner.setup is None


def test_no_seed_ik_prevents_all_motion(tmp_path):
    robot, cam, sess, scanner = create(tmp_path)
    robot.q = np.array([0.415, -0.17, .91, 0, 0, 0], float)
    with patch.object(robot, 'ik', return_value=[]):
        with pytest.raises(RuntimeError, match='IK'):
            scanner.prepare(.4,10,3,3)
    assert robot.moves == []
