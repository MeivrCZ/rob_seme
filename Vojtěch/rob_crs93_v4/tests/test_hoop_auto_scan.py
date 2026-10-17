"""End-to-end auto scan under simulated geometry, no physical device."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from homography_capture import HomographySession
from hoop_auto_scan import HoopAutoScanner


class FakeRobot:
    mode = 'real'
    released = False
    real_motion_enabled = True

    def __init__(self):
        self.q = np.array([.02, 0., .50, 0., 0., 0.], float)
        self._table_guard = None
        self._hoop_offset_tcp = None
        self.moves = []

    def assert_stationary(self): pass
    def get_q(self): return self.q.copy()
    def require(self): return self
    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q[:3], float)
        return T
    def ik(self, pose):
        return [np.array([*pose[:3, 3], 0, 0, 0], float)]
    def get_joint_limits_deg(self):
        return None, np.array([[-180,180]]*6)
    def set_table_z(self, val):
        self._table_guard = type('Guard', (), {'table_z_m': float(val)})()
    def set_hoop_offset(self, val):
        self._hoop_offset_tcp = tuple(val)
    def check_table_clearance(self, q):
        return self.fk(q)[:3, 3] + np.array([.135,0,0])
    def preview_move(self, q):
        return self.q.copy(), np.array(q, float)
    def move_to_q(self, q, *, expected_q=None):
        np.testing.assert_allclose(self.q, expected_q)
        self.q = np.asarray(q, float).copy()
        self.moves.append(self.q.copy())


class FakeCamera:
    mode = 'real'
    def capture(self):
        return np.zeros((1200,1920,3), np.uint8)


def test_complete_auto_scan_and_solve(tmp_path):
    robot, cam = FakeRobot(), FakeCamera()
    session = HomographySession(folder=tmp_path)
    session.target_offset_tcp_m = (.135, 0., 0.)
    scanner = HoopAutoScanner(robot, cam, session)
    plan = scanner.make_plan(.15,.45,-.10,.10,.20,30, nx=5, ny=5, offset_verified=True)
    assert len(plan.points) == 25 and plan.total_moves > 25
    assert robot.moves == []  # PREVIEW must not move hardware.
    def simulated_detection(frame):
        T = robot.fk(robot.get_q())
        x,y,_ = T[:3,3] + np.array([.135,0,0])
        return 800 + 1400*x, 600 + 1400*y
    with patch('hoop_auto_scan.detect_single_circle', side_effect=simulated_detection):
        assert scanner.run() == 25
    assert len(robot.moves) == len(plan.segments)
    assert session.homography is not None
    assert session.homography.rms_px < .01
    x,y = session.pixel_to_world(800+1400*.35, 600+1400*.02)
    assert x == pytest.approx(.35, abs=.001)
    assert y == pytest.approx(.02, abs=.001)
    assert scanner.plan is None


def test_failed_detection_does_not_calibrate(tmp_path):
    robot, cam = FakeRobot(), FakeCamera()
    session = HomographySession(folder=tmp_path)
    session.target_offset_tcp_m = (.135,0,0)
    scanner = HoopAutoScanner(robot, cam, session)
    scanner.make_plan(.15,.45,-.10,.10,.2,30, nx=3, ny=3, offset_verified=True)
    with patch('hoop_auto_scan.detect_single_circle', return_value=None):
        with pytest.raises(RuntimeError, match='nenasel'):
            scanner.run()
    assert not session.samples
    assert session.homography is None
    assert scanner.plan is None
