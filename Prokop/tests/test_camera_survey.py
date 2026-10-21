"""tools/camera_survey.py on a fake robot whose camera shows the arm with a known
delay."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import camera_survey  # noqa: E402


class FakeRobot:
    """Joint 1 moves with a trapezoid-like smooth profile (0.8 s) after move_to_q;
    the image brightness is a nonlinear function of the joint angle @delay
    seconds ago; a grab takes @frame_time."""

    def __init__(self, delay: float, frame_time: float = 0.03):
        self.q_home = np.zeros(6)
        self.delay, self.frame_time = delay, frame_time
        self._moves = [(-1.0, 0.0, 0.0)]  # (t_start, q_from, q_to)

    def set_speed_relative(self, fraction):
        pass

    def _q1(self, t: float) -> float:
        q = 0.0
        for t0, a, b in self._moves:
            if t >= t0:
                s = float(np.clip((t - t0) / 0.8, 0.0, 1.0))
                q = a + (b - a) * (3 * s**2 - 2 * s**3)  # slow start and end
        return q

    def move_to_q(self, q):
        now = time.monotonic()
        self._moves.append((now + 0.05, self._q1(now), float(q[0])))

    def get_q(self):
        return np.array([self._q1(time.monotonic()), 0, 0, 0, 0, 0])

    def in_motion(self) -> bool:
        return time.monotonic() < self._moves[-1][0] + 0.8

    def grab_image(self):
        time.sleep(self.frame_time)
        q1 = self._q1(time.monotonic() - self.delay)
        level = 40 + 150 * np.sqrt(abs(q1) / np.deg2rad(10))  # nonlinear, monotonic
        return np.full((60, 80, 3), level).astype(np.uint8)


@pytest.mark.parametrize("delay", [0.0, 0.08, 0.2])
def test_delay_estimate(delay):
    frames, states = camera_survey.record_out_and_back(FakeRobot(delay))
    assert frames.shape[1] == 2 and states.shape[1] == 2
    r = camera_survey.estimate_delay(frames, states)
    assert r["delay_s"] == pytest.approx(delay, abs=0.03)
    assert r["mismatch"] <= r["mismatch_at_0"]


def test_grab_timing(tmp_path):
    r = camera_survey.grab_timing(FakeRobot(0.0, frame_time=0.01), tmp_path)
    assert (r["width"], r["height"], r["channels"]) == (80, 60, 3)
    assert r["grab_s_median"] >= 0.01
    assert (tmp_path / "q_home.png").is_file()


def test_survey_pose_visible_and_clear(robot):
    from cell import CAMERA_RESOLUTION, TABLE_Z
    from collision import ArmChecker
    checker = ArmChecker()
    q, info = camera_survey.survey_pose(robot, checker)
    swing = np.zeros(6)
    swing[0] = np.deg2rad(camera_survey.DELAY_MOVE_DEG)
    for qq in (q, q + swing):
        assert robot.in_limits(qq)
        h = checker.hoop_center(qq)
        uv = camera_survey.project(h)
        assert np.all(uv > 0) and np.all(uv < CAMERA_RESOLUTION)
        assert camera_survey.HOOP_HEIGHT[0] <= h[2] - TABLE_Z <= camera_survey.HOOP_HEIGHT[1]
    assert all(d >= camera_survey.MIN_CLEARANCE for d, _ in info["clearance_m"].values())
