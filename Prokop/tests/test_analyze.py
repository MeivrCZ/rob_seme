"""tools/analyze_motion.py and tools/analyze_survey.py on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_motion  # noqa: E402
import analyze_survey  # noqa: E402


def synthetic_log(t0=0.12, v=np.deg2rad(30), a=np.deg2rad(250), d=np.deg2rad(30),
                  dt=0.06) -> np.ndarray:
    """Joint 1 out by @d with a trapezoid starting at @t0, sampled every @dt."""
    t = np.arange(0.05, 2.0, dt)
    log = np.zeros((len(t), 8))
    log[:, 0] = t
    log[:, 1] = analyze_motion.trapezoid(t, t0, v, a, d)
    log[:, 7] = 0.05
    return log


def test_fit_recovers_the_profile():
    f = analyze_motion.fit_first_move(synthetic_log(), 0)
    assert f["start_s"] == pytest.approx(0.12, abs=0.01)
    assert f["speed_deg_s"] == pytest.approx(30, rel=0.02)
    assert f["accel_deg_s2"] == pytest.approx(250, rel=0.3)  # 1-2 samples per ramp


def test_trapezoid_reaches_the_distance():
    s = analyze_motion.trapezoid(np.array([0.0, 10.0]), 0.0, 1.0, 2.0, 3.0)
    assert s[0] == 0.0 and s[1] == pytest.approx(3.0)


def test_out_leg_of_a_single_move():
    leg = analyze_motion.out_leg(synthetic_log(), [0])
    assert leg["duration_s"] == pytest.approx(30 / 30 + 30 / 250, abs=0.13)


def test_delays():
    d = analyze_motion.delays({"a": synthetic_log()})
    assert d["get_q_s"] == pytest.approx(0.05) and d["loop_s"] == pytest.approx(0.06)


def test_find_ring_ignores_other_blue():
    cv2 = pytest.importorskip("cv2")
    img = np.full((1200, 1920, 3), 230, np.uint8)
    cv2.circle(img, (600, 650), 215, (180, 60, 30), 10)  # blue ring (BGR)
    cv2.rectangle(img, (900, 0), (1020, 1200), (180, 60, 30), -1)  # sticker strip
    centre, diameter = analyze_survey.find_ring(img, np.array([620.0, 600.0]))
    assert np.allclose(centre, [600, 650], atol=1.5)
    assert diameter == pytest.approx(430, abs=6)
