"""solver/contracts.py: the reference turntable chain against the simulator, the fakes."""

import numpy as np
import pytest
from placement import Placement, TaskScene
from solver.contracts import (OMEGA, ModelTracker, PhaseEstimate, TurntableModel, example_model,
                              example_plan, fake_measurements, rot_z)


@pytest.fixture(scope="module")
def scene():
    return TaskScene()


@pytest.mark.parametrize("theta", [0.0, 0.7, -2.0, 4.0])
def test_puzzle_T_matches_the_simulator(scene, theta):
    """The contract's chain equals the simulator's placement of the puzzle."""
    m = TurntableModel(axis_xy=np.array([0.5, 0.1]), base_offset=np.array([0.03, -0.02]),
                       base_yaw=1.1, direction=1)
    p = Placement(m.axis_xy, m.base_offset, m.base_yaw)
    sim = (p.turntable_T() @ rot_z(theta) @ scene.T_top @ p.base_T_top()
           @ scene.T_mount)
    assert np.allclose(m.puzzle_T(theta), sim, atol=1e-12)


def test_theta_and_direction():
    for d in (1, -1):
        m = example_model(direction=d, t_ref=10.0)
        assert m.theta_at(10.0) == pytest.approx(m.theta_ref)
        assert m.theta_at(10.0 + 28.0) - m.theta_ref == pytest.approx(d * 2 * np.pi)
    assert OMEGA == pytest.approx(2 * np.pi / 28.0)


def test_measurement_moves_on_a_circle_around_the_axis():
    m = example_model()
    r = [np.linalg.norm(m.measurement_at(t)[0] - m.axis_xy) for t in np.linspace(0, 28, 9)]
    assert np.allclose(r, np.linalg.norm(m.base_offset))
    _, yaw0 = m.measurement_at(0.0)
    _, yaw1 = m.measurement_at(1.0)
    assert (yaw1 - yaw0) == pytest.approx(OMEGA, rel=1e-9)


def test_fake_measurements_noise_and_missing():
    m = example_model()
    times = np.arange(0, 7, 0.2)
    exact = fake_measurements(m, times, sigma_xy=0, sigma_yaw=0)
    assert len(exact) == len(times)
    for meas in exact:
        xy, yaw = m.measurement_at(meas.t_capture)
        assert np.allclose(meas.xy, xy) and meas.yaw == pytest.approx(yaw)
    some = fake_measurements(m, times, p_missing=0.5)
    assert 0 < len(some) < len(times)
    assert all(a.t_capture < b.t_capture for a, b in zip(some, some[1:]))


def test_checkpoint_T_composes(scene, puzzle_path):
    m, path = example_model(), puzzle_path
    T = m.checkpoint_T(path, 3, 0.5)
    assert np.allclose(T, m.puzzle_T(0.5) @ path.frames[3])


def test_phase_estimate_prediction():
    est = PhaseEstimate(t=5.0, theta=1.0, omega=-OMEGA, covariance=np.diag([1e-4, 1e-6]))
    theta, sd = est.predict(7.0)
    assert theta == pytest.approx(1.0 - 2 * OMEGA)
    assert sd == pytest.approx(np.sqrt(1e-4 + 4e-6))


def test_example_plan_is_consistent():
    plan = example_plan()
    assert plan.feasible and plan.steps
    assert all(a.t_rel < b.t_rel for a, b in zip(plan.steps, plan.steps[1:]))
    assert all(s.q.shape == (6,) for s in plan.steps)


def test_model_tracker_reports_the_model():
    m = example_model(direction=-1, t_ref=3.0)
    tracker = ModelTracker(m, clock=lambda: 10.0)
    est = tracker.estimate()
    assert est.t == 10.0 and est.theta == pytest.approx(m.theta_at(10.0))
    assert est.omega == pytest.approx(-OMEGA)
    assert est.predict(12.0)[0] == pytest.approx(m.theta_at(12.0))
