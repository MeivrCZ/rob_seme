"""solver/executor.py (P3.3): executing a plan in the simulator with the puzzle spinning.

The tests act as the evaluation harness: they may read the simulator's truth.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import simulation
from ctu_crs import CRS93
from placement import Placement
from solver.contracts import (OMEGA, ModelTracker, PhaseEstimate, Plan, PlanStep, TurntableModel,
                              example_plan)
from solver.executor import Executor, _ExecutionRejected
from solver.timing import RobotTiming

D = np.deg2rad
Q_SURVEY = D([-7.5, -4.04, -91.71, 0.0, -84.26, 0.0])  # flange down over the work area


def spinning_sim(model_params, direction):
    """A real-time sim with the turntable placed and spinning; its truth as a model."""
    sim = simulation.CRS93Sim(port=0, keep_alive=False, collisions=False,
                              q0=Q_SURVEY)
    sim.initialize(home=False)
    axis, offset, yaw = model_params
    sim.viewer.set_placement(Placement(np.array(axis), np.array(offset), yaw))
    sim.viewer.set_spin(direction * OMEGA)
    t_ref = time.monotonic()
    truth = TurntableModel(axis_xy=np.array(axis), base_offset=np.array(offset),
                           base_yaw=yaw, direction=direction,
                           theta_ref=sim.viewer.turntable_angle(t_ref - sim._t_start),
                           t_ref=t_ref)
    return sim, truth


def puzzle_fixed_plan(truth: TurntableModel, theta_start: float, speed: float) -> tuple:
    """A plan of small moves near Q_SURVEY whose targets are fixed in the puzzle frame."""
    crs = CRS93(tty_dev=None)
    qs = [Q_SURVEY + D([1.5 * i, (-1) ** i * 1.0, 0.0, 0.0, 0.0, 2.0 * i]) for i in range(1, 7)]
    arrivals, _ = RobotTiming(crs, speed).schedule(Q_SURVEY, qs)
    steps = [PlanStep(index=i, q=q, roll=0.0, t_rel=t, margin=1.0)
             for i, (q, t) in enumerate(zip(qs, arrivals))]
    plan = Plan(feasible=True, speed=speed, direction=truth.direction,
                theta_start=theta_start, q_start=Q_SURVEY.copy(), steps=steps,
                approach=[Q_SURVEY.copy()], min_margin=1.0)
    in_puzzle = [np.linalg.inv(truth.puzzle_T(theta_start + truth.direction * OMEGA * s.t_rel))
                 @ crs.fk(s.q) for s in steps]
    return plan, in_puzzle


@pytest.mark.parametrize("direction", [1, -1])
def test_hoop_follows_the_spinning_puzzle(direction):
    sim, truth = spinning_sim(((0.45, 0.02), (0.02, -0.01), 0.4), direction)
    try:
        theta_start = truth.theta_at(time.monotonic() + 2.0)  # start in ~2 s
        plan, in_puzzle = puzzle_fixed_plan(truth, theta_start, speed=0.3)
        sent_end = []  # the simulator's end time of every move (truth)
        move = sim.move_to_q

        def recording_move(q):
            move(q)
            sent_end.append(sim._motion_end() + sim._t_start)

        sim.move_to_q = recording_move
        executor = Executor(sim, truth, plan, ModelTracker(truth),
                            RobotTiming(sim, plan.speed), log=lambda *a: None)
        result = executor.run()
        assert result.success, result.reason
        traverse_ends = sent_end[-len(plan.steps):]
        for rec, t_end, X in zip(result.steps, traverse_ends, in_puzzle):
            # the timing model predicted the arrival (python overhead: a few ms)
            assert rec.t_arrival == pytest.approx(t_end, abs=0.03)
            # at arrival the hoop target sits where the plan wants it on the puzzle
            want = truth.puzzle_T(truth.theta_at(t_end)) @ X
            got = sim.fk(rec.q)
            assert np.linalg.norm(got[:3, 3] - want[:3, 3]) < 0.003  # 3 mm
        # the traversal started at the planned phase
        first = result.steps[0]
        assert abs(first.theta_predicted - first.theta_planned) < D(2)
    finally:
        sim.viewer.server.stop()


def test_infeasible_plan_is_reported_and_nothing_moves():
    sim = simulation.CRS93Sim(port=0, keep_alive=False, collisions=False)
    try:
        sim.initialize(home=False)
        q0 = sim.get_q()
        messages = []
        result = Executor(sim, TurntableModel(np.zeros(2), np.zeros(2), 0.0, 1),
                          Plan(feasible=False, reason="no start phase reachable"),
                          ModelTracker(TurntableModel(np.zeros(2), np.zeros(2), 0.0, 1)),
                          RobotTiming(sim, 0.2), log=messages.append).run()
        assert not result.success and "unreachable" in result.reason
        assert messages and "Unreachable" in messages[0]
        assert np.allclose(sim.get_q(), q0)
    finally:
        sim.viewer.server.stop()


def test_large_phase_error_aborts():
    """A plan for a phase the turntable never matches (wrong direction) aborts."""
    sim, truth = spinning_sim(((0.45, 0.0), (0.0, 0.0), 0.0), 1)
    try:
        plan = example_plan(speed=0.3)
        plan.direction = -1  # planned for the other direction: the error grows
        plan.theta_start = truth.theta_at(time.monotonic() + 1.0)
        result = Executor(sim, truth, plan, ModelTracker(truth), RobotTiming(sim, 0.3),
                          log=lambda *a: None).run()
        assert not result.success and "not acceptable" in result.reason
        assert len(result.steps) <= 2  # the error reaches 12 deg at step 0, 24 at step 1
    finally:
        sim.viewer.server.stop()


@pytest.mark.parametrize("reject_second, late_second", [(False, False), (True, False), (False, True)])
def test_validated_stationary_plan_starts_without_a_tracker(reject_second, late_second):
    class Robot:
        q = Q_SURVEY.copy()
        pending = None

        def get_q(self):
            return self.q.copy()

        def set_speed_relative(self, _speed):
            pass

        def move_to_q(self, q):
            assert self.pending is None
            self.pending = q.copy()

        def wait_for_motion_stop(self):
            if self.pending is not None:
                self.q, self.pending = self.pending, None

    clock_time = 10.0

    def sleep(duration):
        nonlocal clock_time
        clock_time += duration

    robot = Robot()
    model = TurntableModel(np.array([0.5, 0.0]), np.zeros(2), 0.0, 1,
                           omega=0.0, theta_ref=0.4)
    targets = [Q_SURVEY + D([2 * end, 0, 0, 0, 0, 0]) for end in (1, 2)]
    timing = RobotTiming(CRS93(tty_dev=None), speed=0.3)
    arrivals, _ = timing.schedule_stopped(robot.q, targets)
    steps = [PlanStep(i, q, 0.0, t, 1.0) for i, (q, t) in enumerate(zip(targets, arrivals))]
    plan = Plan(True, speed=timing.speed, theta_start=model.theta_ref,
                q_start=robot.q.copy(), steps=steps)
    calls = []

    def validate(move, theta_at):
        assert clock_time < move.t0
        assert theta_at(move.t_end) == model.theta_ref
        calls.append(move)
        if late_second and len(calls) >= 2:
            sleep(0.60)
        return len(calls) == 1 or not reject_second

    result = Executor(robot, model, plan, None, timing, clock=lambda: clock_time,
                      sleep=sleep, validate_move=validate).run()
    assert len(calls) == (3 if late_second else 2)
    assert result.success is not reject_second
    if reject_second:
        assert "collision check" in result.reason
        assert len(result.steps) == 1
        assert robot.q == pytest.approx(targets[0])
    else:
        assert len(result.steps) == 2
        assert robot.q == pytest.approx(targets[-1])
        if late_second:
            assert calls[2].hold_from == result.steps[0].t_arrival
            assert result.steps[1].t_arrival - result.t_start > arrivals[1]
        else:
            assert [step.t_arrival - result.t_start for step in result.steps] == pytest.approx(arrivals)


def test_occlusion_restricts_start_and_uses_uncertainty_during_traversal():
    model = TurntableModel(np.array([0.5, 0.0]), np.zeros(2), 0.0, 1)
    estimate = PhaseEstimate(10.0, 0.0, OMEGA, np.diag(D([0.4, 0.2]) ** 2))
    tracker = SimpleNamespace(estimate=lambda: estimate)
    executor = Executor(None, model, example_plan(), tracker, None)

    with pytest.raises(_ExecutionRejected, match="too old to start"):
        executor._current_phase(12.0)
    assert executor._current_phase(12.0, require_fresh=False).t == 10.0
    with pytest.raises(_ExecutionRejected, match="phase uncertainty"):
        executor._current_phase(26.0, require_fresh=False)


def test_exit_to_waiting_pose_is_fixed_even_when_the_puzzle_turns():
    model = TurntableModel(np.array([0.5, 0.0]), np.zeros(2), 0.0, 1)
    timing = RobotTiming(CRS93(tty_dev=None), speed=0.3)
    goal = Q_SURVEY + D([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plan = Plan(True, speed=timing.speed)
    step = PlanStep(1, goal, 0.0, 1.0, 0.03, track_phase=False)
    estimate = PhaseEstimate(10.0, np.pi / 2, OMEGA, np.zeros((2, 2)))
    executor = Executor(None, model, plan, None, timing, clock=lambda: 10.0)

    move, record = executor._prepare_stopped_move(step, Q_SURVEY, 10.2, estimate)

    assert move.q_to == pytest.approx(goal)
    assert record.dtheta == 0.0
    assert record.theta_predicted == pytest.approx(estimate.predict(move.t_end)[0])
