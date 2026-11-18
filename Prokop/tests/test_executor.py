"""solver/executor.py (P3.3): executing a plan in the simulator with the puzzle spinning.

The tests act as the evaluation harness: they may read the simulator's truth.
"""

import time

import numpy as np
import pytest
import simulation
from ctu_crs import CRS93
from placement import Placement
from solver.contracts import (OMEGA, ModelTracker, Plan, PlanStep, TurntableModel,
                              example_plan)
from solver.executor import Executor
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
