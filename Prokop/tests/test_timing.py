"""solver/timing.py (P3.2): the same timing as CRS93Sim and the real logs."""

import numpy as np
import pytest
import simulation
from ctu_crs import CRS93
from solver.timing import (BLEND_TOL, COMMAND_DELAY, SEND_DELAY, RobotTiming, TimedMove,
                           same_direction)

D = np.deg2rad


@pytest.fixture(scope="module")
def crs():
    return CRS93(tty_dev=None)


def test_constants_equal_the_simulator():
    assert COMMAND_DELAY == simulation.COMMAND_DELAY
    assert SEND_DELAY == simulation.SEND_DELAY
    assert BLEND_TOL == simulation.BLEND_TOL


@pytest.mark.parametrize("speed", [0.0, 0.25, 0.5, 1.0])
def test_profile_equals_the_simulator(crs, speed, rng):
    """Same durations as CRS93Sim._profile at the same speed setting (no viewer)."""
    sim = object.__new__(simulation.CRS93Sim)  # only the timing attributes
    CRS93.__init__(sim, tty_dev=None)
    simulation.CRS93Sim.set_speed(sim, sim._min_speed_irc256_per_ms + speed * (
        sim._max_speed_irc256_per_ms - sim._min_speed_irc256_per_ms))
    simulation.CRS93Sim.set_acceleration(sim, sim._default_acceleration_irc_per_ms)
    timing = RobotTiming(crs, speed)
    for _ in range(20):
        dq = rng.uniform(-1.0, 1.0, 6) * rng.uniform(0.01, 1.0)
        assert timing.profile(dq) == pytest.approx(simulation.CRS93Sim._profile(sim, dq))


def test_matches_the_real_robot_speeds(crs):
    """Peak joint speeds of the real logs (E5): j1 17.6 / 29.8 deg/s, j6 147.1 deg/s."""
    assert np.rad2deg(RobotTiming(crs, 0.25).v_max[0]) == pytest.approx(17.6, abs=0.2)
    assert np.rad2deg(RobotTiming(crs, 0.5).v_max[0]) == pytest.approx(29.8, abs=0.2)
    assert np.rad2deg(RobotTiming(crs, 0.5).v_max[5]) == pytest.approx(147.1, abs=1.0)


def test_timed_move_inverse(crs):
    T, tau = RobotTiming(crs, 0.3).profile(D([30, 0, 0, 0, 0, 0]))
    move = TimedMove(1.0, T, tau, np.zeros(6), D([30, 0, 0, 0, 0, 0]))
    for s in (0.0, 0.05, 0.3, 0.5, 0.9, 1.0):
        assert move.progress(move.time_at(s)) == pytest.approx(s, abs=1e-9)


def test_same_direction_chain_blends(crs):
    """Ten queued 3 deg steps arrive like one 30 deg move (E6)."""
    timing = RobotTiming(crs, 0.5)
    q0 = crs.q_home
    chain = [q0 + D([3 * (i + 1), 0, 0, 0, 0, 0]) for i in range(10)]
    arrivals, moves = timing.schedule(q0, chain)
    assert len(moves) == 1
    single = COMMAND_DELAY + timing.move_duration(q0, chain[-1])
    assert arrivals[-1] == pytest.approx(single)
    assert all(a < b for a, b in zip(arrivals, arrivals[1:]))


def test_corner_and_reversal_stop(crs):
    timing = RobotTiming(crs, 0.5)
    q0 = crs.q_home
    for second in ([20, 0, 0, 40, 0, 0], [5, 0, 0, 0, 0, 0]):
        a = q0 + D([20, 0, 0, 0, 0, 0])
        b = q0 + D(second)
        arrivals, moves = timing.schedule(q0, [a, b])
        assert len(moves) == 2 and moves[1].t0 == pytest.approx(moves[0].t_end)
        assert arrivals[1] == pytest.approx(moves[0].t_end + timing.move_duration(a, b))


def test_late_command_does_not_blend(crs):
    """A same-direction command arriving while the move slows down starts anew."""
    timing = RobotTiming(crs, 0.5)
    q0 = crs.q_home
    a, b = q0 + D([10, 0, 0, 0, 0, 0]), q0 + D([20, 0, 0, 0, 0, 0])
    _, moves = timing.schedule(q0, [a], t_call=0.0)
    late = moves[0].t_end - 0.5 * moves[0].tau - COMMAND_DELAY
    _, moves = timing.schedule(q0, [a, b], call_times=[0.0, late])
    assert len(moves) == 2


def test_schedule_matches_the_simulator(crs):
    """The sim (real time) executes a queued chain with the scheduled timing."""
    sim = simulation.CRS93Sim(port=0, keep_alive=False, collisions=False)
    try:
        sim.initialize(home=False)
        sim.set_speed_relative(0.5)
        timing = RobotTiming(crs, 0.5)
        q0 = sim.get_q()
        targets = [q0 + D([10, 0, 0, 0, 0, 0]), q0 + D([20, 0, 0, 0, 0, 0]),
                   q0 + D([20, 0, 0, 30, 0, 0])]
        t_call = sim._now()
        for q in targets:
            sim.move_to_q(q)
        with sim._lock:
            ends = [seg.t_end - t_call for seg in sim._segments if seg.t_end > t_call]
        arrivals, moves = timing.schedule(q0, targets)
        assert len(ends) == len(moves) == 2
        # python overhead between the calls shifts the sim by a few ms
        assert np.allclose(ends, [m.t_end for m in moves], atol=0.02)
        sim.wait_for_motion_stop()
    finally:
        sim.viewer.server.stop()


def test_same_direction():
    assert same_direction(np.array([1.0, 0, 0]), np.array([2.0, 0, 0]))
    assert not same_direction(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))
    assert not same_direction(np.array([1.0, 0, 0]), np.array([1.0, 0.01, 0]))
    assert not same_direction(np.zeros(3), np.array([1.0, 0, 0]))
