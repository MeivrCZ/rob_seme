"""simulation.CRS93Sim: the ctu_crs.CRS93 interface, shapes, joint limits."""

import time

import numpy as np
import pytest
from conftest import free_port
from simulation import CRS93Sim

TIME_SCALE = 50.0  # sim seconds per wall second, keeps the tests fast
IRC_TOL = 1e-4  # rad, targets are rounded to IRC like the real commands


def make_sim(**kwargs) -> CRS93Sim:
    """A fast sim without the collision monitor (it is tested on its own in
    test_collision_monitor.py and would slow the runs at TIME_SCALE)."""
    return CRS93Sim(port=free_port(), time_scale=TIME_SCALE, keep_alive=False,
                    collisions=False, **kwargs)


@pytest.fixture(scope="module")
def sim():
    s = make_sim()
    s.initialize(home=False)
    yield s
    s.stop()


@pytest.fixture
def stopped(sim):
    """The module sim at rest in q_home."""
    sim.initialize(home=False)  # drops queued motion
    sim.move_to_q(sim.q_home)
    sim.wait_for_motion_stop()
    return sim


def move(sim, q):
    sim.move_to_q(q)
    sim.wait_for_motion_stop()
    return sim.get_q()


# --- shapes and types ---

def test_get_q_shape(stopped):
    q = stopped.get_q()
    assert isinstance(q, np.ndarray) and q.shape == (6,) and q.dtype == np.float64


def test_in_motion_type(stopped):
    assert stopped.in_motion() is False


def test_fk_of_state(stopped):
    T = stopped.fk(stopped.get_q())
    assert T.shape == (4, 4)


def test_list_input_accepted(stopped):
    q = move(stopped, list(stopped.q_home + 0.1))
    assert np.allclose(q, stopped.q_home + 0.1, atol=IRC_TOL)


def test_wrong_number_of_joints(stopped):
    with pytest.raises(ValueError):
        stopped.move_to_q(np.zeros(5))


def test_set_speed_wrong_length(stopped):
    with pytest.raises(AssertionError):
        stopped.set_speed(np.ones(5))


# --- initialization ---

def test_move_before_initialize():
    s = make_sim()
    try:
        with pytest.raises(AssertionError):
            s.move_to_q(s.q_home)
    finally:
        s.stop()


def test_hard_home_ends_at_measured_pose():
    """Hard home ends where the real CRS93 did (2026-10-21), in degrees."""
    s = make_sim()
    try:
        s.hard_home()
        assert np.allclose(np.rad2deg(s.get_q()),
                           [163.4, 0.3, -56.0, 176.7, 14.9, -172.0], atol=0.15)
    finally:
        s.stop()


def test_initialize_homes_to_q_home():
    s = make_sim(q0=np.zeros(6))
    try:
        s.initialize(home=True)
        assert not s.in_motion()
        assert np.allclose(s.get_q(), s.q_home, atol=IRC_TOL)
    finally:
        s.stop()


# --- motion ---

def test_reaches_target(stopped):
    q = stopped.q_home + np.deg2rad([20, -10, 15, 30, -20, 45])
    assert np.allclose(move(stopped, q), q, atol=IRC_TOL)


def test_in_motion_while_moving(stopped):
    stopped.move_to_q(stopped.q_home + np.deg2rad([30, 0, 0, 0, 0, 0]))
    assert stopped.in_motion()
    stopped.wait_for_motion_stop()
    assert not stopped.in_motion()


def test_queued_moves_end_at_last_target(stopped):
    targets = [stopped.q_home + np.deg2rad([d, 0, 0, 0, 0, 0]) for d in (10, -10, 5)]
    for q in targets:
        stopped.move_to_q(q)
    stopped.wait_for_motion_stop()
    assert np.allclose(stopped.get_q(), targets[-1], atol=IRC_TOL)


def test_motion_is_continuous(stopped):
    """Sampled during a move, the configuration changes smoothly and stays
    between start and target."""
    start = stopped.get_q()
    target = stopped.q_home + np.deg2rad([60, 0, 0, 0, 0, 0])
    stopped.move_to_q(target)
    samples = [stopped.get_q() for _ in range(50)]
    stopped.wait_for_motion_stop()
    q1 = np.array(samples)[:, 0]
    assert np.all(np.diff(q1) >= -1e-12)  # monotonic in the moving joint
    # the target is rounded to IRC, like the real command
    assert q1.min() >= start[0] - IRC_TOL and q1.max() <= target[0] + IRC_TOL
    assert np.allclose(np.array(samples)[:, 1:], start[1:])


def test_slower_speed_takes_longer(stopped):
    def duration(fraction: float) -> float:
        stopped.set_speed_relative(fraction)
        stopped.move_to_q(stopped.q_home + np.deg2rad([30, 0, 0, 0, 0, 0]))
        d = stopped._motion_end() - stopped._now()
        stopped.wait_for_motion_stop()
        move(stopped, stopped.q_home)
        return d

    assert duration(0.2) > duration(1.0)


# --- blending of queued moves (measured on the CRS93, 2026-10-21) ---

def _queued_duration(sim, offsets_deg) -> float:
    """Sim time from the first move_to_q until the last queued move ends."""
    t0 = sim._now()
    for dq in offsets_deg:
        sim.move_to_q(sim.q_home + np.deg2rad(dq))
    d = sim._motion_end() - t0
    sim.wait_for_motion_stop()
    move(sim, sim.q_home)
    return d


@pytest.fixture(scope="module")
def realtime():
    """A sim at real-time speed: blending needs the next command to arrive before
    the running move slows down, which script overhead scaled by TIME_SCALE would
    prevent (as on the real robot, a late command means a stop)."""
    s = CRS93Sim(port=free_port(), time_scale=1.0, keep_alive=False)
    s.initialize(home=False)
    yield s
    s.stop()


def test_same_direction_moves_blend(realtime):
    """Ten queued 3 deg steps take as long as one 30 deg move."""
    single = _queued_duration(realtime, [[30, 0, 0, 0, 0, 0]])
    chain = _queued_duration(realtime, [[3 * (i + 1), 0, 0, 0, 0, 0] for i in range(10)])
    assert chain == pytest.approx(single, abs=0.01)


def test_corner_and_reversal_stop(stopped):
    """A corner or a reversal starts from standstill: longer than blended."""
    def stops(offsets):
        stopped.move_to_q(stopped.q_home + np.deg2rad(offsets[0]))
        stopped.move_to_q(stopped.q_home + np.deg2rad(offsets[1]))
        with stopped._lock:
            segs = list(stopped._segments[-2:])
        stopped.wait_for_motion_stop()
        move(stopped, stopped.q_home)
        return len(segs) == 2 and segs[1].t0 == pytest.approx(segs[0].t_end)

    assert stops([[20, 0, 0, 0, 0, 0], [20, 0, 0, 40, 0, 0]])  # corner
    assert stops([[20, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])  # reversal


def test_blended_motion_is_continuous(realtime):
    """Merging into a running move does not make the configuration jump."""
    realtime.move_to_q(realtime.q_home + np.deg2rad([15, 0, 0, 0, 0, 0]))
    time.sleep(0.2)  # moving
    before = realtime._q_at(realtime._now())
    realtime.move_to_q(realtime.q_home + np.deg2rad([30, 0, 0, 0, 0, 0]))
    after = realtime._q_at(realtime._now())
    last = realtime._segments[-1]  # merged: one move 0 -> 30 deg
    assert np.rad2deg(last.q_to[0] - last.q_from[0]) == pytest.approx(30, abs=0.01)
    realtime.wait_for_motion_stop()
    assert after[0] >= before[0] and after[0] - before[0] < np.deg2rad(2)
    assert np.allclose(realtime.get_q(), realtime.q_home + np.deg2rad([30, 0, 0, 0, 0, 0]),
                       atol=IRC_TOL)
    move(realtime, realtime.q_home)


def test_move_to_q_blocks_for_send_delay(stopped):
    t = time.monotonic()
    stopped.move_to_q(stopped.q_home + np.deg2rad([5, 0, 0, 0, 0, 0]))
    assert (time.monotonic() - t) * TIME_SCALE >= stopped.send_delay * 0.9
    stopped.wait_for_motion_stop()


# --- joint limits ---

@pytest.mark.parametrize("side", ["min", "max"])
def test_move_to_all_limits(stopped, side):
    """All joints exactly at their limits: accepted, reached, still in limits."""
    q_lim = stopped.q_min if side == "min" else stopped.q_max
    q = move(stopped, q_lim)
    assert stopped.in_limits(q)
    assert np.allclose(q, q_lim, atol=IRC_TOL)


@pytest.mark.parametrize("side", ["min", "max"])
@pytest.mark.parametrize("j", range(6))
def test_beyond_limit_rejected(stopped, j, side):
    """Just beyond a limit: AssertionError, nothing queued, robot stays."""
    q = stopped.q_home.copy()
    q[j] = stopped.q_max[j] + 1e-6 if side == "max" else stopped.q_min[j] - 1e-6
    before = stopped.get_q()
    with pytest.raises(AssertionError):
        stopped.move_to_q(q)
    assert not stopped.in_motion()
    assert np.array_equal(stopped.get_q(), before)


# --- stubs ---

def test_grab_image_format(stopped):
    image = stopped.grab_image()
    assert image.shape == (1200, 1920, 3) and image.dtype == np.uint8


def test_grab_image_timing_and_age():
    """Real-time sim: the first call blocks ~FIRST_GRAB_TIME, later ones
    ~GRAB_TIME; the image shows the robot as it was camera_delay before the
    call returned (here during a slow joint 1 move)."""
    from simulation import CAMERA_DELAY, FIRST_GRAB_TIME, GRAB_TIME
    s = CRS93Sim(port=free_port(), time_scale=1.0, keep_alive=False)
    try:
        s.initialize(home=False)
        seen = []
        t = s._now()
        s.grab_image()
        first = s._now() - t
        renderer = s._renderer
        render = renderer.render
        renderer.render = lambda q, *a: seen.append(np.array(q)) or render(q, *a)
        s.set_speed_relative(0.0)
        s.move_to_q(s.q_home + np.deg2rad([20, 0, 0, 0, 0, 0]))
        s._sleep(1.0)  # well into the move
        t = s._now()
        s.grab_image()
        t_return = s._now()
        # The first call also pays the first render while the collision
        # monitor runs: up to ~0.15 s late (the real one was measured once).
        assert first == pytest.approx(FIRST_GRAB_TIME, abs=0.25)
        assert t_return - t == pytest.approx(GRAB_TIME, abs=0.06)
        expected = s._q_at(t_return - CAMERA_DELAY)
        speed = np.deg2rad(6)  # j1 at the lowest speed setting, ~5 deg/s
        assert abs(seen[-1][0] - expected[0]) < speed * 0.03
    finally:
        s.stop()


def test_gripper_stub(stopped):
    stopped.gripper.control_position_relative(0.5)
    lo, hi = stopped.gripper.bounds
    assert stopped.gripper.position == int(lo + (hi - lo) * 0.5)


# --- drop-in for ctu_crs.CRS93: same interface, ctu_crs kinematics untouched ---

KINEMATICS = ["fk", "ik", "fk_flange_pos", "in_limits", "dh_to_se3",
              "_ik_flange_pos", "_irc_to_joint_values", "_joint_values_to_irc"]


@pytest.mark.parametrize("name", KINEMATICS)
def test_kinematics_not_overridden(name):
    from ctu_crs import CRS93
    assert getattr(CRS93Sim, name) is getattr(CRS93, name)


def test_same_public_interface():
    import inspect

    from ctu_crs import CRS93
    for name, member in inspect.getmembers(CRS93, inspect.isfunction):
        if name.startswith("_"):
            continue
        assert hasattr(CRS93Sim, name), name
        if name != "__init__":
            assert (inspect.signature(getattr(CRS93Sim, name))
                    == inspect.signature(member)), name


def test_same_parameters_as_real_robot(stopped, robot):
    for attr in ["q_home", "q_min", "q_max", "dh_d", "dh_a", "dh_alpha",
                 "dh_offset", "link_lengths"]:
        assert np.array_equal(getattr(stopped, attr), getattr(robot, attr)), attr


def test_ik_same_as_ctu(stopped, robot, rng):
    """ik() returns the ctu_crs solutions as they are (order is random in
    ctu_crs, so compared as sets)."""
    for _ in range(20):
        T = robot.fk(rng.uniform(robot.q_min, robot.q_max))
        a = np.array(sorted(map(tuple, np.round(stopped.ik(T), 9))))
        b = np.array(sorted(map(tuple, np.round(robot.ik(T), 9))))
        assert np.array_equal(a, b)
