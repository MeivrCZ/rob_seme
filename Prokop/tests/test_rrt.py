"""Tests of solver.rrt (joint-space RRT with timed edges, shortcutting)."""

import numpy as np

from solver.rrt import path_times, rrt, shortcut

LO, HI = np.full(2, -3.0), np.full(2, 3.0)


def duration(a, b):
    return float(np.abs(b - a).max())  # 1 rad/s on the slowest joint


def wall(a, b, t):
    """A wall x = 0 for |y| < 1, sampled finely along the edge."""
    for s in np.linspace(0, 1, 50):
        p = a + s * (b - a)
        if abs(p[0]) < 0.05 and abs(p[1]) < 1.0:
            return False
    return True


def free_path(path, edge_free, t0=0.0):
    times = path_times(path, duration, t0)
    return all(edge_free(a, b, t) for a, b, t in zip(path, path[1:], times))


def test_straight_when_free():
    a, b = np.array([-1.0, 2.0]), np.array([1.0, 2.0])
    assert len(rrt(a, b, LO, HI, duration, wall)) == 2


def test_around_a_wall_and_shortcut():
    rng = np.random.default_rng(0)
    a, b = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    path = rrt(a, b, LO, HI, duration, wall, rng=rng, max_nodes=2000)
    assert path is not None and np.allclose(path[0], a) and np.allclose(path[-1], b)
    assert free_path(path, wall)
    short = shortcut(path, duration, wall, rng=rng, tries=200)
    assert len(short) <= len(path) and free_path(short, wall)


def test_time_aware_edges():
    """A gate that is closed until t = 5: the straight edge at t = 0 is not free."""
    def gate(a, b, t):
        return t >= 5.0 or not (a[0] < 0 < b[0] or b[0] < 0 < a[0])
    a, b = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    assert rrt(a, b, LO, HI, duration, gate, t0=6.0) == [a, b] or \
        len(rrt(a, b, LO, HI, duration, gate, t0=6.0)) == 2
    path = rrt(a, b, LO, HI, duration, gate, t0=0.0, rng=np.random.default_rng(1),
               max_nodes=3000)
    assert path is None or free_path(path, gate)
