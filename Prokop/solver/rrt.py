"""Joint-space RRT between two configurations, with time-aware edges, and shortcutting.

Lab 6 (RRT + random shortcut), with time: the puzzle may spin, so an edge is
collision free only at the time the robot drives it. Every tree node carries
its arrival time from the root (the robot stops at every waypoint: the moves
are straight in joint space and queued; a bend in the path is a stop), and an
edge is checked by the caller's `edge_free(q_from, q_to, t_start)`, e.g. the
planner's continuous soft-model check with the turntable angle of each moment.

1. `rrt(q_start, q_goal, ...)`: a single tree from q_start; samples are drawn
   from a box around the two configurations (`margin` wider on every joint,
   within the limits), with GOAL_BIAS towards q_goal; extend by at most STEP.
   Lazy: growing checks only the new configuration (`node_free`, cheap); the
   edges are checked (continuously, `edge_free`) only on a candidate path to
   the goal, tried from every node that comes closer to the goal (Euclidean
   joint distance) than any verified before, and every GOAL_EVERY nodes from
   the closest node. A failed edge kills its child node (nothing grows from it or its
   descendants). If the box gives nothing within max_nodes, it is widened
   (BOX_MARGINS); everything stops after `time_limit`.
2. `shortcut(path, ...)`: random pairs of waypoints joined by a straight edge
   where it is free (the times of the waypoints after it change, so the caller
   checks the final path again as a whole).

Example:
    ::

        path = rrt(qa, qb, lower, upper, duration, edge_free, t0=0.0, rng=rng)
        path = shortcut(path, duration, edge_free, t0=0.0, rng=rng)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

STEP = np.deg2rad(20.0)  # rad, largest joint change of one extension
GOAL_BIAS = 0.2  # share of samples that are the goal
BOX_MARGINS = np.deg2rad([40.0, 90.0, 180.0])  # rad, sampling boxes tried in turn
MAX_NODES = 400
TIME_LIMIT = 5.0  # s, per call of rrt
SHORTCUT_TRIES = 30
GOAL_EVERY = 25  # nodes between tries from the closest node

EdgeFree = Callable[[np.ndarray, np.ndarray, float], bool]
NodeFree = Callable[[np.ndarray, float], bool]


def rrt(q_start: np.ndarray, q_goal: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        duration: Callable[[np.ndarray, np.ndarray], float], edge_free: EdgeFree,
        t0: float = 0.0, rng: np.random.Generator | None = None,
        max_nodes: int = MAX_NODES, margins=BOX_MARGINS, node_free: NodeFree | None = None,
        time_limit: float = TIME_LIMIT, stats: dict | None = None) -> list[np.ndarray] | None:
    """Path of waypoints from q_start to q_goal with free edges, or None.

    Args:
        q_start, q_goal: Configurations [rad].
        lower, upper: Joint limits [rad].
        duration: Time of the straight move between two configurations [s].
        edge_free: Whether the straight move q_from -> q_to started at time t is free.
        t0: Time at q_start [s] (the robot is there, at rest).
        rng: Random numbers.
        max_nodes: Tree nodes per sampling box.
        margins: Sampling boxes tried in turn (margin beyond the two configurations).
        node_free: Whether a configuration is free at a time (default: always).
        time_limit: Give up after this long [s, wall clock].
        stats: Filled with "nodes", "edges" (checked), "seconds".

    Returns:
        The waypoints, q_start first and q_goal last.
    """
    import time as _time
    rng = rng or np.random.default_rng()
    stats = {} if stats is None else stats
    stats.update(nodes=0, edges=0, seconds=0.0)
    t_wall = _time.monotonic()
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    node_free = node_free or (lambda q, t: True)

    def checked_edge(a, b, t) -> bool:
        stats["edges"] += 1
        return edge_free(a, b, t)

    path = None
    if checked_edge(q_start, q_goal, t0):
        path = [q_start, q_goal]
    else:
        deadline = t_wall + time_limit
        for margin in np.atleast_1d(margins):
            lo = np.maximum(np.minimum(q_start, q_goal) - margin, lower)
            hi = np.minimum(np.maximum(q_start, q_goal) + margin, upper)
            path = _grow(q_start, q_goal, lo, hi, duration, checked_edge, node_free, t0, rng,
                         max_nodes, deadline, stats)
            if path is not None or _time.monotonic() > deadline:
                break
    stats["seconds"] = _time.monotonic() - t_wall
    return path


def _grow(q_start, q_goal, lo, hi, duration, edge_free, node_free, t0, rng, max_nodes,
          deadline, stats):
    """One lazy tree in the sampling box [lo, hi] (see rrt)."""
    import time as _time
    nodes, parents, times, dead = [q_start], [-1], [t0], [False]
    verified: set[int] = {0}  # nodes whose edge from the parent is checked free
    best = np.linalg.norm(q_goal - q_start)
    while len(nodes) < max_nodes and _time.monotonic() < deadline:
        sample = q_goal if rng.random() < GOAL_BIAS else rng.uniform(lo, hi)
        alive = np.flatnonzero(~np.array(dead))
        tree = np.array(nodes)[alive]
        near = int(alive[np.argmin(np.abs(tree - sample).max(axis=1))])
        d = sample - nodes[near]
        scale = min(1.0, STEP / max(np.abs(d).max(), 1e-12))
        q_new = nodes[near] + scale * d
        t_new = times[near] + duration(nodes[near], q_new)
        if not node_free(q_new, t_new):
            continue
        nodes.append(q_new)
        parents.append(near)
        times.append(t_new)
        dead.append(False)
        stats["nodes"] += 1
        gap = np.linalg.norm(q_goal - q_new)
        if gap >= best and stats["nodes"] % GOAL_EVERY:
            continue
        if gap >= best:  # the periodic try: from the closest live node
            alive = np.flatnonzero(~np.array(dead))
            closest = int(alive[np.argmin(np.linalg.norm(np.array(nodes)[alive] - q_goal,
                                                         axis=1))])
            if closest == 0:
                continue
            q_new, t_new = nodes[closest], times[closest]
            gap = np.linalg.norm(q_goal - q_new)
            last = closest
        else:
            last = len(nodes) - 1
        # A candidate path: verify its edges from the root (lazily, once each);
        # only a verified node counts as the closest so far.
        chain = []
        k = last
        while k > 0:
            chain.append(k)
            k = parents[k]
        ok = True
        for k in reversed(chain):
            if dead[k]:
                ok = False
                break
            if k not in verified:
                if edge_free(nodes[parents[k]], nodes[k], times[parents[k]]):
                    verified.add(k)
                else:
                    dead[k] = True
                    ok = False
                    break
        if not ok:
            dead[last] = True
            continue
        best = min(best, gap)
        if edge_free(q_new, q_goal, t_new):
            return [nodes[k] for k in [0] + list(reversed(chain))] + [q_goal]
    return None


def path_times(path: list[np.ndarray], duration: Callable[[np.ndarray, np.ndarray], float],
               t0: float = 0.0) -> list[float]:
    """Arrival time at every waypoint, stopping at each [s]."""
    times = [t0]
    for a, b in zip(path, path[1:]):
        times.append(times[-1] + duration(a, b))
    return times


def shortcut(path: list[np.ndarray], duration: Callable[[np.ndarray, np.ndarray], float],
             edge_free: EdgeFree, t0: float = 0.0, rng: np.random.Generator | None = None,
             tries: int = SHORTCUT_TRIES) -> list[np.ndarray]:
    """Random shortcuts: waypoints i < j joined straight where the edge is free.

    Returns:
        The shorter path (same start and end).
    """
    rng = rng or np.random.default_rng()
    path = list(path)
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i < 2:
            continue
        t_i = path_times(path, duration, t0)[i]
        if edge_free(path[i], path[j], t_i):
            path = path[:i + 1] + path[j:]
    return path
