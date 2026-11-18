"""Task P2.7: collision-free approach and retreat paths (lab 6: RRT + random shortcut).

Moves between q_home and the plan's pre-entry configuration q_start. The puzzle
keeps spinning while the robot approaches, so the arm is checked against the
volume the turntable, base and puzzle sweep in a full turn (from the estimated
TurntableModel), besides the cell, the table and itself.

The planner follows the semantics of the lab-6 template (robotics_labs
planning/rrt.py and utils/configuration_utils.py): RRT(robot, delta_q,
p_sample_goal) with goal-biased sampling, steering by at most delta_q along the
straight line in joint space, and random_shortcut(path, max_iterations) that
replaces a part of the path by a straight segment when that is free.

Example:
    ::

        checker = ApproachChecker(model)
        path = plan_approach(robot, checker, robot.q_home, plan.q_start)  # or None
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import trimesh
from cell import TABLE_Z, cell_meshes
from collision import SceneCollider, arm_parts
from placement import Placement, TaskScene

from solver.contracts import TurntableModel

MARGIN = 0.02  # m, required clearance of the arm on the approach
CHECK_STEP = np.deg2rad(1.0)  # rad, largest joint change between checked configurations
# RRT sampling boxes around start and goal [rad per joint]; None = all joint limits.
SAMPLE_MARGINS = (np.deg2rad(30.0), np.deg2rad(60.0), None)


def distance_between_configurations(a: np.ndarray, b: np.ndarray) -> float:
    """Joint-space distance (lab 6 configuration_utils semantics)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def interpolate(a: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
    """Configuration at distance @d from @a towards @b (lab 6 semantics)."""
    return a + d * (b - a) / distance_between_configurations(a, b)


class ApproachChecker:
    """Clearance of the arm (with the hoop) for the approach.

    Obstacles: the cell, the table, the volume swept by the spinning task, and
    the arm itself.

    Attributes:
        margin: Required clearance [m].
    """

    def __init__(self, model: TurntableModel, margin: float = MARGIN,
                 extra_obstacles: dict[str, trimesh.Trimesh] | None = None,
                 task: TaskScene | None = None) -> None:
        """Builds the obstacles.

        Args:
            model: The estimated turntable model (its swept volume is an obstacle).
            margin: Required clearance [m].
            extra_obstacles: More static obstacles by name, robot frame (tests).
            task: The task models; loaded if None.
        """
        self.margin = margin
        task = task or TaskScene()
        self._self = SceneCollider(task)
        self.urdf, self._parts = arm_parts()
        self._arm = trimesh.collision.CollisionManager()
        for _, name, mesh, _ in self._parts:
            self._arm.add_object(name, mesh)
        self._obstacles = trimesh.collision.CollisionManager()
        placement = Placement(np.asarray(model.axis_xy, float),
                              np.asarray(model.base_offset, float), model.base_yaw)
        obstacles = {**cell_meshes(), "task_swept": task.swept_task_mesh(placement),
                     **(extra_obstacles or {})}
        for name, mesh in obstacles.items():
            self._obstacles.add_object(name, mesh)
        self._hull = {name: mesh.convex_hull.vertices for link, name, mesh, _ in self._parts
                      if link != self.urdf.base_link}
        self._free: dict[tuple, bool] = {}

    def clearance(self, q: np.ndarray) -> tuple[float, str]:
        """Clearance of the arm in a configuration.

        Args:
            q: Joint configuration [rad].

        Returns:
            Distance [m] to the nearest obstacle (0 if touching or in
            self-collision) and its name ("table", "self", "task_swept", a cell
            mesh name or an extra obstacle).
        """
        if self._self.self_hits(q):
            return 0.0, "self"
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        poses = {}
        for link, name, _, offset in self._parts:
            poses[name] = self.urdf.get_transform(link, self.urdf.base_link) @ offset
            self._arm.set_transform(name, poses[name])
        d, (_, what) = self._arm.min_distance_other(self._obstacles, return_names=True)
        for name, points in self._hull.items():
            z = trimesh.transform_points(points, poses[name])[:, 2].min() - TABLE_Z
            if z < d:
                d, what = z, "table"
        return max(0.0, float(d)), what

    def is_free(self, q: np.ndarray) -> bool:
        """Whether configuration @q keeps the required clearance (cached)."""
        key = tuple(np.round(np.asarray(q, float), 9))
        if key not in self._free:
            self._free[key] = self.clearance(q)[0] >= self.margin
        return self._free[key]

    def segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether the straight joint-space segment a -> b keeps the clearance.

        Checked every CHECK_STEP of the largest joint change, endpoints included,
        in bisection order (the middle first), so that a blocked segment fails
        after few checks.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.abs(b - a).max() / CHECK_STEP)))
        return all(self.is_free(a + k / n * (b - a)) for k in _bisection_order(n))


def _bisection_order(n: int) -> list[int]:
    """Indices 0..n ordered coarse to fine: ends, middle, quarters, ...

    Args:
        n: Number of steps of the segment.

    Returns:
        Every index of 0..n once.
    """
    order, seen, step = [0, n], {0, n}, n
    while step > 1:
        step = (step + 1) // 2
        for k in range(step, n, step):
            if k not in seen:
                seen.add(k)
                order.append(k)
    order += [k for k in range(n + 1) if k not in seen]
    return order


class RRT:
    """RRT in joint space (lab 6 semantics), checked with an ApproachChecker."""

    def __init__(self, robot, checker: ApproachChecker, delta_q: float = 0.2,
                 p_sample_goal: float = 0.5, rng: np.random.Generator | None = None
                 ) -> None:
        """Sets up the planner.

        Args:
            robot: CRS93 / CRS93Sim (joint limits q_min, q_max).
            checker: Collision checker.
            delta_q: Maximum distance between two configurations of the tree [rad].
            p_sample_goal: Probability of sampling the goal as q_rand.
            rng: Random generator (default seed 0, for reproducible plans).
        """
        self.robot, self.checker = robot, checker
        self.delta_q, self.p_sample_goal = delta_q, p_sample_goal
        self.rng = rng or np.random.default_rng(0)

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray,
             max_iterations: int = 10000, sample_margin: float | None = None
             ) -> list[np.ndarray]:
        """RRT from @q_start to @q_goal.

        Args:
            q_start: Start configuration (free) [rad].
            q_goal: Goal configuration (free) [rad].
            max_iterations: Samples before giving up.
            sample_margin: If given, random samples are drawn from the box
                spanned by start and goal, widened by this much per joint [rad]
                (and clipped to the joint limits), instead of the whole joint
                range: an approach detour is local.

        Returns:
            The path from start to goal (configurations at most delta_q apart),
            or an empty list if none was found.
        """
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        assert self.checker.is_free(q_start) and self.checker.is_free(q_goal)
        lo, hi = self.robot.q_min, self.robot.q_max
        if sample_margin is not None:
            lo = np.maximum(lo, np.minimum(q_start, q_goal) - sample_margin)
            hi = np.minimum(hi, np.maximum(q_start, q_goal) + sample_margin)
        nodes, parents = [q_start], [-1]
        blocked_to_goal: set[int] = set()  # nodes whose extension to the goal failed
        for _ in range(max_iterations):
            to_goal = self.rng.uniform() < self.p_sample_goal
            q_rand = q_goal if to_goal else self.rng.uniform(lo, hi)
            i_near = int(np.argmin([distance_between_configurations(n, q_rand)
                                    for n in nodes]))
            if to_goal and i_near in blocked_to_goal:
                continue  # the same extension again would fail again
            q_near = nodes[i_near]
            d = distance_between_configurations(q_near, q_rand)
            if d == 0:
                continue
            q_new = q_rand if d <= self.delta_q else interpolate(q_near, q_rand,
                                                                 self.delta_q)
            if not self.robot.in_limits(q_new) or not self.checker.segment_free(q_near, q_new):
                if to_goal:
                    blocked_to_goal.add(i_near)
                continue
            nodes.append(q_new)
            parents.append(i_near)
            if (distance_between_configurations(q_new, q_goal) <= self.delta_q
                    and self.checker.segment_free(q_new, q_goal)):
                path, i = [q_goal], len(nodes) - 1
                while i >= 0:
                    path.append(nodes[i])
                    i = parents[i]
                return path[::-1] if np.any(path[-1] != q_goal) else path
        return []

    def random_shortcut(self, path: list[np.ndarray], max_iterations: int = 100
                        ) -> list[np.ndarray]:
        """Random shortcut (lab 6 semantics).

        Picks two configurations on the path at random and replaces the part
        between them by the straight segment if that is free.

        Args:
            path: The path.
            max_iterations: Number of attempts.

        Returns:
            The shortened path (a copy).
        """
        out = deepcopy(path)
        for _ in range(max_iterations):
            if len(out) < 3:
                break
            i, j = sorted(self.rng.choice(len(out), 2, replace=False))
            if j - i < 2:
                continue
            if self.checker.segment_free(out[i], out[j]):
                out = out[:i + 1] + out[j:]
        return out


def plan_approach(robot, checker: ApproachChecker, q_from: np.ndarray,
                  q_to: np.ndarray, max_iterations: int = 5000,
                  rng: np.random.Generator | None = None) -> list[np.ndarray] | None:
    """Collision-free joint-space path between two configurations.

    One straight move if it is free; otherwise RRT (sampling near start and goal
    first, then wider, see SAMPLE_MARGINS) and random shortcut. Each
    configuration of the result is meant as one move_to_q (the robot stops at
    the corners).

    Args:
        robot: CRS93 / CRS93Sim (joint limits).
        checker: Collision checker for the current turntable model.
        q_from: Start configuration [rad].
        q_to: Goal configuration [rad].
        max_iterations: RRT samples before giving up.
        rng: Random generator.

    Returns:
        The configurations from q_from to q_to (both included), or None if the
        start or goal is not free or no path was found.
    """
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    if not (checker.is_free(q_from) and checker.is_free(q_to)):
        return None
    if checker.segment_free(q_from, q_to):
        return [q_from, q_to]
    rrt = RRT(robot, checker, rng=rng)
    for margin in SAMPLE_MARGINS:  # local first, wider if that fails
        path = rrt.plan(q_from, q_to, max_iterations // len(SAMPLE_MARGINS), margin)
        if path:
            return rrt.random_shortcut(path, 200)
    return None
