"""P2.5: choose hoop roll, tilt and IK branches along the Puzzle D checkpoint path.

This module samples tool orientations, obtains valid IK solutions for each
checkpoint, and uses dynamic programming to maximize the worst configuration
margin while penalizing orientation, joint, and IK-branch transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

try:
    from .hoop_targets import flange_target
    from .plan_ik import all_solutions
except ImportError:
    from hoop_targets import flange_target
    from plan_ik import all_solutions


MarginFunction = Callable[[np.ndarray, np.ndarray, int, float, float], float]


@dataclass(frozen=True)
class SelectionCandidate:
    """One hoop orientation and joint configuration at a checkpoint."""

    index: int
    roll: float
    tilt: float
    q: np.ndarray
    flange_target: np.ndarray
    margin: float
    branch_key: tuple[int, ...]


@dataclass(frozen=True)
class SelectionResult:
    """Dynamic-programming result for the complete checkpoint path."""

    feasible: bool
    candidates: tuple[SelectionCandidate, ...]
    min_margin: float
    reason: str = ""


def discretized_rolls(roll_limits: np.ndarray | None = None,
                      roll_step: float = np.deg2rad(10.0)) -> np.ndarray:
    """Create a roll grid, optionally restricted to one allowed range."""
    if roll_step <= 0.0 or not np.isfinite(roll_step):
        raise ValueError("roll_step must be positive and finite")

    if roll_limits is None:
        lower, upper = -np.pi, np.pi
    else:
        limits = np.asarray(roll_limits, dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)):
            raise ValueError("roll_limits must contain finite [lower, upper]")
        lower, upper = float(limits[0]), float(limits[1])
        if lower > upper:
            raise ValueError("roll lower limit must not exceed upper limit")

    count = max(1, int(np.floor((upper - lower) / roll_step)))
    rolls = lower + np.arange(count + 1) * (upper - lower) / count
    if not np.any(np.isclose(rolls, 0.0, atol=1e-12)):
        rolls = np.append(rolls, 0.0)
    return np.unique(np.clip(rolls, lower, upper))


def branch_key(q: ArrayLike) -> tuple[int, ...]:
    """Stable, coarse IK-branch identifier independent of 2pi wrist wraps."""
    configuration = np.asarray(q, dtype=float)
    if configuration.shape != (6,) or not np.all(np.isfinite(configuration)):
        raise ValueError("q must be a finite six-joint configuration")

    # q[:3] distinguish arm/elbow position branches. q4/q5/q6 distinguish
    # the wrist branch while also ignoring exact +/-2pi whole-turn variants.
    return tuple(int(np.sign(configuration[i])) for i in range(3)) + (
        int(round(configuration[3] / (np.pi / 2.0))),
        int(np.sign(configuration[4])),
        int(round(configuration[5] / (np.pi / 2.0))),
    )


def circular_roll_distance(a: float, b: float | np.ndarray) -> float | np.ndarray:
    """Shortest angular distance between a and each value in b."""
    return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def default_margin(robot, q: np.ndarray, checker=None, placement=None,
                   turntable_angle: float = 0.0) -> float:
    """Default P2.4 margin without requiring a placement/collision setup."""
    try:
        from .safety_metrics import configuration_metrics
    except ImportError:
        from safety_metrics import configuration_metrics

    metric = configuration_metrics(robot, q, checker=checker,
                                   placement=placement,
                                   turntable_angle=turntable_angle)
    return float(metric.margin)


def select_roll_branch(
    robot,
    checkpoint_frames: Sequence[np.ndarray],
    roll_limits: np.ndarray | None = None,
    roll_step: float = np.deg2rad(10.0),
    tilt: float = 0.0,
    q_start: ArrayLike | None = None,
    margin_function: MarginFunction | None = None,
    checker=None,
    placement=None,
    turntable_angle: float = 0.0,
    joint_weight: float = 1.0,
    roll_weight: float = 0.25,
    branch_switch_penalty: float = 2.0,
    transition_allowed: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    *,
    tilt_options: ArrayLike | None = None,
    tilt_weight: float = 0.25,
    max_joint_step: float | None = None,
) -> SelectionResult:
    """Select roll, tilt and IK branch for every checkpoint.

    Args:
        robot: CRS93/CRS93Sim interface (ik, fk, in_limits).
        checkpoint_frames: Target poses in the robot frame, shape (N, 4, 4).
        roll_limits: Per-path limits (N, 2), or None for [-pi, pi].
        roll_step: Roll-grid spacing [rad].
        tilt: Constant tilt around each checkpoint x axis [rad] when there are
            no tilt_options.
        q_start: Optional first seed configuration for branch continuity.
        margin_function: Optional (q, flange, index, roll, tilt) -> margin.
        checker/placement/turntable_angle: Optional P2.4 collision context.
        joint_weight: Weight of Euclidean joint transition distance.
        roll_weight: Weight of circular roll transition distance [rad].
        branch_switch_penalty: Additive cost for changing the coarse IK branch.
        transition_allowed: Optional test for the move between consecutive
            checkpoint configurations. The approach to the first is separate.
        tilt_options: Optional grid of tilts to try at each checkpoint [rad].
        tilt_weight: Weight of circular tilt transition distance [rad].
        max_joint_step: Optional maximum change in one joint between
            consecutive checkpoints [rad].

    Returns:
        The chosen candidate sequence, or a reason no feasible path was found.
    """
    frames = [np.asarray(frame, dtype=float) for frame in checkpoint_frames]
    if not frames:
        return SelectionResult(False, (), 0.0, "no checkpoints")
    if any(frame.shape != (4, 4) for frame in frames):
        raise ValueError("every checkpoint frame must have shape (4, 4)")
    tilts = np.asarray([tilt] if tilt_options is None else tilt_options, dtype=float)
    if (tilts.ndim != 1 or tilts.size == 0 or not np.isfinite(tilts).all()
            or np.any(np.abs(tilts) > np.pi)):
        raise ValueError("tilts must be a nonempty sequence of finite angles between -pi and pi")
    if max_joint_step is not None and (not np.isfinite(max_joint_step) or max_joint_step <= 0):
        raise ValueError("max_joint_step must be positive and finite")

    limits = None
    if roll_limits is not None:
        limits = np.asarray(roll_limits, dtype=float)
        if limits.shape != (len(frames), 2):
            raise ValueError("roll_limits must have shape (number of checkpoints, 2)")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("a roll lower limit exceeds its upper limit")

    previous_seed = None
    if q_start is not None:
        previous_seed = np.asarray(q_start, dtype=float)
        if previous_seed.shape != np.asarray(robot.q_min).shape:
            raise ValueError("q_start does not match the robot joint configuration shape")

    states: list[tuple[tuple[float, float], list[SelectionCandidate]]] = []

    for index, frame in enumerate(frames):
        checkpoint_limits = None if limits is None else limits[index]
        rolls = discretized_rolls(checkpoint_limits, roll_step)
        candidates: list[SelectionCandidate] = []

        for candidate_tilt in tilts:
            for roll in rolls:
                target = flange_target(frame, roll=float(roll), tilt=float(candidate_tilt))
                for q in all_solutions(robot, target, q_previous=previous_seed):
                    if margin_function is None:
                        margin = default_margin(robot, q, checker, placement, turntable_angle)
                    else:
                        margin = float(margin_function(q, target, index,
                                                       float(roll), float(candidate_tilt)))
                    if np.isfinite(margin):
                        candidates.append(SelectionCandidate(
                            index=index,
                            roll=float(roll),
                            tilt=float(candidate_tilt),
                            q=np.asarray(q, dtype=float).copy(),
                            flange_target=target,
                            margin=margin,
                            branch_key=branch_key(q),
                        ))

        if not candidates:
            return SelectionResult(False, (), 0.0,
                                   f"no reachable safe IK solution at checkpoint {index}")

        next_states = []
        if index == 0:
            for candidate in candidates:
                cost = 0.0
                if previous_seed is not None:
                    joint_cost = float(np.linalg.norm(candidate.q - previous_seed))
                    cost = joint_weight * joint_cost
                next_states.append(((candidate.margin, -cost), [candidate]))
        else:
            previous = [path[-1] for _, path in states]
            previous_q = np.array([candidate.q for candidate in previous])
            previous_rolls = np.array([candidate.roll for candidate in previous])
            previous_tilts = np.array([candidate.tilt for candidate in previous])
            previous_branches = np.array([candidate.branch_key for candidate in previous])
            previous_margins, previous_costs = np.array([score for score, _ in states]).T
            for candidate in candidates:
                delta = candidate.q - previous_q
                reachable = np.arange(len(states))
                if max_joint_step is not None:
                    reachable = reachable[np.max(np.abs(delta), axis=1) <= max_joint_step]

                transition_costs = (
                    joint_weight * np.linalg.norm(delta, axis=1)
                    + roll_weight * circular_roll_distance(candidate.roll, previous_rolls)
                    + tilt_weight * circular_roll_distance(candidate.tilt, previous_tilts)
                    + branch_switch_penalty * np.any(previous_branches != candidate.branch_key, axis=1)
                )
                margins = np.minimum(previous_margins, candidate.margin)
                costs = previous_costs - transition_costs
                ranked = np.lexsort((-costs[reachable], -margins[reachable]))
                # Test segments in score order, stopping at the first valid one.
                for predecessor in reachable[ranked]:
                    if transition_allowed is None or transition_allowed(previous_q[predecessor], candidate.q):
                        score = (margins[predecessor], costs[predecessor])
                        next_states.append((score, states[predecessor][1] + [candidate]))
                        break

        if not next_states:
            return SelectionResult(False, (), 0.0,
                                   f"no continuous selection at checkpoint {index}")
        states = next_states
        previous_seed = None

    best_score, best_path = max(states, key=lambda state: state[0])
    return SelectionResult(True, tuple(best_path), float(best_score[0]), "")


# Descriptive aliases for downstream use.
select_path = select_roll_branch
viterbi_select = select_roll_branch
dynamic_programming_select = select_roll_branch
