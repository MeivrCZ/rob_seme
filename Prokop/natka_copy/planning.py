"""P2.6: timed traversal planning on the moving Puzzle D turntable.

For each candidate start phase and speed, this module iterates between
checkpoint arrival times and the moving checkpoint poses. The final IK/roll
choices come from P2.5; P2.4 supplies configuration and segment safety margins.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROKOP_DIR = PROJECT_ROOT / "Prokop"
if str(PROKOP_DIR) not in sys.path:
    sys.path.insert(0, str(PROKOP_DIR))

from solver.contracts import Plan, PlanStep, PuzzlePath, TurntableModel

try:
    from .selection import SelectionResult, select_roll_branch
except ImportError:
    from selection import SelectionResult, select_roll_branch


class RobotTimingProtocol(Protocol):
    speed: float

    def schedule(self, q_start: np.ndarray, targets: list[np.ndarray],
                 t_call: float = 0.0,
                 call_times: list[float] | None = None
                 ) -> tuple[list[float], list]: ...


class CollisionContext(Protocol):
    pass


def _mod_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def fixed_point_arrivals(timing: RobotTimingProtocol,
                         q_start: np.ndarray,
                         moving_frames: Callable[[list[float]], np.ndarray],
                         model: TurntableModel,
                         path: PuzzlePath,
                         direction: int,
                         theta_start: float,
                         roll_step: float,
                         tilt: float,
                         checker=None,
                         placement=None,
                         margin_function: Callable | None = None,
                         max_iterations: int = 12,
                         tolerance: float = 1e-3,
                         pre_entry: bool = False,
                         continuity: tuple[float, float] | None = None,
                         progress: Callable[[str], None] | None = None,
                         margin_floor: float | None = None,
                         closest: bool = False,
                         speed_cap: bool = False,
                         ) -> tuple[SelectionResult, list[float], np.ndarray]:
    """Iterate arrival times and moving checkpoint targets to a fixed point.

    At iteration k, arrival estimates determine the puzzle angle and thus the
    checkpoint poses. P2.5 chooses a configuration sequence for those poses;
    RobotTiming.schedule() then computes the new arrival estimates.

    Returns:
        (P2.5 selection, arrival times [s], final moving checkpoint frames).
    """
    count = len(path.frames)
    arrivals = np.zeros(count, dtype=float)

    # PROKOP COPY: with pre_entry, path index 0 is the pre-entry pose: chosen by
    # P2.5 like a checkpoint (no seed), the robot waits there for theta_start
    # (t = 0), and the timing starts there instead of at q_start.
    seed = None if pre_entry else q_start

    def schedule(candidates):
        qs = [candidate.q for candidate in candidates]
        if not pre_entry:
            return np.asarray(timing.schedule(q_start, qs)[0], dtype=float)
        return np.concatenate(([0.0], timing.schedule(qs[0], qs[1:])[0]))

    selection = SelectionResult(False, (), 0.0,
                               "timed fixed-point iteration did not start")

    say = progress if progress is not None else (lambda text: None)  # PROKOP COPY

    # PROKOP COPY: speed cap. No joint can move more than v_max * dt between
    # two checkpoints, dt from the previous pass's arrivals (pass 0: no cap).
    v_max = getattr(getattr(timing, "timing", timing), "v_max", None)
    if speed_cap and v_max is None:
        raise ValueError("speed_cap needs a timing model with v_max")

    def step_limit(times, first_pass):
        if not speed_cap or first_pass:
            return None
        dt = np.diff(np.asarray(times, dtype=float), prepend=0.0)
        return np.outer(np.maximum(dt, 0.0), np.asarray(v_max, dtype=float))

    def select_options(first_pass):
        return dict(continuity=continuity, margin_floor=margin_floor,
                    closest=closest, step_limit=step_limit(arrivals, first_pass))
    for iteration in range(max_iterations):
        t_iteration = time.monotonic()
        angles = theta_start + direction * model.omega * arrivals
        frames = np.asarray([model.checkpoint_T(path, i, float(angles[i]))
                             for i in range(count)], dtype=float)

        def arrival_margin(q: np.ndarray, target: np.ndarray, index: int,
                           roll: float, candidate_tilt: float) -> float:
            if margin_function is not None:
                return float(margin_function(q, target, index, float(arrivals[index]),
                                             roll, candidate_tilt))
            try:
                from .safety_metrics import configuration_metrics
            except ImportError:
                from safety_metrics import configuration_metrics
            metric = configuration_metrics(
                selection_robot, q, checker=checker, placement=placement,
                turntable_angle=float(angles[index]),
            )
            return float(metric.margin)

        # A P2.5 margin callback needs a robot. It is supplied explicitly for
        # normal planning when available; otherwise P2.5's own fallback uses
        # joint-limit and Jacobian checks without a collision object.
        selection_robot = getattr(timing, "robot", None)
        if selection_robot is None:
            selection_robot = getattr(checker, "robot", None)

        if selection_robot is None:
            return SelectionResult(False, (), 0.0,
                                   "planning needs a robot kinematics interface"), \
                   arrivals.tolist(), frames
        else:
            selection = select_roll_branch(
                selection_robot, frames,
                roll_limits=path.roll_limits,
                roll_step=roll_step,
                tilt=tilt,
                q_start=seed,
                **select_options(iteration == 0),
                margin_function=(margin_function
                                 if margin_function is not None
                                 else arrival_margin),
            )

        if not selection.feasible:
            say(f"    pass {iteration}: {selection.reason} "
                f"({time.monotonic() - t_iteration:.0f} s)")
            return selection, arrivals.tolist(), frames

        new_arrivals = schedule(selection.candidates)
        qs = np.array([candidate.q for candidate in selection.candidates])
        steps_deg = np.rad2deg(np.abs(np.diff(qs, axis=0)).max(axis=1))
        say(f"    pass {iteration}: feasible, margin {selection.min_margin:.3f}, "
            f"last arrival {new_arrivals[-1]:.1f} s (change "
            f"{np.max(np.abs(new_arrivals - arrivals)):.2f} s), largest joint step "
            f"{steps_deg.max():.1f} deg at index {int(np.argmax(steps_deg)) + 1} "
            f"({time.monotonic() - t_iteration:.0f} s)")
        if np.max(np.abs(new_arrivals - arrivals)) <= tolerance:
            arrivals = new_arrivals
            break
        arrivals = new_arrivals

    # Rebuild final moving poses and rerun P2.5 at the converged times.
    angles = theta_start + direction * model.omega * arrivals
    frames = np.asarray([model.checkpoint_T(path, i, float(angles[i]))
                         for i in range(count)], dtype=float)

    selection_robot = getattr(timing, "robot", None)
    if selection_robot is None:
        selection_robot = getattr(checker, "robot", None)
    if selection_robot is None:
        return SelectionResult(False, (), 0.0,
                               "planning needs a robot kinematics interface"), \
               arrivals.tolist(), frames

    def final_margin(q: np.ndarray, target: np.ndarray, index: int,
                     roll: float, candidate_tilt: float) -> float:
        if margin_function is not None:
            return float(margin_function(q, target, index,
                                         float(arrivals[index]), roll,
                                         candidate_tilt))
        try:
            from .safety_metrics import configuration_metrics
        except ImportError:
            from safety_metrics import configuration_metrics
        metric = configuration_metrics(selection_robot, q, checker=checker,
                                       placement=placement,
                                       turntable_angle=float(angles[index]))
        return float(metric.margin)

    t_iteration = time.monotonic()
    selection = select_roll_branch(
        selection_robot, frames,
        roll_limits=path.roll_limits,
        roll_step=roll_step,
        tilt=tilt,
        q_start=seed,
        **select_options(False),
        margin_function=final_margin,
    )
    if selection.feasible:
        arrivals = schedule(selection.candidates)
    say(f"    final pass: {'feasible, margin %.3f' % selection.min_margin if selection.feasible else selection.reason} "
        f"({time.monotonic() - t_iteration:.0f} s)")
    return selection, list(arrivals), frames


class _RobotNotAvailable:
    """Fails clearly if P2.6 is called without the robot used to build timing."""

    def ik(self, pose):
        return []


class RobotTimingWithRobot:
    """Small adapter that attaches a robot to an existing timing protocol."""

    def __init__(self, robot, timing):
        self.robot = robot
        self.timing = timing
        self.speed = float(getattr(timing, "speed", 0.0))

    def schedule(self, q_start: np.ndarray, targets: list[np.ndarray],
                 t_call: float = 0.0,
                 call_times: list[float] | None = None):
        return self.timing.schedule(q_start, targets, t_call=t_call,
                                    call_times=call_times)


def timed_plan(
    robot,
    timing: RobotTimingProtocol,
    model: TurntableModel,
    path: PuzzlePath,
    start_phases: Sequence[float] | None = None,
    directions: Sequence[int] | None = None,
    speeds: Sequence[float] | None = None,
    q_start: np.ndarray | None = None,
    roll_step: float = np.deg2rad(10.0),
    tilt: float = 0.0,
    checker=None,
    placement=None,
    margin_function: Callable | None = None,
    max_fixed_point_iterations: int = 12,
    pre_entry: float | None = None,
    continuity: tuple[float, float] | None = None,
    progress: Callable[[str], None] | None = None,
    margin_floor: float | None = None,
    closest: bool = False,
    speed_cap: bool = False,
) -> Plan:
    """Search start phase, direction, and speed for the max-min-margin plan.

    PROKOP COPY: pre_entry [m] (None = off, the original behaviour: timing
    from q_start): the plan starts at a pose this far before checkpoint 0
    along its direction of travel (0 = at checkpoint 0 itself, which already
    lies outside the bar); q_start is its configuration, theta_start the
    puzzle angle when the hoop is there (the robot moves there and waits).
    continuity: (alpha, beta [rad]) first-order continuity limit of P2.5
    (selection.select_roll_branch), None = no limit. progress: called with a
    line of text per fixed-point pass and per start phase. margin_floor,
    closest: see selection.select_roll_branch. speed_cap: limit each joint's
    change between checkpoints to v_max * (arrival difference of the previous
    fixed-point pass).
    """
    extra_pose = pre_entry is not None and pre_entry > 0
    if extra_pose:
        back = np.eye(4)
        back[2, 3] = -float(pre_entry)
        frames = np.concatenate(([path.frames[0] @ back], path.frames))
        limits = (None if path.roll_limits is None
                  else np.concatenate(([path.roll_limits[0]], path.roll_limits)))
        final = path.final_index % len(path.frames) + 1
        path = PuzzlePath(frames=frames,
                          clearance=np.concatenate(([path.clearance[0]], path.clearance)),
                          roll_limits=limits, final_index=final)
    if start_phases is None:
        start_phases = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    if directions is None:
        directions = (-1, 1)
    if speeds is None:
        speeds = (float(getattr(timing, "speed", 0.2)),)

    start_phases = [float(_mod_angle(theta)) for theta in start_phases]
    directions = [int(direction) for direction in directions if direction in (-1, 1)]
    if not directions:
        directions = (-1, 1)

    if q_start is None:
        if hasattr(robot, "q_home"):
            q_start = np.asarray(robot.q_home, dtype=float)
        else:
            q_start = np.zeros(6, dtype=float)
    q_start = np.asarray(q_start, dtype=float)

    if hasattr(timing, "schedule") and not hasattr(timing, "robot"):
        timing = RobotTimingWithRobot(robot, timing)

    best: Plan | None = None
    best_score = -np.inf
    failures: list[str] = []

    for direction in directions:
        for theta_start in start_phases:
            for speed in speeds:
                if speed is not None:
                    timing_speed = getattr(timing, "speed", None)
                    if timing_speed is not None and abs(float(timing_speed) - float(speed)) > 1e-12:
                        try:
                            from solver.timing import RobotTiming
                            candidate_timing = RobotTiming(robot, float(speed))
                            timing = RobotTimingWithRobot(robot, candidate_timing)
                        except Exception as error:
                            failures.append(f"speed {speed}: {error}")
                            continue

                t_phase = time.monotonic()
                if progress is not None:
                    progress(f"  theta_start {np.rad2deg(theta_start):+.0f} deg, "
                             f"direction {direction:+d}:")
                selection, arrivals, _ = fixed_point_arrivals(
                    timing, q_start, lambda times: np.asarray(times),
                    model, path, direction, theta_start,
                    roll_step, tilt, checker=checker, placement=placement,
                    margin_function=margin_function,
                    max_iterations=max_fixed_point_iterations,
                    pre_entry=pre_entry is not None, continuity=continuity,
                    progress=progress, margin_floor=margin_floor,
                    closest=closest, speed_cap=speed_cap,
                )
                if progress is not None:
                    progress(f"  -> {'feasible, min margin %.3f' % selection.min_margin if selection.feasible else 'failed: ' + selection.reason} "
                             f"({time.monotonic() - t_phase:.0f} s)")
                if not selection.feasible:
                    failures.append(selection.reason)
                    continue

                steps = [
                    PlanStep(index=candidate.index,
                             q=candidate.q.copy(),
                             roll=candidate.roll,
                             t_rel=float(arrivals[index]),
                             margin=float(candidate.margin))
                    for index, candidate in enumerate(selection.candidates)
                ]
                plan_start = q_start
                if pre_entry is not None:  # PROKOP COPY: start at the first pose
                    plan_start = steps[0].q
                if extra_pose:  # the pre-entry pose is not a checkpoint
                    steps = [PlanStep(index=st.index - 1, q=st.q, roll=st.roll,
                                      t_rel=st.t_rel, margin=st.margin)
                             for st in steps[1:]]
                plan = Plan(
                    feasible=True,
                    speed=float(getattr(timing, "speed", 0.0)),
                    direction=direction,
                    theta_start=float(theta_start),
                    q_start=plan_start.copy(),
                    steps=steps,
                    approach=[plan_start.copy()],
                    min_margin=float(selection.min_margin),
                )
                if selection.min_margin > best_score:
                    best_score = selection.min_margin
                    best = plan

    if best is not None:
        return best

    reason = "no feasible start phase"
    if failures:
        reason += ": " + " | ".join(list(dict.fromkeys(failures))[-4:] )
    return Plan(feasible=False, reason=reason, speed=0.0,
                direction=1, theta_start=0.0,
                q_start=q_start.copy(), steps=[], approach=[q_start.copy()],
                min_margin=0.0)
