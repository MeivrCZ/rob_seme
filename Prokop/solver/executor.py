"""Task P3.3: executing a Plan on the robot (or the simulator) while the puzzle spins.

Only the ctu_crs CRS93 interface is used (move_to_q, get_q, wait_for_motion_stop),
so the same code runs on CRS93Sim and the real robot.

Sequence:

1. Report and stop if the plan is infeasible ("unreachable").
2. Approach: the plan's approach path to q_start, one move per configuration.
3. Wait for the phase: from the tracker's estimate, the next time the turntable
   reaches Plan.theta_start (at least LEAD from now); the first traversal
   command is sent then.
4. Traverse: every step is sent just in time, GUARD before the previous move
   ends (so the queue never runs dry and the newest estimate is used). Its
   arrival time is predicted with the timing model; the plan's configuration is
   corrected by the difference between the predicted and the planned turntable
   angle at that arrival (solver.correction, fixed-point: the corrected
   configuration changes the move's duration). Abort, i.e. stop sending, if the
   estimate is too uncertain, a correction is not acceptable or the attempt runs
   out of time; the robot then finishes the move it already has.
5. Done when the last step arrived; the hoop stays where the plan ends.

Plans created with ``RobotTiming.schedule_stopped`` pass ``validate_move``.
They can prepare the next corrected move during the previous one, but confirm
that the robot has stopped before sending it. A stationary model starts after
the approach and first check.

Example:
    ::

        executor = Executor(robot, model, plan, tracker, RobotTiming(robot, plan.speed))
        result = executor.run()
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from solver.contracts import PhaseEstimate, PhaseTracker, Plan, PlanStep, TurntableModel
from solver.correction import MAX_CORRECTION, correct_for_rotation
from solver.timing import STOPPED_STEP_LEAD, RobotTiming, TimedMove

LEAD = 0.5  # s, at least this long from now to the start of the traversal
GUARD = 0.15  # s, a command is sent this long before the previous move ends
MAX_PHASE_SD = np.deg2rad(3.0)  # rad, abort above this uncertainty at arrival
CHECKED_ROTATION_LIMIT = np.deg2rad(22.5)  # each corrected segment is checked before sending
TIME_LIMIT = 120.0  # s, one attempt (assignment)
FIXED_POINT_ITERATIONS = 3
MAX_START_PHASE_AGE = 1.5  # s since the last accepted exposure before entering the puzzle
COMMAND_TIME_TOL = 0.025  # s of permitted dispatch jitter after validation
ARRIVAL_TOL = 0.008  # s allowed in the correction's fixed-point calculation
FIRST_MOVE_LEAD = 0.85  # s to check the longer entry move with the simulator and camera running


class _ExecutionRejected(RuntimeError):
    """A phase, time or collision check prevented a command."""


def wrap(angle: float) -> float:
    """Wraps an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class StepRecord:
    """What the executor did for one step.

    Attributes:
        index: Step number in the plan (0-based).
        t_send: Time the command was sent [s].
        t_arrival: Predicted arrival [s].
        theta_planned: Planned turntable angle at arrival [rad].
        theta_predicted: Predicted angle at arrival [rad].
        sd: Standard deviation of the prediction [rad].
        dtheta: Correction applied [rad].
        q: Configuration sent [rad].
        t_capture: Timestamp of the phase estimate, when available [s].
        t_stopped: Time the stop was confirmed, when available [s].
    """

    index: int
    t_send: float
    t_arrival: float
    theta_planned: float
    theta_predicted: float
    sd: float
    dtheta: float
    q: np.ndarray
    t_capture: float | None = None
    t_stopped: float | None = None


@dataclass
class ExecutionResult:
    """Outcome of Executor.run.

    Attributes:
        success: All steps were sent and arrived.
        reason: Why not (empty on success).
        t_start: Time the traversal started (first command) [s], or None.
        t_end: Time the motion ended [s].
        steps: One record per step sent.
    """

    success: bool
    reason: str = ""
    t_start: float | None = None
    t_end: float = 0.0
    steps: list[StepRecord] = field(default_factory=list)


class Executor:
    """Runs one Plan on a robot (see the module doc for the sequence)."""

    def __init__(self, robot, model: TurntableModel, plan: Plan,
                 tracker: PhaseTracker | None, timing: RobotTiming, clock=time.monotonic,
                 sleep=time.sleep, log=print, *,
                 validate_move: Callable[[TimedMove, Callable[[float], float]], bool] | None = None,
                 scene_changed: Callable[[], bool] | None = None,
                 ) -> None:
        """Sets up the execution.

        Args:
            robot: CRS93 or CRS93Sim, initialized, at the start of the plan's
                approach; its speed must equal plan.speed (set by run()).
            model: The turntable model (its axis is needed for corrections).
            plan: The plan to run.
            tracker: Source of the turntable's rotation, optional when stationary.
            timing: Timing model at plan.speed.
            clock: Time source [s], the same as the tracker's.
            sleep: Sleep function [s].
            log: Function for progress messages.
            validate_move: Checks an absolute timed move against a phase prediction.
                Enables the stop-at-waypoint schedule, including fixed scenes.
            scene_changed: Reports a fixture change since planning, for checked runs.
        """
        self.robot, self.model, self.plan = robot, model, plan
        self.tracker, self.timing = tracker, timing
        self.clock, self.sleep, self.log = clock, sleep, log
        self.validate_move = validate_move
        self.scene_changed = scene_changed
        self.t_attempt = clock()

    def _sleep_until(self, t: float) -> None:
        dt = t - self.clock()
        if dt > 0:
            self.sleep(dt)

    def run(self) -> ExecutionResult:
        """Approach, wait for the phase, traverse.

        Returns:
            The result; on an infeasible plan it carries the plan's reason and
            the robot does not move.
        """
        if not self.plan.feasible:
            self.log(f"Unreachable: {self.plan.reason}")
            return ExecutionResult(False, f"unreachable: {self.plan.reason}")
        assert abs(self.timing.speed - self.plan.speed) < 1e-12, "timing speed != plan speed"
        self.t_attempt = self.clock()
        if self.validate_move is not None or self.model.omega == 0.0:
            return self._run_stopped()
        self.robot.set_speed_relative(self.plan.speed)
        self.approach()
        t_go = self.wait_for_phase()
        return self.traverse(t_go)

    def approach(self) -> None:
        """Moves along the plan's approach path to q_start and waits there."""
        for q in self.plan.approach:
            self.robot.move_to_q(q)
        self.robot.move_to_q(self.plan.q_start)
        self.robot.wait_for_motion_stop()

    def next_start_time(self, earliest: float) -> float:
        """Next time at or after @earliest when the turntable is at theta_start.

        Args:
            earliest: Earliest acceptable time [s].

        Returns:
            The time [s], from the tracker's latest estimate.
        """
        est = self.tracker.estimate()
        theta = est.predict(earliest)[0]
        direction = 1.0 if est.omega >= 0 else -1.0
        ahead = ((self.plan.theta_start - theta) * direction) % (2 * np.pi)
        return earliest + ahead / abs(est.omega)

    def wait_for_phase(self) -> float:
        """Waits until shortly before the traversal can start.

        Returns:
            The start time of the traversal [s] (re-estimated shortly before).
        """
        t_go = self.next_start_time(self.clock() + LEAD)
        self.log(f"Waiting {t_go - self.clock():.1f} s for the turntable phase")
        self._sleep_until(t_go - LEAD)
        t_go = self.next_start_time(self.clock() + 0.5 * LEAD)  # fresher estimate
        self._sleep_until(t_go)
        return t_go

    def traverse(self, t_go: float) -> ExecutionResult:
        """Sends the plan's steps just in time, corrected for the predicted rotation.

        Args:
            t_go: Start of the traversal (first command) [s].

        Returns:
            The result.
        """
        result = ExecutionResult(False, t_start=self.clock())
        plan, timing = self.plan, self.timing
        direction = plan.direction
        q_last, queue_end = np.asarray(plan.q_start, float), None
        for k, step in enumerate(plan.steps):
            if queue_end is not None:
                self._sleep_until(queue_end - timing.command_delay - GUARD)
            t_send = self.clock()
            if t_send - self.t_attempt > TIME_LIMIT:
                return self._abort(result, f"time limit at step {k}")
            est = self.tracker.estimate()
            theta_planned = plan.theta_start + direction * self.model.omega * step.t_rel
            start = t_send + timing.command_delay
            if queue_end is not None:
                start = max(start, queue_end)
            q, dtheta, t_arrival, theta_pred, sd = step.q, 0.0, start, 0.0, 0.0
            for _ in range(FIXED_POINT_ITERATIONS):
                t_arrival = start + timing.move_duration(q_last, q)
                theta_pred, sd = est.predict(t_arrival)
                dtheta = wrap(theta_pred - theta_planned)
                q, ok = correct_for_rotation(self.robot, step.q, dtheta, self.model.axis_xy)
                if not ok:
                    return self._abort(result, f"step {k}: correction of "
                                       f"{np.rad2deg(dtheta):.1f} deg not acceptable")
            if sd > MAX_PHASE_SD:
                return self._abort(result, f"step {k}: phase uncertainty "
                                   f"{np.rad2deg(sd):.1f} deg")
            self.robot.move_to_q(q)
            result.steps.append(StepRecord(k, t_send, t_arrival, theta_planned,
                                           theta_pred, sd, dtheta, q.copy()))
            q_last, queue_end = q, t_arrival
        self.robot.wait_for_motion_stop()
        result.success, result.t_end = True, self.clock()
        self.log(f"Traversal done in {result.t_end - result.t_start:.1f} s, "
                 f"{len(result.steps)} steps")
        return result

    def _abort(self, result: ExecutionResult, reason: str) -> ExecutionResult:
        """Stops sending; the robot finishes the move it already has."""
        self.log(f"Abort: {reason}")
        self.robot.wait_for_motion_stop()
        result.reason, result.t_end = reason, self.clock()
        return result

    def _require_time(self, t: float) -> None:
        if t - self.t_attempt > TIME_LIMIT:
            raise _ExecutionRejected("time limit for the attempt")

    def _phase_at(self, estimate: PhaseEstimate, t: float) -> tuple[float, float]:
        theta, sd = estimate.predict(t)
        if not np.isfinite([theta, sd]).all() or sd > MAX_PHASE_SD:
            raise _ExecutionRejected(f"phase uncertainty {np.rad2deg(sd):.1f} deg")
        return theta, sd

    def _current_phase(self, now: float, *, require_fresh: bool = True) -> PhaseEstimate:
        if self.scene_changed is not None and self.scene_changed():
            raise _ExecutionRejected("the scene changed since planning")
        if self.model.omega == 0.0:
            return PhaseEstimate(now, self.model.theta_at(now), 0.0, np.zeros((2, 2)))
        if self.tracker is None:
            raise _ExecutionRejected("a phase tracker is required for a rotating turntable")
        try:
            estimate = self.tracker.estimate()
        except RuntimeError as error:
            raise _ExecutionRejected(str(error)) from error
        if not np.isfinite([estimate.t, estimate.theta, estimate.omega]).all():
            raise _ExecutionRejected("invalid phase estimate")
        if estimate.t - now > COMMAND_TIME_TOL:
            raise _ExecutionRejected("phase time is ahead of the executor clock")
        if require_fresh and now - estimate.t > MAX_START_PHASE_AGE:
            raise _ExecutionRejected("phase estimate is too old to start")
        if estimate.omega * self.plan.direction <= 0.0:
            raise _ExecutionRejected("turntable direction does not match the plan")
        self._phase_at(estimate, now)
        return estimate

    def _stopped_start(self) -> float:
        if self.model.omega == 0.0:
            return self.clock() + FIRST_MOVE_LEAD
        first = True
        while True:
            now = self.clock()
            self._require_time(now)
            estimate = self._current_phase(now)
            theta, _ = self._phase_at(estimate, now)
            ahead = ((self.plan.theta_start - theta) * self.plan.direction) % (2 * np.pi)
            remaining = ahead / abs(estimate.omega)
            if remaining < FIRST_MOVE_LEAD / 2:
                remaining += 2 * np.pi / abs(estimate.omega)
            self._require_time(now + remaining)
            if first:
                self.log(f"Waiting {remaining:.1f} s for the turntable phase")
                first = False
            if remaining <= FIRST_MOVE_LEAD + COMMAND_TIME_TOL:
                return now + remaining
            self.sleep(min(0.1, remaining - FIRST_MOVE_LEAD))

    def _prepare_stopped_move(self, step: PlanStep, q_from: np.ndarray, t_send: float,
                              estimate: PhaseEstimate) -> tuple[TimedMove, StepRecord]:
        theta_planned = self.plan.theta_start + self.plan.direction * self.model.omega * step.t_rel
        move = self.timing.stopped_move(q_from, step.q, t_send)
        correction = 0.0
        if self.model.omega != 0.0 and step.track_phase:
            arrival = move.t_end
            previous_delta = 0.0
            rotation_limit = CHECKED_ROTATION_LIMIT if self.validate_move is not None else MAX_CORRECTION
            for _ in range(12):
                phase, _ = self._phase_at(estimate, arrival)
                correction = wrap(phase - theta_planned)
                q, ok = correct_for_rotation(self.robot, step.q, correction, self.model.axis_xy,
                                             max_rotation=rotation_limit)
                if not ok:
                    raise _ExecutionRejected(f"step {step.index}: correction of "
                                             f"{np.rad2deg(correction):.1f} deg not acceptable")
                move = self.timing.stopped_move(q_from, q, t_send)
                delta = move.t_end - arrival
                if abs(delta) <= ARRIVAL_TOL:
                    break
                # Damp oscillations between longer and shorter corrections.
                arrival += delta / 2 if delta * previous_delta < 0 else delta
                previous_delta = delta
            else:
                raise _ExecutionRejected(f"step {step.index}: arrival time did not converge")
        self._require_time(move.t_end)
        phase, sd = self._phase_at(estimate, move.t_end)
        record = StepRecord(step.index, t_send, move.t_end, theta_planned, phase, sd,
                            correction, move.q_to.copy())
        if self.model.omega != 0.0:
            record.t_capture = estimate.t
        return move, record

    def _run_stopped(self) -> ExecutionResult:
        """Execute a plan timed with schedule_stopped, checking each corrected move."""
        result = ExecutionResult(False)
        pending = None
        try:
            self._current_phase(self.clock())
            self.robot.set_speed_relative(self.plan.speed)
            self.robot.wait_for_motion_stop()
            q_last = self.robot.get_q()
            for q in (*self.plan.approach, self.plan.q_start):
                if np.allclose(q_last, q, rtol=0.0, atol=1e-6):
                    continue
                self._require_time(self.clock())
                self._current_phase(self.clock())
                self.robot.move_to_q(q)
                self.robot.wait_for_motion_stop()
                q_last = q

            next_call = self._stopped_start()
            previous_target = np.asarray(self.plan.q_start, dtype=float)
            for step in self.plan.steps:
                starting = result.t_start is None
                if self.model.omega and pending is not None:
                    # Aim for the original timetable again after a delay.
                    # Start no earlier than the previous predicted stop, and
                    # confirm the real stop before sending the command below.
                    nominal_duration = self.timing.move_duration(previous_target, step.q)
                    nominal_call = result.t_start + step.t_rel - nominal_duration - self.timing.command_delay
                    next_call = max(nominal_call, pending.t_arrival + COMMAND_TIME_TOL,
                                    pending.t_send + self.timing.send_delay)
                for _ in range(2):
                    now = self.clock()
                    # Once threaded, a covered marker is expected. The
                    # arrival-uncertainty guard still applies to every move.
                    estimate = self._current_phase(now, require_fresh=starting)
                    t_send = max(next_call, now + STOPPED_STEP_LEAD / 2)
                    move, record = self._prepare_stopped_move(step, q_last, t_send, estimate)
                    move.hold_from = pending.t_arrival if pending is not None else None
                    if self.validate_move is not None and not self.validate_move(
                        move, lambda t: estimate.theta + estimate.omega * (t - estimate.t),
                    ):
                        raise _ExecutionRejected(f"step {step.index}: timed move fails collision check")
                    preparation_time = self.clock() - now
                    if pending is not None and pending.t_stopped is None:
                        self.robot.wait_for_motion_stop()
                        pending.t_stopped = self.clock()
                    self._sleep_until(t_send)
                    actual_send = self.clock()
                    if actual_send - t_send <= COMMAND_TIME_TOL:
                        break
                    # Rebuild and recheck the move at a later phase with room
                    # for the time this joint span just took to validate.
                    validation_lead = max(STOPPED_STEP_LEAD, preparation_time + 0.07)
                    next_call = actual_send + validation_lead
                else:
                    raise _ExecutionRejected(f"step {step.index}: validation missed the send deadline")

                self._require_time(actual_send)
                self._current_phase(actual_send, require_fresh=starting)
                self.robot.move_to_q(move.q_to)
                if result.t_start is None:
                    result.t_start = actual_send
                record.t_send = actual_send
                record.t_arrival += actual_send - t_send
                record.theta_predicted, record.sd = estimate.predict(record.t_arrival)
                result.steps.append(record)
                pending = record
                q_last = move.q_to
                previous_target = step.q
                next_call = max(record.t_arrival, actual_send + self.timing.send_delay) + STOPPED_STEP_LEAD
            if pending is not None:
                self.robot.wait_for_motion_stop()
                pending.t_stopped = self.clock()
            self._require_time(self.clock())
        except _ExecutionRejected as error:
            result = self._abort(result, str(error))
            if pending is not None and pending.t_stopped is None:
                pending.t_stopped = result.t_end
            return result

        result.success, result.t_end = True, self.clock()
        self.log(f"Traversal done, {len(result.steps)} steps")
        return result
