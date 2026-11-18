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

Example:
    ::

        executor = Executor(robot, model, plan, tracker, RobotTiming(robot, plan.speed))
        result = executor.run()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from solver.contracts import PhaseTracker, Plan, TurntableModel
from solver.correction import correct_for_rotation
from solver.timing import RobotTiming

LEAD = 0.5  # s, at least this long from now to the start of the traversal
GUARD = 0.15  # s, a command is sent this long before the previous move ends
MAX_PHASE_SD = np.deg2rad(3.0)  # rad, abort above this uncertainty at arrival
TIME_LIMIT = 120.0  # s, one attempt (assignment)
FIXED_POINT_ITERATIONS = 3


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
    """

    index: int
    t_send: float
    t_arrival: float
    theta_planned: float
    theta_predicted: float
    sd: float
    dtheta: float
    q: np.ndarray


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
                 tracker: PhaseTracker, timing: RobotTiming, clock=time.monotonic,
                 sleep=time.sleep, log=print) -> None:
        """Sets up the execution.

        Args:
            robot: CRS93 or CRS93Sim, initialized, at the start of the plan's
                approach; its speed must equal plan.speed (set by run()).
            model: The turntable model (its axis is needed for corrections).
            plan: The plan to run.
            tracker: Source of the turntable's rotation.
            timing: Timing model at plan.speed.
            clock: Time source [s], the same as the tracker's.
            sleep: Sleep function [s].
            log: Function for progress messages.
        """
        self.robot, self.model, self.plan = robot, model, plan
        self.tracker, self.timing = tracker, timing
        self.clock, self.sleep, self.log = clock, sleep, log
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
