"""Plan a D traversal on a stopped or spinning turntable."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from Natka.hoop_targets import flange_target, hoop_ring_pose_from_flange, so3_rx
from Natka.plan_ik import all_solutions, continue_ik
from Natka.selection import SelectionResult, select_roll_branch
from collision import PATH_STEP, SceneCollider
from placement import Placement, TaskScene
from solver.approach import ApproachChecker, plan_approach
from solver.contracts import Plan, PlanStep, PuzzlePath, TurntableModel
from solver.feasibility import ring_frames
from solver.ik_batch import batch_ik, expand_turns, in_limits
from solver.timing import COMMAND_DELAY, STOPPED_STEP_LEAD, RobotTiming, TimedMove


MIN_CLEARANCE = 0.001  # m, required estimated scene clearance along the traversal
PHASE_STEP = np.deg2rad(1.0)
ARRIVAL_TOLERANCE = 0.008  # s, for timing and IK at moving waypoints
LOW_POINT_HEIGHT = 0.11  # m in puzzle coordinates, around the lower bend of D
LOW_POINT_LEAD = 0.60  # s for stop confirmation and collision checking near the base
ENTRY_POINTS = 2  # just outside the free end and at the tip
FORWARD_TILTS = np.deg2rad([-60, -55, -50, -45, -30, -15, 0, 15, 30, 45, 50, 55, 60])
REVERSE_TILTS = np.where(FORWARD_TILTS <= 0, FORWARD_TILTS + np.pi, FORWARD_TILTS - np.pi)


@dataclass
class _State:
    q: np.ndarray
    call: float
    roll: float
    tilt: float
    cost: float
    moves: tuple[tuple[TimedMove, float], ...] = ()


class PuzzleCollisionChecker:
    """Check the robot's timed joint motion against the measured puzzle pose."""

    def __init__(self, model: TurntableModel, command_delay: float = COMMAND_DELAY) -> None:
        self.model = model
        self.placement = Placement(model.axis_xy, model.base_offset, model.base_yaw)
        self.collider = SceneCollider(TaskScene())
        self.command_delay = command_delay
        self._contacts: dict[tuple, bool] = {}
        self._clearances: dict[tuple, float] = {}

    @staticmethod
    def _key(q: np.ndarray, theta: float) -> tuple:
        return (*q.round(9), round(theta % (2 * np.pi), 9))

    def clearance(self, q: np.ndarray, theta: float) -> float:
        key = self._key(q, theta)
        if key not in self._clearances:
            hits, clearance = self.collider.check(q, self.placement, theta)
            self._clearances[key] = 0.0 if hits else clearance
            self._contacts[key] = bool(hits)
        return self._clearances[key]

    def _free(self, q: np.ndarray, theta: float) -> bool:
        key = self._key(q, theta)
        if key not in self._contacts:
            self._contacts[key] = self.collider.in_collision(q, self.placement, theta)
        return not self._contacts[key]

    def is_free(self, q: np.ndarray) -> bool:
        """Fixed-pose check for approach planning with a stationary puzzle."""
        return self._free(q, self.model.theta_ref)

    def segment_free(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        count = max(1, int(np.ceil(np.abs(q_to - q_from).max() / PATH_STEP)))
        return all(self.is_free(q) for q in np.linspace(q_from, q_to, count + 1))

    def move_clearance(self, move: TimedMove, theta_at: Callable[[float], float], *,
                       hold: float | None = None) -> float:
        """Return the minimum sampled clearance including the hold before motion."""
        minimum = np.inf
        for q, theta in self._move_samples(move, theta_at, hold):
            minimum = min(minimum, self.clearance(q, theta))
            if minimum < MIN_CLEARANCE:
                break
        return minimum

    def _move_samples(self, move: TimedMove, theta_at: Callable[[float], float], hold: float | None):
        hold = self.command_delay + STOPPED_STEP_LEAD if hold is None else hold
        begin = move.t0 - hold if move.hold_from is None else min(move.t0, move.hold_from)
        count_q = max(1, int(np.ceil(np.abs(move.q_to - move.q_from).max() / PATH_STEP)))
        count_theta = max(1, int(np.ceil(abs(theta_at(move.t_end) - theta_at(begin)) / PHASE_STEP)))
        times = sorted(set(np.linspace(begin, move.t_end, count_theta + 1))
                       | {move.time_at(k / count_q) for k in range(count_q + 1)})
        # Joint progress and phase are monotone. Merge adjacent checks when
        # both of the original displacement limits still hold.
        last = previous = (move.q_at(begin), theta_at(begin))
        yield last
        for t in times[1:]:
            sample = (move.q_at(float(t)), theta_at(float(t)))
            if (np.abs(sample[0] - last[0]).max() > PATH_STEP + 1e-9
                    or abs(sample[1] - last[1]) > PHASE_STEP + 1e-9):
                if previous is not last:
                    yield previous
                    last = previous
            previous = sample
        if previous is not last:
            yield previous

    def validate_move(self, move: TimedMove, theta_at: Callable[[float], float]) -> bool:
        """Check timed contact and keep a positive margin at the target."""
        return (all(self._free(q, theta) for q, theta in self._move_samples(move, theta_at, None))
                and self.clearance(move.q_to, theta_at(move.t_end)) >= MIN_CLEARANCE)


def _phase_order(robot, model: TurntableModel, path: PuzzlePath, q_start: np.ndarray,
                 phases: Sequence[float]) -> list[float]:
    # Prefer phases where the free tip moves away from the waiting hoop.
    ring = hoop_ring_pose_from_flange(robot.fk(q_start))[:2, 3]
    omega = model.direction * model.omega

    def separation(phase: float) -> float:
        tip = model.checkpoint_T(path, min(1, len(path.frames) - 1), phase + omega * 1.4)[:2, 3]
        rel = tip - model.axis_xy
        velocity = omega * np.array([-rel[1], rel[0]])
        return float(velocity @ (tip - ring) / max(np.linalg.norm(tip - ring), 1e-9))

    return sorted(phases, key=separation, reverse=True)


def _waiting_poses(robot, model: TurntableModel, path: PuzzlePath,
                   checker: ApproachChecker, phases: Sequence[float], q_now: np.ndarray):
    # Try pre-entry poses back along the tangent and above the swept puzzle.
    omega = model.direction * model.omega
    for phase in phases[:4]:
        frame = model.checkpoint_T(path, 0, phase + 0.65 * omega)
        for distance, height in ((0.08, 0.0), (0.12, 0.0), (0.12, 0.09), (0.2, 0.0)):
            wait = frame.copy()
            wait[:3, 3] -= distance * frame[:3, 2]
            wait[2, 3] += height
            choices = [q for tilt in (0.0, np.pi) for roll in np.deg2rad(np.arange(-180, 180, 45))
                       for q in all_solutions(robot, flange_target(wait, roll=roll, tilt=tilt), q_now)]
            nearest = next((q for q in sorted(choices, key=lambda q: np.linalg.norm(q - q_now))
                            if checker.is_free(q)), None)
            if nearest is not None:
                yield nearest, phase
                break
    yield q_now, None
    if not np.allclose(robot.q_home, q_now):
        yield np.asarray(robot.q_home, dtype=float), None


def _selection(robot, path: PuzzlePath, model: TurntableModel, checker: PuzzleCollisionChecker,
               q_start: np.ndarray, tilts: Sequence[float], roll_step: float,
               transition_allowed=None) -> SelectionResult:
    frames = model.puzzle_T(model.theta_ref) @ path.frames

    def margin(q: np.ndarray, _flange: np.ndarray, index: int, _roll: float, tilt: float) -> float:
        if not checker.is_free(q):
            return -np.inf
        return abs(np.cos(tilt)) if index < ENTRY_POINTS else 1.0

    return select_roll_branch(
        robot, frames, path.roll_limits, roll_step, q_start=q_start,
        margin_function=margin, tilt_options=tilts, max_joint_step=np.deg2rad(60),
        transition_allowed=transition_allowed,
    )


def _fixed_route(robot, timing: RobotTiming, model: TurntableModel, path: PuzzlePath,
                 checker: PuzzleCollisionChecker, q_now: np.ndarray,
                 tilts: Sequence[float], roll_step: float) -> Plan | None:
    theta = model.theta_ref
    rejected_edges = set()

    def key(q: np.ndarray) -> tuple:
        return tuple(q.round(8))

    def edge_allowed(q_from: np.ndarray, q_to: np.ndarray) -> bool:
        return (key(q_from), key(q_to)) not in rejected_edges

    for _ in range(8):
        selection = _selection(robot, path, model, checker, q_now,
                               tilts, roll_step, edge_allowed if rejected_edges else None)
        if not selection.feasible:
            return None
        targets = [candidate.q for candidate in selection.candidates]
        hold_q = targets[0]
        theta_at = lambda _t: theta
        arrivals, moves = timing.schedule_stopped(hold_q, targets)
        index = next((i for i, move in enumerate(moves) if not checker.validate_move(move, theta_at)), None)
        if index is None:
            margins = [checker.move_clearance(move, theta_at, hold=timing.command_delay if i == 0 else None)
                       for i, move in enumerate(moves)]
            index = next((i for i, margin in enumerate(margins) if margin < MIN_CLEARANCE), None)
        if index is None:
            # A stationary scene permits an RRT approach to the first checkpoint.
            route = plan_approach(robot, checker, q_now, hold_q, max_iterations=3000,
                                  rng=np.random.default_rng(0))
            if route is None:
                return None
            steps = [PlanStep(item.index, q, item.roll, arrival, margin) for
                     item, q, arrival, margin in zip(selection.candidates, targets, arrivals, margins)]
            return Plan(True, "", timing.speed, model.direction, theta, hold_q.copy(),
                        steps, route[1:-1], min(margins))
        if index > 0:
            rejected_edges.add((key(selection.candidates[index - 1].q), key(selection.candidates[index].q)))
        else:
            return None
    return None


def _beam_route(robot, timing: RobotTiming, model: TurntableModel, path: PuzzlePath,
                checker: PuzzleCollisionChecker, q_start: np.ndarray,
                phase: float, approach: list[np.ndarray], *, width: int = 5,
                return_to_waiting: bool = False) -> Plan | None:
    """Explore continuous branches at the arrival time of each candidate move."""
    omega = model.direction * model.omega

    def theta_at(t: float) -> float:
        return phase + omega * t

    to_flange = flange_target(np.eye(4))
    states = [_State(q_start, 0.0, 0.0, 0.0, 0.0)]
    for index in range(len(path.frames)):
        next_states = []
        for state in states:
            initial = state.call + (1.1 if not index else 0.55)
            if not index:
                rolls = np.deg2rad(np.arange(-180.0, 180.0, 30.0))
                tilts = np.deg2rad([0.0, 180.0, -25.0, 25.0, -155.0, 155.0])
            else:
                rolls = state.roll + np.deg2rad([-30.0, -15.0, 0.0, 15.0, 30.0])
                tilts = state.tilt + np.deg2rad([-18.0, 0.0, 18.0])
                tilts = tilts[np.abs(np.cos(tilts)) >= 0.50]
            if path.roll_limits is not None:
                lower, upper = path.roll_limits[index]
                rolls = np.unique(np.clip(rolls, lower, upper))
            frame = model.checkpoint_T(path, index, theta_at(initial))
            flanges = []
            for tilt in tilts:
                tilted = frame.copy()
                tilted[:3, :3] = frame[:3, :3] @ so3_rx(tilt)
                flanges.extend(ring_frames(tilted, rolls, 1) @ to_flange)
            Q, _ = batch_ik(robot, np.asarray(flanges))
            Q = expand_turns(robot, Q)
            valid = in_limits(robot, Q)
            fi, qi = np.nonzero(valid)
            if not len(fi):
                continue
            Q = Q[fi, qi]
            diffs = Q - state.q
            roll = rolls[fi % len(rolls)]
            tilt = tilts[fi // len(rolls)]
            score = np.linalg.norm(diffs, axis=1) + 0.12 * (1.0 - np.abs(np.cos(tilt)))
            if index:
                score += 0.15 * (abs(roll - state.roll) + abs(tilt - state.tilt))
            winners = 0
            for k in np.argsort(score)[:(28 if not index else 12)]:
                if index and np.abs(diffs[k]).max() > np.deg2rad(60):
                    continue
                q = Q[k]
                move = timing.stopped_move(state.q, q, state.call)
                for _ in range(10):
                    frame = model.checkpoint_T(path, index, theta_at(move.t_end))
                    q = continue_ik(robot, flange_target(frame, roll[k], tilt[k]), q)
                    if q is None or (index and np.abs(q - state.q).max() > np.deg2rad(60)):
                        break
                    refined = timing.stopped_move(state.q, q, state.call)
                    if abs(refined.t_end - move.t_end) <= ARRIVAL_TOLERANCE:
                        move = refined
                        if state.moves:
                            move.hold_from = state.moves[-1][0].t_end
                        if checker.validate_move(move, theta_at):
                            difficulty = float(np.linalg.norm(move.q_to - state.q))
                            alignment = 0.25 * (1.0 - abs(np.cos(tilt[k])))
                            lead = LOW_POINT_LEAD if path.positions[index, 2] <= LOW_POINT_HEIGHT else STOPPED_STEP_LEAD
                            next_states.append(_State(
                                move.q_to, max(move.t_end, state.call + timing.send_delay) + lead,
                                float(roll[k]), float(tilt[k]),
                                state.cost + difficulty + alignment, (*state.moves, (move, float(roll[k]))),
                            ))
                            winners += 1
                        break
                    # Damping the estimated arrival helps when travel speed and
                    # tip speed are similar in an early approach.
                    move = TimedMove(move.t0, (move.T + refined.T) / 2, refined.tau, state.q, q)
                if winners >= (width if not index else 2):
                    break
        if not next_states:
            return None
        states = []
        for state in sorted(next_states, key=lambda item: item.cost):
            if not any(np.linalg.norm(state.q - prior.q) < 0.06 and abs(state.call - prior.call) < 0.025
                       for prior in states):
                states.append(state)
            if len(states) == width:
                break
    if return_to_waiting:
        # The starting configuration was already cleared against the full
        # swept volume. Check the timed move back to it while the puzzle turns.
        exit_states = []
        for state in states:
            move = timing.stopped_move(state.q, q_start, state.call)
            move.hold_from = state.moves[-1][0].t_end
            if checker.validate_move(move, theta_at):
                exit_states.append((*state.moves, (move, 0.0)))
    else:
        exit_states = [state.moves for state in states]

    def endpoint_clearance(moves: tuple[tuple[TimedMove, float], ...]) -> float:
        return min(checker.clearance(move.q_to, theta_at(move.t_end)) for move, _ in moves)

    best = None
    for moves in sorted(exit_states, key=endpoint_clearance, reverse=True):
        # Endpoint clearance is an upper bound on clearance along a route.
        if best is not None and endpoint_clearance(moves) <= best.min_margin:
            break
        margins = []
        for i, (move, _) in enumerate(moves):
            margin = checker.move_clearance(move, theta_at, hold=timing.command_delay if i == 0 else None)
            if margin < MIN_CLEARANCE or (best is not None and margin <= best.min_margin):
                break
            margins.append(margin)
        else:
            steps = [PlanStep(i, move.q_to, roll, move.t_end, margins[i], track_phase=i < len(path.frames))
                     for i, (move, roll) in enumerate(moves)]
            best = Plan(True, "", timing.speed, model.direction, phase, q_start.copy(),
                        steps, approach, min(margins))
    return best


def plan_puzzle(robot, model: TurntableModel, path: PuzzlePath, speed: float, *,
                start_phases: Sequence[float] | None = None,
                preferred_phase: float | None = None,
                return_to_waiting: bool = False, log=print) -> Plan:
    """Choose a waiting pose and a collision-checked traversal up to final_index."""
    q_now = np.asarray(robot.get_q(), dtype=float)
    count = path.final_index + 1 if path.final_index >= 0 else len(path.frames)
    if not 0 < count <= len(path.frames):
        raise ValueError("průjezd musí mít platný koncový bod")
    path = PuzzlePath(path.frames[:count], path.clearance[:count],
                      path.roll_limits[:count] if path.roll_limits is not None else None, count - 1)
    timing = RobotTiming(robot, speed)
    checker = PuzzleCollisionChecker(model, timing.command_delay)
    approach_checker = ApproachChecker(model) if model.omega else checker
    reason = "nenalezena časovaná bezkolizní trasa"
    if not approach_checker.is_free(q_now):
        reason = "výchozí poloha zasahuje do překážky nebo do prostoru rotace"
    elif not model.omega:
        log("Plánuji průjezd pro změřenou pevnou polohu.")
        for tilts, roll_degrees in (
                ((np.pi,), 30), ((0.0,), 30), ((np.pi,), 15), ((0.0,), 15),
                (REVERSE_TILTS[::3], 30), (FORWARD_TILTS[::3], 30),
                (REVERSE_TILTS, 15), (FORWARD_TILTS, 15)):
            candidate = _fixed_route(robot, timing, model, path, checker, q_now,
                                     tilts, np.deg2rad(roll_degrees))
            if candidate is not None:
                return candidate
    else:
        phases = (list(start_phases) if start_phases is not None else
                  list(np.linspace(0, 2 * np.pi, 16, endpoint=False) - model.base_yaw))
        ranked_phases = _phase_order(robot, model, path, q_now, phases)
        if preferred_phase is not None and ranked_phases:
            nearest = min(range(len(ranked_phases)), key=lambda index:
                          abs(np.angle(np.exp(1j * (ranked_phases[index] - preferred_phase)))))
            ranked_phases.insert(0, ranked_phases.pop(nearest))
        for wait_index, (q_start, entry_phase) in enumerate(
                _waiting_poses(robot, model, path, approach_checker, ranked_phases, q_now)):
            approach = plan_approach(robot, approach_checker, q_now, q_start,
                                     max_iterations=1500, rng=np.random.default_rng(0))
            if approach is None:
                continue
            phases_here = (entry_phase,) if entry_phase is not None else ranked_phases
            log(f"Zkouším čekací polohu {wait_index + 1}, počet fází: {len(phases_here)}")
            for phase in phases_here:
                candidate = _beam_route(robot, timing, model, path, checker, q_start, phase, approach[1:-1],
                                        return_to_waiting=return_to_waiting)
                if candidate is not None:
                    return candidate
    return Plan(False, reason, timing.speed, model.direction, model.theta_ref, q_now, [], [], 0.0)


def plan_withdrawal(robot, stopped_model: TurntableModel, path: PuzzlePath, speed: float,
                    entry: Sequence[np.ndarray]) -> Plan:
    """Withdraw through the same checkpoints in a stationary puzzle frame."""
    count = len(entry)
    if stopped_model.omega != 0.0 or not 2 <= count <= len(path.frames):
        raise ValueError("Výjezd vyžaduje zastavenou točnu a dokončený průjezd.")
    start = np.asarray(robot.get_q(), dtype=float)
    timing = RobotTiming(robot, speed)
    checker = PuzzleCollisionChecker(stopped_model, timing.command_delay)
    theta = stopped_model.theta_ref
    previous = start
    targets = []
    for q in entry[-2::-1]:
        if not checker.segment_free(previous, q):
            break
        targets.append(q)
        previous = q
    else:
        arrivals, moves = timing.schedule_stopped(start, targets)
        margins = [checker.move_clearance(move, lambda _t: theta) for move in moves]
        if min(margins) >= MIN_CLEARANCE:
            steps = [PlanStep(count - 2 - i, q, 0.0, arrival, margin)
                     for i, (q, arrival, margin) in enumerate(zip(targets, arrivals, margins))]
            return Plan(True, "", speed, stopped_model.direction, theta, start, steps,
                        min_margin=min(margins))

    # Search other rolls/tilts if the quantized pose cannot retrace the plan.
    indices = slice(count - 1, None, -1)
    reverse_path = PuzzlePath(path.frames[indices], path.clearance[indices],
                              path.roll_limits[indices] if path.roll_limits is not None else None,
                              count - 1)
    return plan_puzzle(robot, stopped_model, reverse_path, speed)
