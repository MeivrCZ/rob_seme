"""Prokop's own traversal planner for puzzle D: checkpoints by tier, checked moves.

For a granularity tier (solver.wire), the planner

1. builds the ring targets of every checkpoint at the turntable angle of its
   arrival. A checkpoint fixes only the ring centre: the orientation is free
   on a grid (orientation_grid) of the ring axis' tilt away from the
   direction of travel (in any direction) and the roll about it, for one
   leading side of the hoop (face -1 = side B: axis against the travel, C13;
   face +1 = side A: along it). With the centre on the wire's centreline the
   wire passes through the ring whenever the ring does not touch it;
2. takes every IK solution within the joint limits as a candidate and picks
   one per checkpoint by dynamic programming: the sequence with the least
   total joint travel (sum of ||q_k - q_(k-1)||), i.e. the closest
   configurations. The choice is then checked lazily (soft model, results
   cached): a chosen candidate that collides at its checkpoint is ruled out,
   a chosen move that collides along the way (as in 4) rules out that
   transition (that pair of configurations), and the selection runs again,
   until everything chosen is free or no sequence is left;
3. times the moves as the robot runs them (solver.timing.RobotTiming.schedule,
   queued back to back from checkpoint 0, where the robot waits) and repeats
   1-3 until the arrival times stop changing (with no spin: after one pass);
4. checks every move along the way, continuously: q(t) from the timing
   model; after each sample the next one is placed so that no point of the
   arm can have moved further than the clearance found (the joints' lever
   arms times their largest speed in the move profile, plus the task's own
   motion if the turntable spins), so nothing can pass between two samples.

All collision checks use the soft model (solver.soft_collision.SoftScene,
boxes posed from the perceived TurntableModel, margins subtracted), never the
simulator's exact meshes.

Checkpoint 0 is where the robot waits for the puzzle to come round: its
configuration must also clear the volume the task sweeps in a full turn
(SoftScene.swept_clearance). plan_tiers puts it at the smallest entry
distance before the open end (ENTRIES) where some configuration can wait.
The selection counts the approach too: the joint travel from q_start (by
default q_home) to checkpoint 0, so it keeps the branch near the start.

The final checkpoint (in the stem, just under the jog) must also meet the
assignment's success condition: the whole ring lower than SUCCESS_HEIGHT
above the pad.

Steps 1-4 run for each face separately (the hoop cannot turn over while it
encircles the bar), and the tier keeps the better face: collision free first,
then the shorter traversal. The checks stay strict to the end: the last
checkpoint lies in the stem, already below the success height, but the
assignment allows touching only the maze after success (never the table,
the turntable or the robot itself), so no contact is tolerated anywhere.

plan_tiers runs tier 0, 1, 2, ... and stops at the first tier whose moves are
all collision free, or when a tier has a checkpoint without candidates (every
finer tier contains the same checkpoints). The turntable may spin (model.omega > 0): the targets and
the collision checks use the angle at each time; with omega = 0 the puzzle
stands still.

Example:
    ::

        results = plan_tiers(kin, RobotTiming(kin, 0.2), model, theta_start)
        best = results[-1]   # the first collision-free tier, if any
        plan = best.plan     # contracts.Plan for the executor
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from solver.contracts import Plan, PlanStep, PuzzlePath, TurntableModel
from solver.hoop_pose import (FACE_NAMES, FACES, ROLL_STEP, TILT_DIRECTION_STEP,  # noqa: F401
                              TILTS, candidates, flange_target, orientation_grid,
                              ring_in_flange)
from solver.soft_collision import SoftClearance, SoftScene
from solver.wire import (SEGMENT_NAMES, SUCCESS_HEIGHT, checkpoint_arclengths,
                         ring_top_above_pad, segment_of, wire_frame_at, wire_path)

SLACK = 1.15  # safety factor on the motion bound (the lever arms change during a step)
MAX_JOINT_STEP = np.deg2rad(5.0)  # rad, largest joint change between two samples
MIN_ALLOW = 0.0005  # m, a clearance below this counts as too close (steps would vanish)
ENTRIES = (0.015, 0.03, 0.05, 0.08)  # m, entry distances tried before the open end
PASS_LOG: Callable[[str], None] | None = None  # set to print a line per timing pass
# Set to a callable to record why the space-time tree rejects an extension:
# REJECT_LOG(checkpoint, kind, pair, clearance [m], time [s]); kind "config" or "move".
REJECT_LOG: Callable[[int, str, tuple, float, float], None] | None = None
MAX_SELECTIONS = 3000  # selections per pass (each rules out what it found colliding)
MAX_TIER = 4
MAX_PASSES = 12  # fixed-point passes (arrival times <-> moving targets)
ARRIVAL_TOL = 1e-3  # s


class Selector:
    """Least-total-joint-travel sequence (DP), with candidates and transitions ruled out.

    The step costs ||q_k - q_(k-1)|| are computed once; ruling something out
    only masks them, so selecting again is cheap.
    """

    def __init__(self, cands: list[np.ndarray], q_first: np.ndarray | None = None) -> None:
        self.cands = [np.asarray(c, dtype=np.float32) for c in cands]
        self.start = (np.zeros(len(cands[0]), np.float32) if q_first is None
                      else np.linalg.norm(self.cands[0] - np.float32(q_first), axis=1))
        self.steps = [np.linalg.norm(cur[:, None, :] - prev[None, :, :], axis=2)
                      for prev, cur in zip(self.cands, self.cands[1:])]  # (M_k, M_k-1)
        self.ok = [np.ones(len(c), bool) for c in cands]

    def ban(self, k: int, i: int) -> None:
        """Rules out candidate i of checkpoint k."""
        self.ok[k][i] = False

    def forbid(self, k: int, a: int, b: int) -> None:
        """Rules out the move from candidate a of checkpoint k - 1 to candidate b of k."""
        self.steps[k - 1][b, a] = np.inf

    def select(self) -> list[int] | None:
        """The best sequence of candidate indices, or None if none is left."""
        cost = np.where(self.ok[0], self.start, np.inf)
        back = []
        for k, step in enumerate(self.steps, start=1):
            total = step + cost[None, :]
            back.append(np.argmin(total, axis=1))
            cost = np.where(self.ok[k], total[np.arange(len(step)), back[-1]], np.inf)
        if not np.isfinite(cost).any():
            return None
        chosen = [int(np.argmin(cost))]
        for b in reversed(back):
            chosen.append(int(b[chosen[-1]]))
        return chosen[::-1]


def select_closest(cands: list[np.ndarray], q_first: np.ndarray | None = None,
                   forbidden: set[tuple[int, int, int]] = frozenset()) -> list[int] | None:
    """Candidate per checkpoint with the least total joint travel (see Selector).

    Args:
        cands: Per checkpoint the candidate configurations (M_k, 6).
        q_first: Fixed configuration of checkpoint 0's predecessor, or None
            (checkpoint 0 is free: the robot is brought there beforehand).
        forbidden: Transitions ruled out, as (k, index at k - 1, index at k).

    Returns:
        The chosen index per checkpoint, or None if a checkpoint has none or
        the forbidden transitions leave no sequence.
    """
    if any(len(c) == 0 for c in cands):
        return None
    selector = Selector(cands, q_first)
    for k, a, b in forbidden:
        selector.forbid(k, a, b)
    return selector.select()


@dataclass
class MoveCheck:
    """The check of the move into one checkpoint."""

    index: int  # the checkpoint the move ends at
    segment: str  # the wire segment it follows
    samples: int
    clearance: float  # m, least soft clearance along the move (margins subtracted)
    hits: list[tuple[str, str]]  # the pair that came too close, at the first such sample
    t_hit: float | None  # s, time of that sample


@dataclass
class TierResult:
    """The plan and its checks at one granularity tier."""

    tier: int
    checkpoints: int
    feasible: bool  # IK found for every checkpoint
    reason: str = ""
    face: int = -1  # leading side of the hoop (FACES)
    passes: int = 0
    ruled_out: int = 0  # transitions ruled out because their move collides
    banned: int = 0  # candidates ruled out because they collide at their checkpoint
    entry: float = 0.0  # m, where the path starts before the open end
    selections: int = 0  # selections run (in the last pass)
    qs: np.ndarray | None = None  # (N, 6)
    orientations: np.ndarray | None = None  # (N, 3): roll, tilt, tilt direction [rad]
    arrivals: np.ndarray | None = None  # s after the first command, checkpoint 0 at 0
    moves: list[MoveCheck] = field(default_factory=list)
    seconds: float = 0.0
    plan: Plan | None = None

    @property
    def collision_free(self) -> bool:
        return self.feasible and all(not m.hits for m in self.moves)

    @property
    def min_clearance(self) -> float:
        return min((m.clearance for m in self.moves), default=np.inf)


def _q_at(moves, q0: np.ndarray, t: float) -> np.ndarray:
    """Configuration at time t of a schedule's moves (at rest before / between)."""
    q = q0
    for move in moves:
        if t < move.t0:
            break
        q = move.q_at(t)
    return q


def _step(t: float, end: float, allow: float, levers: np.ndarray, moves, rates,
          task_speed: float) -> float:
    """Longest time step from t in which nothing moves further than allow [s]."""
    dt = end - t
    for _ in range(4):  # the moves overlapping [t, t + dt] decide the speed
        active = [r for m, r in zip(moves, rates) if m.t0 < t + dt and m.t_end > t]
        rate = np.max(active, axis=0) if active else np.zeros(6)
        speed = SLACK * float(levers @ rate) + task_speed
        limit = dt
        if speed > 0:
            limit = allow / speed
        if rate.max() > 0:
            limit = min(limit, MAX_JOINT_STEP / rate.max())
        if limit >= dt:
            break
        dt = limit
    return max(dt, 1e-5)


def check_moves(scene: SoftScene, angle: Callable[[float], float], omega: float, timing,
                qs: np.ndarray, segments: np.ndarray, swept: bool = False
                ) -> tuple[list[MoveCheck], np.ndarray]:
    """Times the moves through the checkpoints and checks each continuously (soft model).

    Args:
        scene: The soft model of the scene.
        angle: Turntable angle [rad] at a time [s] after the first command.
        omega: Turntable speed [rad/s] (0: static).
        timing: RobotTiming.
        qs: Configurations of the checkpoints (N, 6); the robot rests at qs[0].
        segments: Wire segment of the move into each checkpoint (N,).
        swept: Also keep clear of the volume the task sweeps in a full turn
            (moves made while the puzzle may be anywhere, e.g. the approach).

    Returns:
        The check of each move (N - 1) and the arrival times (N,) [s].
    """
    arrivals, moves = timing.schedule(qs[0], list(qs[1:]))
    arrivals = np.concatenate(([0.0], arrivals))
    # Largest joint speeds of each move: the trapezoid peaks at 1 / (T - tau) of the move.
    rates = [np.abs(m.q_to - m.q_from) / max(m.T - m.tau, 1e-9) for m in moves]
    task_speed = abs(omega) * scene.task_reach(angle(0.0)) if omega else 0.0
    checks = []
    for k in range(1, len(qs)):
        t, end = arrivals[k - 1], arrivals[k]
        samples, least, hits, t_hit = 0, np.inf, [], None
        while True:
            q = _q_at(moves, qs[0], t)
            c = scene.clearance(q, angle(t))
            if swept:
                d, pair = scene.swept_clearance(q)
                if d < c.distance:
                    c = SoftClearance(d, pair, c.self_gap, c.self_pair)
            samples += 1
            least = min(least, c.distance)
            allow = min(c.distance, c.self_gap / 2)  # two parts close in on each other
            if allow < MIN_ALLOW:
                hits = [c.pair if c.distance <= c.self_gap / 2 else c.self_pair]
                t_hit = float(t)
                break
            if t >= end:
                break
            t = min(t + _step(t, end, allow, scene.lever_arms(q), moves, rates, task_speed),
                    end)
        checks.append(MoveCheck(k, SEGMENT_NAMES[segments[k]], samples, float(least),
                                hits, t_hit))
    return checks, arrivals


def plan_tier(kin, timing, model: TurntableModel, theta_start: float, tier: int,
              scene: SoftScene, orientations: np.ndarray | None = None,
              faces: tuple[int, ...] = FACES, entry: float | dict[int, float] = ENTRIES[0],
              q_start: np.ndarray | None = None) -> TierResult:
    """Plans and checks one granularity tier for each face; keeps the better one.

    entry: one entry distance for all faces, or one per face.
    """
    t_start = time.monotonic()
    orientations = orientation_grid() if orientations is None else orientations
    entries = entry if isinstance(entry, dict) else {face: entry for face in faces}
    results = []
    for face in faces:
        results.append(_plan_face(kin, timing, model, theta_start, tier, scene, orientations,
                                  face, entries[face], q_start))
        results[-1].entry = entries[face]
    feasible = [r for r in results if r.feasible]
    if feasible:
        best = min(feasible, key=lambda r: (not r.collision_free, r.arrivals[-1]))
    else:
        best = results[0]
        best.reason = "; ".join(f"side {FACE_NAMES[r.face]}: {r.reason}" for r in results)
    best.seconds = time.monotonic() - t_start
    return best


def waits_free(scene: SoftScene, q: np.ndarray, theta: float) -> bool:
    """Whether the robot can hold q while the task turns (free now and of the swept volume)."""
    return scene.clearance(q, theta).free and scene.swept_clearance(q)[0] > 0


def _plan_face(kin, timing, model: TurntableModel, theta_start: float, tier: int,
               scene: SoftScene, orientations: np.ndarray, face: int, entry: float,
               q_start: np.ndarray | None) -> TierResult:
    """Plans and checks the traversal at one tier with one leading face."""
    path: PuzzlePath = wire_path(tier, entry)
    n = len(path.frames)
    result = TierResult(tier=tier, checkpoints=n, feasible=False, face=face)
    segments = segment_of(tier)

    def angle(t: float) -> float:
        return theta_start + model.direction * model.omega * t

    def ik_candidates(i: int, t: float) -> tuple[np.ndarray, np.ndarray]:
        qs, rs = candidates(kin, model.checkpoint_T(path, i, angle(t)), orientations, face)
        if i == n - 1:  # the end: the whole ring lower than SUCCESS_HEIGHT above the pad
            to_puzzle = np.linalg.inv(model.puzzle_T(angle(t)))
            keep = [ring_top_above_pad(to_puzzle @ kin.fk(q) @ ring_in_flange())
                    < SUCCESS_HEIGHT for q in qs]
            qs, rs = qs[keep], rs[keep]
        return qs, rs

    def move_free(k: int, qa: np.ndarray, qb: np.ndarray, t0: float) -> bool:
        """Whether the move from checkpoint k - 1 to k, starting at t0, is free."""
        check = check_moves(scene, lambda t: angle(t0 + t), model.omega, timing,
                            np.array([qa, qb]), np.array([-1, segments[k]]))[0][0]
        return not check.hits

    arrivals = np.zeros(n)
    for result.passes in range(1, MAX_PASSES + 1):
        cands, cand_orients = zip(*(ik_candidates(i, arrivals[i]) for i in range(n)))
        cands = list(cands)
        missing = [i for i, c in enumerate(cands) if len(c) == 0]
        if missing:
            result.reason = f"no IK solution within the limits at checkpoint(s) {missing}"
            return result
        selector = Selector(cands, q_start)
        node_free: dict[tuple[int, int], bool] = {}
        free_moves: dict[tuple[int, int, int], bool] = {}
        chosen, last = None, None  # last: the last choice with free candidates
        result.ruled_out = result.banned = 0
        for result.selections in range(1, MAX_SELECTIONS + 1):
            chosen = selector.select()
            if chosen is None:
                break
            bad_nodes = []
            for k in range(n):
                key = (k, chosen[k])
                if key not in node_free:
                    q, theta = cands[k][chosen[k]], angle(arrivals[k])
                    node_free[key] = (waits_free(scene, q, theta) if k == 0
                                      else scene.clearance(q, theta).free)
                if not node_free[key]:
                    bad_nodes.append(key)
            if bad_nodes:
                for k, i in bad_nodes:
                    selector.ban(k, i)
                result.banned += len(bad_nodes)
                continue
            last = chosen
            bad = []
            for k in range(1, n):
                key = (k, chosen[k - 1], chosen[k])
                if key not in free_moves:
                    free_moves[key] = move_free(k, cands[k - 1][key[1]], cands[k][key[2]],
                                                arrivals[k - 1])
                if not free_moves[key]:
                    bad.append(key)
            if not bad:
                break
            for key in bad:
                selector.forbid(*key)
            result.ruled_out += len(bad)
        if chosen is None:
            if last is None:  # every sequence has a colliding candidate
                dead = [k for k in range(n) if not selector.ok[k].any()]
                result.reason = ("no collision-free candidate at checkpoint(s) "
                                 f"{dead or 'any sequence'}")
                return result
            chosen = last  # no sequence of free moves: keep the last (its moves collide)
        qs = np.array([c[j] for c, j in zip(cands, chosen)])
        result.orientations = np.array([o[j] for o, j in zip(cand_orients, chosen)])
        new = np.concatenate(([0.0], timing.schedule(qs[0], list(qs[1:]))[0]))
        done = np.max(np.abs(new - arrivals)) <= ARRIVAL_TOL
        if PASS_LOG is not None:
            PASS_LOG(f"tier {tier} side {FACE_NAMES[face]} pass {result.passes}: arrivals "
                     f"{np.round(new, 2).tolist()} (change {np.max(np.abs(new - arrivals)):.2f} s), "
                     f"{result.selections} selections, {result.banned} candidates and "
                     f"{result.ruled_out} moves ruled out")
        arrivals = new
        if done or model.omega == 0.0:
            break
    result.feasible, result.qs = True, qs
    result.moves, result.arrivals = check_moves(scene, angle, model.omega, timing, qs,
                                                segments)
    result.plan = Plan(
        feasible=result.collision_free,
        reason="" if result.collision_free else "collision along a move",
        speed=float(getattr(timing, "speed", 0.0)), direction=model.direction,
        theta_start=theta_start, q_start=qs[0],
        steps=[PlanStep(index=i, q=qs[i], roll=float(result.orientations[i, 0]),
                        t_rel=float(arrivals[i]),
                        margin=result.moves[i - 1].clearance if i else np.inf)
               for i in range(n)],
        min_margin=result.min_clearance)
    return result


def plan_tiers(kin, timing, model: TurntableModel, theta_start: float,
               max_tier: int = MAX_TIER, orientations: np.ndarray | None = None,
               scene: SoftScene | None = None, faces: tuple[int, ...] = FACES,
               progress: Callable[[TierResult], None] | None = None,
               q_start: np.ndarray | None = None,
               entries: tuple[float, ...] = ENTRIES, min_tier: int = 0) -> list[TierResult]:
    """Tier 0, 1, ... up to max_tier; stops at the first collision-free one.

    Args:
        q_start: Where the robot comes from (its travel to checkpoint 0
            counts in the selection); default kin.q_home.
        entries: Entry distances tried (the first where the robot can wait).
        min_tier: The first tier tried (a finer one even if coarser ones work).

    Returns:
        The result of every tier tried (the last one is collision free if any is).
    """
    scene = scene or SoftScene(model)
    q_start = np.asarray(kin.q_home if q_start is None else q_start, dtype=float)
    orientations = orientation_grid() if orientations is None else orientations
    # The entry per face: a side that can wait closer in must not move the other one.
    entry = {face: choose_entry(kin, scene, model, theta_start, orientations, (face,), entries)
             for face in faces}
    faces = tuple(face for face in faces if entry[face] is not None)
    if not faces:
        return [TierResult(tier=0, checkpoints=0, feasible=False,
                           reason=f"no configuration can wait at the entry "
                                  f"({', '.join(f'{1000 * e:.0f}' for e in entries)} mm "
                                  f"before the open end) clear of the swept volume")]
    results = []
    for tier in range(min_tier, max(max_tier, min_tier) + 1):
        results.append(plan_tier(kin, timing, model, theta_start, tier, scene,
                                 orientations, faces, entry, q_start))
        if progress is not None:
            progress(results[-1])
        if results[-1].collision_free or not results[-1].feasible:
            break
    return results


def choose_entry(kin, scene: SoftScene, model: TurntableModel, theta: float,
                 orientations: np.ndarray, faces: tuple[int, ...],
                 entries: tuple[float, ...] = ENTRIES) -> float | None:
    """The smallest entry distance where some configuration can wait (waits_free)."""
    for entry in entries:
        checkpoint = model.checkpoint_T(wire_path(0, entry), 0, theta)
        for face in faces:
            qs, _ = candidates(kin, checkpoint, orientations, face)
            if any(waits_free(scene, q, theta) for q in qs):
                return entry
    return None


PRE_ENTRY_DIRECTIONS = ("back", "up", "back and up")
PRE_ENTRY_DISTANCES = (0.05, 0.10, 0.15)  # m


def plan_approach(kin, scene: SoftScene, model: TurntableModel, theta: float, timing,
                  q_from: np.ndarray, q0: np.ndarray, entry: float
                  ) -> tuple[list[np.ndarray] | None, str]:
    """Moves from q_from to checkpoint 0's configuration q0, clear of the swept volume.

    The puzzle may be anywhere while the robot approaches, so every move is
    checked continuously against the scene AND the volume the task sweeps in a
    full turn (check_moves with swept=True). Tried in order: one direct joint
    move; then two moves through a pre-entry pose: the ring as at checkpoint 0,
    moved back along the wire, up, or back and up (PRE_ENTRY_DISTANCES), with
    the IK solution closest to q0 (itself a waiting pose: waits_free).

    Returns:
        The configurations from q_from to q0 (None if nothing works) and how.
    """
    def free(qs: list[np.ndarray]) -> float | None:
        """Least clearance of the moves through qs (scene and swept volume), None if any hits."""
        checks = check_moves(scene, lambda t: theta, 0.0, timing, np.array(qs),
                             np.zeros(len(qs), int), swept=True)[0]
        return None if any(c.hits for c in checks) else min(c.clearance for c in checks)

    q_from, q0 = np.asarray(q_from, float), np.asarray(q0, float)
    least = free([q_from, q0])
    if least is not None:
        return [q_from, q0], f"one direct joint move, soft clearance {1000 * least:.1f} mm"
    ring = kin.fk(q0) @ ring_in_flange()
    back = -model.checkpoint_T(wire_path(0, entry), 0, theta)[:3, 2]  # against the travel
    up = np.array([0.0, 0.0, 1.0])
    directions = {"back": back, "up": up, "back and up": back + up}
    for name in PRE_ENTRY_DIRECTIONS:
        direction = directions[name] / np.linalg.norm(directions[name])
        for distance in PRE_ENTRY_DISTANCES:
            target = ring.copy()
            target[:3, 3] += distance * direction
            sols = [q for q in kin.ik(target @ np.linalg.inv(ring_in_flange()))
                    if kin.in_limits(q)]
            if not sols:
                continue
            q_pre = min(sols, key=lambda q: np.linalg.norm(q - q0))
            least = free([q_from, q_pre, q0])
            if least is not None:
                return ([q_from, q_pre, q0], f"through a pre-entry pose {100 * distance:.0f} cm "
                        f"{name}, soft clearance {1000 * least:.1f} mm")
    return None, ("no direct move or pre-entry pose (back / up / back and up, 5-15 cm) "
                  "clears the scene and the swept volume")


# --- The spinning puzzle: when to start ---
#
# One start angle per placement: the robot waits at checkpoint 0 (clear of the
# swept volume) until the puzzle comes round to it, at most one turn. Rotating
# the whole scene about the robot axis maps plans onto each other, so a fixed
# relative geometry at the start (the puzzle facing the robot half way through
# the traversal) is the natural choice; no search over start angles.

START_STEP = np.deg2rad(30.0)  # rad, spacing of the start angles tried
TRAVERSAL_GUESS = 7.0  # s, typical traversal (centres the start angles)


def facing_angle(model: TurntableModel) -> float:
    """Turntable angle at which the puzzle's open end (its +x) points at the robot."""
    theta = 0.0
    for _ in range(4):  # the puzzle origin moves with theta when the base is off the axis
        T = model.puzzle_T(theta)
        theta += np.arctan2(-T[1, 3], -T[0, 3]) - np.arctan2(T[1, 0], T[0, 0])
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def start_angles(model: TurntableModel, traversal: float = TRAVERSAL_GUESS,
                 count: int = 12, step: float = START_STEP) -> list[float]:
    """Start angles to try, best first.

    The puzzle should face the robot about half way through the traversal, so
    the first candidate is the facing angle minus half the traversal's turn;
    then alternately later and earlier ones, `step` apart (`count` cover a
    full turn with the default step).
    """
    centre = facing_angle(model) - model.direction * model.omega * traversal / 2
    out = [centre]
    for k in range(1, count):
        out.append(centre + (k + 1) // 2 * step * (1 if k % 2 else -1))
    return [float(np.arctan2(np.sin(a), np.cos(a))) for a in out]


def plan_spinning(kin, timing, model: TurntableModel, candidates: list[float] | None = None,
                  bend_points: int = 3, q_start: np.ndarray | None = None,
                  progress: Callable[[float, TierResult], None] | None = None
                  ) -> tuple[float | None, TierResult | None]:
    """Plans the traversal of a spinning puzzle at its start angle.

    Args:
        model: The (fitted) turntable model; its omega and direction are used.
        candidates: Start angles to try in order (default: only the first of
            start_angles(model), the puzzle facing the robot half way through).
        bend_points: Checkpoints on the bend (plan_rrt: RRT between checkpoints).
        progress: Called with (start angle, the last tier's result) per candidate.

    Returns:
        The start angle and the collision-free result, or (None, None).
    """
    scene = SoftScene(model)
    for theta in (start_angles(model, count=1) if candidates is None else candidates):
        result = (plan_spacetime(kin, timing, model, theta, bend_points=bend_points,
                                 scene=scene, q_start=q_start) if model.omega
                  else plan_rrt(kin, timing, model, theta, bend_points=bend_points, scene=scene,
                                q_start=q_start))
        if progress is not None:
            progress(theta, result)
        if result.collision_free:
            return theta, result
    return None, None


# --- RRT between checkpoints (lab 6) ---
#
# Checkpoints: the segment ends and `bend_points` on the bend. The selection
# picks one configuration per checkpoint as above; between consecutive ones a
# joint-space RRT (solver.rrt) finds the moves, every edge timed from the start
# of the traversal and checked continuously against the puzzle at the angle of
# that moment. A pair the RRT cannot connect is ruled out and the selection
# runs again. With spin, the passes repeat with the new arrival times; from
# the second pass on, each checkpoint's candidates are those within
# WARM_RADIUS of the previous pass's configuration (so the passes converge and
# get cheap). Finally random shortcuts over the whole path (the checkpoints
# between may be skipped: a free continuous motion cannot unthread the ring)
# and one continuous check of the final path with its real timing.

WARM_RADIUS = np.deg2rad(25.0)  # rad, warm start: largest joint change from the last pass


def _checkpoint_candidates(kin, model: TurntableModel, path: PuzzlePath, i: int, theta: float,
                           orientations: np.ndarray, face: int) -> tuple[np.ndarray, np.ndarray]:
    """IK candidates at checkpoint i (at the end: only those meeting the success height)."""
    qs, os_ = candidates(kin, model.checkpoint_T(path, i, theta), orientations, face)
    if i == len(path.frames) - 1:
        to_puzzle = np.linalg.inv(model.puzzle_T(theta))
        keep = [ring_top_above_pad(to_puzzle @ kin.fk(q) @ ring_in_flange()) < SUCCESS_HEIGHT
                for q in qs]
        qs, os_ = qs[keep], os_[keep]
    return qs, os_


def plan_rrt(kin, timing, model: TurntableModel, theta_start: float, bend_points: int = 3,
             scene: SoftScene | None = None, orientations: np.ndarray | None = None,
             faces: tuple[int, ...] = FACES, q_start: np.ndarray | None = None,
             entries: tuple[float, ...] = ENTRIES, seed: int = 0) -> TierResult:
    """Plans the traversal with RRT moves between checkpoints (see above); best face."""
    from solver import rrt as rrt_mod
    t_start = time.monotonic()
    scene = scene or SoftScene(model)
    q_start = np.asarray(kin.q_home if q_start is None else q_start, dtype=float)
    orientations = orientation_grid() if orientations is None else orientations
    results = []
    for face in faces:
        entry = choose_entry(kin, scene, model, theta_start, orientations, (face,), entries)
        if entry is None:
            r = TierResult(tier=0, checkpoints=0, feasible=False, face=face)
            r.reason = "no configuration can wait at the entry clear of the swept volume"
            results.append(r)
            continue
        r = _plan_rrt_face(kin, timing, model, theta_start, bend_points, scene, orientations,
                           face, entry, q_start, np.random.default_rng(seed), rrt_mod)
        r.entry = entry
        results.append(r)
    feasible = [r for r in results if r.feasible]
    if feasible:
        best = min(feasible, key=lambda r: (not r.collision_free, r.arrivals[-1]))
    else:
        best = results[0]
        best.reason = "; ".join(f"side {FACE_NAMES[r.face]}: {r.reason}" for r in results)
    best.seconds = time.monotonic() - t_start
    return best


def _plan_rrt_face(kin, timing, model, theta_start, bend_points, scene, orientations, face,
                   entry, q_start, rng, rrt_mod) -> TierResult:
    """plan_rrt for one leading face."""
    path = wire_path(0, entry, bend_points)
    n = len(path.frames)
    result = TierResult(tier=0, checkpoints=n, feasible=False, face=face)

    def angle(t: float) -> float:
        return theta_start + model.direction * model.omega * t

    def duration(a: np.ndarray, b: np.ndarray) -> float:
        return timing.move_duration(a, b)

    # Pass 1 plans with the puzzle frozen at the start angle (all arrival times
    # are still 0, the targets are computed for it); from pass 2 on, the links
    # are checked with the turntable turning (the targets follow the arrivals).
    frozen = {"on": True}

    def link_angle(t: float) -> float:
        return theta_start if frozen["on"] else angle(t)

    def edge_free(a: np.ndarray, b: np.ndarray, t: float) -> bool:
        omega = 0.0 if frozen["on"] else model.omega
        check = check_moves(scene, lambda s: link_angle(t + s), omega, timing,
                            np.array([a, b]), np.array([-1, 0]))[0][0]
        return not check.hits

    def config_free(q: np.ndarray, t: float) -> bool:
        return scene.clearance(q, link_angle(t)).free

    arrivals = np.zeros(n)
    previous = None  # the last pass's checkpoint configurations
    for result.passes in range(1, MAX_PASSES + 1):
        frozen["on"] = result.passes == 1 and model.omega != 0.0
        cands, cand_orients = [], []
        for i in range(n):
            qs, os_ = _checkpoint_candidates(kin, model, path, i, angle(arrivals[i]),
                                             orientations, face)
            if previous is not None and len(qs):  # warm start near the last pass
                near = np.abs(qs - previous[i]).max(axis=1) <= WARM_RADIUS
                if near.any():
                    qs, os_ = qs[near], os_[near]
            cands.append(qs)
            cand_orients.append(os_)
        missing = [i for i, c in enumerate(cands) if len(c) == 0]
        if missing:
            result.reason = f"no IK solution within the limits at checkpoint(s) {missing}"
            return result
        selector = Selector(cands, q_start)
        node_free: dict[tuple[int, int], bool] = {}
        links: dict[tuple[int, int, int], list[np.ndarray] | None] = {}
        chosen, segs = None, None
        result.ruled_out = result.banned = 0
        for result.selections in range(1, MAX_SELECTIONS + 1):
            chosen = selector.select()
            if chosen is None:
                break
            bad_nodes = []
            for k in range(n):
                key = (k, chosen[k])
                if key not in node_free:
                    q, theta = cands[k][chosen[k]], angle(arrivals[k])
                    node_free[key] = (waits_free(scene, q, theta) if k == 0
                                      else scene.clearance(q, theta).free)
                if not node_free[key]:
                    bad_nodes.append(key)
            if bad_nodes:
                for k, i in bad_nodes:
                    selector.ban(k, i)
                result.banned += len(bad_nodes)
                continue
            segs, t, bad = [], 0.0, None
            for k in range(1, n):
                key = (k, chosen[k - 1], chosen[k])
                if key not in links:
                    stats: dict = {}
                    # Checked around this pass's arrival estimate, the time the
                    # targets were computed for (pass 1: the puzzle at the start
                    # angle); the final path is checked with its real timing.
                    links[key] = rrt_mod.rrt(cands[k - 1][key[1]], cands[k][key[2]],
                                             kin.q_min, kin.q_max, duration, edge_free,
                                             t0=arrivals[k - 1], rng=rng,
                                             node_free=config_free, stats=stats)
                    if PASS_LOG is not None:
                        link = links[key]
                        PASS_LOG(f"  link {k - 1}->{k} at t {arrivals[k - 1]:.1f} s: "
                                 f"{'straight' if link is not None and len(link) == 2 else (f'{len(link)} waypoints' if link else 'NONE')}"
                                 f", {stats['nodes']} nodes, {stats['edges']} edges checked, "
                                 f"{stats['seconds']:.1f} s")
                if links[key] is None:
                    bad = key
                    break
                segs.append(links[key])
                t = rrt_mod.path_times(links[key], duration, t)[-1]
            if bad is None:
                break
            selector.forbid(*bad)
            result.ruled_out += 1
            segs = None
        if segs is None:
            if chosen is None and not any(selector.ok[k].any() for k in range(n)):
                result.reason = "no collision-free candidate at the checkpoints"
            else:
                result.reason = "no sequence of checkpoints the RRT can connect"
            return result
        waypoints = [segs[0][0]]
        at_checkpoint = [0]
        for seg in segs:
            waypoints += list(seg[1:])
            at_checkpoint.append(len(waypoints) - 1)
        previous = np.array([waypoints[i] for i in at_checkpoint])
        result.orientations = np.array([o[j] for o, j in zip(cand_orients, chosen)])
        new = np.array(rrt_mod.path_times(waypoints, duration))[at_checkpoint]
        done = np.max(np.abs(new - arrivals)) <= ARRIVAL_TOL
        if PASS_LOG is not None:
            PASS_LOG(f"rrt side {FACE_NAMES[face]} pass {result.passes}: checkpoint arrivals "
                     f"{np.round(new, 2).tolist()} (change {np.max(np.abs(new - arrivals)):.2f} "
                     f"s), {len(waypoints)} waypoints, {result.selections} selections, "
                     f"{result.banned} candidates and {result.ruled_out} links ruled out")
        arrivals = new
        if done or model.omega == 0.0:
            break
    waypoints = rrt_mod.shortcut(waypoints, duration, edge_free, rng=rng)
    qs = np.array(waypoints)
    result.feasible, result.qs = True, qs
    result.moves, result.arrivals = check_moves(scene, angle, model.omega, timing, qs,
                                                np.zeros(len(qs), int))
    result.plan = Plan(
        feasible=result.collision_free,
        reason="" if result.collision_free else "collision along the final path",
        speed=float(getattr(timing, "speed", 0.0)), direction=model.direction,
        theta_start=theta_start, q_start=qs[0],
        steps=[PlanStep(index=i, q=qs[i], roll=0.0, t_rel=float(result.arrivals[i]),
                        margin=result.moves[i - 1].clearance if i else np.inf)
               for i in range(len(qs))],
        min_margin=result.min_clearance)
    return result


# --- The spinning puzzle: one space-time tree (user design, 2026-11-25) ---
#
# The robot waits at checkpoint 0 until the puzzle is at the start angle: that
# is t = 0. From there a tree grows in (q, t): a node's time is its parent's
# plus the move's duration (timing model, the robot stops at each node), and
# the move to it is checked continuously against the puzzle turning during it
# (check_moves). The checkpoints are subgoals in turn; a reached one is the root
# for the next. A checkpoint's goal configuration moves with the puzzle: the
# ring's pose relative to the puzzle is fixed (the orientation chosen by the
# static plan), and the goal is the IK solution, closest to the node, at the
# moment the robot would arrive there. No timing passes: all times are real.
# The ring's roll about its own axis is free at every moment (it is round:
# only the tilt and its direction relative to the bar decide whether it fits),
# so the hoop need not turn with the puzzle.
# Few, fast moves: the robot stops at every node (it blends only moves in the
# same joint direction), and while it stands the puzzle turns on; many small
# steps fall behind the turning bar until the pose leaves the arm's reach
# (diagnosed 2026-11-25). So the steps are large (ST_STEP), but no move may
# take longer than ST_MAX_MOVE, so the puzzle turns little during any one.
#
# Sampling: the free space with the ring around the bar is a thin tube along
# the wire, which uniform samples almost never hit. So most extensions are
# guided along the wire: from a tree node (each knows how far along the wire it
# is), aim at a point further along it, up to the next checkpoint, the ring's
# orientation interpolated between the two checkpoints', solved at the time the
# robot would get there (IK closest to the node, plus noise). The rest: the goal
# (ST_GOAL_BIAS) and uniform samples. A new configuration is first checked alone
# at its arrival time (cheap) before the move to it is checked continuously.
# A checkpoint fixes only the ring centre: a free node with the ring centre
# within ST_AT_CHECKPOINT of it has reached it, in whatever orientation (the
# last checkpoint also needs the success height); the exact goal pose (the
# static plan's tilt, any roll) is one more way to arrive.
#
# Backtracking: how the ring arrives at a checkpoint (configuration, time)
# decides whether the next one can be reached. If checkpoint k + 1 cannot be
# reached, the planner goes back to k for a different arrival (a new search,
# accepted only if it differs by ST_DIFFERENT), up to ST_ARRIVALS per
# checkpoint, then further back.

ST_STEP = np.deg2rad(25.0)  # rad, largest joint change of one extension
ST_MAX_MOVE = 1.2  # s, longest move (the puzzle turns ~15 deg meanwhile)
ST_GOAL_BIAS = 0.2
ST_BOX = np.deg2rad(40.0)  # rad, sampling box beyond the node and the goal
ST_MAX_NODES = 600  # per checkpoint
ST_TIME_LIMIT = 6.0  # s, per tree search (one orientation at one checkpoint)
ST_TOTAL_LIMIT = 150.0  # s, the whole space-time planning
ST_ARRIVALS = 3  # different arrivals tried at a checkpoint before backtracking further
ST_RETRY_ORIENTATIONS = 3  # orientations tried when searching a checkpoint again
ST_DIFFERENT = (0.3, np.deg2rad(10.0))  # s, rad: an arrival this different is a new one
ST_ALTERNATIVES = 5  # other ring orientations tried at a checkpoint that cannot be reached
ST_MAX_GOAL_JUMP = np.deg2rad(60.0)  # rad, a goal further from the node is another branch
ST_WIRE_SHARE = 0.6  # share of the extensions guided along the wire (after the goal bias)
ST_WIRE_NOISE = np.deg2rad(0.5)  # rad, noise on a guided target (none at the checkpoint)
ST_AIM_CHECKPOINT = 0.3  # share of the guided extensions aimed at the checkpoint itself
ST_AT_CHECKPOINT = 0.005  # m, a free node with the ring centre this close has reached it
ST_RECENT = 20  # guided extensions start from one of the last nodes this often (half)


def plan_spacetime(kin, timing, model: TurntableModel, theta_start: float,
                   bend_points: int = 3, scene: SoftScene | None = None,
                   q_start: np.ndarray | None = None, seed: int = 0) -> TierResult:
    """Plans the traversal of the spinning puzzle with a space-time tree (see above).

    Returns:
        A TierResult (qs = the waypoints, arrivals their times, plan the Plan).
    """
    from dataclasses import replace
    from solver import rrt as rrt_mod
    t_wall = time.monotonic()
    scene = scene or SoftScene(model)
    rng = np.random.default_rng(seed)
    # 1. The static plan at the start angle: side, entry, waiting pose, orientations.
    static = plan_rrt(kin, timing, replace(model, omega=0.0), theta_start, bend_points,
                      scene=scene, q_start=q_start, seed=seed)
    if not static.collision_free:
        static.reason = f"static plan at the start angle: {static.reason or 'collides'}"
        return static
    face, entry = static.face, static.entry
    path = wire_path(0, entry, bend_points)
    n = len(path.frames)
    arclength = checkpoint_arclengths(0, entry, bend_points)
    result = TierResult(tier=0, checkpoints=n, feasible=False, face=face)
    result.entry = entry

    def angle(t: float) -> float:
        return theta_start + model.direction * model.omega * t

    def config_free(q: np.ndarray, t: float) -> bool:
        return scene.clearance(q, angle(t)).free

    rolls = np.arange(-np.pi, np.pi, ROLL_STEP)

    def ik_any_roll(frame: np.ndarray, orient: np.ndarray, q_near: np.ndarray
                    ) -> np.ndarray | None:
        """IK closest to q_near putting the ring on frame with the orientation's tilt and
        tilt direction, at ANY roll: the ring is round, so its roll about its own axis
        does not change whether it fits around the bar (it need not turn with the puzzle)."""
        _, tilt, direction = (float(x) for x in orient)
        sols = [x for roll in rolls
                for x in kin.ik(flange_target(frame, float(roll), face, tilt, direction))
                if kin.in_limits(x)]
        if not sols:
            return None
        return min(sols, key=lambda x: float(np.abs(x - q_near).max()))

    def wire_target(s_wire: float, orient: np.ndarray, q_near: np.ndarray,
                    t_near: float) -> np.ndarray | None:
        """IK (closest to q_near, any roll) putting the ring at distance s_wire along the
        wire, at the time the robot would get there from q_near."""
        q = q_near
        for _ in range(2):
            t = t_near + duration(q_near, q)
            q = ik_any_roll(model.puzzle_T(angle(t)) @ wire_frame_at(s_wire, entry), orient,
                            q_near)
            if q is None:
                return None
        return q if np.abs(q - q_near).max() <= ST_MAX_GOAL_JUMP else None

    def blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
        """Orientation between a and b (angles interpolated the short way round)."""
        return a + w * np.angle(np.exp(1j * (b - a)))

    def duration(a: np.ndarray, b: np.ndarray) -> float:
        return timing.move_duration(a, b)

    def move_free(a: np.ndarray, b: np.ndarray, t: float) -> bool:
        check = check_moves(scene, lambda s: angle(t + s), model.omega, timing,
                            np.array([a, b]), np.array([-1, 0]))[0][0]
        return not check.hits

    def goal_at(k: int, orient: np.ndarray, q_near: np.ndarray, t: float) -> np.ndarray | None:
        """Configuration putting the ring on checkpoint k at time t, closest to q_near
        (the orientation's tilt and tilt direction, any roll)."""
        q = ik_any_roll(model.checkpoint_T(path, k, angle(t)), orient, q_near)
        if q is None:
            return None
        return q if np.abs(q - q_near).max() <= ST_MAX_GOAL_JUMP else None

    to_ring = ring_in_flange()

    def at_checkpoint(k: int, q: np.ndarray, t: float) -> bool:
        """Whether the ring centre is on checkpoint k at time t (and, at the last
        checkpoint, the whole ring below the success height)."""
        ring = kin.fk(q) @ to_ring
        P = model.puzzle_T(angle(t))
        if np.linalg.norm(ring[:3, 3] - (P @ path.frames[k])[:3, 3]) > ST_AT_CHECKPOINT:
            return False
        if k == n - 1:
            return ring_top_above_pad(np.linalg.inv(P) @ ring) < SUCCESS_HEIGHT
        return True

    def reach(k: int, orient: np.ndarray, q0: np.ndarray, t0: float,
              orient_from: np.ndarray):
        """Tree from (q0, t0) to checkpoint k; the path [(q, t), ...] or None."""
        deadline = time.monotonic() + ST_TIME_LIMIT
        nodes, times, parents = [q0], [t0], [-1]
        progress = [arclength[k - 1]]  # how far along the wire each node is
        best = np.inf

        def try_goal(i: int):
            q, t = nodes[i], times[i]
            goal = goal_at(k, orient, q, t)
            for _ in range(2):  # the arrival time depends on the move
                if goal is None:
                    return None
                goal = goal_at(k, orient, q, t + duration(q, goal))
            if goal is None or duration(q, goal) > ST_MAX_MOVE:  # no long moves
                return None
            if not scene.clearance(goal, angle(t + duration(q, goal))).free:
                return None
            if not move_free(q, goal, t):
                return None
            chain = [i]
            while parents[chain[-1]] >= 0:
                chain.append(parents[chain[-1]])
            out = [(nodes[j], times[j]) for j in reversed(chain)]
            return out + [(goal, t + duration(q, goal))]

        span = arclength[k] - arclength[k - 1]
        found = try_goal(0)
        while found is None and len(nodes) < ST_MAX_NODES and time.monotonic() < deadline:
            draw = rng.random()
            s_new = None
            if draw < ST_GOAL_BIAS:  # the goal, as seen from the root
                sample = goal_at(k, orient, q0, t0)
                if sample is None:
                    continue
                near = int(np.argmin(np.linalg.norm(np.array(nodes) - sample, axis=1)))
            elif draw < ST_GOAL_BIAS + ST_WIRE_SHARE:  # guided along the wire
                near = (len(nodes) - 1 - int(rng.integers(min(len(nodes), ST_RECENT)))
                        if rng.random() < 0.5 else int(rng.integers(len(nodes))))
                s_new = (arclength[k] if rng.random() < ST_AIM_CHECKPOINT
                         else rng.uniform(progress[near], arclength[k]))
                w = (s_new - arclength[k - 1]) / span if span > 0 else 1.0
                target = wire_target(s_new, blend(orient_from, orient, w), nodes[near],
                                     times[near])
                if target is None:
                    continue
                sample = target if s_new >= arclength[k] else \
                    target + rng.normal(0.0, ST_WIRE_NOISE, 6)
            else:  # uniform around the tree and the goal
                ref = goal_at(k, orient, q0, t0)
                pts = np.array(nodes + ([ref] if ref is not None else []))
                lo = np.maximum(pts.min(axis=0) - ST_BOX, kin.q_min)
                hi = np.minimum(pts.max(axis=0) + ST_BOX, kin.q_max)
                sample = rng.uniform(lo, hi)
                near = int(np.argmin(np.linalg.norm(np.array(nodes) - sample, axis=1)))
            d = sample - nodes[near]
            scale = min(1.0, ST_STEP / max(np.abs(d).max(), 1e-12))
            while scale > 0.05 and duration(nodes[near], nodes[near] + scale * d) > ST_MAX_MOVE:
                scale *= 0.7
            q_new = nodes[near] + scale * d
            t_new = times[near] + duration(nodes[near], q_new)
            if not config_free(q_new, t_new):  # cheap first
                if REJECT_LOG is not None:
                    c = scene.clearance(q_new, angle(t_new))
                    pair = c.pair if c.distance <= c.self_gap else c.self_pair
                    REJECT_LOG(k, "config", pair, min(c.distance, c.self_gap), t_new)
                continue
            if not move_free(nodes[near], q_new, times[near]):
                if REJECT_LOG is not None:
                    chk = check_moves(scene, lambda x: angle(times[near] + x), model.omega,
                                      timing, np.array([nodes[near], q_new]),
                                      np.array([-1, 0]))[0][0]
                    REJECT_LOG(k, "move", tuple(chk.hits[0]) if chk.hits else ("", ""),
                               chk.clearance, times[near] + (chk.t_hit or 0.0))
                continue
            nodes.append(q_new)
            times.append(t_new)
            parents.append(near)
            if at_checkpoint(k, q_new, t_new):
                chain = [len(nodes) - 1]
                while parents[chain[-1]] >= 0:
                    chain.append(parents[chain[-1]])
                found = [(nodes[j], times[j]) for j in reversed(chain)]
                progress.append(arclength[k])
                break
            # Progress along the wire: a guided step earns the part of its target's
            # advance it covered (the step may stop short of the target).
            progress.append(progress[near] if s_new is None
                            else progress[near] + scale * (s_new - progress[near]))
            ref = goal_at(k, orient, q_new, times[-1])
            gap = np.linalg.norm(ref - q_new) if ref is not None else np.inf
            if gap < best or len(nodes) % 25 == 0:
                best = min(best, gap)
                found = try_goal(len(nodes) - 1)
        reach_progress[0] = ((max(progress) - arclength[k - 1]) / span) if span > 0 else 1.0
        return found, len(nodes)

    # Alternative orientations per checkpoint: one per tilt (the static plan's
    # tilt first, then the nearer ones), each with the roll and tilt direction
    # closest to the static plan's choice.
    grid = orientation_grid()

    def alternatives(chosen: np.ndarray) -> list[np.ndarray]:
        out = [chosen]
        tilts = sorted(set(np.round(grid[:, 1], 9)), key=lambda a: abs(a - chosen[1]))
        for tilt in tilts:
            same = grid[np.isclose(grid[:, 1], tilt)]
            diff = np.abs(np.angle(np.exp(1j * (same - chosen))))
            best = same[int(np.argmin(diff[:, 0] + diff[:, 2]))]
            if not np.allclose(best, chosen):
                out.append(best)
        return out[:ST_ALTERNATIVES + 1]

    reach_progress = [0.0]  # the last tree's furthest progress (for the log)
    start = (np.asarray(static.qs[0], float), 0.0)  # the waiting pose at t = 0
    segments = {}  # checkpoint k -> (its path from k - 1's arrival, its orientation)
    tried = {k: [] for k in range(1, n)}  # arrivals (q, t) found so far at each checkpoint
    k = 1
    while k < n:
        if time.monotonic() - t_wall > ST_TOTAL_LIMIT:
            result.reason = f"time limit ({ST_TOTAL_LIMIT:.0f} s) at checkpoint {k}"
            result.seconds = time.monotonic() - t_wall
            return result
        if len(tried[k]) >= ST_ARRIVALS:  # exhausted here: back further
            tried[k] = []
            k -= 1
            if k == 0:
                break
            continue
        q0, t0 = segments[k - 1][0][-1] if k > 1 else start
        orient_from = segments[k - 1][1] if k > 1 else static.orientations[0]
        order = alternatives(static.orientations[k])
        if tried[k]:  # a repeat search: fewer orientations
            order = order[:ST_RETRY_ORIENTATIONS]
        found = None
        for orient in order:
            segment, nodes = reach(k, orient, q0, t0, orient_from)
            new = segment is not None and all(
                abs(segment[-1][1] - a[1]) > ST_DIFFERENT[0]
                or np.abs(segment[-1][0] - a[0]).max() > ST_DIFFERENT[1]
                for a in tried[k] if a is not None)
            if PASS_LOG is not None:
                what = ("NOT reached" if segment is None else
                        f"reached at t {segment[-1][1]:.2f} s" + ("" if new else " (same as before)"))
                PASS_LOG(f"  checkpoint {k} try {len(tried[k]) + 1} (tilt "
                         f"{np.rad2deg(orient[1]):.0f} deg) from t {t0:.2f} s: {what}, {nodes} "
                         f"nodes, furthest {100 * reach_progress[0]:.0f} % of the way")
            if new:
                found = (segment, orient)
                break
        if found is None:
            tried[k] = [None] * ST_ARRIVALS  # nothing new here: back
            continue
        tried[k].append(found[0][-1])
        segments[k] = found
        k += 1
        if k < n:
            tried[k] = []  # a new arrival before it: its searches start afresh
    if k < n:
        result.reason = "the tree cannot get round the puzzle (backtracked to the start)"
        result.seconds = time.monotonic() - t_wall
        return result
    waypoints = [start]
    for k in range(1, n):
        waypoints += segments[k][0][1:]

    # 3. Shortcuts over the whole path (times are recomputed), then the final check.
    qs = [q for q, _ in waypoints]
    qs = rrt_mod.shortcut(qs, duration, move_free, rng=rng)
    qs = np.array(qs)
    result.feasible, result.qs = True, qs
    result.orientations = static.orientations
    result.moves, result.arrivals = check_moves(scene, angle, model.omega, timing, qs,
                                                np.zeros(len(qs), int))
    result.plan = Plan(
        feasible=result.collision_free,
        reason="" if result.collision_free else "collision along the final path",
        speed=float(getattr(timing, "speed", 0.0)), direction=model.direction,
        theta_start=theta_start, q_start=qs[0],
        steps=[PlanStep(index=i, q=qs[i], roll=0.0, t_rel=float(result.arrivals[i]),
                        margin=result.moves[i - 1].clearance if i else np.inf)
               for i in range(len(qs))],
        min_margin=result.min_clearance)
    result.seconds = time.monotonic() - t_wall
    return result

