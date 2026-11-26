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
ANALYTIC_RING = False  # the ring against the bar in closed form (soft_collision; 9.12)
PER_PART_STEP = False  # each part's gap at its own speed (see _step): sound, but no gain (the hoop is the closest and the fastest part; 9.11)
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


def _step(t: float, end: float, allow, levers: np.ndarray, moves, rates,
          task_speed) -> float:
    """Longest time step from t in which nothing moves further than allow [s].

    allow / levers / task_speed: one bound (a float, levers (6,)), or one per part
    (allows (P,), levers (P, 6), task speeds (P,)): each part may close in only by its
    own gap, at its own speed (the per-part bound: a far, fast part does not shorten
    the step of a near, slow one)."""
    dt = end - t
    allow = np.atleast_1d(np.asarray(allow, float))
    levers = np.atleast_2d(levers)
    task_speed = np.broadcast_to(np.asarray(task_speed, float), allow.shape)
    for _ in range(4):  # the moves overlapping [t, t + dt] decide the speed
        active = [r for m, r in zip(moves, rates) if m.t0 < t + dt and m.t_end > t]
        rate = np.max(active, axis=0) if active else np.zeros(6)
        speed = SLACK * (levers @ rate) + task_speed
        limit = dt
        moving = speed > 0
        if moving.any():
            limit = float(np.min(allow[moving] / speed[moving]))
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
            if PER_PART_STEP and c.part_gaps is not None:
                # Per part: each part's gap at its own speed (the task turns: every part
                # vs the task gets the task's speed), plus the arm against itself with
                # the whole arm's bound (both parts of a self pair move).
                gaps = c.part_gaps
                near = np.isfinite(gaps)
                L = scene.part_lever_arms(q)[near]
                allows = np.append(gaps[near], c.self_gap / 2)
                levers = np.vstack([L, scene.lever_arms(q)])
                tasks = np.append(np.full(int(near.sum()), task_speed), 0.0)
                if swept:  # the swept volume's distance is not per part: the whole arm
                    allows = np.append(allows, c.distance)
                    levers = np.vstack([levers, scene.lever_arms(q)])
                    tasks = np.append(tasks, task_speed)
                dt_step = _step(t, end, allows, levers, moves, rates, tasks)
            else:
                dt_step = _step(t, end, allow, scene.lever_arms(q), moves, rates, task_speed)
            t = min(t + dt_step, end)
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


# --- The approach: rings for the turning task (user design, 2026-11-26) ---
#
# The approach (q_home -> the waiting pose at checkpoint 0) must stay clear of the
# task whatever its angle. The cell walls and the camera bar are out of reach in any
# configuration and the move from q_home to the task does not come near the table, so
# only the task counts: each of its parts (turntable, base with the puzzle's foot,
# stem, vertical run + jog,
# bend, top bar) sweeps a ring about the turntable axis in a full turn, with an inner
# and an outer radius and a height range. A ring does not depend on the angle, and
# the distance of an arm box from it is a few vectorised numbers (the old check took
# the full soft model plus every task box swept, 20-30 s for an approach). Checkpoint
# 0 can lie inside a ring (e.g. the top bar's), so the approach is home -> a
# pre-pre-entry pose outside the rings (checked against the rings) -> checkpoint 0 (a
# few cm, checked with the full swept volume as before). The arm against itself is
# checked every APPROACH_SELF_STEP.

APPROACH_RING_GROUPS = (
    ("turntable", ()), ("base", ("base", "puzzle_foot")),  # the foot sits in the base
    ("stem", ("puzzle_stem",)),
    ("vertical run and jog", ("puzzle_vertical_run", "puzzle_jog")),
    ("bend", ("puzzle_bend",)), ("top bar", ("puzzle_top_bar",)))
PRE_PRE_DIRECTIONS = ("toward the robot", "outward", "up", "outward and up", "back",
                      "back and up", "toward the robot and up")
APPROACH_MIN_GAP = 0.003  # m, the approach need not squeeze: a smaller ring gap is blocked
APPROACH_LAST_MAX = np.deg2rad(40.0)  # rad, longest last move (pre-pre-entry -> waiting pose)
# The pre-pre-entry pose fixes only the ring's position (a via point): orientations
# around the waiting pose's own, roll about the ring axis x tilt about two horizontals.
PRE_PRE_ROLLS = np.deg2rad(np.arange(0.0, 360.0, 30.0))
PRE_PRE_TILTS = np.deg2rad([-25.0, 0.0, 25.0])
PRE_PRE_DISTANCES = (0.05, 0.10, 0.15)  # m
APPROACH_SELF_STEP = np.deg2rad(5.0)  # rad, self-collision samples along an approach move
_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def _corners(boxes) -> np.ndarray:
    """(n, 8, 3) corners of soft boxes."""
    return boxes.c[:, None] + np.einsum("nij,nkj->nki", boxes.R, _SIGNS[None] * boxes.h[:, None])


class ApproachRings:
    """The rings the task's parts sweep about the turntable axis (see above)."""

    def __init__(self, scene: SoftScene) -> None:
        self.scene = scene
        axis = np.asarray(scene.model.axis_xy, float)
        self.axis = axis
        obs = scene.obstacle_boxes(0.0)
        n = len(scene._obs_leaves)
        names = [str(x) for x in scene.obs.name[:n]]
        corners_obs = _corners(type(obs)(obs.c[:n], obs.R[:n], obs.h[:n]))
        corners_tt = _corners(scene._tt_boxes)
        self.names, rows = [], []
        for name, prefixes in APPROACH_RING_GROUPS:
            if not prefixes:
                cs = corners_tt
            else:
                idx = [k for k, nm in enumerate(names) if nm.startswith(prefixes)]
                cs = corners_obs[idx]
            r = np.linalg.norm(cs[..., :2] - axis, axis=2)  # (boxes, 8)
            centre = cs.mean(axis=1)[:, :2]
            rho = np.linalg.norm(cs[..., :2] - centre[:, None], axis=2).max(axis=1)
            r_in = max(0.0, float((np.linalg.norm(centre - axis, axis=1) - rho).min()))
            rows.append((r_in, float(r.max()), float(cs[..., 2].min()), float(cs[..., 2].max())))
            self.names.append(name)
        self.r_in, self.r_out, self.z0, self.z1 = (np.array(c) for c in zip(*rows))
        self.leaves = scene._arm_leaves
        self.margin = scene.margin

    def clearance(self, q: np.ndarray) -> tuple[float, str]:
        """Least gap [m] (margin subtracted, a lower bound) of the arm to the rings and
        the ring; <= 0: too close."""
        arm = self.scene.arm_boxes(q)
        sub = type(arm)(arm.c[self.leaves], arm.R[self.leaves], arm.h[self.leaves])
        cs = _corners(sub)  # (n, 8, 3)
        centre = sub.c[:, :2]
        rho = np.linalg.norm(cs[..., :2] - centre[:, None], axis=2).max(axis=1)
        dc = np.linalg.norm(centre - self.axis, axis=1)
        d_lo = np.maximum(0.0, dc - rho)[:, None]  # nearest a box can come to the axis
        d_hi = np.linalg.norm(cs[..., :2] - self.axis, axis=2).max(axis=1)[:, None]
        z_lo, z_hi = cs[..., 2].min(axis=1)[:, None], cs[..., 2].max(axis=1)[:, None]
        radial = np.maximum(self.r_in[None] - d_hi, d_lo - self.r_out[None])
        vertical = np.maximum(self.z0[None] - z_hi, z_lo - self.z1[None])
        gaps = np.maximum(radial, vertical) - self.margin  # (boxes, rings)
        a, b = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
        return float(gaps[a, b]), self.names[b]

    def move_free(self, a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
        """Whether the straight joint move a -> b stays clear of the rings (continuous:
        the next sample is no further than the gap allows), and its least gap."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        dq = np.abs(b - a)
        u, least = 0.0, np.inf
        while True:
            q = a + min(u, 1.0) * (b - a)
            gap, _ = self.clearance(q)
            least = min(least, gap)
            if gap <= APPROACH_MIN_GAP:
                return False, least
            if u >= 1.0:
                return True, least
            reach = float(self.scene.lever_arms(q) @ dq)  # metres per unit of u
            u += gap / (SLACK * max(reach, 1e-9))

    def self_free(self, a: np.ndarray, b: np.ndarray, theta: float = 0.0) -> bool:
        """The arm against itself along a -> b, every APPROACH_SELF_STEP."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.abs(b - a).max() / APPROACH_SELF_STEP)))
        return all(self.scene.clearance(a + (b - a) * (k / n), theta).self_gap > 0
                   for k in range(n + 1))


def plan_approach(kin, scene: SoftScene, model: TurntableModel, theta: float, timing,
                  q_from: np.ndarray, q0: np.ndarray, entry: float
                  ) -> tuple[list[np.ndarray] | None, str]:
    """Moves from q_from to checkpoint 0's configuration q0, clear of the turning task
    (see above: rings for the long move, the full swept volume for the last few cm).

    Tried in order: one direct joint move (against the rings); then through a
    pre-pre-entry pose: the ring as at checkpoint 0 moved outward from the turntable
    axis, up, outward and up, back along the wire, or back and up (PRE_PRE_DISTANCES),
    the IK solution closest to q0, outside the rings; home -> it checked against the
    rings, it -> q0 against the full swept volume.

    Returns:
        The configurations from q_from to q0 (None if nothing works) and how.
    """
    rings = getattr(scene, "_approach_rings", None)
    if rings is None:
        rings = scene._approach_rings = ApproachRings(scene)

    def swept_free(qs: list[np.ndarray]) -> float | None:
        """Least clearance of the moves through qs (scene and swept volume), None if any hits."""
        checks = check_moves(scene, lambda t: theta, 0.0, timing, np.array(qs),
                             np.zeros(len(qs), int), swept=True)[0]
        return None if any(c.hits for c in checks) else min(c.clearance for c in checks)

    q_from, q0 = np.asarray(q_from, float), np.asarray(q0, float)
    ok, least = rings.move_free(q_from, q0)
    if ok and rings.self_free(q_from, q0, theta):
        return [q_from, q0], f"one direct joint move, ring gap {1000 * least:.1f} mm"
    ring = kin.fk(q0) @ ring_in_flange()
    p = ring[:3, 3]
    outward = np.append(p[:2] - rings.axis, 0.0)
    outward = outward / np.linalg.norm(outward) if np.linalg.norm(outward) > 1e-6 else None
    back = -model.checkpoint_T(wire_path(0, entry), 0, theta)[:3, 2]  # against the travel
    up = np.array([0.0, 0.0, 1.0])
    directions = {"up": up, "back": back, "back and up": back + up}
    if outward is not None:
        directions.update({"outward": outward, "outward and up": outward + up})
    to_robot = np.append(-p[:2], 0.0)
    if np.linalg.norm(to_robot) > 1e-6:
        to_robot /= np.linalg.norm(to_robot)
        directions.update({"toward the robot": to_robot, "toward the robot and up": to_robot + up})
    flange_of_ring = np.linalg.inv(ring_in_flange())

    def rot(axis, a):
        c, s_ = np.cos(a), np.sin(a)
        x, y, z = axis
        return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s_, x * z * (1 - c) + y * s_],
                         [y * x * (1 - c) + z * s_, c + y * y * (1 - c), y * z * (1 - c) - x * s_],
                         [z * x * (1 - c) - y * s_, z * y * (1 - c) + x * s_, c + z * z * (1 - c)]])

    def ik_position(position):
        """In-limits IK closest to q0 putting the ring centre at position, the ring's
        orientation free around the waiting pose's (see PRE_PRE_ROLLS / _TILTS)."""
        best, best_d = None, np.inf
        R0 = ring[:3, :3]
        for ta in PRE_PRE_TILTS:
            for tb in PRE_PRE_TILTS:
                Rt = rot((1.0, 0.0, 0.0), ta) @ rot((0.0, 1.0, 0.0), tb)
                for roll in PRE_PRE_ROLLS:
                    T = np.eye(4)
                    T[:3, :3] = Rt @ R0 @ rot((0.0, 0.0, 1.0), roll)
                    T[:3, 3] = position
                    for q in kin.ik(T @ flange_of_ring):
                        if kin.in_limits(q):
                            d = float(np.abs(q - q0).max())
                            if d < best_d:
                                best, best_d = q, d
        return best

    # All candidates first, then the cheap checks, then the full check of the last move
    # in order of its length in joint space (a pose with a turned hoop can be far from
    # the waiting pose: its "last few cm" are then a long, slow, usually failing check).
    candidates = []
    for name in PRE_PRE_DIRECTIONS:
        if name not in directions:
            continue
        direction = directions[name] / np.linalg.norm(directions[name])
        for distance in PRE_PRE_DISTANCES:
            q_pre = ik_position(p + distance * direction)
            if q_pre is None or np.abs(q_pre - q0).max() > APPROACH_LAST_MAX:
                continue
            if rings.clearance(q_pre)[0] <= APPROACH_MIN_GAP:
                continue  # not outside the rings
            candidates.append((float(np.abs(q_pre - q0).max()), name, distance, q_pre))
    for _, name, distance, q_pre in sorted(candidates, key=lambda c: c[0]):
        ok, least = rings.move_free(q_from, q_pre)
        if not ok or not rings.self_free(q_from, q_pre, theta):
            continue
        last = swept_free([q_pre, q0])  # the last move: the full check
        if last is not None:
            return ([q_from, q_pre, q0], f"through a pre-pre-entry pose {100 * distance:.0f}"
                    f" cm {name}: ring gap {1000 * least:.1f} mm, then {1000 * last:.1f} mm "
                    f"soft clearance into the waiting pose")
    return None, ("no direct move or pre-pre-entry pose (toward the robot / outward / up / "
                  "back, 5-15 cm) clears the task's rings")


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
                  progress: Callable[[float, TierResult], None] | None = None,
                  workers: int | None = None, approach_from: np.ndarray | None = None,
                  deadline: float | None = None,
                  rank: Callable[[float, TierResult], tuple] | None = None,
                  margins=None,
                  accept: Callable[[float, TierResult], bool] | None = None,
                  window: Callable[[float, TierResult], float] | None = None
                  ) -> tuple[float | None, TierResult | None]:
    """Plans the traversal of a spinning puzzle at its start angle.

    Args:
        model: The (fitted) turntable model; its omega and direction are used.
        candidates: Spinning: END angles to try in order (default end_angles(model):
            plan_backward, the start angle follows). Static: start angles (default
            the first of start_angles(model)).
        bend_points: Checkpoints on the bend (plan_rrt: RRT between checkpoints).
        progress: Called with (start angle, the last tier's result) per candidate.
        workers: Spinning: end angles planned at the same time, one process each
            (the first collision-free plan wins; 1 = one after another; default
            BW_WORKERS).
        approach_from: Spinning: if given (e.g. q_home), a plan counts only if
            plan_approach also finds the approach from there to its waiting pose
            (result.approach, result.approach_how); else the next end angle.
        deadline: Spinning, parallel: time [s, time.monotonic()] by which the
            choice is made (the workers are stopped then); also clamps the
            BW_SCORE_WINDOW after the first usable plan.
        accept: Spinning, parallel: whether a usable plan counts (e.g. it ends in
            time); one that does not is ignored.
        window: Spinning, parallel: how long [s] to wait for other plans after the
            first accepted one (default BW_SCORE_WINDOW), clamped by deadline.
        margins: solver.uncertainty.Margins (the hoop-puzzle and joint-limit
            margins from the run's uncertainties); default the fixed ones.
        rank: Spinning, parallel: sort key (largest best) of (start angle, result)
            for choosing between the usable plans; default _plan_rank (distance
            from the joint limits and singularities).

    Returns:
        The start angle and the collision-free result, or (None, None).
    """
    scene = _margin_scene(model, margins)
    joint_margin = margins.joint_limit if margins is not None else 0.0
    if model.omega:  # backwards in time from the end angles; the start angle falls out
        angles = end_angles(model) if candidates is None else list(candidates)
        workers = BW_WORKERS if workers is None else workers
        if workers > 1:
            return _plan_backward_parallel(timing, model, angles, bend_points, workers,
                                           progress, approach_from, deadline, rank, margins,
                                           accept, window)
        for theta_end in angles:
            theta, result = plan_backward(kin, timing, model, theta_end,
                                          bend_points=bend_points, scene=scene,
                                          joint_margin=joint_margin)
            _with_approach(kin, timing, model, theta, result, approach_from, scene)
            if progress is not None:
                progress(theta_end, result)
            if _usable(theta, result, approach_from):
                return theta, result
        return None, None
    for theta in (start_angles(model, count=1) if candidates is None else candidates):
        result = plan_rrt(kin, timing, model, theta, bend_points=bend_points, scene=scene,
                          q_start=q_start)
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
ST_GOAL_BIAS = 0.5
ST_BOX = np.deg2rad(40.0)  # rad, sampling box beyond the node and the goal
ST_MAX_NODES = 600  # per checkpoint
ST_TIME_LIMIT = 6.0  # s, per tree search (one orientation at one checkpoint)
ST_TOTAL_LIMIT = 150.0  # s, the whole space-time planning
ST_ARRIVALS = 3  # different arrivals tried at a checkpoint before backtracking further
ST_RETRY_ORIENTATIONS = 3  # orientations tried when searching a checkpoint again
ST_DIFFERENT = (0.3, np.deg2rad(10.0))  # s, rad: an arrival this different is a new one
ST_ALTERNATIVES = 5  # other ring orientations tried at a checkpoint that cannot be reached
ST_MAX_GOAL_JUMP = np.deg2rad(60.0)  # rad, a goal further from the node is another branch
ST_WIRE_SHARE = 0.4  # share of the extensions guided along the wire (after the goal bias)
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



# --- The spinning puzzle backwards in time (user design, 2026-11-25) ---
#
# The robot can wait only at the start (checkpoint 0), so the free end of the
# problem is the start, and the hard part (the end of the bend, the vertical
# run, the stem: the turning top bar meets the forearm there) is at the end.
# So the tree grows BACKWARDS in time: the root is the ring in the stem at
# t = 0 with the puzzle at an end angle chosen so the top bar is out of the
# arm's way; a new node is EARLIER than its parent by the move's duration (the
# move new -> parent is checked in forward time against the turning puzzle);
# the checkpoints are subgoals in reverse, down to checkpoint 0, which must be
# a waiting pose (clear of the swept volume). The puzzle's angle there IS the
# start angle (the robot waits for it). Moves are time-symmetric, so the same
# timing model and checks apply. Otherwise as plan_spacetime: wire-guided
# sampling (walking back along the wire), position-only checkpoints with free
# roll, moves <= ST_MAX_MOVE, backtracking.

BW_END_ANGLES = 6  # end angles tried (the top bar away from the robot first, 30 deg apart)
BW_ROOTS = 3  # final configurations tried per end angle
BW_TOTAL_LIMIT = 120.0  # s, per end angle
BW_TILT = np.deg2rad(20.0)  # rad, the default ring tilt at a checkpoint (any roll)
BISECT_DEPTH = 8  # binary subdivision levels at most (a safety cap; 0: the tree only)
BISECT_MIN_SPACING = 0.010  # m of wire: a piece shorter than this is not split again
# (user: a smallest distance between the stops instead of a fixed depth; a 10 mm piece
# of the R50 bend is within s^2 / 8R = 0.25 mm of its chord)
BISECT_LOG = None  # hook: called with a line per subdivision failure (diagnostics)
BISECT_WIDE_TILT = np.deg2rad(50.0)  # rad, the grid fallback's largest tilt (every direction; as TILTS)
BISECT_IK_TRIES = 12  # nearest IK solutions checked for collisions at a stop
BISECT_START_CLEARANCE = 0.005  # m: start poses with this much clearance are preferred
BISECT_DISTINCT = np.deg2rad(5.0)  # rad: candidates closer than this are the same branch
BISECT_ALTERNATIVES = ST_ARRIVALS  # start candidates tried per checkpoint (the retries)
BW_TREE = False  # the backward tree as a fallback (off: the subdivision only)
BISECT_WIRE_STEP = 0.002  # m, the wire sampled for the perpendicular projection
BISECT_END_GAP = 0.005  # m, a projected stop keeps this far from the stretch's ends
BW_END_CHECKED = 40  # final configurations (most clearance first) checked for waiting at the end
BW_END_STEP = np.deg2rad(5.0)  # rad, angle step of that check (a full turn)


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def se3_log(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The twist (w, v) of a rigid transform: T = exp([w, v])."""
    R, p = T[:3, :3], T[:3, 3]
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3), p.copy()
    if np.pi - th < 1e-6:  # half a turn: the axis from R + I
        k = int(np.argmax(np.diag(R)))
        axis = (R[:, k] + np.eye(3)[:, k]) / np.sqrt(2.0 * (1.0 + R[k, k]))
        w = th * axis
    else:
        w = th / (2.0 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                                                 R[1, 0] - R[0, 1]])
    W = _skew(w)
    V_inv = np.eye(3) - 0.5 * W + (1.0 / th ** 2) * (
        1.0 - th * np.sin(th) / (2.0 * (1.0 - np.cos(th)))) * (W @ W)
    return w, V_inv @ p


def se3_exp(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The rigid transform of the twist (w, v)."""
    th = float(np.linalg.norm(w))
    W = _skew(w)
    T = np.eye(4)
    if th < 1e-9:
        T[:3, :3] = np.eye(3) + W
        T[:3, 3] = v
        return T
    A, B = np.sin(th) / th, (1.0 - np.cos(th)) / th ** 2
    T[:3, :3] = np.eye(3) + A * W + B * (W @ W)
    T[:3, 3] = (np.eye(3) + B * W + (th - np.sin(th)) / th ** 3 * (W @ W)) @ v
    return T


def se3_interpolate(Ta: np.ndarray, Tb: np.ndarray, u: float) -> np.ndarray:
    """The pose a fraction u along the geodesic from Ta to Tb: Ta exp(u log(Ta^-1 Tb))."""
    w, v = se3_log(np.linalg.inv(Ta) @ Tb)
    return Ta @ se3_exp(u * w, u * v)


def end_angles(model: TurntableModel, count: int = BW_END_ANGLES,
               step: float = START_STEP) -> list[float]:
    """End angles to try: the top bar pointing away from the robot first, then alternately
    later and earlier ones, `step` apart."""
    away = facing_angle(model) + np.pi
    out = [away] + [away + (k + 1) // 2 * step * (1 if k % 2 else -1) for k in range(1, count)]
    return [float(np.arctan2(np.sin(a), np.cos(a))) for a in out]


def plan_backward(kin, timing, model: TurntableModel, theta_end: float,
                  bend_points: int = 3, scene: SoftScene | None = None, face: int = -1,
                  seed: int = 0, joint_margin: float = 0.0) -> tuple[float | None, TierResult]:
    """Plans the spinning traversal backwards in time from the end angle (see above).

    Returns:
        The start angle (the puzzle's angle when the robot leaves checkpoint 0) and the
        result (qs, arrivals from t = 0 at checkpoint 0, plan); (None, result) if none.

    joint_margin: every configuration keeps this from the joint limits [rad]
    (solver.uncertainty: room for the executor's corrections). The IK filters and
    the sampling box are the limits shrunk by it; a box is convex, so every step
    between nodes stays inside too.
    """
    from solver import rrt as rrt_mod
    t_wall = time.monotonic()
    scene = scene or SoftScene(model, restricted=True, analytic_ring=ANALYTIC_RING)
    q_lo, q_hi = kin.q_min + joint_margin, kin.q_max - joint_margin

    def in_room(q) -> bool:
        return bool(np.all(q >= q_lo) and np.all(q <= q_hi))
    rng = np.random.default_rng(seed)
    grid = orientation_grid()
    to_ring = ring_in_flange()
    rolls = np.arange(-np.pi, np.pi, ROLL_STEP)

    def angle(t: float) -> float:  # t <= 0, the end at t = 0
        return theta_end + model.direction * model.omega * t

    def duration(a, b):
        return timing.move_duration(a, b)

    def config_free(q, t):
        return scene.clearance(q, angle(t)).free

    def move_free(a, b, t):  # a -> b starting at t (forward in time)
        check = check_moves(scene, lambda s: angle(t + s), model.omega, timing,
                            np.array([a, b]), np.array([-1, 0]))[0][0]
        return not check.hits

    # The wire laid out from the LARGEST entry: checkpoint 0 may be any of the entry
    # distances (the ring leaves the bar back along the straight top-bar axis; the
    # start angle, which decides where a pose can wait, is known only at the end).
    for entry in (max(ENTRIES),):
        stops = [(e, entry - e) for e in sorted(ENTRIES)]  # (entry, arclength), nearest first
        path = wire_path(0, entry, bend_points)
        n = len(path.frames)
        arclength = checkpoint_arclengths(0, entry, bend_points)
        result = TierResult(tier=0, checkpoints=n, feasible=False, face=face)
        result.entry = entry

        def ik_any_roll(frame, orient, q_near):
            _, tilt, direction = (float(x) for x in orient)
            sols = [x for roll in rolls
                    for x in kin.ik(flange_target(frame, float(roll), face, tilt, direction))
                    if in_room(x)]
            if not sols:
                return None
            q = min(sols, key=lambda x: float(np.abs(x - q_near).max()))
            return q if np.abs(q - q_near).max() <= ST_MAX_GOAL_JUMP else None

        def before(frame_at, orient, q_near, t_near):
            """Configuration on frame_at(t) at the time t the robot would leave it to reach
            (q_near, t_near): t = t_near - duration (solved by iteration)."""
            q = q_near
            for _ in range(3):
                t = t_near - duration(q, q_near)
                q = ik_any_roll(frame_at(t), orient, q_near)
                if q is None:
                    return None, None
            return q, t_near - duration(q, q_near)

        def waiting_entry(q, t):
            """The entry distance whose checkpoint 0 the ring is on, if q can wait there
            (clear of the swept volume); else None."""
            ring = kin.fk(q) @ to_ring
            P = model.puzzle_T(angle(t))
            for e, s0 in stops:
                if np.linalg.norm(ring[:3, 3] - (P @ wire_frame_at(s0, entry))[:3, 3]) \
                        <= ST_AT_CHECKPOINT:
                    return e if waits_free(scene, q, angle(t)) else None
            return None

        def at_checkpoint(k, q, t):
            if k == 0:
                return waiting_entry(q, t) is not None
            ring = kin.fk(q) @ to_ring
            P = model.puzzle_T(angle(t))
            return np.linalg.norm(ring[:3, 3] - (P @ path.frames[k])[:3, 3]) <= ST_AT_CHECKPOINT

        def orient_for(tilt):
            same = grid[np.isclose(grid[:, 1], tilt)]
            return same[0]  # roll is free; the tilt direction of the grid's first entry

        tilts = sorted(set(np.round(grid[:, 1], 9)), key=lambda a: abs(a - BW_TILT))
        orientations = [orient_for(tilt) for tilt in tilts][:ST_ALTERNATIVES + 1]

        def reach(k, orient, q_from, t_from, orient_from):
            """Tree backwards from (q_from, t_from) at checkpoint k + 1 to checkpoint k;
            the path [(q, t), ...] in forward order, or None."""
            deadline = time.monotonic() + ST_TIME_LIMIT
            nodes, times, parents = [q_from], [t_from], [-1]
            progress = [arclength[k + 1]]  # how far along the wire (decreasing)
            span = arclength[k + 1] - arclength[k]

            def chain_from(i):  # forward order: node i, its parent, ..., the root
                out = [i]
                while parents[out[-1]] >= 0:
                    out.append(parents[out[-1]])
                return [(nodes[j], times[j]) for j in out]

            def try_goal(i):
                aims = [s0 for _, s0 in stops] if k == 0 else [arclength[k]]
                for s_goal in aims:
                    goal, t_goal = before(lambda t: model.puzzle_T(angle(t))
                                          @ wire_frame_at(s_goal, entry),
                                          orient, nodes[i], times[i])
                    if goal is None or duration(goal, nodes[i]) > ST_MAX_MOVE:
                        continue
                    if not at_checkpoint(k, goal, t_goal) or not config_free(goal, t_goal):
                        continue
                    if not move_free(goal, nodes[i], t_goal):
                        continue
                    return [(goal, t_goal)] + chain_from(i)
                return None

            found = try_goal(0)
            best = np.inf
            while found is None and len(nodes) < ST_MAX_NODES and time.monotonic() < deadline:
                draw = rng.random()
                s_new = None
                if draw < ST_GOAL_BIAS + ST_WIRE_SHARE:  # guided back along the wire
                    near = (len(nodes) - 1 - int(rng.integers(min(len(nodes), ST_RECENT)))
                            if rng.random() < 0.5 else int(rng.integers(len(nodes))))
                    aimed = draw < ST_GOAL_BIAS or rng.random() < ST_AIM_CHECKPOINT
                    s_goal = stops[int(rng.integers(len(stops)))][1] if k == 0 else arclength[k]
                    s_new = s_goal if aimed else rng.uniform(arclength[k], progress[near])
                    w = (arclength[k + 1] - s_new) / span if span > 0 else 1.0
                    o = orient_from + w * np.angle(np.exp(1j * (orient - orient_from)))
                    target, _ = before(lambda t: model.puzzle_T(angle(t))
                                       @ wire_frame_at(s_new, entry), o, nodes[near], times[near])
                    if target is None:
                        continue
                    sample = target if aimed else target + rng.normal(0.0, ST_WIRE_NOISE, 6)
                else:  # uniform around the tree
                    pts = np.array(nodes)
                    lo = np.maximum(pts.min(axis=0) - ST_BOX, q_lo)
                    hi = np.minimum(pts.max(axis=0) + ST_BOX, q_hi)
                    sample = rng.uniform(lo, hi)
                    near = int(np.argmin(np.linalg.norm(pts - sample, axis=1)))
                if not in_room(sample):  # a noisy wire sample just over the margin
                    sample = np.clip(sample, q_lo, q_hi)
                d = sample - nodes[near]
                scale = min(1.0, ST_STEP / max(np.abs(d).max(), 1e-12))
                while scale > 0.05 and duration(nodes[near] + scale * d, nodes[near]) > ST_MAX_MOVE:
                    scale *= 0.7
                q_new = nodes[near] + scale * d
                t_new = times[near] - duration(q_new, nodes[near])
                if not config_free(q_new, t_new):
                    if REJECT_LOG is not None:
                        c = scene.clearance(q_new, angle(t_new))
                        pair = c.pair if c.distance <= c.self_gap else c.self_pair
                        REJECT_LOG(k, "config", pair, min(c.distance, c.self_gap), t_new)
                    continue
                if not move_free(q_new, nodes[near], t_new):
                    if REJECT_LOG is not None:
                        chk = check_moves(scene, lambda x: angle(t_new + x), model.omega, timing,
                                          np.array([q_new, nodes[near]]), np.array([-1, 0]))[0][0]
                        REJECT_LOG(k, "move", tuple(chk.hits[0]) if chk.hits else ("", ""),
                                   chk.clearance, t_new + (chk.t_hit or 0.0))
                    continue
                nodes.append(q_new)
                times.append(t_new)
                parents.append(near)
                if at_checkpoint(k, q_new, t_new):
                    found = chain_from(len(nodes) - 1)
                    progress.append(arclength[k])
                    break
                progress.append(progress[near] if s_new is None
                                else progress[near] + scale * (s_new - progress[near]))
                gap = progress[-1] - arclength[k]
                if gap < best or len(nodes) % 25 == 0:
                    best = min(best, gap)
                    found = try_goal(len(nodes) - 1)
            reach_progress[0] = ((arclength[k + 1] - min(progress)) / span) if span > 0 else 1.0
            return found, len(nodes)

        # --- Binary subdivision (user design, 2026-11-26): the straight joint move
        # between the ends of a stretch of wire; if it collides, its midpoint's ring
        # centre is projected perpendicularly onto the wire (within the stretch), a
        # stop is put there (position-only IK closest to the move's midpoint) and both
        # halves are done the same way, the later half first (backwards in time: what
        # is later stays fixed; the earlier start is re-solved at its new time).
        # Every accepted piece is one straight joint move: a polyline through the 4D
        # wire (space and time), refined only where a straight move cannot follow it.
        wire_s = np.arange(0.0, arclength[-1] + 1e-9, BISECT_WIRE_STEP)
        wire_pts = np.array([wire_frame_at(x, entry)[:3, 3] for x in wire_s])
        bisect_stats = {"pieces": 0, "fallback": 0, "early_backtracks": 0}
        bisect_fail = [None]  # why the last subdivision failed ("start": no start pose)

        def project(ring_world, t, s_lo, s_hi) -> float:
            """Arclength of the foot of the perpendicular from the ring centre onto the
            wire within (s_lo, s_hi); the middle of the stretch if the closest point is
            at an end of it (no foot inside: the subdivision would stall)."""
            local = (np.linalg.inv(model.puzzle_T(angle(t))) @ np.append(ring_world, 1.0))[:3]
            inside = (wire_s > s_lo + BISECT_END_GAP) & (wire_s < s_hi - BISECT_END_GAP)
            if not inside.any():
                return 0.5 * (s_lo + s_hi)
            idx = np.flatnonzero(inside)
            best = idx[int(np.argmin(np.linalg.norm(wire_pts[idx] - local, axis=1)))]
            if best in (idx[0], idx[-1]):
                return 0.5 * (s_lo + s_hi)
            return float(wire_s[best])

        flange_of_ring = np.linalg.inv(to_ring)
        roll_R = [np.array([[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0],
                            [0.0, 0.0, 1.0]]) for r in rolls]
        wide = [o for o in grid if o[1] <= BISECT_WIDE_TILT + 1e-9
                and np.isclose(o[0], grid[0, 0])]  # one roll each (the roll is looped)

        def ring_in_puzzle(q, t) -> np.ndarray:
            return np.linalg.inv(model.puzzle_T(angle(t))) @ kin.fk(q) @ to_ring

        def carried(q_next, t_next, s_next, s_point) -> np.ndarray:
            """q_next's ring orientation (puzzle frame) carried along the wire to s_point
            (turned with the wire's tangent)."""
            R_next = ring_in_puzzle(q_next, t_next)[:3, :3]
            F_next = wire_frame_at(s_next, entry)[:3, :3]
            F_here = wire_frame_at(s_point, entry)[:3, :3]
            return F_here @ F_next.T @ R_next

        def solve_at(s_point, q_next, t_next, q_ref, start_of=None, preferred=(), alt=0):
            """A configuration with the ring on the wire at s_point, timed so it reaches
            (q_next, t_next); position only, in the joint room, collision free at its
            time, the IK closest to q_ref. Orientations: the preferred ones (rotations in
            the puzzle frame, the roll about the ring axis free), then the grid up to
            BISECT_WIDE_TILT in every tilt direction. With start_of = 0: checkpoint 0
            (any entry distance, a waiting pose).

            The candidates (the nearest BISECT_IK_TRIES free solutions per place, distinct
            branches) are ranked: those with at least BISECT_START_CLEARANCE first, the
            closest to q_ref first; then the rest, the most clearance first (a pose
            nearly touching makes every move from it touch at once: seed 1's jog). alt
            picks the alt-th (0: the best; the retries take the next ones, instead of the
            tree)."""
            points = [s0 for _, s0 in stops] if start_of == 0 else [s_point]
            ranked = []
            # Cheap first (the profile: 4000 IK calls per pose, 78 % of the time): the
            # preferred orientations, the wide grid only if they give too few; the
            # candidates generated once at the first time estimate, the time's fixed
            # point refined only for the candidates used, each with its own orientation.
            t_guess = t_next - duration(q_ref, q_next)
            for sp in points:
                p_local = wire_frame_at(sp, entry)[:3, 3]
                F_local = wire_frame_at(sp, entry)

                def target(spec, t):
                    P = model.puzzle_T(angle(t))
                    if spec[0] == "R":
                        T = np.eye(4)
                        T[:3, :3], T[:3, 3] = spec[1], p_local
                        return P @ T @ flange_of_ring
                    _, roll, tilt, direction = spec
                    return flange_target(P @ F_local, roll, face, tilt, direction)

                groups = ([("R", R @ Rr) for R in preferred for Rr in roll_R],
                          [("grid", float(r), float(o[1]), float(o[2]))
                           for o in wide for r in rolls])
                for group in groups:
                    if not group:
                        continue
                    cands = [(x, spec) for spec in group
                             for x in kin.ik(target(spec, t_guess)) if in_room(x)]
                    cands.sort(key=lambda c: float(np.abs(c[0] - q_ref).max()))
                    here = []
                    for x, spec in cands:
                        if len(here) >= BISECT_IK_TRIES:
                            break
                        if any(np.abs(x - y).max() < BISECT_DISTINCT for y, *_ in here):
                            continue  # the same branch as one already kept
                        for _ in range(2):  # its time depends on its move to q_next
                            tx = t_next - duration(x, q_next)
                            sols = [y for y in kin.ik(target(spec, tx)) if in_room(y)]
                            if not sols:
                                x = None
                                break
                            x = min(sols, key=lambda y: float(np.abs(y - x).max()))
                        if x is None:
                            continue
                        tx = t_next - duration(x, q_next)
                        c = scene.clearance(x, angle(tx))
                        if not c.free:
                            continue
                        if start_of == 0 and waiting_entry(x, tx) is None:
                            continue
                        here.append((x, tx, float(np.abs(x - q_ref).max()),
                                     min(c.distance, c.self_gap)))
                    roomy = sorted((h for h in here if h[3] >= BISECT_START_CLEARANCE),
                                   key=lambda h: h[2])
                    tight = sorted((h for h in here if h[3] < BISECT_START_CLEARANCE),
                                   key=lambda h: -h[3])
                    ranked += roomy + tight
                    if len(ranked) > alt:
                        break  # enough from the preferred ones: the grid is not needed
                if len(ranked) > alt:
                    break  # the nearest entry that works (checkpoint 0)
            if alt < len(ranked):
                return ranked[alt][0], ranked[alt][1]
            return None

        def bisect(s_a, q_next, t_next, s_next, depth, k_start=None, q_ref=None,
                   preferred=(), alt=0):
            """Forward path [(q, t), ...] from the ring at s_a to (q_next, t_next), or None.
            The start's IK is the one closest to q_ref (default q_next); its orientation:
            the preferred ones, then q_next's carried along the wire, then the grid."""
            start = solve_at(s_a, q_next, t_next, q_next if q_ref is None else q_ref,
                             start_of=k_start,
                             preferred=list(preferred) + [carried(q_next, t_next, s_next, s_a)],
                             alt=alt if depth == BISECT_DEPTH else 0)
            if start is None:
                if depth == BISECT_DEPTH:  # no pose at the stretch's start at all
                    bisect_fail[0] = "start"
                if BISECT_LOG is not None:
                    BISECT_LOG(f"{'  ' * (BISECT_DEPTH - depth)}no free IK on the wire at "
                               f"s {1000 * s_a:.0f} mm (stretch to {1000 * s_next:.0f} mm)")
                return None
            q_a, t_a = start
            if move_free(q_a, q_next, t_a):
                bisect_stats["pieces"] += 1
                return [(q_a, t_a), (q_next, t_next)]
            if BISECT_LOG is not None:
                chk = check_moves(scene, lambda x: angle(t_a + x), model.omega, timing,
                                  np.array([q_a, q_next]), np.array([-1, 0]))[0][0]
                BISECT_LOG(f"{'  ' * (BISECT_DEPTH - depth)}straight s {1000 * s_a:.0f} -> "
                           f"{1000 * s_next:.0f} mm collides: {chk.hits[0] if chk.hits else '?'}"
                           f" at {chk.t_hit or 0:.2f} s of {duration(q_a, q_next):.2f} s")
            if depth == 0 or s_next - s_a < 2.0 * BISECT_MIN_SPACING:
                return None  # the halves would be shorter than the smallest spacing
            # The midpoint in task space (user): the ring's pose relative to the puzzle
            # halfway along the geodesic Ta exp(1/2 log(Ta^-1 Tb)); its position
            # projected perpendicularly onto the wire, its orientation tried first.
            q_mid = 0.5 * (q_a + q_next)
            t_mid = t_a + 0.5 * duration(q_a, q_next)
            L_mid = se3_interpolate(ring_in_puzzle(q_a, t_a), ring_in_puzzle(q_next, t_next),
                                    0.5)
            ring_world = (model.puzzle_T(angle(t_mid)) @ L_mid)[:3, 3]
            s_m = project(ring_world, t_mid, s_a, s_next)
            later = bisect(s_m, q_next, t_next, s_next, depth - 1, q_ref=q_mid,
                           preferred=[L_mid[:3, :3]])
            if later is None:
                return None
            q_m, t_m = later[0]
            earlier = bisect(s_a, q_m, t_m, s_m, depth - 1, k_start, q_ref=q_a)
            if earlier is None:
                return None
            return earlier + later[1:]

        # The roots: free final configurations with the ring low enough, most clearance first.
        final = model.checkpoint_T(path, n - 1, angle(0.0))
        roots = []
        for o in grid:
            for q in kin.ik(flange_target(final, float(o[0]), face, float(o[1]), float(o[2]))):
                if not in_room(q):
                    continue
                ring = kin.fk(q) @ to_ring
                if ring_top_above_pad(np.linalg.inv(model.puzzle_T(angle(0.0))) @ ring) \
                        >= SUCCESS_HEIGHT:
                    continue
                c = scene.clearance(q, angle(0.0))
                if c.free:
                    roots.append((c.distance, q, o))
        roots.sort(key=lambda r: -r[0])

        # At the end the robot stays put while the puzzle keeps turning: prefer final
        # configurations that stay free for a whole turn (the user: if none exists, plan
        # anyway and warn; the puzzle is hollow plastic on foam, the hoop magnetic).
        def stays_free(q) -> bool:
            return all(scene.clearance(q, angle(0.0) + model.direction * a).free
                       for a in np.arange(BW_END_STEP, 2 * np.pi, BW_END_STEP))

        waiting = {i for i in range(min(BW_END_CHECKED, len(roots))) if stays_free(roots[i][1])}
        order = sorted(range(len(roots)), key=lambda i: (i not in waiting, i))
        roots = [roots[i] + (i in waiting,) for i in order]
        reach_progress = [0.0]
        for _, q_root, o_root, root_waits in roots[:BW_ROOTS]:
            segments = {}  # checkpoint k -> its path to checkpoint k + 1's arrival, orientation
            tried = {k: [] for k in range(n - 1)}
            k = n - 2
            arrival = {n - 1: ((q_root, 0.0), o_root)}
            while 0 <= k < n - 1:
                if time.monotonic() - t_wall > BW_TOTAL_LIMIT:
                    break
                if len(tried[k]) >= ST_ARRIVALS:
                    tried[k] = []
                    k += 1
                    if k >= n - 1:
                        break
                    continue
                (q_from, t_from), orient_from = arrival[k + 1]
                order = orientations if not tried[k] else orientations[:ST_RETRY_ORIENTATIONS]
                found = None
                skip_tree = not BW_TREE
                if BISECT_DEPTH > 0:
                    # The subdivision from the stretch's start candidates in turn (the
                    # retries after a backtrack continue with the next ones): every one
                    # gives a different arrival; deterministic, no sampling.
                    s_from = arclength[k + 1]
                    for alt in range(len(tried[k]), BISECT_ALTERNATIVES):
                        bisect_fail[0] = None
                        segment = bisect(arclength[k], q_from, t_from, s_from, BISECT_DEPTH,
                                         k_start=0 if k == 0 else None, alt=alt)
                        if PASS_LOG is not None:
                            PASS_LOG(f"  back to checkpoint {k} by subdivision (start "
                                     f"candidate {alt + 1}) from t {t_from:.2f} s: "
                                     + ("NOT found" if segment is None else
                                        f"{len(segment) - 1} straight move(s), reached at t "
                                        f"{segment[0][1]:.2f} s"))
                        if segment is not None and all(
                                abs(segment[0][1] - a[1]) > ST_DIFFERENT[0]
                                or np.abs(segment[0][0] - a[0]).max() > ST_DIFFERENT[1]
                                for a in tried[k] if a is not None):
                            found = (segment, orientations[0])
                            break
                        if bisect_fail[0] == "start":  # no further candidate at the start
                            bisect_stats["early_backtracks"] += 1
                            break
                    if found is None and BW_TREE:
                        bisect_stats["fallback"] += 1
                for orient in (order if found is None and not skip_tree else ()):
                    segment, count = reach(k, orient, q_from, t_from, orient_from)
                    new = segment is not None and all(
                        abs(segment[0][1] - a[1]) > ST_DIFFERENT[0]
                        or np.abs(segment[0][0] - a[0]).max() > ST_DIFFERENT[1]
                        for a in tried[k] if a is not None)
                    if PASS_LOG is not None:
                        what = ("NOT reached" if segment is None else
                                f"reached at t {segment[0][1]:.2f} s"
                                + ("" if new else " (same as before)"))
                        PASS_LOG(f"  back to checkpoint {k} try {len(tried[k]) + 1} (tilt "
                                 f"{np.rad2deg(orient[1]):.0f} deg) from t {t_from:.2f} s: {what}, "
                                 f"{count} nodes, furthest {100 * reach_progress[0]:.0f} % of the way")
                    if new:
                        found = (segment, orient)
                        break
                if found is None:
                    tried[k] = [None] * ST_ARRIVALS
                    continue
                tried[k].append(found[0][0])
                segments[k] = found[0]
                arrival[k] = (found[0][0], found[1])
                k -= 1
                if k >= 0:
                    tried[k] = []
            if k >= 0:
                continue  # this root failed: the next one
            # Forward order, time shifted so checkpoint 0 is at t = 0.
            waypoints = list(segments[0])
            for j in range(1, n - 1):
                waypoints += segments[j][1:]
            t0 = waypoints[0][1]
            theta_start = angle(t0)
            result.entry = waiting_entry(*waypoints[0])  # the entry the plan waits at
            result.end_waits = root_waits  # whether the final pose stays free while it turns

            def fwd(t: float) -> float:
                return theta_start + model.direction * model.omega * t

            def path_free(path) -> bool:
                moves, _ = check_moves(scene, fwd, model.omega, timing, np.array(path),
                                       np.zeros(len(path), int))
                return not any(m.hits for m in moves)

            # No global shortcut pass: the subdivision's pieces are straight moves
            # already (it rechecked the whole path per shortcut: ~30 % of the time).
            qs = np.array([q for q, _ in waypoints])
            result.bisect_stats = dict(bisect_stats)
            result.feasible, result.qs = True, qs
            result.orientations = np.array([arrival[j][1] for j in range(n)])
            result.moves, result.arrivals = check_moves(scene, fwd, model.omega, timing, qs,
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
            return theta_start, result
        if time.monotonic() - t_wall > BW_TOTAL_LIMIT:
            break
    result.reason = (f"backwards from end angle {np.rad2deg(theta_end):.0f} deg: no way back to "
                     f"a waiting pose ({len(roots)} final configurations)")
    result.seconds = time.monotonic() - t_wall
    return None, result


BW_WORKERS = BW_END_ANGLES  # processes planning end angles at the same time (~300 MB each)


# --- Choosing between finished plans: distance from singularities and joint limits ---
#
# User, 2026-11-26: when several end angles give a plan, take the one with the most
# leeway from the singularities. The executor corrects every step by numerical IK
# (the flange pose rotated with the turntable): near a singularity a small correction
# needs large joint changes, and the aborts so far were joint-limit violations. So a
# plan is scored by the worst, along its whole path (every PLAN_SCORE_STEP of joint
# motion), of (1) the smallest singular value of the flange Jacobian, its rotation rows
# scaled by PLAN_SCORE_LENGTH so they compare with metres, and (2) the smallest distance
# of any joint from its limits. Plans keeping PLAN_MIN_LIMIT_MARGIN come first; among
# them the largest (1) wins. The others get BW_SCORE_WINDOW after the first usable plan
# (clamped by BW_TOTAL_LIMIT), or until all end angles are done.

BW_SCORE_WINDOW = 15.0  # s after the first usable plan to wait for others
PLAN_SCORE_LENGTH = 0.1  # m, scale of the Jacobian's rotation rows (the hoop's lever)
PLAN_SCORE_STEP = np.deg2rad(2.0)  # rad, sampling along the path
PLAN_MIN_LIMIT_MARGIN = np.deg2rad(10.0)  # rad, joint-limit room a correction may need


def plan_quality(kin, qs: np.ndarray) -> tuple[float, float]:
    """(smallest scaled Jacobian singular value [m], smallest joint-limit margin [rad])
    over the whole path (straight joint moves between the waypoints)."""
    try:
        from plan_ik import jacobian_finite_forward_differences
    except ImportError:  # a script that did not put Natka/ on the path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Natka"))
        from plan_ik import jacobian_finite_forward_differences
    qs = np.asarray(qs, float)
    samples = [qs[0]]
    for a, b in zip(qs, qs[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / PLAN_SCORE_STEP)))
        samples += [a + (b - a) * (j / n) for j in range(1, n + 1)]
    sigma, margin = np.inf, np.inf
    for q in samples:
        J = jacobian_finite_forward_differences(kin, q).copy()
        J[3:] *= PLAN_SCORE_LENGTH
        sigma = min(sigma, float(np.linalg.svd(J, compute_uv=False)[-1]))
        margin = min(margin, float(np.min(np.minimum(q - kin.q_min, kin.q_max - q))))
    return sigma, margin


def _plan_rank(result) -> tuple:
    """Sort key (largest best): enough joint-limit room first, then the singular value."""
    sigma, margin = result.quality
    return (margin >= PLAN_MIN_LIMIT_MARGIN, sigma, margin)


def _with_approach(kin, timing, model, theta, result, approach_from, scene=None) -> None:
    """With approach_from: plans the approach to a collision-free result's waiting pose
    (result.approach, result.approach_how); a plan without one does not count
    (_usable), so another end angle is tried (user, 2026-11-26: the first plan whose
    waiting pose could not be approached made the whole placement unreachable)."""
    if approach_from is None or theta is None or not result.collision_free:
        return
    approach, how = plan_approach(kin, scene or SoftScene(model), model, theta, timing,
                                  approach_from, result.qs[0], result.entry)
    result.approach, result.approach_how = approach, how
    if approach is None:
        result.reason = f"plan found, but no approach: {how}"


def _usable(theta, result, approach_from) -> bool:
    return (theta is not None and result.collision_free
            and (approach_from is None or getattr(result, "approach", None) is not None))


def _margin_scene(model, margins) -> SoftScene:
    """The soft scene with the run's hoop-puzzle margin (solver.uncertainty)."""
    # restricted: only the pairs that can touch (validated: experiments/
    # validate_restricted_pairs.py, 0 missed collisions in 4130 poses near the task)
    if margins is None:
        return SoftScene(model, restricted=True, analytic_ring=ANALYTIC_RING)
    return SoftScene(model, hoop_puzzle_margin=margins.hoop_puzzle, restricted=True,
                     analytic_ring=ANALYTIC_RING)


def _backward_job(job):
    """One end angle in a worker process (its own kinematics; see plan_spinning)."""
    timing, model, theta_end, bend_points, seed, approach_from, margins = job
    from ctu_crs import CRS93
    kin = CRS93(tty_dev=None)
    scene = _margin_scene(model, margins)
    theta, result = plan_backward(kin, timing, model, theta_end, bend_points=bend_points,
                                  seed=seed, scene=scene,
                                  joint_margin=margins.joint_limit if margins else 0.0)
    _with_approach(kin, timing, model, theta, result, approach_from, scene)
    if _usable(theta, result, approach_from):
        result.quality = plan_quality(kin, result.qs)
    return theta_end, theta, result


def _plan_backward_parallel(timing, model, angles, bend_points, workers, progress,
                            approach_from=None, deadline=None, rank=None, margins=None,
                            accept=None, window=None):
    """plan_backward for the end angles in `workers` processes; the first usable plan
    (collision free, and with an approach if approach_from is given) wins and the
    other workers are stopped."""
    import multiprocessing
    jobs = [(timing, model, a, bend_points, i, approach_from, margins)
            for i, a in enumerate(angles)]
    import os
    pool = multiprocessing.get_context("spawn").Pool(
        max(1, min(workers, len(jobs), os.cpu_count() or 1)))
    t0 = time.monotonic()
    limit = min(t0 + BW_TOTAL_LIMIT, np.inf if deadline is None else deadline)
    usable, deadline = [], limit
    try:
        results = pool.imap_unordered(_backward_job, jobs)
        for _ in jobs:
            try:
                theta_end, theta, result = results.next(
                    timeout=max(0.0, deadline - time.monotonic()))
            except multiprocessing.TimeoutError:
                break  # the window (or the limit) is over: the others are stopped
            if progress is not None:
                progress(theta_end, result)
            result.theta_end = theta_end
            if _usable(theta, result, approach_from) and (accept is None
                                                           or accept(theta, result)):
                usable.append((theta, result))
                if len(usable) == 1:  # the first: the others get the window
                    wait = BW_SCORE_WINDOW if window is None else window(theta, result)
                    deadline = min(time.monotonic() + max(0.0, wait), limit)
        if not usable:
            return None, None
        theta, best = max(usable, key=(lambda u: rank(*u)) if rank is not None
                          else (lambda u: _plan_rank(u[1])))
        best.candidates = [(r.theta_end, r.quality) for _, r in usable]
        return theta, best
    finally:
        pool.terminate()
        pool.join()
