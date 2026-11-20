"""Task P2.8: the feasibility funnel, cheap screening before the plan search (P2.6).

Three stages, cheapest first, so that the expensive timed simulation only sees
start phases that can work and an unreachable placement is reported with a
reason:

A. Reach: is the wrist centre of each checkpoint's flange target within the
   arm's reach, for some turntable angle, hoop roll and hoop face? A checkpoint
   that fails for all of them makes the placement infeasible at once.
B. IK feasibility map: for every checkpoint x turntable angle (THETA_STEP) x
   hoop face x roll (ROLL_STEP), the analytical IK solutions within the joint
   limits (vectorized, solver.ik_batch; lab 5), each with a conditioning value
   in [0, 1]: the minimum of the wrist term |sin q5| (0 at the wrist
   singularity), the elbow term (0 with the arm fully stretched) and the joint
   limit term (0 at a limit). Computed once per placement.
C. Start-phase screening: for every start phase theta0 (THETA_STEP) and spin
   direction, the checkpoints are walked in order with nominal timing: a
   checkpoint's solutions connect to the previous checkpoint's if no joint moves
   more than MAX_STEP_CHANGE (a larger jump means a branch switch). A state is
   (checkpoint, grid angle, solution) with its own nominal arrival time
   (solver.timing); a transition counts only if the arrival time it gives falls
   on the next solution's grid angle, so every path is consistent to half a
   grid step. Per checkpoint the best MAX_STATES states are kept (conditioning
   first, then arrival time). Per solution the best bottleneck conditioning along
   the way is kept (max-min, like the plan's objective). theta0 survives if the
   last checkpoint is reached.

Target convention (contracts): the ring centre on the checkpoint, the ring's
normal (the hoop's local z) along the checkpoint z (face +1) or against it
(face -1), rotated by the roll about the ring axis from the checkpoint x. At
roll 0 and face +1 the hoop's handle (towards the mount) points along the
checkpoint +y. Clearance does not depend on roll or face (the ring is round).

Example:
    ::

        result = screen(robot, model, path, speed=0.2)
        if not result.feasible:
            print(result.reason)
        best = result.candidates[0]   # theta0, direction, conditioning, times, path
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass, field

import numpy as np

from solver.contracts import OMEGA, PuzzlePath, TurntableModel
from solver.ik_batch import batch_ik, expand_turns, in_limits
from solver.timing import RobotTiming

THETA_STEP = np.deg2rad(5.0)  # rad, turntable angle grid
ROLL_STEP = np.deg2rad(15.0)  # rad, hoop roll grid
FACES = (1, -1)  # ring normal along / against the direction of travel
MAX_STEP_CHANGE = np.deg2rad(25.0)  # rad, largest joint change between checkpoints
MAX_STATES = 60  # states kept per checkpoint in stage C (more: slower, same best)
ELBOW_SCALE = 0.1  # elbow term = (1 - reach / max reach) / ELBOW_SCALE, clipped to 1
LIMIT_SCALE = np.deg2rad(20.0)  # limit term = distance to the nearest limit / this


def hoop_offset() -> np.ndarray:
    """Pose of the ring centre frame (the hoop's hoop_center) in the flange frame.

    Returns:
        4x4 transform, from the robot URDF (adapter, turned 180 deg on the CRS93)
        and the hoop URDF at its nominal mount.
    """
    from ctu_crs import CRS93
    from placement import HOOP_MOUNT, MODELS, load_urdf
    robot = CRS93(tty_dev=None)
    urdf = load_urdf(MODELS / "crs93.urdf")
    urdf.update_cfg(robot.q_home)
    hoop = load_urdf(MODELS / "hoop.urdf")
    return (np.linalg.inv(robot.fk(robot.q_home))
            @ urdf.get_transform("adapter_front", urdf.base_link) @ HOOP_MOUNT
            @ hoop.get_transform("hoop_center", hoop.base_link))


def ring_frames(checkpoint_T: np.ndarray, rolls: np.ndarray, face: int) -> np.ndarray:
    """Desired ring centre frames at a checkpoint.

    Args:
        checkpoint_T: Checkpoint frame, robot frame (4x4).
        rolls: Rolls [rad], shape (R,).
        face: +1 ring normal along the checkpoint z, -1 against it.

    Returns:
        Ring centre frames (the hoop's hoop_center), shape (R, 4, 4).
    """
    c, s = np.cos(rolls), np.sin(rolls)
    Rz = np.zeros((len(rolls), 4, 4))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = c, -s, s, c
    Rz[:, 2, 2] = Rz[:, 3, 3] = 1.0
    F = np.diag([1.0, -1.0, -1.0, 1.0]) if face < 0 else np.eye(4)  # flip about x
    return checkpoint_T @ F @ Rz


def conditioning(robot, Q: np.ndarray) -> np.ndarray:
    """Conditioning of configurations in [0, 1] (0 singular or at a limit).

    Args:
        robot: CRS93 / CRS93Sim.
        Q: Configurations [rad], shape (..., 6).

    Returns:
        min(wrist, elbow, limit) terms, shape (...).
    """
    wrist = np.abs(np.sin(Q[..., 4]))
    a1, d3 = robot.dh_a[1], robot.dh_d[3]
    # law of cosines: wrist-centre distance from the shoulder from q3
    reach = np.sqrt(a1**2 + d3**2 + 2 * a1 * d3 * np.cos(Q[..., 2]))
    elbow = np.clip((1 - reach / (a1 + d3)) / ELBOW_SCALE, 0, 1)
    limit = np.clip(np.min(np.minimum(Q - robot.q_min, robot.q_max - Q), axis=-1)
                    / LIMIT_SCALE, 0, 1)
    return np.minimum(np.minimum(wrist, elbow), limit)


@dataclass
class FeasibilityMap:
    """Stage A and B results: IK solutions per checkpoint x angle x (face, roll).

    Attributes:
        thetas: Turntable angle grid [rad], shape (A,).
        rolls: Roll grid [rad], shape (R,).
        Q: Solutions [rad], shape (N, A, 2, R, K, 6), NaN where none.
        cond: Their conditioning, shape (N, A, 2, R, K), NaN where none.
        reach: Stage A: wrist centre within reach, shape (N, A, 2, R).
    """

    thetas: np.ndarray
    rolls: np.ndarray
    Q: np.ndarray
    cond: np.ndarray
    reach: np.ndarray

    def angle_index(self, theta: float) -> int:
        """Index of the grid angle nearest to @theta (any turn)."""
        return int(np.round((theta % (2 * np.pi)) / THETA_STEP)) % len(self.thetas)

    def reachable(self) -> np.ndarray:
        """Per checkpoint x angle: whether any solution exists, shape (N, A)."""
        return np.any(np.isfinite(self.cond), axis=(2, 3, 4))


@dataclass
class Candidate:
    """A start phase that passed the screening.

    Attributes:
        theta0: Turntable angle when the traversal starts [rad].
        direction: Spin direction (+1 / -1).
        conditioning: Best bottleneck conditioning along the checkpoints.
        times: Nominal arrival times at the checkpoints [s], shape (N,).
        thetas: Grid turntable angle of each checkpoint [rad], shape (N,): q is
            exact for these, and theta0 + direction * OMEGA * times is within
            half a grid step of them.
        q: Configurations of the best path [rad], shape (N, 6).
        face: Hoop face of the best path per checkpoint, shape (N,).
        roll: Roll of the best path per checkpoint [rad], shape (N,).
    """

    theta0: float
    direction: int
    conditioning: float
    times: np.ndarray
    thetas: np.ndarray
    q: np.ndarray
    face: np.ndarray
    roll: np.ndarray


@dataclass
class FeasibilityResult:
    """Outcome of the funnel.

    Attributes:
        feasible: Whether any start phase survived.
        reason: Why not (empty if feasible).
        candidates: Survivors, best conditioning first.
        map: The feasibility map (None if stage A already failed).
        stage: Stage that decided: "A", "B", "C" or "" (feasible).
    """

    feasible: bool
    reason: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    map: FeasibilityMap | None = None
    stage: str = ""


def build_map(robot, model: TurntableModel, path: PuzzlePath,
              offset: np.ndarray | None = None) -> FeasibilityMap:
    """Stages A and B: reach and IK solutions on the grid.

    Args:
        robot: CRS93 / CRS93Sim.
        model: The turntable model (theta only is varied; theta_ref, t_ref unused).
        path: The checkpoints.
        offset: Ring centre in the flange frame (hoop_offset() if None).

    Returns:
        The map.
    """
    offset = hoop_offset() if offset is None else offset
    to_flange = np.linalg.inv(offset)
    thetas = np.arange(0, 2 * np.pi - 1e-9, THETA_STEP)
    rolls = np.arange(0, 2 * np.pi - 1e-9, ROLL_STEP)
    N, A, R = len(path.frames), len(thetas), len(rolls)
    flanges = np.empty((N, A, 2, R, 4, 4))
    for a, theta in enumerate(thetas):
        P = model.puzzle_T(theta)
        for i in range(N):
            C = P @ path.frames[i]
            for f, face in enumerate(FACES):
                flanges[i, a, f] = ring_frames(C, rolls, face) @ to_flange
    # stage A: wrist centre within reach of the shoulder
    d, a1 = robot.dh_d, robot.dh_a[1]
    wc = flanges[..., :3, 3] - d[5] * flanges[..., :3, 2]
    dist = np.linalg.norm(wc - np.array([0.0, 0.0, d[0]]), axis=-1)
    reach = (dist < d[3] + a1) & (dist > abs(d[3] - a1))
    # stage B: IK only where stage A passed, one checkpoint at a time, keeping only
    # the in-limit solutions (most of the 8 x 9 slots with the +-2 pi variants of
    # joints 4 and 6 are empty)
    Qs, conds = [], []
    for i in range(N):
        flat = flanges[i].reshape(-1, 4, 4)
        ok = reach[i].reshape(-1)
        Q = np.full((len(flat), 8, 6), np.nan)
        if ok.any():
            Q[ok], _ = batch_ik(robot, flat[ok])
        Q = _compact(robot, Q)
        Q = _compact(robot, expand_turns(robot, Q))
        Qs.append(Q)
        conds.append(np.where(np.isfinite(Q[..., 0]),
                              conditioning(robot, np.nan_to_num(Q)), np.nan))
    K = max(1, max(q.shape[1] for q in Qs))
    Q = np.full((N, A * 2 * len(rolls), K, 6), np.nan)
    cond = np.full((N, A * 2 * len(rolls), K), np.nan)
    for i in range(N):
        Q[i, :, :Qs[i].shape[1]] = Qs[i]
        cond[i, :, :conds[i].shape[1]] = conds[i]
    return FeasibilityMap(thetas, rolls, Q.reshape(N, A, 2, len(rolls), K, 6),
                          cond.reshape(N, A, 2, len(rolls), K), reach)


def _compact(robot, Q: np.ndarray) -> np.ndarray:
    """Keeps only in-limit solutions, moved to the front, cut to the largest count.

    Args:
        robot: CRS93 / CRS93Sim (joint limits).
        Q: Solutions, shape (M, K, 6), NaN where none.

    Returns:
        Shape (M, K', 6) with K' the largest number of in-limit solutions of any
        target (at least 1); the rest NaN.
    """
    ok = in_limits(robot, Q)
    order = np.argsort(~ok, axis=1, kind="stable")
    Q = np.take_along_axis(Q, order[..., None], axis=1)
    ok = np.take_along_axis(ok, order, axis=1)
    K = max(1, int(ok.sum(axis=1).max()))
    Q = Q[:, :K].copy()
    Q[~ok[:, :K]] = np.nan
    return Q


def _durations(timing: RobotTiming, dQ: np.ndarray) -> np.ndarray:
    """Durations of many coordinated moves at once (RobotTiming.profile, vectorized).

    Args:
        timing: Timing model.
        dQ: Joint displacements, shape (..., 6).

    Returns:
        Durations [s], shape (...).
    """
    d = np.abs(dQ)
    ramp = timing.v_max / timing.a_max
    T = np.where(d > timing.v_max * ramp, d / timing.v_max + ramp,
                 2 * np.sqrt(d / timing.a_max))
    return T.max(axis=-1)


def _states(fmap: FeasibilityMap) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """The valid solutions of every checkpoint and grid angle, gathered once.

    Args:
        fmap: The map.

    Returns:
        states[i][a] = (keys into the flattened (face, roll, solution) axes, q
        (S, 6), conditioning (S,)).
    """
    N, A = fmap.Q.shape[:2]
    flat_Q = fmap.Q.reshape(N, A, -1, 6)
    flat_c = fmap.cond.reshape(N, A, -1)
    out = []
    for i in range(N):
        row = []
        for a in range(A):
            k = np.flatnonzero(np.isfinite(flat_c[i, a]))
            row.append((k, flat_Q[i, a, k], flat_c[i, a, k]))
        out.append(row)
    return out


def _walk(fmap: FeasibilityMap, states, timing: RobotTiming, theta0: float,
          direction: int) -> Candidate | int:
    """Stage C for one start phase.

    Args:
        fmap: The map.
        states: _states(fmap).
        timing: Timing model (nominal arrival times).
        theta0: Start phase [rad].
        direction: Spin direction.

    Returns:
        The candidate, or the index of the first checkpoint that cannot be
        reached from the previous one.
    """
    N, A = fmap.Q.shape[:2]
    omega = direction * OMEGA
    longest = float(_durations(timing, np.full(6, MAX_STEP_CHANGE)))
    offsets = np.arange(0.0, longest + THETA_STEP / OMEGA, 0.4 * THETA_STEP / OMEGA)

    def angle_of(t: np.ndarray) -> np.ndarray:
        return np.round(((theta0 + omega * t) % (2 * np.pi)) / THETA_STEP).astype(int) % A

    a0 = int(angle_of(np.array(0.0)))
    keys, Q_prev, b_prev = states[0][a0]
    if len(keys) == 0:
        return 0
    t_prev = np.zeros(len(keys))
    layers = [(np.full(len(keys), a0), keys, np.full(len(keys), -1))]
    for i in range(1, N):
        # grid angles each previous state's next arrival can fall on
        reach_a = np.zeros((len(t_prev), A), dtype=bool)
        reach_a[np.arange(len(t_prev))[:, None],
                angle_of(t_prev[:, None] + offsets[None, :])] = True
        cand = []
        for a in np.flatnonzero(reach_a.any(axis=0)):
            k, Q_cur, c_cur = states[i][a]
            if len(k) == 0:
                continue
            prev = np.flatnonzero(reach_a[:, a])  # only those that can land on a
            dQ = Q_cur[:, None, :] - Q_prev[None, prev, :]  # (cur, prev, 6)
            t_arr = t_prev[None, prev] + _durations(timing, dQ)
            ok = (np.abs(dQ).max(-1) <= MAX_STEP_CHANGE) & (angle_of(t_arr) == a)
            if not ok.any():
                continue
            score = np.where(ok, np.minimum(b_prev[None, prev], c_cur[:, None]), -np.inf)
            order = np.lexsort((np.where(ok, t_arr, np.inf), -score), axis=1)[:, 0]
            rows = np.arange(len(k))
            alive = np.isfinite(score[rows, order])
            cand.append((np.full(alive.sum(), a), k[alive], Q_cur[alive],
                         t_arr[rows, order][alive], score[rows, order][alive],
                         prev[order[alive]]))
        if not cand:
            return i
        a_i, k_i, Q_prev, t_prev, b_prev, p_i = (np.concatenate(x) for x in zip(*cand))
        keep = np.lexsort((t_prev, -b_prev))[:MAX_STATES]
        a_i, k_i, Q_prev, t_prev, b_prev, p_i = (x[keep] for x in
                                                 (a_i, k_i, Q_prev, t_prev, b_prev, p_i))
        layers.append((a_i, k_i, p_i))
    s = int(np.lexsort((t_prev, -b_prev))[0])
    final = float(b_prev[s])
    chain = []
    for a_l, k_l, p_l in reversed(layers):
        chain.append((int(a_l[s]), int(k_l[s])))
        s = int(p_l[s])
    chain.reverse()
    flat_Q = fmap.Q.reshape(N, A, -1, 6)
    q, face, roll, thetas = [], [], [], []
    for i, (a, key) in enumerate(chain):
        f, r_, _ = np.unravel_index(key, fmap.cond.shape[2:])
        q.append(flat_Q[i, a, key])
        face.append(FACES[f])
        roll.append(fmap.rolls[r_])
        thetas.append(fmap.thetas[a])
    q = np.array(q)
    times = np.concatenate([[0.0], np.cumsum(_durations(timing, np.diff(q, axis=0)))])
    return Candidate(float(theta0), direction, final, times, np.array(thetas), q,
                     np.array(face), np.array(roll))


_SHARED: tuple | None = None  # (fmap, states, timing) inherited by forked workers


def _walk_job(job: tuple[float, int]) -> Candidate | int:
    """Worker task: stage C for one (theta0, direction) on the shared map."""
    fmap, states, timing = _SHARED
    return _walk(fmap, states, timing, *job)


def screen(robot, model: TurntableModel, path: PuzzlePath, speed: float,
           directions: tuple[int, ...] | None = None,
           fmap: FeasibilityMap | None = None,
           workers: int | None = None) -> FeasibilityResult:
    """The feasibility funnel: stages A, B and C.

    Args:
        robot: CRS93 / CRS93Sim (kinematics, limits, speed parameters).
        model: The estimated turntable model.
        path: The checkpoints.
        speed: Relative robot speed (nominal timing).
        directions: Spin directions to screen; (model.direction,) by default.
        fmap: A map from an earlier call for the same placement and path.
        workers: Processes for stage C (the start phases are independent);
            default: the number of CPUs; 1 runs in this process. Uses fork, so
            parallel only on Linux / macOS.

    Returns:
        The result: candidates (best first) or the reason why none.
    """
    fmap = fmap or build_map(robot, model, path)
    never = np.flatnonzero(~fmap.reach.any(axis=(1, 2, 3)))
    if len(never):
        return FeasibilityResult(False, f"checkpoint(s) {never.tolist()} out of the "
                                 "arm's reach at every turntable angle", map=fmap, stage="A")
    no_ik = np.flatnonzero(~fmap.reachable().any(axis=1))
    if len(no_ik):
        return FeasibilityResult(False, f"checkpoint(s) {no_ik.tolist()} have no IK "
                                 "solution within the joint limits at any turntable angle",
                                 map=fmap, stage="B")
    global _SHARED
    timing = RobotTiming(robot, speed)
    _SHARED = (fmap, _states(fmap), timing)
    jobs = [(float(theta0), d) for d in directions or (model.direction,)
            for theta0 in fmap.thetas]
    workers = workers or os.cpu_count() or 1
    if workers > 1 and "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            outs = pool.map(_walk_job, jobs, chunksize=max(1, len(jobs) // (4 * workers)))
    else:
        outs = [_walk_job(job) for job in jobs]
    _SHARED = None
    candidates = [o for o in outs if isinstance(o, Candidate)]
    failures = [o for o in outs if not isinstance(o, Candidate)]
    if not candidates:
        worst = np.bincount(failures).argmax()
        return FeasibilityResult(False, "no start phase works: the path breaks most "
                                 f"often at checkpoint {worst} (unreachable there or "
                                 "only by switching the IK branch)", map=fmap, stage="C")
    candidates.sort(key=lambda c: -c.conditioning)
    return FeasibilityResult(True, candidates=candidates, map=fmap)
