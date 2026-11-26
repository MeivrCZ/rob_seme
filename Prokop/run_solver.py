"""Runs the whole solver (variant D) phase by phase on the simulator, with diagnostics.

Calls the phases in order and hands each one the previous phase's
contract types (solver/contracts.py). Every phase prints what it produced
and, on the simulator, how far that is from the truth. The run stops at the
first phase that fails, so the output shows where the pipeline breaks.

    Phase 1  Turntable model   images -> BaseMeasurement (Vojtech, P1.2)
                               -> TurntableModel (Vojtech, P1.3 fit_orbit)
    Phase 2  Plan              default (--planner prokop): Prokop's planner,
                               the wire's checkpoints planned backwards in time
                               from end angles in parallel (solver.planner.
                               plan_spinning) -> approach clear of the swept
                               volume (plan_approach). --planner natka / copy:
                               Natka's P2.1-P2.7 (or the experimental copy)
    Phase 3  Execution         Executor (P3.3) with a PhaseTracker: default
                               Vojtech's CameraPhaseTracker (keeps measuring
                               the base while the robot works); --tracker
                               fitted: ModelTracker on the fitted model

The pipeline uses only the ctu_crs CRS93 interface (move_to_q, get_q,
grab_image, fk, ik, ...). The simulator runs standalone in its own process
(sim_process.SimProcess): the pipeline gets only its robot interface; the scene
setup and the Truth class (the ground truth that judges the results, read
through the SimProcess) belong to this runner, nothing in the pipeline uses
them. On the real robot, `--real` creates CRS93() instead and skips the truth
checks (untested).

Usage (from the repo root; the viewer runs on http://localhost:8080):

    uv run Prokop/run_solver.py                                # the whole pipeline
    uv run Prokop/run_solver.py --axis 0.487 0 --offset 0.02 0  # a chosen placement
    uv run Prokop/run_solver.py --seed 3 --direction -1
    uv run Prokop/run_solver.py --stop-after 2      # only perception + planning
    uv run Prokop/run_solver.py --hold              # keep the viewer open at the end

Each run writes its log, the camera frames and a summary.json to
Prokop/runs/<time>/.
"""

from __future__ import annotations

import time

T_SCRIPT = time.monotonic()  # the program starts: the assignment's 120 s count from here

import argparse
import contextlib
import json
import sys
import threading
from dataclasses import replace
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for directory in (HERE, REPO / "Vojtěch" / "ArucoFinder", REPO / "Vojtěch" / "Homografy",
                  REPO / "Natka"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cv2  # noqa: E402
from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from camera import Camera  # noqa: E402
from cell import TABLE_Z, WORK_AREA_XY  # noqa: E402
from crs_puzzle_d_hoop_path import puzzle_path_contract  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from homography import Homography  # noqa: E402
from turntable_orbit import (MARKER_HEIGHT_ABOVE_TABLE_M, capture_series,  # noqa: E402
                             draw_orbit, fit_orbit, measure_base)

from solver.approach import ApproachChecker, plan_line, ring_in_flange  # noqa: E402
from solver.contracts import (OMEGA, BaseMeasurement, ModelTracker, Plan,  # noqa: E402
                              PuzzlePath, TurntableModel)
from solver.executor import LEAD, ExecutionResult, Executor  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from solver.wire import SUCCESS_HEIGHT, ring_top_above_pad, wire_path  # noqa: E402

RUNS = HERE / "runs"
RUN_LIMIT = 120.0  # s, program start -> hoop < 4 cm above the base, threaded (assignment)
# Planning goes on as long as a plan could still end in time: until RUN_LIMIT minus the
# least the rest can take (a short approach, no wait, a short traversal, the executor's
# overheads: tracker ready, LEAD, the step-0 compute budget); a plan counts only if
# its predicted end (with its real wait for its start angle) is within RUN_LIMIT.
APPROACH_MIN = 2.0  # s
TRAVERSAL_MIN = 4.0  # s
EXEC_OVERHEAD = 4.0  # s, camera tracker ready + LEAD + compute budget (measured: see logs)
TIME_MARGIN = 3.0  # s
PLAN_PHASES = 6  # start phases handed to the planner (an even grid over a turn)
FINAL_HEIGHT = 0.04  # m, success: hoop less than this above the base (assignment)


TRACKER_LOST_AFTER = 1.0  # s, age of the camera tracker's latest measurement that counts as lost


def watch_tracker(camera, stop, log=print) -> None:
    """Logs once when the camera tracker loses the markers (its latest measurement is
    older than TRACKER_LOST_AFTER: it predicts from the model) and once when it finds
    them again (CameraPhaseTracker itself is Vojtech's code and is not changed)."""
    lost = False
    while not stop.wait(0.1):
        try:
            age = time.monotonic() - camera.estimate().t
        except RuntimeError:
            return
        if not lost and age > TRACKER_LOST_AFTER:
            log("camera tracker: lost aruco, believing model")
            lost = True
        elif lost and age <= TRACKER_LOST_AFTER:
            log("camera tracker: found aruco again, updating model")
            lost = False


class PhaseFailed(RuntimeError):
    """A phase did not produce its output; the run stops."""


class Log:
    """Prints and writes to the run's log file."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("w", encoding="utf-8")

    def __call__(self, *parts) -> None:
        text = " ".join(str(p) for p in parts)
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()


def mm(x) -> str:
    return f"{1000 * float(x):.1f} mm"


def deg(x) -> str:
    return f"{np.rad2deg(float(x)):+.2f} deg"


def wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


# --- Ground truth (simulator only, never used by the pipeline) ---

class Truth:
    """What the simulator knows: the true placement, turntable angle and arm motion.

    Read through the SimProcess (the scene's ground truth, not the robot
    interface). Times are time.monotonic() like the pipeline's.
    """

    def __init__(self, sim, placement, direction: int) -> None:
        self.sim, self.placement, self.direction = sim, placement, direction

    def theta_at(self, t: float) -> float:
        return self.sim.truth_angle(t)

    def model(self) -> TurntableModel:
        """The true TurntableModel (same chain as the contracts, tested equal)."""
        t = time.monotonic()
        p = self.placement
        return TurntableModel(axis_xy=np.asarray(p.turntable_xy, float),
                              base_offset=np.asarray(p.base_xy, float),
                              base_yaw=float(p.base_yaw), direction=self.direction,
                              omega=OMEGA, theta_ref=self.theta_at(t), t_ref=t)

    @property
    def track(self) -> list[tuple[float, np.ndarray]]:
        """The recorded true arm configurations (time, q)."""
        return self.sim.truth_track(0.0)

    def stop(self) -> None:
        pass


def simulated_calibration() -> Homography:
    """3x4 camera matrix fitted to the simulator's camera model (camera.Camera).

    The same as Vojtech's Mainframe `simulated_homography` (a stand-in for the
    calibration from hoop images): the "calibration" of a perfect camera.
    """
    camera = Camera()
    model = Homography(x_range_m=tuple(WORK_AREA_XY[0]), y_range_m=tuple(WORK_AREA_XY[1]),
                       z_levels_m=(0.03, 0.10, 0.20), table_z_m=TABLE_Z, nx=4, ny=4)
    for index, xyz in enumerate(model.points):
        model.record(index, tuple(camera.project(xyz)), tuple(xyz))
    model.calculate(validation_fraction=0.0, ransac=False)
    return model


# --- Phases ---

@contextlib.contextmanager
def phase(log: Log, name: str, summary: dict):
    """Runs a phase: header, duration, and any exception turned into PhaseFailed."""
    log(f"\n=== {name} ===")
    t0 = time.monotonic()
    record = summary.setdefault("phases", {}).setdefault(name, {})
    try:
        yield record
        record["ok"] = True
    except PhaseFailed as error:
        record.update(ok=False, reason=str(error))
        log(f"--> FAILED: {error}")
        raise
    except Exception as error:  # a bug or an unexpected input in a module
        record.update(ok=False, reason=f"{type(error).__name__}: {error}")
        log(f"--> FAILED with an exception:\n{traceback.format_exc()}")
        raise PhaseFailed(record["reason"]) from error
    finally:
        record["seconds"] = round(time.monotonic() - t0, 2)
        log(f"({name}: {record['seconds']:.1f} s)")


def phase1_turntable(robot, calibration: Homography, count: int, interval: float,
                     out: Path, log: Log, truth: Truth | None, record: dict
                     ) -> TurntableModel:
    """Images of the spinning puzzle -> BaseMeasurements -> TurntableModel."""
    detector = ArucoBlockDetector()
    marker_z = calibration.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
    measurements: list[BaseMeasurement] = []
    xy_err, yaw_err, image = [], [], None
    log(f"P1.2: {count} images, {interval:.1f} s apart")
    for k, (t_capture, image) in enumerate(
            capture_series(robot.grab_image, count, interval, robot.camera_delay)):
        cv2.imwrite(str(out / f"frame_{k:02d}.png"), image)
        m = measure_base(image, t_capture, detector, calibration, marker_z)
        if m is None:
            log(f"  frame {k:02d}: no markers")
            continue
        measurements.append(m)
        line = f"  frame {k:02d}: {m.n_markers} marker(s), xy {np.round(m.xy, 4)}, yaw {deg(m.yaw)}"
        if truth is not None:
            xy_t, yaw_t = truth.model().measurement_at(t_capture)
            xy_err.append(np.linalg.norm(m.xy - xy_t))
            yaw_err.append(abs(wrap(m.yaw - yaw_t)))
            line += f" | truth error {mm(xy_err[-1])}, {deg(yaw_err[-1])}"
        log(line)
    record["detections"] = f"{len(measurements)}/{count}"
    if xy_err:
        record["measurement_error"] = {"xy_max_mm": 1000 * max(xy_err),
                                       "yaw_max_deg": float(np.rad2deg(max(yaw_err)))}
        log(f"P1.2 vs truth: xy error max {mm(max(xy_err))}, yaw error max {deg(max(yaw_err))}")
    if len(measurements) < 4:
        raise PhaseFailed(f"only {len(measurements)} detections (fit needs >= 4)")

    log("P1.3: fit_orbit")
    fit = fit_orbit(measurements)
    model = fit.model
    cv2.imwrite(str(out / "orbit.png"), draw_orbit(image, fit, measurements, calibration, marker_z))
    log(f"  axis {np.round(model.axis_xy, 4)}, offset {np.round(model.base_offset, 4)} "
        f"(r = {mm(fit.radius_m)}), base_yaw {deg(model.base_yaw)}, direction {model.direction:+d}, "
        f"period {fit.period_s:.2f} s; residuals {mm(fit.xy_rmse_m)}, {deg(fit.yaw_rmse_rad)}")
    record["model"] = {"axis_xy": model.axis_xy.tolist(), "base_offset": model.base_offset.tolist(),
                       "base_yaw": model.base_yaw, "direction": model.direction,
                       "omega": model.omega, "period_s": fit.period_s}
    if truth is not None:
        true = truth.model()
        log(f"  truth: axis {np.round(true.axis_xy, 4)}, offset r = "
            f"{mm(np.linalg.norm(true.base_offset))}, direction {true.direction:+d}")
        t_now = time.monotonic()
        errors = {}
        for dt in (0.0, 10.0, 30.0):  # the prediction drifts with a wrong omega
            t = t_now + dt
            P, P_t = model.puzzle_T(model.theta_at(t)), true.puzzle_T(true.theta_at(t))
            dp = np.linalg.norm(P[:3, 3] - P_t[:3, 3])
            da = np.arccos(np.clip((np.trace(P[:3, :3].T @ P_t[:3, :3]) - 1) / 2, -1, 1))
            errors[f"+{dt:.0f}s"] = {"pos_mm": 1000 * dp, "rot_deg": float(np.rad2deg(da))}
            log(f"  puzzle pose error at now+{dt:.0f} s: {mm(dp)}, {np.rad2deg(da):.2f} deg")
        record["model_error"] = errors
        if model.direction != true.direction:
            raise PhaseFailed("fitted spin direction is wrong")
    record["fit_for_margins"] = {"xy_rmse_m": float(fit.xy_rmse_m),
                                 "yaw_rmse_rad": float(fit.yaw_rmse_rad),
                                 "times": [float(m.t_capture) for m in measurements]}
    return model


def plan_consistency(robot, model: TurntableModel, path: PuzzlePath, plan: Plan) -> float:
    """Largest distance [m] of the planned ring centre from its checkpoint, with
    the puzzle where the plan's own timing says it is (model, not truth)."""
    tool = ring_in_flange()
    worst = 0.0
    for step in plan.steps:
        theta = plan.theta_start + plan.direction * model.omega * step.t_rel
        target = model.checkpoint_T(path, step.index, theta)[:3, 3]
        worst = max(worst, float(np.linalg.norm((robot.fk(step.q) @ tool)[:3, 3] - target)))
    return worst


def phase2_plan(kin: CRS93, model: TurntableModel, speed: float, n_phases: int, planner: str,
                pre_entry: float, continuity: list[float] | None, log: Log, record: dict,
                margin_floor: float | None = None, closest: bool = False,
                speed_cap: bool = False) -> tuple[Plan, PuzzlePath]:
    """PuzzlePath -> timed Plan -> approach line."""
    log("P2.1: puzzle path (Natka)")
    path = puzzle_path_contract()
    log(f"  {len(path.frames)} checkpoints, clearance min {mm(np.min(path.clearance))}, "
        f"final index {path.final_index}")
    record["checkpoints"] = len(path.frames)

    phases = np.linspace(-np.pi, np.pi, n_phases, endpoint=False)
    log(f"P2.6: timed plan ({planner}) over {len(phases)} start phases "
        f"{', '.join(deg(p) for p in phases)}")
    t0 = time.monotonic()
    extra = {}
    if planner == "copy":  # Prokop's experimental copy: side B leads, pre-entry start
        from natka_copy.planning import timed_plan
        extra = {"pre_entry": pre_entry, "continuity": None if not continuity else
                 (continuity[0], *np.deg2rad(continuity[1:])), "progress": log,
                 "margin_floor": margin_floor, "closest": closest, "speed_cap": speed_cap}
        log(f"  copy: start {mm(pre_entry)} before checkpoint 0; continuity "
            + ("off" if not continuity else
               f"||dq - dq*|| <= {continuity[0]} ||dq*|| + {continuity[1]} deg"
               + (f", capped at {continuity[2]} deg" if len(continuity) > 2 else "")))
        log(f"  copy: margin > {margin_floor}; rank {'closest first' if closest else 'margin first'}; "
            f"speed cap {'on' if speed_cap else 'off'}")
        if pre_entry > 0:
            log("  (planner indices: 0 = the pre-entry pose, k = checkpoint k - 1)")
    else:
        from planning import timed_plan
    plan = timed_plan(kin, RobotTiming(kin, speed), model, path,
                      start_phases=phases, directions=(model.direction,),
                      speeds=(speed,), q_start=kin.q_home, **extra)
    log(f"  {time.monotonic() - t0:.1f} s; feasible {plan.feasible} {plan.reason}")
    record["plan"] = {"feasible": plan.feasible, "reason": plan.reason}
    if not plan.feasible:
        raise PhaseFailed(f"planner: {plan.reason}")
    first = plan.steps[0]
    log(f"  theta_start {deg(plan.theta_start)}, {len(plan.steps)} steps, min margin "
        f"{plan.min_margin:.3f}, traversal {plan.steps[-1].t_rel:.1f} s")
    log(f"  q_start {np.round(np.rad2deg(plan.q_start), 1)} deg; first step moves "
        f"{np.rad2deg(np.abs(first.q - plan.q_start).max()):.1f} deg (largest joint) "
        f"and arrives at t_rel {first.t_rel:.2f} s")
    jumps = [np.rad2deg(np.abs(b.q - a.q).max()) for a, b in zip(plan.steps, plan.steps[1:])]
    log(f"  joint change between steps: max {max(jumps):.1f} deg (step {int(np.argmax(jumps)) + 1})")
    err = plan_consistency(kin, model, path, plan)
    log(f"  plan replay (ring centre vs checkpoint at the planned angle): max {mm(err)}")
    record["plan"].update(theta_start=plan.theta_start, steps=len(plan.steps),
                          min_margin=plan.min_margin, traversal_s=plan.steps[-1].t_rel,
                          replay_error_mm=1000 * err, max_step_jump_deg=float(max(jumps)))

    log("P2.7: approach line q_home -> q_start")
    line = plan_line(kin, kin.q_home, plan.q_start, ApproachChecker(model))
    if line.path is None:
        raise PhaseFailed(f"approach: {line.reason}")
    plan.approach = line.path[1:-1]
    log(f"  {len(line.path)} samples, max deviation from the line {mm(line.max_deviation)}")
    record["approach_samples"] = len(line.path)
    return plan, path


def predicted_end(timing, model: TurntableModel, theta_start: float, result,
                  t_now: float) -> float:
    """When a plan would end if execution started now: the approach, the executor's
    overheads, the wait for its start angle, its traversal [s, time.monotonic()]."""
    path = [np.asarray(q) for q in result.approach]
    approach = sum(timing.move_duration(a, b) + timing.command_delay
                   for a, b in zip(path, path[1:]))
    t_ready = t_now + EXEC_OVERHEAD + approach
    ahead = ((theta_start - model.theta_at(t_ready)) * model.direction) % (2 * np.pi)
    return t_ready + ahead / model.omega + float(result.arrivals[-1])


def phase2_prokop(kin: CRS93, model: TurntableModel, speed: float, log: Log,
                  record: dict, t_base: float = T_SCRIPT,
                  fit: dict | None = None) -> tuple[Plan, PuzzlePath]:
    """Prokop's planner: the wire backwards in time from end angles (in parallel), then
    the approach from q_home clear of the swept volume. Planning stops when the rest
    could no longer finish within RUN_LIMIT of t_base; of the plans found, the one that
    ends soonest is taken (enough joint-limit room first, the distance from the
    singularities between plans ending within a second of each other)."""
    from solver import planner
    from solver.planner import FACE_NAMES, plan_approach, plan_spinning
    from solver.soft_collision import SoftScene
    from solver.uncertainty import Margins
    timing = RobotTiming(kin, speed)
    margins = None
    if fit is not None:  # the run's uncertainties -> the margins (k sigma)
        margins = Margins.from_fit(fit["xy_rmse_m"], fit["yaw_rmse_rad"], fit["times"],
                                   reach=SoftScene(model).task_reach(0.0))
        log(f"P2: margins from the uncertainties: {margins.describe()}")
        record["margins"] = {"hoop_puzzle_mm": 1000 * margins.hoop_puzzle,
                             "joint_limit_deg": float(np.rad2deg(margins.joint_limit)),
                             "sd_angle_deg": float(np.rad2deg(margins.sd_angle))}
    # The reachability map (tools/build_reachability_map.py): a quick screen first.
    from solver.reachability import MAP_PATH, ReachabilityMap
    if MAP_PATH.exists():
        t_map = time.monotonic()
        reach_ok, reach_text = ReachabilityMap.load().check(model)
        log(f"P2: reachability map ({1000 * (time.monotonic() - t_map):.0f} ms): {reach_text}")
        record["reachability"] = reach_text
        if not reach_ok:
            raise PhaseFailed(f"unreachable (reachability map): {reach_text}")
    angles = planner.end_angles(model)
    log(f"P2: backwards in time from {len(angles)} end angles "
        f"({', '.join(deg(a) for a in angles)}), {planner.BW_WORKERS} processes, speed {speed}")
    t0 = time.monotonic()
    t_limit = t_base + RUN_LIMIT
    plan_deadline = t_limit - (APPROACH_MIN + TRAVERSAL_MIN + EXEC_OVERHEAD + TIME_MARGIN)
    log(f"  time: {t0 - t_base:.1f} s used, planning may take up to {plan_deadline - t0:.1f} s "
        f"(a plan counts only if it is predicted to end within {RUN_LIMIT:.0f} s)")
    if plan_deadline <= t0:
        raise PhaseFailed("no time left for planning")

    def accept(theta, r):
        end = predicted_end(timing, model, theta, r, time.monotonic())
        if end > t_limit - TIME_MARGIN:
            log(f"  end angle {deg(r.theta_end if hasattr(r, 'theta_end') else 0.0)}: plan "
                f"found, but predicted to end {end - t_base:.0f} s after the start: too late")
            return False
        return True

    def window(theta, r):
        """How long to wait for other plans: at most BW_SCORE_WINDOW, and never so long
        that this plan misses its start angle (its end would jump a turn) or the limit."""
        now = time.monotonic()
        end = predicted_end(timing, model, theta, r, now)
        t_go = end - float(r.arrivals[-1])
        path = [np.asarray(q) for q in r.approach]
        approach = sum(timing.move_duration(a, b) + timing.command_delay
                       for a, b in zip(path, path[1:]))
        to_start = t_go - (now + EXEC_OVERHEAD + approach)  # until its angle comes round
        return min(planner.BW_SCORE_WINDOW, to_start - 1.0, t_limit - TIME_MARGIN - end - 1.0)

    def rank(theta, r):
        end = predicted_end(timing, model, theta, r, time.monotonic())
        r.predicted_end = end
        sigma, _ = r.quality  # the joint-limit margin is kept by every plan (margins)
        return (end <= t_limit, -round(end), sigma)

    def progress(theta_end, r):
        quality = getattr(r, "quality", None)
        log(f"  end angle {deg(theta_end)}: "
            + (r.reason if r.collision_free and r.reason else
               'collision free' if r.collision_free else r.reason or 'collides')
            + f" ({r.seconds:.0f} s)"
            + (f"; score: Jacobian sigma_min {1000 * quality[0]:.1f} mm, joint-limit margin "
               f"{np.rad2deg(quality[1]):.1f} deg" if quality else ""))

    theta_start, result = plan_spinning(kin, timing, model, progress=progress,
                                        approach_from=kin.q_home, deadline=plan_deadline,
                                        rank=rank, margins=margins, accept=accept,
                                        window=window)
    if result is None:
        raise PhaseFailed(f"no plan that ends within {RUN_LIMIT:.0f} s "
                          f"({time.monotonic() - t0:.0f} s of planning)")
    record["plan"] = {"seconds": round(time.monotonic() - t0, 1), "theta_start": theta_start,
                      "waypoints": len(result.qs), "traversal_s": float(result.arrivals[-1]),
                      "entry_mm": 1000 * result.entry,
                      "soft_clearance_mm": 1000 * result.min_clearance,
                      "end_pose_waits": bool(getattr(result, "end_waits", True))}
    log(f"  plan in {time.monotonic() - t0:.0f} s: start angle {deg(theta_start)}, "
        f"{len(result.qs)} waypoints, side {FACE_NAMES[result.face]}, entry {mm(result.entry)}, "
        f"traversal {result.arrivals[-1]:.1f} s, tilts "
        f"{np.round(np.rad2deg(result.orientations[:, 1])).astype(int).tolist()} deg, least "
        f"soft clearance {mm(result.min_clearance)}")
    end = predicted_end(timing, model, theta_start, result, time.monotonic())
    log(f"  predicted end {end - t_base:.1f} s after the start (limit {RUN_LIMIT:.0f} s)")
    record["plan"]["predicted_end_s"] = round(end - t_base, 1)
    if getattr(result, "candidates", None):
        log(f"  chosen from {len(result.candidates)} plan(s): the soonest end (then the "
            f"farthest from singularities): end angle {deg(result.theta_end)}")
        record["plan"]["candidates"] = [
            {"end_angle_deg": float(np.rad2deg(a)), "sigma_min_mm": 1000 * q[0],
             "limit_margin_deg": float(np.rad2deg(q[1]))} for a, q in result.candidates]
    if not record["plan"]["end_pose_waits"]:
        log("  WARNING: no final pose stays clear of the turning puzzle: after the end the "
            "puzzle will hit the arm (harmless: hollow puzzle on foam, magnetic hoop)")

    log("P2: approach q_home -> the waiting pose, clear of the swept volume")
    approach, how = result.approach, result.approach_how  # planned with the plan
    log(f"  {how}")
    record["approach"] = how
    # The executor moves through the approach's pre-entry pose(s) to the plan's start and
    # sends the traversal steps (the first step is the start pose itself).
    plan = replace(result.plan, steps=result.plan.steps[1:],
                   approach=[np.asarray(q) for q in approach[1:-1]])
    return plan, wire_path(4, result.entry)  # a fine path for judging the ring's motion


class FreezingTracker:
    """The camera tracker until the traversal starts, then the fitted model re-anchored
    on the tracker's last angle (the user: over a few seconds of traversal the model
    alone is accurate enough).

    freeze() stops the camera: nothing needs images during the traversal, so the
    robot's commands do not compete with camera grabs (on the simulator they share one
    link; on the real robot a grab could hold up the interpreter), and late images
    cannot pull the angle off (docs/PLANNER_EXPERIMENTS.md 9.6).
    """

    def __init__(self, camera, model: TurntableModel) -> None:
        self.camera, self.model = camera, model
        self._frozen: ModelTracker | None = None

    def estimate(self):
        return self._frozen.estimate() if self._frozen is not None else self.camera.estimate()

    def freeze(self):
        """Switches to the model; returns the camera's last estimate."""
        est = self.camera.estimate()
        self._frozen = ModelTracker(replace(self.model, theta_ref=est.theta, t_ref=est.t))
        self.camera.__exit__(None, None, None)  # stops the camera thread
        return est


class FreezingExecutor(Executor):
    """Executor that freezes a FreezingTracker LEAD before the traversal starts, at the
    start of the final wait, unconditionally (stopping the camera waits for the grab in
    progress, up to 0.2 s: done at the start would delay step 0). An earlier version
    froze only if the re-estimated start was within LEAD + 0.05 s, which an estimate
    jumping when the markers were lost and found skipped: the traversal then ran on
    the live camera tracker and aborted on its uncertainty (batch seed 5)."""

    def wait_for_phase(self) -> float:
        t_go = self.next_start_time(self.clock() + LEAD)
        self.log(f"Waiting {t_go - self.clock():.1f} s for the turntable phase")
        self._sleep_until(t_go - LEAD)
        if isinstance(self.tracker, FreezingTracker):
            est = self.tracker.freeze()
            self.log(f"camera tracker stopped for the traversal: the fitted model from its "
                     f"last angle ({self.clock() - est.t:.2f} s old)")
        t_go = self.next_start_time(self.clock() + 0.5 * LEAD)  # from the frozen model
        self._sleep_until(t_go - self.compute_budget)  # step 0 is solved before t_go
        return t_go


class ExecutionProbe:
    """Simulator-only diagnostics of phase 3 (observes, changes nothing).

    * every new estimate of the phase tracker against the true angle (the
      tracker's angle is in the fitted model's convention: compared by the base
      orientation, angle + base yaw);
    * every robot/camera call through the simulator link: when it was asked
      for and how long it took (the link serves one call at a time, so a robot
      command can wait behind a camera grab).
    """

    def __init__(self, robot, model: TurntableModel, truth: Truth) -> None:
        self.model, self.truth = model, truth
        self.yaw_true = float(truth.placement.base_yaw)
        self.estimates: list[tuple[float, float, float]] = []  # (wall, est.t, error rad)
        self.calls: list[tuple[str, float, float]] = []  # (name, t_request, t_done)
        self._stop = threading.Event()
        self._tracker = None
        link = getattr(robot, "_link", None)
        if link is not None:
            inner = link.call

            def timed(kind, name, *a, **k):
                t0 = time.monotonic()
                try:
                    return inner(kind, name, *a, **k)
                finally:
                    self.calls.append((name, t0, time.monotonic()))
            link.call = timed
            self._link, self._inner = link, inner

    def error(self, theta_fitted: float, t: float) -> float:
        """Fitted-convention angle minus the truth at t, by base orientation [rad]."""
        return wrap(theta_fitted + self.model.base_yaw - self.truth.theta_at(t) - self.yaw_true)

    def watch(self, tracker) -> None:
        self._tracker = tracker
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        last = None
        while not self._stop.wait(0.02):
            try:
                e = self._tracker.estimate()
            except RuntimeError:
                return
            if e.t != last:
                last = e.t
                self.estimates.append((time.monotonic(), e.t, self.error(e.theta, e.t)))

    def stop(self) -> None:
        self._stop.set()
        if getattr(self, "_link", None) is not None:
            self._link.call = self._inner

    def dump(self, robot, plan: Plan, result, executor, path: Path) -> None:
        """Saves what an offline analysis needs (experiments/execution_replay.py)."""
        import pickle
        from dataclasses import asdict
        t_now = time.monotonic()
        m, pl = self.model, self.truth.placement
        data = {
            "calls": self.calls, "estimates": self.estimates,
            "steps": [asdict(s) for s in (result.steps if result else [])],
            "reason": getattr(result, "reason", "") if result else "exception",
            "t_start": getattr(result, "t_start", None) if result else None,
            "t_attempt": executor.t_attempt,
            "track": self.truth.track,
            "grabs": self.truth.sim.truth_grabs(executor.t_attempt - 120.0),
            "truth_angle": (t_now, self.truth.theta_at(t_now)),
            "direction": self.truth.direction, "omega_true": OMEGA,
            "placement": {"turntable_xy": np.asarray(pl.turntable_xy, float),
                          "base_xy": np.asarray(pl.base_xy, float),
                          "base_yaw": float(pl.base_yaw)},
            "model": {"axis_xy": m.axis_xy, "base_offset": m.base_offset,
                      "base_yaw": m.base_yaw, "direction": m.direction, "omega": m.omega,
                      "theta_ref": m.theta_ref, "t_ref": m.t_ref},
            "plan": {"theta_start": plan.theta_start, "q_start": plan.q_start,
                     "steps": [(st.t_rel, np.asarray(st.q)) for st in plan.steps]},
            "camera_delay": float(robot.camera_delay),
            "command_delay": float(executor.timing.command_delay),
        }
        path.write_bytes(pickle.dumps(data))

    def report(self, robot, plan: Plan, result, log: Log, out: Path) -> None:
        log("\nDiagnostics (simulator truth):")
        # 1. The tracker: each new estimate's error at its own time; gaps between them.
        if self.estimates:
            with (out / "tracker.csv").open("w") as f:
                f.write("wall_s,estimate_t_s,error_deg\n")
                for w, t, err in self.estimates:
                    f.write(f"{w:.3f},{t:.3f},{np.rad2deg(err):.3f}\n")
            t0 = self.estimates[0][1]
            prev = None
            for w, t, err in self.estimates:
                gap = None if prev is None else t - prev
                if gap is None or gap > TRACKER_LOST_AFTER or abs(np.rad2deg(err)) > 1.0:
                    log(f"  tracker estimate at t+{t - t0:6.2f} s: error {np.rad2deg(err):+6.2f} deg"
                        + ("" if gap is None else f" (after a gap of {gap:.2f} s)")
                        + f", available {w - t:.2f} s after its image")
                prev = t
            errs = np.rad2deg([e for _, _, e in self.estimates])
            log(f"  tracker: {len(errs)} estimates, error median {np.median(errs):+.2f} deg, "
                f"range {errs.min():+.2f} .. {errs.max():+.2f} deg (written to tracker.csv)")
        # 2. The link: robot commands that waited.
        moves = [(t0, t1) for name, t0, t1 in self.calls if name == "move_to_q"]
        grabs = [t1 - t0 for name, t0, t1 in self.calls if name == "grab_image"]
        if moves:
            waits = [t1 - t0 for t0, t1 in moves]
            log(f"  link: {len(moves)} move_to_q took {1000 * min(waits):.0f}..{1000 * max(waits):.0f} ms "
                f"(the simulated command itself blocks 25 ms); {len(grabs)} grab_image "
                f"{1000 * np.median(grabs):.0f} ms median")
        # 3. The steps: the executor's correction vs what the truth needed.
        track = self.truth.track
        for s in (result.steps if result else []):
            planned = s.theta_planned
            need_pred = wrap(self.truth.theta_at(s.t_arrival) + self.yaw_true
                             - self.model.base_yaw - planned)
            arrived = next((t for t, q in track if t >= s.t_send
                            and np.abs(np.asarray(q) - s.q).max() < np.deg2rad(0.3)), None)
            line = (f"  step {s.index}: correction {np.rad2deg(s.dtheta):+6.2f} deg; the truth "
                    f"needed {np.rad2deg(need_pred):+6.2f} deg at the predicted arrival")
            if arrived is not None:
                need_real = wrap(self.truth.theta_at(arrived) + self.yaw_true
                                 - self.model.base_yaw - planned)
                line += (f", {np.rad2deg(need_real):+6.2f} deg at the real one "
                         f"({arrived - s.t_arrival:+.2f} s late)")
            log(line)
        # 4. The abort: which check of the correction failed.
        reason = getattr(result, "reason", "") if result else ""
        if reason.startswith("step ") and "correction of" in reason:
            from solver.correction import (MAX_CORRECTION, MAX_JOINT_CHANGE, ik_numerical,
                                           rotation_about_axis)
            k = int(reason.split()[1].rstrip(":"))
            dtheta = np.deg2rad(float(reason.split("correction of")[1].split()[0]))
            q_plan = plan.steps[k].q
            T = rotation_about_axis(self.model.axis_xy, dtheta) @ robot.fk(q_plan)
            q, converged = ik_numerical(robot, T, q_plan)
            change = np.abs(q - q_plan)
            log(f"  abort at step {k} ({np.rad2deg(dtheta):+.1f} deg): |correction| <= "
                f"{np.rad2deg(MAX_CORRECTION):.0f} deg {abs(dtheta) <= MAX_CORRECTION}; IK converged "
                f"{converged}; within limits {bool(robot.in_limits(q))}; largest joint change "
                f"{np.rad2deg(change.max()):.1f} deg (joint {int(change.argmax()) + 1}, limit "
                f"{np.rad2deg(MAX_JOINT_CHANGE):.0f})")
            if self.estimates:
                w, t, err = self.estimates[-1]
                log(f"  the tracker's last estimate before the end: error {np.rad2deg(err):+.2f} deg")


def phase3_execute(robot, model: TurntableModel, plan: Plan, path: PuzzlePath, log: Log,
                   truth: Truth | None, record: dict, tracker_kind: str = "fitted",
                   calibration: Homography | None = None,
                   out: Path | None = None, t_base: float = T_SCRIPT) -> ExecutionResult:
    """Executor with a tracker; then, on the simulator, the hoop vs the true puzzle."""
    camera = watch_stop = None
    probe = ExecutionProbe(robot, model, truth) if truth is not None else None
    if tracker_kind == "camera":
        from advanced_aruco import AdvancedArucoFinder
        from camera_phase_tracker import CameraPhaseTracker
        log("P3.1: tracker = CameraPhaseTracker (Vojtech): measures the base while it works")
        camera = CameraPhaseTracker(AdvancedArucoFinder(ArucoBlockDetector(), calibration, model),
                                    robot.grab_image, robot.camera_delay)
        camera.__enter__()
        if not camera.wait_ready(timeout=15.0):
            camera.__exit__(None, None, None)
            raise PhaseFailed("camera tracker: no confirmed rotation within 15 s")
        log("  camera tracker ready")
        watch_stop = threading.Event()
        threading.Thread(target=watch_tracker, args=(camera, watch_stop, log), daemon=True).start()
        tracker = FreezingTracker(camera, model)
    else:
        log("P3.1: tracker = ModelTracker(fitted model)  [no updates]")
        tracker = ModelTracker(model)
    if probe is not None:
        probe.watch(camera if camera is not None else tracker)
    executor = FreezingExecutor(robot, model, plan, tracker, RobotTiming(robot, plan.speed),
                                log=log, deadline=t_base + RUN_LIMIT)
    result = None
    try:
        result = executor.run()
    except Exception as error:  # e.g. simulation.CollisionError
        record["collision"] = f"{type(error).__name__}: {error}"
        raise PhaseFailed(record["collision"]) from error
    finally:
        if probe is not None:
            probe.stop()
        if camera is not None:
            watch_stop.set()
            camera.__exit__(None, None, None)
        if probe is not None:
            probe.report(robot, plan, result, log, out or RUNS)
            probe.dump(robot, plan, result, executor, (out or RUNS) / "execution.pkl")
            log(f"  saved for experiments/execution_replay.py: {(out or RUNS) / 'execution.pkl'}")
        if truth is not None:
            evaluate_traversal(robot, path, executor.t_attempt,
                               result.steps if result else [], truth, log, record)
    record.update(success=result.success, reason=result.reason, steps_sent=len(result.steps))
    if result.t_end is not None:
        total = result.t_end - t_base
        record["end_after_start_s"] = round(total, 1)
        log(f"Time: the traversal ended {total:.1f} s after the start "
            f"({'within' if total <= RUN_LIMIT else 'OVER'} the {RUN_LIMIT:.0f} s limit)")
    if not result.success:
        raise PhaseFailed(f"executor: {result.reason}")
    return result


def evaluate_traversal(robot, path: PuzzlePath, t_attempt: float, steps: list,
                       truth: Truth, log: Log, record: dict) -> None:
    """The true hoop motion against the true, spinning puzzle."""
    true = truth.model()
    tool = ring_in_flange()
    points = path.positions  # checkpoint positions, puzzle frame
    samples = [(t, q) for t, q in truth.track if t >= t_attempt]
    if not samples:
        return
    dist = []
    for t, q in samples:  # ring centre in the true puzzle frame -> distance to the path
        ring = (robot.fk(q) @ tool)[:3, 3]
        P = true.puzzle_T(true.theta_at(t))
        local = (np.linalg.inv(P) @ np.append(ring, 1.0))[:3]
        a, b = points[:-1], points[1:]
        u = np.clip(np.einsum("ij,ij->i", local - a, b - a)
                    / np.einsum("ij,ij->i", b - a, b - a), 0, 1)
        dist.append(np.linalg.norm(a + u[:, None] * (b - a) - local, axis=1).min())
    dist = np.array(dist)
    near = dist < 0.05  # while the ring is around the path at all
    if near.any():
        log(f"Truth: ring centre off the path while within 5 cm of it: max {mm(dist[near].max())}, "
            f"median {mm(np.median(dist[near]))} (the ring's inner radius is 30.0 mm)")
        record["ring_off_path_max_mm"] = 1000 * float(dist[near].max())
    else:
        log("Truth: the ring never came within 5 cm of the path")
    for s in steps:
        theta_true = true.theta_at(s.t_arrival)
        log(f"  step {s.index:2d}: planned theta {deg(s.theta_planned)}, predicted "
            f"{deg(s.theta_predicted)}, true {deg(theta_true)} (correction {deg(s.dtheta)})")
    t_end, q_end = samples[-1]
    ring = (robot.fk(q_end) @ tool)[:3, 3]
    height = ring[2] - (TABLE_Z + MARKER_HEIGHT_ABOVE_TABLE_M)  # base top = marker plane
    log(f"Truth: final ring centre {mm(height)} above the base top (success < {mm(FINAL_HEIGHT)})")
    record["final_height_mm"] = 1000 * float(height)
    top = ring_top_above_pad(np.linalg.inv(true.puzzle_T(true.theta_at(t_end)))
                             @ robot.fk(q_end) @ tool)
    log(f"Truth: whole ring's top {mm(top)} above the pad -> "
        f"{'SUCCESS' if top < SUCCESS_HEIGHT else 'NOT DONE'} (< {mm(SUCCESS_HEIGHT)})")
    record["ring_top_above_pad_mm"] = 1000 * float(top)


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="random placement (simulator)")
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1,
                        help="spin direction (simulator): +1 CCW, -1 CW from above")
    parser.add_argument("--axis", type=float, nargs=2, metavar=("X", "Y"),
                        help="turntable axis [m] instead of the seed's placement (simulator)")
    parser.add_argument("--offset", type=float, nargs=2, metavar=("X", "Y"), default=(0.02, 0.0),
                        help="base offset from the axis [m] (with --axis)")
    parser.add_argument("--yaw", type=float, default=0.0, help="base yaw [deg] (with --axis)")
    parser.add_argument("--speed", type=float, default=0.5, help="relative robot speed")
    parser.add_argument("--tracker", choices=("camera", "fitted"), default="camera",
                        help="phase 3: Vojtech's camera tracker or the fitted model alone")
    parser.add_argument("--frames", type=int, default=8, help="images for phase 1")
    parser.add_argument("--interval", type=float, default=1.0, help="s between images")
    parser.add_argument("--pre-entry", type=float, default=0.0,
                        help="copy planner: start this far [m] before checkpoint 0 "
                             "(0 = at checkpoint 0, already outside the bar)")
    parser.add_argument("--continuity", type=float, nargs="*", default=[0.5, 2.0, 20.0],
                        metavar="ALPHA BETA_DEG [CAP_DEG]",
                        help="copy planner: first-order continuity limit; no values = off")
    parser.add_argument("--margin-floor", type=float, default=0.0,
                        help="copy planner: keep only candidates with margin above this")
    parser.add_argument("--rank", choices=("closest", "margin"), default="closest",
                        help="copy planner: closest configurations first, or the "
                             "original largest worst margin first")
    parser.add_argument("--speed-cap", action="store_true",
                        help="copy planner: limit joint steps to v_max * dt of the "
                             "previous fixed-point pass")
    parser.add_argument("--phases", type=int, default=PLAN_PHASES,
                        help="start phases tried by the planner (even grid over a turn)")
    parser.add_argument("--planner", choices=("prokop", "natka", "copy"), default="prokop",
                        help="phase 2: Prokop's planner, Natka/, or the experimental "
                             "Prokop/natka_copy/")
    parser.add_argument("--stop-after", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--calibration", type=Path,
                        help="Homography JSON (default on the simulator: from its camera)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hold", action="store_true", help="keep the viewer open at the end")
    parser.add_argument("--real", action="store_true", help="the real robot (untested)")
    args = parser.parse_args()

    out = RUNS / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt")
    summary: dict = {"args": {k: str(v) for k, v in vars(args).items()}}
    log(f"Run folder: {out}")

    truth = None
    t_base = T_SCRIPT  # the real robot: from the program start
    if args.real:
        if args.calibration is None:
            parser.error("--real needs --calibration")
        robot = CRS93()
        robot.initialize(home=False)  # never home from a script
        # ctu_crs.CRS93 has no camera_delay (the simulator's attribute): the measured
        # age of an image when grab_image returns (docs/EXPERIMENTS.md E8).
        from simulation import CAMERA_DELAY
        robot.camera_delay = CAMERA_DELAY
        if np.abs(robot.get_q() - robot.q_home).max() > np.deg2rad(2):
            parser.error("the robot must stand at q_home (home it by hand first)")
    else:  # the simulator in its own process: only the robot interface for the pipeline
        from placement import TaskScene
        from sim_process import SimProcess
        t_sim = time.monotonic()
        sim = SimProcess(port=args.port)
        robot = sim.robot
        robot.initialize(home=False)
        from placement import Placement
        placement = (TaskScene().random_placement(np.random.default_rng(args.seed), WORK_AREA_XY)
                     if args.axis is None else
                     Placement(np.array(args.axis), np.array(args.offset), np.deg2rad(args.yaw)))
        sim.set_placement(placement)
        sim.set_spin(args.direction * OMEGA)
        truth = Truth(sim, placement, args.direction)
        # The simulator's own start-up does not exist on the real robot: not counted.
        t_base = T_SCRIPT + (time.monotonic() - t_sim)
        log(f"Time: the simulator took {time.monotonic() - t_sim:.1f} s to start (not "
            f"counted in the {RUN_LIMIT:.0f} s); {t_sim - T_SCRIPT:.1f} s before it (imports)")
        log(f"Simulator: http://localhost:{args.port}; placement seed {args.seed}: turntable "
            f"{np.round(placement.turntable_xy, 3)}, base offset {np.round(placement.base_xy, 3)}, "
            f"yaw {deg(placement.base_yaw)}, spin {args.direction:+d}")
        summary["placement"] = {"turntable_xy": np.asarray(placement.turntable_xy).tolist(),
                                "base_xy": np.asarray(placement.base_xy).tolist(),
                                "base_yaw": float(placement.base_yaw)}
    kin = CRS93(tty_dev=None)  # kinematics for planning (no robot calls)
    calibration = (Homography.load(args.calibration) if args.calibration
                   else simulated_calibration())

    failed = None
    try:
        with phase(log, "Phase 1: turntable model", summary) as rec:
            model = phase1_turntable(robot, calibration, args.frames, args.interval, out, log,
                                     truth, rec)
        if args.stop_after >= 2:
            with phase(log, "Phase 2: plan", summary) as rec:
                if args.planner == "prokop":
                    plan, path = phase2_prokop(
                        kin, model, args.speed, log, rec, t_base,
                        summary["phases"]["Phase 1: turntable model"].get("fit_for_margins"))
                else:
                    plan, path = phase2_plan(kin, model, args.speed, args.phases, args.planner, args.pre_entry,
                                         args.continuity, log, rec, args.margin_floor,
                                         args.rank == "closest", args.speed_cap)
        if args.stop_after >= 3:
            with phase(log, "Phase 3: execution", summary) as rec:
                phase3_execute(robot, model, plan, path, log, truth, rec, args.tracker,
                               calibration, out, t_base)
    except PhaseFailed as error:
        failed = str(error)
    finally:
        summary["result"] = "stopped: " + failed if failed else "all phases ran"
        log(f"\n=== Result: {summary['result']} ===")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        if truth is not None:
            truth.stop()
        if args.hold and not args.real:
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input("Viewer still running; press Enter to quit. ")
        if not args.real:
            sim.stop()


if __name__ == "__main__":
    main()
