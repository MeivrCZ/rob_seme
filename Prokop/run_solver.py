"""Runs the whole solver (variant D) phase by phase on the simulator, with diagnostics.

Calls the phases in order and hands each one the previous phase's
contract types (solver/contracts.py). Every phase prints what it produced
and, on the simulator, how far that is from the truth. The run stops at the
first phase that fails, so the output shows where the pipeline breaks.

    Phase 1  Turntable model   images -> BaseMeasurement (Vojtech, P1.2)
                               -> TurntableModel (Vojtech, P1.3 fit_orbit)
    Phase 2  Plan              PuzzlePath (Natka, P2.1) -> timed Plan (Natka,
                               P2.2-P2.6) -> approach line (P2.7)
    Phase 3  Execution         Executor (P3.3) with a PhaseTracker (P3.1 is
                               not done yet: ModelTracker on the fitted model)

The pipeline uses only the ctu_crs CRS93 interface (move_to_q, get_q,
grab_image, fk, ik, ...). The simulator runs standalone in its own process
(sim_process.SimProcess): the pipeline gets only its robot interface; the scene
setup and the Truth class (the ground truth that judges the results, read
through the SimProcess) belong to this runner, nothing in the pipeline uses
them. On the real robot, `--real` creates CRS93() instead and skips the truth
checks (untested).

Usage (from the repo root; the viewer runs on http://localhost:8080):

    uv run Prokop/run_solver.py --seed 3 --direction -1
    uv run Prokop/run_solver.py --stop-after 2      # only perception + planning
    uv run Prokop/run_solver.py --hold              # keep the viewer open at the end

Each run writes its log, the camera frames and a summary.json to
Prokop/runs/<time>/.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
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
from solver.executor import ExecutionResult, Executor  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402

RUNS = HERE / "runs"
PLAN_PHASES = 6  # start phases handed to the planner (an even grid over a turn)
FINAL_HEIGHT = 0.04  # m, success: hoop less than this above the base (assignment)


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


def phase3_execute(robot, model: TurntableModel, plan: Plan, path: PuzzlePath, log: Log,
                   truth: Truth | None, record: dict) -> ExecutionResult:
    """Executor with a tracker; then, on the simulator, the hoop vs the true puzzle."""
    log("P3.1: tracker = ModelTracker(fitted model)  [the Kalman tracker is not done yet]")
    tracker = ModelTracker(model)
    executor = Executor(robot, model, plan, tracker, RobotTiming(robot, plan.speed), log=log)
    result = None
    try:
        result = executor.run()
    except Exception as error:  # e.g. simulation.CollisionError
        record["collision"] = f"{type(error).__name__}: {error}"
        raise PhaseFailed(record["collision"]) from error
    finally:
        if truth is not None:
            evaluate_traversal(robot, path, executor.t_attempt,
                               result.steps if result else [], truth, log, record)
    record.update(success=result.success, reason=result.reason, steps_sent=len(result.steps))
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


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="random placement (simulator)")
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1,
                        help="spin direction (simulator): +1 CCW, -1 CW from above")
    parser.add_argument("--speed", type=float, default=0.2, help="relative robot speed")
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
    parser.add_argument("--planner", choices=("natka", "copy"), default="natka",
                        help="P2.2-P2.6: Natka/ or the experimental Prokop/natka_copy/")
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
    if args.real:
        if args.calibration is None:
            parser.error("--real needs --calibration")
        robot = CRS93()
        robot.initialize(home=False)  # never home from a script
        if np.abs(robot.get_q() - robot.q_home).max() > np.deg2rad(2):
            parser.error("the robot must stand at q_home (home it by hand first)")
    else:  # the simulator in its own process: only the robot interface for the pipeline
        from placement import TaskScene
        from sim_process import SimProcess
        sim = SimProcess(port=args.port)
        robot = sim.robot
        robot.initialize(home=False)
        placement = TaskScene().random_placement(np.random.default_rng(args.seed), WORK_AREA_XY)
        sim.set_placement(placement)
        sim.set_spin(args.direction * OMEGA)
        truth = Truth(sim, placement, args.direction)
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
                plan, path = phase2_plan(kin, model, args.speed, args.phases, args.planner, args.pre_entry,
                                         args.continuity, log, rec, args.margin_floor,
                                         args.rank == "closest", args.speed_cap)
        if args.stop_after >= 3:
            with phase(log, "Phase 3: execution", summary) as rec:
                phase3_execute(robot, model, plan, path, log, truth, rec)
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
