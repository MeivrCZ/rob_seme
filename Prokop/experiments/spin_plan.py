"""The spinning puzzle end to end: perception, planning, execution, evaluation.

On the standalone simulator (sim_process.SimProcess), with a random placement
in the work area and the turntable spinning:

1. perception: images while it spins -> ArUco base measurements (Vojtech's
   measure_base) -> orbit fit (fit_orbit): the TurntableModel, speed included;
2. planning on the fitted model: solver.planner.plan_spinning (one start
   angle: the puzzle facing the robot half way through; RRT between the
   checkpoints, plan_rrt) and plan_approach;
3. execution: solver.executor.Executor with a PhaseTracker: ModelTracker on
   the fitted model (the Kalman tracker P3.1 is not done yet: no updates), or
   with --tracker truth a perfect tracker read from the simulator;
4. evaluation: how far the tracker's angle is from the truth at the start, the
   simulator's collision monitor, the ring's height above the pad at the end.

    uv run python Prokop/experiments/spin_plan.py [--seed N] [--direction 1|-1]
        [--tracker fitted|truth] [--port P]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))
sys.path.insert(0, str(PROKOP / "experiments"))

from run_solver import simulated_calibration  # noqa: E402  (puts Vojtech's dirs on the path)
from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from cell import WORK_AREA_XY  # noqa: E402
from ctu_crs import CRS93  # noqa: E402
from placement import TaskScene  # noqa: E402
from sim_process import SimProcess  # noqa: E402
from simulation import CollisionError  # noqa: E402
from turntable_orbit import (MARKER_HEIGHT_ABOVE_TABLE_M, capture_series,  # noqa: E402
                             fit_orbit, measure_base)

from solver.contracts import OMEGA, ModelTracker, TurntableModel  # noqa: E402
from solver.executor import Executor  # noqa: E402
from solver.hoop_pose import ring_in_flange  # noqa: E402
from solver import planner  # noqa: E402
from solver.planner import FACE_NAMES, plan_approach, plan_spinning  # noqa: E402
from solver.soft_collision import SoftScene  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from solver.wire import SUCCESS_HEIGHT, ring_top_above_pad  # noqa: E402


def wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class TruthTracker:
    """A perfect PhaseTracker read from the simulator (to separate tracking errors).

    Reports the true rotation in the FITTED model's convention: a fit sets the
    angle's zero at its own reference (and puts the base's yaw into base_yaw),
    so the angle that gives the true base orientation with the fitted base_yaw
    is the true angle + true base_yaw - fitted base_yaw.
    """

    def __init__(self, sim, direction: int, true_base_yaw: float, model: TurntableModel) -> None:
        self.sim, self.direction = sim, direction
        self.offset = true_base_yaw - model.base_yaw

    def estimate(self):
        from solver.contracts import PhaseEstimate
        t = time.monotonic()
        return PhaseEstimate(t=t, theta=self.sim.truth_angle(t) + self.offset,
                             omega=self.direction * OMEGA, covariance=np.zeros((2, 2)))


def yaw_error(model: TurntableModel, truth: TurntableModel, t: float) -> float:
    """Error of the model's base orientation (angle + base yaw) against the truth at t [rad]."""
    return wrap(model.theta_at(t) + model.base_yaw - truth.theta_at(t) - truth.base_yaw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1, help="random placement")
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--speed", type=float, default=0.5, help="relative robot speed")
    parser.add_argument("--tracker", choices=("fitted", "truth"), default="fitted")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, args.speed)
    placement = TaskScene().random_placement(np.random.default_rng(args.seed), WORK_AREA_XY)
    sim = SimProcess(port=args.port)
    robot = sim.robot
    robot.initialize(home=False)
    robot.set_speed_relative(args.speed)
    sim.set_placement(placement)
    sim.set_spin(args.direction * OMEGA)
    print(f"placement seed {args.seed}: turntable {np.round(placement.turntable_xy, 3)}, base "
          f"offset {np.round(placement.base_xy, 3)}, spin {args.direction:+d}; viewer "
          f"http://localhost:{args.port}", flush=True)

    def true_model(t: float) -> TurntableModel:
        return TurntableModel(axis_xy=np.asarray(placement.turntable_xy, float),
                              base_offset=np.asarray(placement.base_xy, float),
                              base_yaw=float(placement.base_yaw), direction=args.direction,
                              omega=OMEGA, theta_ref=sim.truth_angle(t), t_ref=t)

    try:
        # 1. Perception.
        calibration = simulated_calibration()
        detector = ArucoBlockDetector()
        marker_z = calibration.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
        found = [m for t, image in capture_series(robot.grab_image, 8, 1.0, robot.camera_delay)
                 if (m := measure_base(image, t, detector, calibration, marker_z)) is not None]
        if len(found) < 4:
            print(f"UNREACHABLE: perception: the base was found in {len(found)} of 8 images "
                  f"(outside the camera's view)", flush=True)
            return
        model = fit_orbit(found).model
        t_fit = time.monotonic()
        truth = true_model(t_fit)
        print(f"perception: {len(found)}/8 images; fitted axis {np.round(model.axis_xy, 4)} "
              f"(truth {np.round(truth.axis_xy, 4)}), period {2 * np.pi / model.omega:.2f} s "
              f"(truth {2 * np.pi / OMEGA:.2f} s), direction {model.direction:+d}; base "
              f"orientation error now {np.rad2deg(yaw_error(model, truth, t_fit)):+.2f} deg",
              flush=True)

        # 2. Planning.
        t0 = time.monotonic()
        planner.PASS_LOG = lambda line: print(f"  [{time.monotonic() - t0:4.0f} s] {line}",
                                              flush=True)

        def progress(theta, r):
            what = "collision free" if r.collision_free else (r.reason or "moves collide")
            print(f"  start angle {np.rad2deg(theta):+6.1f} deg: {what} "
                  f"({r.seconds:.0f} s)", flush=True)

        theta_start, result = plan_spinning(kin, timing, model, progress=progress)
        if result is None:
            print(f"UNREACHABLE: no start angle gives a collision-free traversal "
                  f"({time.monotonic() - t0:.0f} s)", flush=True)
            return
        approach, how = plan_approach(kin, SoftScene(model), model, theta_start, timing,
                                      kin.q_home, result.qs[0], result.entry)
        if approach is None:
            print(f"UNREACHABLE: approach: {how}", flush=True)
            return
        print(f"plan ({time.monotonic() - t0:.0f} s): start angle {np.rad2deg(theta_start):+.1f} "
              f"deg, {len(result.qs)} waypoints, {result.checkpoints} checkpoints, side "
              f"{FACE_NAMES[result.face]}, entry {1000 * result.entry:.0f} mm, traversal "
              f"{result.arrivals[-1]:.1f} s, tilts "
              f"{np.round(np.rad2deg(result.orientations[:, 1])).astype(int).tolist()} deg; "
              f"approach: {how}", flush=True)

        # 3. Execution: the executor sends the traversal steps (not the start pose) and
        # moves through the approach's pre-entry pose(s) to the start.
        plan = replace(result.plan, steps=result.plan.steps[1:],
                       approach=[np.asarray(q) for q in approach[1:-1]])
        tracker = (ModelTracker(model) if args.tracker == "fitted"
                   else TruthTracker(sim, args.direction, float(placement.base_yaw), model))
        executor = Executor(robot, model, plan, tracker, timing,
                            log=lambda s: print("  executor: " + s, flush=True))
        collision = None
        try:
            outcome = executor.run()
        except CollisionError as error:
            collision, outcome = error, None
        t_end = time.monotonic()

        # 4. Evaluation.
        if outcome is not None and outcome.t_start is not None:
            t_go = outcome.t_start
            drift = yaw_error(model, true_model(t_go), t_go)
            print(f"start: {t_go - t_fit:.0f} s after the fit; the fitted model's base "
                  f"orientation was off by {np.rad2deg(drift):+.2f} deg at the start", flush=True)
        true_now = true_model(t_end)
        ring = np.linalg.inv(true_now.puzzle_T(true_now.theta_at(t_end))) \
            @ kin.fk(robot.get_q() if collision is None else collision.q_stop) @ ring_in_flange()
        height = ring_top_above_pad(ring)
        if collision is not None:
            print(f"COLLISION: {collision}", flush=True)
        elif not outcome.success:
            print(f"ABORTED: {outcome.reason}", flush=True)
        print(f"end: ring top {1000 * height:.1f} mm above the pad (success < "
              f"{1000 * SUCCESS_HEIGHT:.0f} mm) -> "
              f"{'SUCCESS' if collision is None and outcome.success and height < SUCCESS_HEIGHT else 'NOT DONE'}",
              flush=True)
        print("the viewer stays open (Ctrl+C to quit)", flush=True)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
