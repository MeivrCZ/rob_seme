"""Shows a plan of Prokop's planner (solver.planner) in the simulator.

Puts a static puzzle facing the robot into the simulator (sim_process.SimProcess:
CRS93Sim in its own process, the truth), measures
it like the real pipeline (camera images -> Vojtech's ArUco detection and
measure_base -> a static TurntableModel), plans on that ESTIMATE, then runs
the plan on the simulated robot with the viewer (http://localhost:PORT):

1. the approach from q_home to checkpoint 0 (solver.planner.plan_approach:
   a direct move or two through a pre-entry pose, clear of the scene and of
   the volume the spinning puzzle sweeps);
2. a pause, then the traversal: every step queued with move_to_q, as the
   executor would send them;
3. a pause at the end, then back out along the same path and home again,
   and over again (--loops).

The simulator's hard collision monitor (exact meshes, true placement) is the
independent judge: on contact it stops the robot and shows the bodies red.
Before running, the plan is also replayed against the exact meshes at the
true placement.

Static perception cannot see the turntable's axis (that needs the spin): the
model puts the axis at the measured base centre, the angle = the measured
yaw (exact for a base centred on the turntable, as here).

    uv run python Prokop/experiments/show_plan.py [--axis X Y] [--loops N] [--port P]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))
sys.path.insert(0, str(PROKOP / "experiments"))

from ctu_crs import CRS93  # noqa: E402
from placement import Placement  # noqa: E402
from sim_process import SimProcess  # noqa: E402
from simulation import CollisionError  # noqa: E402
from static_plan import replay_exact  # noqa: E402
from static_puzzle import facing_model  # noqa: E402
from run_solver import simulated_calibration  # noqa: E402
from aruco_block_detector import ArucoBlockDetector  # noqa: E402
from turntable_orbit import MARKER_HEIGHT_ABOVE_TABLE_M, capture_series, measure_base  # noqa: E402

from solver.planner import FACE_NAMES, plan_approach, plan_tiers  # noqa: E402
from solver.soft_collision import SoftScene  # noqa: E402
from solver.timing import RobotTiming  # noqa: E402
from solver.contracts import TurntableModel  # noqa: E402


class PerceptionError(RuntimeError):
    """The base was not found in the images: the puzzle is not where it can be measured."""


def perceive(robot, images: int = 5) -> tuple[TurntableModel, float, list]:
    """Static TurntableModel and angle from camera images (module doc), and the measurements.

    Raises:
        PerceptionError: no image shows the base's markers (the puzzle is outside
            the camera's view, i.e. outside the work area): report unreachable.
    """
    # A perfect calibration of the simulator's camera model (on the station: from hoop images).
    calibration = simulated_calibration()
    detector = ArucoBlockDetector()
    marker_z = calibration.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
    found = [m for t, image in capture_series(robot.grab_image, images, 0.3, robot.camera_delay)
             if (m := measure_base(image, t, detector, calibration, marker_z)) is not None]
    if not found:
        raise PerceptionError(f"no base markers in {images} images: the puzzle is outside "
                              "the camera's view (the work area) - unreachable")
    xy = np.mean([m.xy for m in found], axis=0)
    yaw = float(np.angle(np.mean([np.exp(1j * m.yaw) for m in found])))
    model = TurntableModel(axis_xy=xy, base_offset=np.zeros(2), base_yaw=0.0, direction=1,
                           omega=0.0)
    # Make the model's base yaw at angle theta equal the measured yaw.
    theta = yaw - float(np.arctan2(*model.base_T(0.0)[[1, 0], 0]))
    return model, theta, found


def pose_error(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """Position [m] and rotation [rad] difference of two poses."""
    angle = np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1.0, 1.0))
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(angle)


def run(robot, qs: list[np.ndarray], pause: float = 0.0) -> None:
    """Queues the configurations, waits until the robot stands, then pauses."""
    for q in qs:
        robot.move_to_q(q)
    robot.wait_for_motion_stop()
    time.sleep(pause)


def plan_job(model: TurntableModel, theta: float, speed: float, true_model: TurntableModel,
             true_theta: float) -> dict:
    """Plans the traversal and the approach on the estimate; replays it at the truth."""
    kin = CRS93(tty_dev=None)
    timing = RobotTiming(kin, speed)
    result = plan_tiers(kin, timing, model, theta)[-1]
    if not result.collision_free:
        return {"error": f"no collision-free plan: {result.reason or 'moves collide'}"}
    approach, how = plan_approach(kin, SoftScene(model), model, theta, timing, kin.q_home,
                                  result.qs[0], result.entry)
    if approach is None:
        return {"error": f"no approach: {how}"}
    print(f"plan: tier {result.tier}, {result.checkpoints} checkpoints, side "
          f"{FACE_NAMES[result.face]}, entry {1000 * result.entry:.0f} mm before the open end, "
          f"traversal {result.arrivals[-1]:.1f} s, tilts "
          f"{np.round(np.rad2deg(result.orientations[:, 1])).astype(int).tolist()} deg; "
          f"q at checkpoint 0 {np.round(np.rad2deg(result.qs[0]), 1).tolist()} deg; "
          f"approach: {how}", flush=True)
    print("at the TRUE placement: ", end="", flush=True)
    replay_exact(true_model, true_theta, timing, result)
    return {"traversal": list(result.qs), "approach": approach}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", type=float, nargs=2, default=[0.487, 0.0], metavar=("X", "Y"),
                        help="turntable axis [m] (default: work-area centre)")
    parser.add_argument("--speed", type=float, default=0.2, help="relative robot speed")
    parser.add_argument("--loops", type=int, default=20, help="how many times to show it")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    true_model, true_theta = facing_model(np.array(args.axis))
    sim = SimProcess(port=args.port)
    sim.set_placement(Placement(np.asarray(true_model.axis_xy, float),
                                np.asarray(true_model.base_offset, float), true_model.base_yaw))
    sim.set_turntable(true_theta, 0.0)
    robot = sim.robot
    robot.initialize(home=False)
    robot.set_speed_relative(args.speed)

    try:
        model, theta, found = perceive(robot)
    except PerceptionError as error:
        sim.stop()
        sys.exit(f"perception: {error}")
    dp, da = pose_error(model.puzzle_T(theta), true_model.puzzle_T(true_theta))
    print(f"perception: {len(found)} images, base at {np.round(model.axis_xy, 4)} m, yaw "
          f"{np.rad2deg(theta):+.2f} deg; puzzle pose error vs truth {1000 * dp:.2f} mm, "
          f"{np.rad2deg(da):.2f} deg", flush=True)

    print("planning on the estimate ...", flush=True)
    t0 = time.monotonic()
    job = plan_job(model, theta, args.speed, true_model, true_theta)
    if "error" in job:
        sim.stop()
        sys.exit(job["error"])
    print(f"planned in {time.monotonic() - t0:.0f} s", flush=True)
    approach, traversal = job["approach"], job["traversal"]

    print(f"open http://localhost:{args.port} (Ctrl+C to quit)", flush=True)
    time.sleep(5.0)  # time to open the viewer

    try:
        for loop in range(1, args.loops + 1):
            print(f"loop {loop}: approach, traversal, back", flush=True)
            run(robot, approach[1:], pause=1.0)
            run(robot, traversal[1:], pause=2.0)
            run(robot, traversal[-2::-1], pause=0.5)
            run(robot, approach[-2::-1], pause=2.0)
    except CollisionError as error:
        print(f"COLLISION: {error}", flush=True)
    except KeyboardInterrupt:
        pass
    try:
        print("done; the viewer stays open (Ctrl+C to quit)", flush=True)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
