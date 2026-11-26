"""The simulated robot in its own process: a standalone environment.

CRS93Sim (simulation.py) runs its scene, renderer, collision monitor and
viewer as threads. In the same process as the solver they compete with it for
Python's interpreter lock and slow planning down many times over. Here the
simulator runs in a separate process and the solver only gets what the real
station offers:

* `SimProcess.robot`: a `ctu_crs.CRS93` with the real interface. Kinematics
  (fk, ik, limits, q_home) are computed locally; the robot and camera commands
  (initialize, move_to_q, get_q, in_motion, wait_for_motion_stop, grab_image,
  speed and acceleration settings, homing, release) go to the simulator
  process. A hard collision there raises simulation.CollisionError here.
* The scene setup (placement, turntable angle and spin) and the ground truth
  for evaluation (true turntable angle, recorded true arm motion) are NOT on
  the robot: they belong to the test script, through the SimProcess itself.

Example:
    ::

        with SimProcess(port=8080) as sim:
            sim.set_placement(Placement(...))
            sim.set_turntable(angle=0.0, omega=0.0)
            robot = sim.robot          # use like ctu_crs.CRS93
            robot.initialize(home=False)
            robot.move_to_q(q)
            image = robot.grab_image()
"""

from __future__ import annotations

import multiprocessing
import threading
import time
import traceback

import numpy as np
from ctu_crs import CRS93

# Robot and camera commands forwarded to the simulator (everything else of the
# CRS93 interface, e.g. fk, ik, in_limits, set_speed_relative, runs locally).
COMMANDS = ("initialize", "move_to_q", "get_q", "in_motion", "wait_for_motion_stop",
            "grab_image", "set_speed", "set_acceleration", "soft_home", "hard_home",
            "release", "reset_motors")
ATTRIBUTES = ("camera_delay",)  # read once from the simulator
RECORD_DT = 0.02  # s, period of the recorded true arm configuration
RECORD_SPAN = 600.0  # s, how long the record reaches back


def _serve(conn, port: int, sim_kwargs: dict) -> None:
    """The simulator process: runs CRS93Sim and answers requests until "stop"."""
    from simulation import CRS93Sim
    robot = CRS93Sim(port=port, keep_alive=False, **sim_kwargs)
    viewer = robot.viewer

    def sim_time(t: float) -> float:  # time.monotonic() -> the simulator's clock
        return (t - robot._t_start) * robot._time_scale

    track: list[tuple[float, np.ndarray]] = []  # (time.monotonic(), true q)
    grab_sent: list[float] = []  # when each grab_image reply was sent (time.monotonic())
    running = True

    def record() -> None:
        while running:
            t = time.monotonic()
            track.append((t, robot._q_at(sim_time(t))))
            while track and track[0][0] < t - RECORD_SPAN:
                track.pop(0)
            time.sleep(RECORD_DT)

    threading.Thread(target=record, daemon=True).start()
    conn.send(("ready", {name: getattr(robot, name) for name in ATTRIBUTES}))
    while True:
        kind, name, args, kwargs = conn.recv()
        try:
            if kind == "stop":
                conn.send(("ok", None))
                break
            if kind == "robot":
                value = getattr(robot, name)(*args, **kwargs)
                if name == "grab_image":  # the reply's send time, for truth_grabs
                    conn.send(("ok", value))
                    grab_sent.append(time.monotonic())
                    continue
            elif name == "set_placement":
                value = viewer.set_placement(*args)
            elif name == "set_turntable":  # static angle and spin from now on
                angle, omega = args
                with viewer._spin_lock:
                    viewer._t_ref, viewer._angle_ref, viewer._omega = viewer.clock(), angle, omega
                for callback in viewer.on_change:
                    callback()
                value = None
            elif name == "set_spin":  # spin from now on, from the current angle
                value = viewer.set_spin(*args)
            elif name == "truth_angle":
                value = float(viewer.turntable_angle(sim_time(args[0])))
            elif name == "truth_track":
                value = [(t, q) for t, q in list(track) if t >= args[0]]
            elif name == "truth_grabs":  # grab_image timings, time.monotonic()
                mono = [tuple(robot._t_start + x / robot._time_scale for x in g)
                        for g in list(robot.grab_log)]
                value = [g + (sent,) for g, sent in zip(mono, grab_sent[-len(mono):])
                         if g[0] >= args[0]] if mono else []
            else:
                raise ValueError(f"unknown scene request {name}")
            conn.send(("ok", value))
        except Exception as error:  # report it to the caller; the simulator keeps running
            record = None
            if type(error).__name__ == "CollisionError":
                record = (error.time, error.hits, error.q_stop, error.q_start)
            conn.send(("error", (type(error).__name__, str(error), record,
                                 traceback.format_exc())))
    running = False
    robot.stop()


class _Link:
    """Request/response over a pipe (one caller at a time)."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    def call(self, kind: str, name: str, *args, **kwargs):
        with self._lock:
            self._conn.send((kind, name, args, kwargs))
            status, value = self._conn.recv()
        if status == "ok":
            return value
        type_name, message, record, trace = value
        if type_name == "CollisionError":
            from simulation import CollisionError, _Collision
            t, hits, q_stop, q_start = record
            raise CollisionError(message, _Collision(t, t, hits, q_stop, q_start))
        raise RuntimeError(f"simulator: {type_name}: {message}\n{trace}")


class RemoteCRS93(CRS93):
    """ctu_crs.CRS93 whose robot and camera commands go to the simulator process."""

    def __init__(self, link: _Link, attributes: dict) -> None:
        super().__init__(tty_dev=None)  # local kinematics, limits and settings
        self._link = link
        for name, value in attributes.items():
            setattr(self, name, value)

    def __getattribute__(self, name: str):
        if name in COMMANDS:
            link = object.__getattribute__(self, "_link")
            return lambda *args, **kwargs: link.call("robot", name, *args, **kwargs)
        return object.__getattribute__(self, name)


class SimProcess:
    """Starts CRS93Sim in its own process; `robot` is the station interface to it."""

    def __init__(self, port: int = 8080, **sim_kwargs) -> None:
        """Starts the simulator (keyword arguments go to CRS93Sim, e.g. time_scale)."""
        context = multiprocessing.get_context("spawn")
        here, there = context.Pipe()
        self._process = context.Process(target=_serve, args=(there, port, sim_kwargs),
                                        daemon=True)
        self._process.start()
        status, attributes = here.recv()
        assert status == "ready", status
        self._link = _Link(here)
        self.port = port
        self.robot = RemoteCRS93(self._link, attributes)

    # --- scene setup (for test scripts, not part of the robot interface) ---

    def set_placement(self, placement) -> None:
        """Places the turntable and the base (placement.Placement)."""
        self._link.call("scene", "set_placement", placement)

    def set_turntable(self, angle: float, omega: float = 0.0) -> None:
        """Turntable at @angle [rad] now, turning at @omega [rad/s] (0: standing still)."""
        self._link.call("scene", "set_turntable", float(angle), float(omega))

    def set_spin(self, omega: float) -> None:
        """Turntable turning at @omega [rad/s] from now on, from its current angle."""
        self._link.call("scene", "set_spin", float(omega))

    # --- ground truth (for evaluating a run, not for the solver) ---

    def truth_angle(self, t: float) -> float:
        """True turntable angle [rad] at time @t [s, time.monotonic()]."""
        return self._link.call("scene", "truth_angle", float(t))

    def truth_grabs(self, since: float = 0.0) -> list[tuple[float, ...]]:
        """grab_image timings since @since [s, time.monotonic()]: (image time, collision
        check done, rendered, planned return, real return, reply sent) per call."""
        return self._link.call("scene", "truth_grabs", float(since))

    def truth_track(self, since: float = 0.0) -> list[tuple[float, np.ndarray]]:
        """True arm configurations recorded since @since [s, time.monotonic()]."""
        return self._link.call("scene", "truth_track", float(since))

    # --- lifetime ---

    def stop(self) -> None:
        """Stops the simulator and its viewer."""
        if self._process.is_alive():
            try:
                self._link.call("stop", "")
            except (EOFError, BrokenPipeError):
                pass
            self._process.join(timeout=10)

    def __enter__(self) -> "SimProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _demo() -> None:
    """Smoke test: move a little, grab an image (uv run python Prokop/sim_process.py)."""
    with SimProcess() as sim:
        robot = sim.robot
        robot.initialize(home=False)
        q = robot.get_q()
        robot.move_to_q(q + np.deg2rad([10, 0, 0, 0, 0, 0]))
        robot.wait_for_motion_stop()
        print("q", np.round(np.rad2deg(robot.get_q()), 1), "image", robot.grab_image().shape)


if __name__ == "__main__":
    _demo()
