"""Simulated CRS93 shown in the Viser task scene, a drop-in for ctu_crs.CRS93.

    from simulation import CRS93Sim     # from ctu_crs import CRS93
    robot = CRS93Sim()                  # robot = CRS93()
    robot.initialize()
    robot.move_to_q(q)
    robot.wait_for_motion_stop()

then open http://localhost:8080. CRS93Sim subclasses CRS93, so kinematics, limits,
q_home, DH parameters etc. are the real ones; only the methods that talk to the
MARS control unit are replaced:
  - move_to_q returns at once and queues a coordinated movement like COORDMV:
    all joints start and stop together; queued moves run one after another,
    each from standstill (the real unit may blend them, see ASSUMED below); the
    queue is unlimited (the real one blocks move_to_q when full),
  - get_q / in_motion / wait_for_motion_stop follow the queued motion in real
    time (time_scale > 1 runs faster), with targets rounded to IRC like the real
    commands,
  - initialize(home=True) moves joint group by joint group in the hard home
    sequence to the hard home pose, where the control unit zeroes the encoders
    (IRC 0, ~(163.5, 0.3, -56.0, 176.8, 14.9, -172.0) deg; confirmed on the
    robot 2026-10-21), then to q_home (soft home). The real joints first drive to
    their home switches, on paths the sim does not know; there is no collision
    checking, so do not rely on the sim to judge whether a pose is safe to home
    from,
  - set_speed / set_acceleration (and the *_relative variants) set the motion
    profile in the same units as on the real robot,
  - the gripper is a stub, grab_image is not simulated (no camera model yet),
    release / reset_motors / close do nothing.
Hardware delays (sim seconds; script execution time is real anyway), defaults
measured on the CRS93 (2026-10-21, tools/log_motion.py), 0 for an ideal robot:
  - command_delay: from move_to_q until the control unit starts the move (the
    call itself returns at once, as the serial write does); 0.12 s,
  - query_delay: round trip of the joint position query; get_q blocks for it
    and returns the configuration from its middle; 0.05 s,
  - status_delay: round trip of the status query (ready / in motion); in_motion
    blocks for it and returns the state from its middle, wait_for_motion_stop
    returns status_delay / 2 after the motion ends (at least status_delay after
    the call); 0.01 s,
  - camera_delay: age of the image when grab_image returns (stored for the
    simulated camera, not used yet); not measured yet, < GRAB_TIME (the real
    grab_image takes 186 ms and the image is taken during the call).
One sim clock (time_scale) drives the robot, the delays and the turntable spin;
the render loop shows the whole scene at one time, atomically.
When the script ends, queued moves finish and the scene stays open (keep_alive)
until Ctrl+C. Each CRS93Sim is its own robot and viewer: scripts cannot share
one (like the real robot's single serial line); another instance needs its own
port.

ASSUMED, to be measured on the real robot: the speed and acceleration units
(speed IRC*256/ms as documented in ctu_crs; acceleration taken as IRC*256/ms^2,
which gives ~0.4 s ramps with the defaults), the trapezoidal velocity profile of
COORDMV and that consecutive queued moves stop in between.
"""

from __future__ import annotations

import atexit
import threading
import time
from dataclasses import dataclass

import numpy as np
from ctu_crs import CRS93
from numpy.typing import ArrayLike
from view_models import SceneViewer

RENDER_FPS = 30.0
# Measured defaults of the hardware delays [s] (see the module doc).
COMMAND_DELAY = 0.12
QUERY_DELAY = 0.05
STATUS_DELAY = 0.01
CAMERA_DELAY = 0.0  # PLACEHOLDER, to be measured (tools/camera_survey.py --delay)
# Real grab_image duration [s] (ctu_crs starts and stops the camera per call), the
# first call opening the camera; for the simulated camera.
GRAB_TIME = 0.186
FIRST_GRAB_TIME = 0.8
HISTORY = 1.0  # s, finished moves kept for state queries reaching into the past


@dataclass
class Segment:
    """One coordinated movement: all joints follow the same trapezoidal profile
    s(t) from @q_from to @q_to, starting at @t0 [s, sim clock], lasting @T with
    ramps of @tau."""

    t0: float
    T: float
    tau: float
    q_from: np.ndarray
    q_to: np.ndarray

    @property
    def t_end(self) -> float:
        return self.t0 + self.T

    def q_at(self, t: float) -> np.ndarray:
        if self.T <= 0:
            return self.q_to.copy()
        t = np.clip(t - self.t0, 0.0, self.T)
        tau, T = self.tau, self.T
        v = 1.0 / (T - tau)  # peak of ds/dt, normalized
        if t < tau:
            s = 0.5 * v / tau * t**2
        elif t <= T - tau:
            s = v * (t - tau / 2)
        else:
            s = 1.0 - 0.5 * v / tau * (T - t) ** 2
        return self.q_from + s * (self.q_to - self.q_from)


class SimGripper:
    """Stub of ctu_crs.gripper.Gripper (no gripper on the hoop task); keeps the
    commanded position only."""

    def __init__(self, bounds=None, axis=None):
        self.bounds = bounds
        self._axis = axis
        self.position = None
        self._initialized = False

    def initialize(self):
        self._initialized = True

    def control_position_relative(self, fraction: float):
        assert self._initialized, "Gripper controller must be initialized first."
        assert self.bounds is not None
        self.control_position(
            int(self.bounds[0] + (self.bounds[1] - self.bounds[0]) * fraction))

    def control_position(self, position: float):
        assert self._initialized, "Gripper controller must be initialized first."
        self.position = position

    def release(self):
        assert self._initialized, "Gripper controller must be initialized first."

    def wait_for_motion_stop(self):
        assert self._initialized, "Gripper controller must be initialized first."
        return True


class CRS93Sim(CRS93):
    def __init__(self, tty_dev: str | None = "/dev/mars", baudrate: int = 19200, *,
                 q0: ArrayLike | None = None, port: int = 8080,
                 time_scale: float = 1.0, keep_alive: bool = True,
                 command_delay: float = COMMAND_DELAY,
                 query_delay: float = QUERY_DELAY,
                 status_delay: float = STATUS_DELAY,
                 camera_delay: float = CAMERA_DELAY):
        """Same arguments as CRS93 (@tty_dev and @baudrate are ignored), plus the
        simulation ones: @q0 initial configuration (default q_home: the arm is
        wherever it was left, homing moves it anyway), Viser @port, @time_scale
        (sim seconds per wall second), @keep_alive (keep the scene open after
        the script ends) and the hardware delays [s] (see the module doc)."""
        super().__init__(tty_dev=None, baudrate=baudrate)
        self.gripper = SimGripper(self.gripper.bounds, self.gripper._axis)
        self._time_scale = time_scale
        self._t_start = time.monotonic()
        self.command_delay = command_delay
        self.query_delay = query_delay
        self.status_delay = status_delay
        self.camera_delay = camera_delay
        self._lock = threading.Lock()
        self._segments: list[Segment] = []
        self._q_rest = np.array(self.q_home if q0 is None else q0, dtype=float)
        self._speed = self._default_speed_irc256_per_ms.astype(float)
        self._acceleration = self._default_acceleration_irc_per_ms.astype(float)

        self.viewer = SceneViewer(port, clock=self._now)  # one clock for the scene
        self._add_status_gui()
        self.viewer.add_turntable_gui(animate=False)  # rendered by _render_loop
        self.viewer.show_q(self._q_rest)
        threading.Thread(target=self._render_loop, daemon=True).start()
        print(f"Simulation: open http://localhost:{self.viewer.server.get_port()}")
        if keep_alive:
            atexit.register(self._keep_alive)

    # --- simulated control unit ---

    def _now(self) -> float:
        """Sim time [s] since start: robot motion, delays and the turntable."""
        return (time.monotonic() - self._t_start) * self._time_scale

    def _sleep(self, dt: float) -> None:
        """Block for @dt sim seconds."""
        if dt > 0:
            time.sleep(dt / self._time_scale)

    def _q_at(self, t: float) -> np.ndarray:
        """Configuration at sim time @t (not older than HISTORY)."""
        with self._lock:
            q = self._q_rest
            for seg in self._segments:
                if t < seg.t0:
                    break
                if t < seg.t_end:
                    return seg.q_at(t)
                q = seg.q_to
            return q.copy()

    def _in_motion_at(self, t: float) -> bool:
        """Moving or a move queued (also not started yet) at sim time @t."""
        with self._lock:
            return bool(self._segments) and self._segments[-1].t_end > t

    def _motion_end(self) -> float:
        with self._lock:
            return self._segments[-1].t_end if self._segments else -np.inf

    def _joint_limits_rad(self) -> tuple[np.ndarray, np.ndarray]:
        """Max joint speed [rad/s] and acceleration [rad/s^2] from the IRC settings
        (units ASSUMED, see the module doc)."""
        irc_per_rad = np.abs(self._deg_to_irc) * 180 / np.pi
        return (self._speed * 1000 / 256 / irc_per_rad,
                self._acceleration * 1e6 / 256 / irc_per_rad)

    def _enqueue(self, q: ArrayLike) -> None:
        """Queue a coordinated movement to @q (rounded to IRC like the real
        command). The duration is the shortest trapezoidal move of the slowest
        joint; the other joints are scaled to the same profile."""
        q_to = self._irc_to_joint_values(self._joint_values_to_irc(q))
        v_max, a_max = self._joint_limits_rad()
        now = self._now()
        with self._lock:
            while self._segments and self._segments[0].t_end < now - HISTORY:
                self._q_rest = self._segments.pop(0).q_to
            last = self._segments[-1] if self._segments else None
            q_from = last.q_to if last else self._q_rest
            t0 = now + self.command_delay
            if last:
                t0 = max(t0, last.t_end)
            d = np.abs(q_to - q_from)
            ramp = v_max / a_max
            T_joint = np.where(d > v_max * ramp, d / v_max + ramp,
                               2 * np.sqrt(d / a_max))
            k = int(np.argmax(T_joint))
            T = float(T_joint[k])
            tau = min(float(ramp[k]), T / 2)
            self._segments.append(Segment(t0, T, tau, q_from.copy(), q_to))

    def _stop(self) -> None:
        """Stop at the current configuration and drop the queued motion."""
        q = self._q_at(self._now())
        with self._lock:
            self._segments.clear()
            self._q_rest = q

    # --- CRSRobot interface ---

    def grab_image(self, timeout=10000):
        raise NotImplementedError("CRS93Sim: the camera is not simulated yet")

    def release(self):
        pass

    def reset_motors(self):
        pass

    def close(self):
        pass

    def initialize(self, home: bool = True):
        self._stop()
        self.set_speed(self._default_speed_irc256_per_ms)
        self.set_acceleration(self._default_acceleration_irc_per_ms)
        self.gripper.initialize()
        if home:
            self.hard_home()
            self.soft_home()
        self._initialized = True

    def set_speed(self, speed_irc256_ms: ArrayLike):
        speed = np.asarray(speed_irc256_ms, dtype=float)
        assert len(speed) == len(self._motors_ids)
        self._speed = np.rint(speed)

    def set_acceleration(self, acceleration_irc_ms: ArrayLike):
        acceleration = np.asarray(acceleration_irc_ms, dtype=float)
        assert len(acceleration) == len(self._motors_ids)
        self._acceleration = np.rint(acceleration)

    def hard_home(self):
        self.set_speed(self._default_speed_irc256_per_ms)
        self.set_acceleration(self._default_acceleration_irc_per_ms)
        if len(self._hh_sequence) == 0:
            raise ValueError("The hard home sequence is not defined for this robot.")
        q = self._q_at(self._now())
        q_hh = self._irc_to_joint_values(np.zeros(len(self._motors_ids)))
        for blk in self._hh_sequence:
            for a in blk:
                i = self._motors_ids.index(a)
                q[i] = q_hh[i]
            self._enqueue(q)
            self.wait_for_motion_stop()

    def soft_home(self):
        self._enqueue(self.q_home)
        self.wait_for_motion_stop()

    def move_to_q(self, q: ArrayLike):
        assert self._initialized, "You need to initialize the robot before moving it."
        assert self.in_limits(q), "Joint limits violated."
        self._enqueue(q)

    def get_q(self) -> np.ndarray:
        t = self._now() + self.query_delay / 2  # the unit answers here
        self._sleep(self.query_delay)
        return self._q_at(t)

    def in_motion(self) -> bool:
        t = self._now() + self.status_delay / 2
        self._sleep(self.status_delay)
        return self._in_motion_at(t)

    def wait_for_motion_stop(self):
        t_return = self._now() + self.status_delay
        while True:  # moves may be queued meanwhile (other threads)
            dt = max(t_return, self._motion_end() + self.status_delay / 2) - self._now()
            if dt <= 0:
                break
            self._sleep(dt)

    # --- viewer ---

    def _add_status_gui(self) -> None:
        with self.viewer.server.gui.add_folder("CRS93 simulation"):
            self._status = self.viewer.server.gui.add_markdown("")

    def _render_loop(self) -> None:
        shown = None
        while True:
            now = self._now()
            q = self._q_at(now)
            moved = shown is None or not np.array_equal(q, shown)
            self.viewer.show(now, q if moved else None)  # whole scene at one time
            shown = q
            with self._lock:
                queued = sum(seg.t_end > now for seg in self._segments)
            self._status.content = (
                f"q [deg]: {np.array2string(np.rad2deg(q), precision=1)}  \n"
                f"flange xyz [m]: {np.array2string(self.fk(q)[:3, 3], precision=3)}  \n"
                f"initialized: {self._initialized}, "
                f"in motion: {self._in_motion_at(now)}, queued moves: {queued}")
            time.sleep(1 / RENDER_FPS)

    def _keep_alive(self) -> None:
        """At exit: let the queued motion finish (the real arm would too), then keep
        the scene open until Ctrl+C."""
        try:
            self.wait_for_motion_stop()
            print("Script finished, simulation still open (Ctrl+C to quit)")
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
