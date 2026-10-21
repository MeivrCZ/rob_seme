"""Simulated CRS93 in the Viser task scene, a drop-in replacement for ctu_crs.CRS93.

Example:
    Swap one line to run a script on the real robot instead::

        from simulation import CRS93Sim     # from ctu_crs import CRS93
        robot = CRS93Sim()                  # robot = CRS93()
        robot.initialize()
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

    then open http://localhost:8080.

CRS93Sim subclasses CRS93, so the kinematics (fk, ik), joint limits, q_home and
unit conversions are ctu_crs's own; only the methods that talk to the MARS
control unit are replaced. The replacements behave like the real robot as
measured on the CRS93 station (docs/EXPERIMENTS.md):

* move_to_q queues a coordinated move (all joints start and stop together, a
  trapezoidal speed profile per move) and blocks only for the serial write.
  A queued move that continues in exactly the same joint-space direction is
  blended into the previous one; any other starts from standstill (the real unit
  stops at corners and reversals and nearly stops at the vias of a bending path).
* get_q, in_motion and wait_for_motion_stop follow the motion on the sim clock,
  with the measured round-trip delays; targets are rounded to encoder counts
  (IRC) like the real commands.
* initialize(home=True) drives each joint group of the hard home sequence to the
  hard home pose (where the control unit zeroes the encoders), then to q_home.
  The real joints first drive to their home switches on paths the sim does not
  know; nothing is collision checked, so the sim cannot tell whether a pose is
  safe to home from.
* The gripper is a stub, grab_image is not simulated yet, release, reset_motors
  and close do nothing.

One sim clock (scaled by time_scale) drives the robot, the delays and the
turntable spin; the render loop shows the whole scene at one instant. Each
CRS93Sim is its own robot and viewer, like the real robot's single serial line:
another instance needs another port. When the script ends, queued moves finish
and the scene stays open until Ctrl+C (keep_alive).

Not simulated: the queue limit of the real unit (move_to_q blocks when full),
the ~0.3 deg overshoot and ~0.25 s settling at the end of a real move.
Acceleration units are assumed (IRC*256/ms^2) and match the real ramps to ~x2.
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
# Hardware delays [s], measured on the CRS93 (docs/MEASUREMENTS.md).
COMMAND_DELAY = 0.12  # move_to_q call -> the joints start moving
SEND_DELAY = 0.025  # move_to_q blocks (serial write of the command)
QUERY_DELAY = 0.05  # get_q round trip
STATUS_DELAY = 0.01  # in_motion round trip
CAMERA_DELAY = 0.11  # age of the image when grab_image returns
# Real grab_image duration [s] (ctu_crs starts and stops the camera per call) and
# of the first call (opens the camera); for the simulated camera.
GRAB_TIME = 0.186
FIRST_GRAB_TIME = 0.8
# Queued moves blend if their unit joint-space directions differ by less than this.
BLEND_TOL = 1e-3
HISTORY = 1.0  # s, finished moves kept for state queries reaching into the past


@dataclass
class Segment:
    """One coordinated move: all joints follow the same trapezoidal profile.

    The normalized progress s(t) rises from 0 to 1 with constant acceleration for
    tau, constant speed, and constant deceleration for the last tau.

    Attributes:
        t0: Start time on the sim clock [s].
        T: Duration [s].
        tau: Duration of the speed-up (and of the slow-down) ramp [s].
        q_from: Start configuration [rad].
        q_to: Target configuration [rad].
    """

    t0: float
    T: float
    tau: float
    q_from: np.ndarray
    q_to: np.ndarray

    @property
    def t_end(self) -> float:
        """End time of the move on the sim clock [s]."""
        return self.t0 + self.T

    def q_at(self, t: float) -> np.ndarray:
        """Configuration during the move.

        Args:
            t: Time on the sim clock [s]; clamped to the move.

        Returns:
            The configuration [rad] at @t.
        """
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


def _same_direction(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two joint-space moves point the same way.

    Args:
        a: First move (joint displacement) [rad].
        b: Second move [rad].

    Returns:
        True if both are nonzero and their unit vectors differ by less than
        BLEND_TOL.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return bool(na > 0 and nb > 0 and np.linalg.norm(a / na - b / nb) < BLEND_TOL)


class SimGripper:
    """Stub of ctu_crs.gripper.Gripper; the hoop task uses no gripper.

    Keeps the commanded position only, with the same initialization checks.

    Attributes:
        bounds: Gripper range in IRC (closed, open), as in ctu_crs.
        position: Last commanded position, or None.
    """

    def __init__(self, bounds=None, axis=None):
        """Creates the stub.

        Args:
            bounds: Gripper range in IRC, from the ctu_crs robot parameters.
            axis: MARS axis letter of the gripper (unused).
        """
        self.bounds = bounds
        self._axis = axis
        self.position = None
        self._initialized = False

    def initialize(self):
        """Marks the gripper as initialized."""
        self._initialized = True

    def control_position_relative(self, fraction: float):
        """Commands a position as a fraction of the gripper range.

        Args:
            fraction: 0 = bounds[0], 1 = bounds[1].
        """
        assert self._initialized, "Gripper controller must be initialized first."
        assert self.bounds is not None
        self.control_position(
            int(self.bounds[0] + (self.bounds[1] - self.bounds[0]) * fraction))

    def control_position(self, position: float):
        """Commands an absolute position.

        Args:
            position: Gripper position in IRC.
        """
        assert self._initialized, "Gripper controller must be initialized first."
        self.position = position

    def release(self):
        """Does nothing (the real one releases the gripper motor)."""
        assert self._initialized, "Gripper controller must be initialized first."

    def wait_for_motion_stop(self):
        """Returns at once.

        Returns:
            True, as the real one when the gripper stopped.
        """
        assert self._initialized, "Gripper controller must be initialized first."
        return True


class CRS93Sim(CRS93):
    """The CRS93 robot simulated in a Viser scene, with the ctu_crs.CRS93 interface.

    Attributes:
        viewer: The SceneViewer showing the robot (simulation only).
        command_delay: Delay from a move_to_q call to the start of the move [s].
        send_delay: How long move_to_q blocks [s].
        query_delay: Round trip of get_q [s].
        status_delay: Round trip of in_motion / wait_for_motion_stop [s].
        camera_delay: Age of the image when grab_image returns [s] (for the
            simulated camera, not used yet).
    """

    def __init__(self, tty_dev: str | None = "/dev/mars", baudrate: int = 19200, *,
                 q0: ArrayLike | None = None, port: int = 8080,
                 time_scale: float = 1.0, keep_alive: bool = True,
                 command_delay: float = COMMAND_DELAY,
                 send_delay: float = SEND_DELAY,
                 query_delay: float = QUERY_DELAY,
                 status_delay: float = STATUS_DELAY,
                 camera_delay: float = CAMERA_DELAY):
        """Starts the simulated robot and its viewer.

        The positional arguments are those of CRS93; the keyword-only ones exist
        in the simulation only. Delays are in sim seconds and default to the
        values measured on the CRS93; 0 gives an ideal robot.

        Args:
            tty_dev: Ignored (serial device of the real control unit).
            baudrate: Ignored.
            q0: Initial configuration [rad]; default q_home (the real arm is
                wherever it was left, homing moves it anyway).
            port: Port of the Viser server.
            time_scale: Sim seconds per wall second (> 1 runs faster; script
                execution then takes relatively less time than on the robot).
            keep_alive: Keep the scene open after the script ends.
            command_delay: See the class attributes.
            send_delay: See the class attributes.
            query_delay: See the class attributes.
            status_delay: See the class attributes.
            camera_delay: See the class attributes.
        """
        super().__init__(tty_dev=None, baudrate=baudrate)
        self.gripper = SimGripper(self.gripper.bounds, self.gripper._axis)
        self._time_scale = time_scale
        self._t_start = time.monotonic()
        self.command_delay = command_delay
        self.send_delay = send_delay
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
        """Returns the sim time [s] since start (robot, delays, turntable)."""
        return (time.monotonic() - self._t_start) * self._time_scale

    def _sleep(self, dt: float) -> None:
        """Blocks for @dt sim seconds."""
        if dt > 0:
            time.sleep(dt / self._time_scale)

    def _q_at(self, t: float) -> np.ndarray:
        """Returns the configuration [rad] at sim time @t (not older than HISTORY)."""
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
        """Whether a move runs or is queued (also not started yet) at sim time @t."""
        with self._lock:
            return bool(self._segments) and self._segments[-1].t_end > t

    def _motion_end(self) -> float:
        """Returns the end of the queued motion on the sim clock (-inf if none)."""
        with self._lock:
            return self._segments[-1].t_end if self._segments else -np.inf

    def _joint_limits_rad(self) -> tuple[np.ndarray, np.ndarray]:
        """Converts the speed and acceleration settings to joint units.

        Returns:
            Max joint speeds [rad/s] and accelerations [rad/s^2] (acceleration
            units assumed, see the module doc).
        """
        irc_per_rad = np.abs(self._deg_to_irc) * 180 / np.pi
        return (self._speed * 1000 / 256 / irc_per_rad,
                self._acceleration * 1e6 / 256 / irc_per_rad)

    def _profile(self, dq: np.ndarray) -> tuple[float, float]:
        """Times the coordinated move by @dq.

        The slowest joint makes its shortest trapezoidal move; the others are
        scaled to the same profile.

        Args:
            dq: Joint displacement [rad].

        Returns:
            Duration T and ramp time tau [s].
        """
        v_max, a_max = self._joint_limits_rad()
        d = np.abs(dq)
        ramp = v_max / a_max
        T_joint = np.where(d > v_max * ramp, d / v_max + ramp, 2 * np.sqrt(d / a_max))
        k = int(np.argmax(T_joint))
        T = float(T_joint[k])
        return T, min(float(ramp[k]), T / 2)

    def _enqueue(self, q: ArrayLike) -> None:
        """Queues a coordinated move to @q, rounded to IRC like the real command.

        The move is blended into the last queued one if it continues in the same
        direction and that one is not slowing down yet when the command arrives:
        both become one move with the same start, continuous because their
        speed-up and cruise phases coincide. Otherwise it starts when the last
        one ends (or command_delay after now).

        Args:
            q: Target configuration [rad].
        """
        q_to = self._irc_to_joint_values(self._joint_values_to_irc(q))
        now = self._now()
        arrival = now + self.command_delay
        with self._lock:
            while self._segments and self._segments[0].t_end < now - HISTORY:
                self._q_rest = self._segments.pop(0).q_to
            last = self._segments[-1] if self._segments else None
            q_from = last.q_to if last else self._q_rest
            if last and arrival < last.t_end - last.tau and _same_direction(
                    last.q_to - last.q_from, q_to - q_from):
                T, tau = self._profile(q_to - last.q_from)
                self._segments[-1] = Segment(last.t0, T, tau, last.q_from, q_to)
                return
            t0 = max(arrival, last.t_end) if last else arrival
            T, tau = self._profile(q_to - q_from)
            self._segments.append(Segment(t0, T, tau, q_from.copy(), q_to))

    def _stop(self) -> None:
        """Stops at the current configuration and drops the queued motion."""
        q = self._q_at(self._now())
        with self._lock:
            self._segments.clear()
            self._q_rest = q

    # --- CRSRobot interface ---

    def grab_image(self, timeout=10000):
        """Not simulated yet.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("CRS93Sim: the camera is not simulated yet")

    def release(self):
        """Does nothing (the real one releases errors of the control unit)."""

    def reset_motors(self):
        """Does nothing (the real one resets the motors)."""

    def close(self):
        """Does nothing (the real one closes the serial connection)."""

    def initialize(self, home: bool = True):
        """Stops any motion, sets the default speed and acceleration, homes.

        Args:
            home: Hard home and soft home (to q_home) afterwards, as the real
                robot does.
        """
        self._stop()
        self.set_speed(self._default_speed_irc256_per_ms)
        self.set_acceleration(self._default_acceleration_irc_per_ms)
        self.gripper.initialize()
        if home:
            self.hard_home()
            self.soft_home()
        self._initialized = True

    def set_speed(self, speed_irc256_ms: ArrayLike):
        """Sets the max speed of each joint for the following moves.

        Args:
            speed_irc256_ms: Speeds in IRC*256/ms, one per joint.
        """
        speed = np.asarray(speed_irc256_ms, dtype=float)
        assert len(speed) == len(self._motors_ids)
        self._speed = np.rint(speed)

    def set_acceleration(self, acceleration_irc_ms: ArrayLike):
        """Sets the acceleration of each joint for the following moves.

        Args:
            acceleration_irc_ms: Accelerations in the control unit's units, one
                per joint.
        """
        acceleration = np.asarray(acceleration_irc_ms, dtype=float)
        assert len(acceleration) == len(self._motors_ids)
        self._acceleration = np.rint(acceleration)

    def hard_home(self):
        """Moves each joint group of the hard home sequence to the hard home pose.

        That is where the real control unit zeroes the encoders, (163.4, 0.3,
        -56.0, 176.7, 14.9, -172.0) deg on the CRS93. Resets speed and
        acceleration to their defaults, as the real one does.

        Raises:
            ValueError: If the robot parameters define no hard home sequence.
        """
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
        """Moves to q_home and waits."""
        self._enqueue(self.q_home)
        self.wait_for_motion_stop()

    def move_to_q(self, q: ArrayLike):
        """Queues a coordinated move to @q; blocks only for send_delay.

        Args:
            q: Target configuration [rad].
        """
        assert self._initialized, "You need to initialize the robot before moving it."
        assert self.in_limits(q), "Joint limits violated."
        self._enqueue(q)
        self._sleep(self.send_delay)

    def get_q(self) -> np.ndarray:
        """Reads the current configuration; blocks for query_delay.

        Returns:
            The configuration [rad] at the middle of the round trip.
        """
        t = self._now() + self.query_delay / 2  # the unit answers here
        self._sleep(self.query_delay)
        return self._q_at(t)

    def in_motion(self) -> bool:
        """Reads whether the robot moves; blocks for status_delay.

        Returns:
            Whether a move runs or is queued at the middle of the round trip.
        """
        t = self._now() + self.status_delay / 2
        self._sleep(self.status_delay)
        return self._in_motion_at(t)

    def wait_for_motion_stop(self):
        """Blocks until the queued motion ends (+ status_delay / 2).

        Blocks at least status_delay, and also waits for moves queued meanwhile
        by other threads.
        """
        t_return = self._now() + self.status_delay
        while True:
            dt = max(t_return, self._motion_end() + self.status_delay / 2) - self._now()
            if dt <= 0:
                break
            self._sleep(dt)

    # --- viewer ---

    def _add_status_gui(self) -> None:
        """Adds the "CRS93 simulation" status panel to the viewer."""
        with self.viewer.server.gui.add_folder("CRS93 simulation"):
            self._status = self.viewer.server.gui.add_markdown("")

    def _render_loop(self) -> None:
        """Shows the whole scene at the current sim time, RENDER_FPS times a second."""
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
        """At exit: lets the queued motion finish, then keeps the scene open.

        The real arm also finishes its queued moves after the script ends. Runs
        until Ctrl+C.
        """
        try:
            self.wait_for_motion_stop()
            print("Script finished, simulation still open (Ctrl+C to quit)")
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
