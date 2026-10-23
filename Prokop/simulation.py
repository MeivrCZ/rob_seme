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
* grab_image returns an image rendered from the simulated scene with the
  camera model (camera.py: pose and intrinsics of cell.py, the attribute
  camera), with the real timing: it blocks 0.19 s (0.8 s the first time) and
  shows the scene as it was camera_delay before it returns. The last image is
  shown in the viewer's "Camera" panel.
* Hard collisions: a monitor checks the exact meshes (collision.SceneCollider:
  the arm with itself, with the cell and the table, with the turntable, base
  and puzzle) along the motion, slightly ahead of the sim clock. On contact the
  robot stops at the last collision-free configuration, the turntable stops,
  the colliding bodies turn red and the next robot call raises CollisionError.
  Until "Release" is pressed in the viewer's "Collision" panel, every motion
  command raises again (get_q and grab_image still work). Release puts the
  robot back where the colliding move started.
* The gripper is a stub; release, reset_motors and close do nothing.

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
from trimesh import transform_points as trimesh_points
from camera import Camera, SceneRenderer
from collision import SceneCollider
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
# Collision monitor (sim seconds / metres / radians).
MONITOR_LOOKAHEAD = 0.15  # s, how far ahead of the clock the motion is checked
MONITOR_MAX_STEP = 0.25  # s, longest step between checks
MONITOR_SUBSTEP = 0.01  # s, resolution of the motion bound
# Max joint change between checks: self-collisions are only possible between
# bulky parts (>= 4 cm), and 3 deg moves the far edge of the hoop ~1.5 cm.
MONITOR_JOINT_STEP = np.deg2rad(3.0)
MONITOR_REFRESH = 0.03  # m, exact distance to a group recomputed below this bound
MONITOR_MIN_MOVE = 0.002  # m, smallest step even when (nearly) touching
ARM_REACH = 1.1  # m, bound on the distance of any arm point from any joint axis
TASK_REACH = 0.15  # m, bound on base + puzzle reach beyond the base offset
MONITOR_WAIT = 5.0  # s (wall), longest wait of a query for the monitor


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


class CollisionError(RuntimeError):
    """The simulated robot hit something (hard collision) and stopped.

    Attributes:
        time: Sim time of the contact [s].
        hits: Colliding pairs (arm part, other body).
        q_stop: Configuration the robot stopped in [rad].
        q_start: Start of the colliding move, where Release puts the robot [rad].
    """

    def __init__(self, message: str, record: "_Collision"):
        super().__init__(message)
        self.time = record.t
        self.hits = record.hits
        self.q_stop = record.q_stop
        self.q_start = record.q_start


@dataclass
class _Collision:
    """A detected collision: contact at @t, the robot frozen from @t_free on."""

    t: float
    t_free: float
    hits: list
    q_stop: np.ndarray
    q_start: np.ndarray


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
        camera_delay: Age of the image when grab_image returns [s].
        camera: Camera model grab_image renders with (pose, intrinsics); change
            it to test perception against a camera that differs from its
            calibration.
    """

    def __init__(self, tty_dev: str | None = "/dev/mars", baudrate: int = 19200, *,
                 q0: ArrayLike | None = None, port: int = 8080,
                 time_scale: float = 1.0, keep_alive: bool = True,
                 command_delay: float = COMMAND_DELAY,
                 send_delay: float = SEND_DELAY,
                 query_delay: float = QUERY_DELAY,
                 status_delay: float = STATUS_DELAY,
                 camera_delay: float = CAMERA_DELAY,
                 collisions: bool = True):
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
            collisions: Monitor hard collisions (stop, CollisionError). Off
                for runs far faster than real time (time_scale >> 1) that do
                not need it: the monitor would slow them down.
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
        self.camera = Camera()
        self._camera_opened: set[int] = set()  # threads that called grab_image
        self._lock = threading.Lock()
        self._running = True  # the threads run until stop()
        self._segments: list[Segment] = []
        self._q_rest = np.array(self.q_home if q0 is None else q0, dtype=float)
        self._speed = self._default_speed_irc256_per_ms.astype(float)
        self._acceleration = self._default_acceleration_irc_per_ms.astype(float)

        self.viewer = SceneViewer(port, clock=self._now)  # one clock for the scene
        self._renderer = SceneRenderer(self.viewer.task, self.camera)
        self._renderer.prepare()  # OpenGL context of the creating thread (~0.6 s)
        # Hard collision monitor: everything up to _checked is collision free.
        self._collider = SceneCollider(self.viewer.task)
        self._collision: _Collision | None = None
        self._collision_reported = False
        self._checked = self._now()
        self._last_free = self._checked  # time of the last collision-free check
        self._dirty = True
        self._red: list = []  # red overlays of colliding bodies
        self.viewer.on_change.append(self._recheck)
        self._add_status_gui()
        self.viewer.add_turntable_gui(animate=False)  # rendered by _render_loop
        self.viewer.show_q(self._q_rest)
        threading.Thread(target=self._render_loop, daemon=True).start()
        self._collisions_on = collisions
        if collisions:
            threading.Thread(target=self._monitor_loop, daemon=True).start()
        else:
            self._collision_text.content = "collision checking off"
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
            if self._collision is not None and t >= self._collision.t_free:
                return self._collision.q_stop.copy()
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
            if self._collision is not None and t >= self._collision.t_free:
                return False
            return bool(self._segments) and self._segments[-1].t_end > t

    def _motion_end(self) -> float:
        """Returns the end of the queued motion on the sim clock (-inf if none)."""
        with self._lock:
            end = self._segments[-1].t_end if self._segments else -np.inf
            if self._collision is not None:
                end = min(end, self._collision.t_free)
            return end

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
            else:
                t0 = max(arrival, last.t_end) if last else arrival
                T, tau = self._profile(q_to - q_from)
                self._segments.append(Segment(t0, T, tau, q_from.copy(), q_to))
        self._recheck()

    def _stop(self) -> None:
        """Stops at the current configuration and drops the queued motion."""
        q = self._q_at(self._now())
        with self._lock:
            self._segments.clear()
            self._q_rest = q
        self._recheck()

    # --- hard collisions ---

    def _recheck(self) -> None:
        """Re-checks from now on (the motion, placement or spin changed)."""
        self._checked = min(self._checked, self._now())
        self._dirty = True

    def _await_checked(self, t: float) -> None:
        """Waits until the collision monitor has checked the scene up to sim time
        @t, so that no state is reported that a collision would still change.

        Only matters when the monitor falls behind the clock (time_scale > 1 or
        a slow computer); gives up after MONITOR_WAIT seconds of wall time.

        Args:
            t: Sim time.
        """
        if not self._collisions_on:
            return
        deadline = time.monotonic() + MONITOR_WAIT
        while (self._checked < t and self._collision is None
               and time.monotonic() < deadline):
            time.sleep(0.002)

    def _guard(self, motion: bool) -> None:
        """Raises CollisionError after a collision.

        Once for any call (the first one after the robot stopped), and for every
        motion command until Release is pressed.

        Args:
            motion: Whether the caller commands motion.

        Raises:
            CollisionError: As described.
        """
        record = self._collision
        if record is None or self._now() < record.t_free:
            return
        if not self._collision_reported:
            self._collision_reported = True
            pairs = ", ".join(f"{a}-{b}" for a, b in record.hits)
            raise CollisionError(
                f"collision at t = {record.t:.2f} s: {pairs}. The robot stopped at "
                f"q = {np.rad2deg(record.q_stop).round(1)} deg. Press 'Release' in "
                "the viewer ('Collision' panel): the robot then goes back to the "
                "start of the colliding move.", record)
        if motion:
            raise CollisionError("the robot is stopped after a collision; press "
                                 "'Release' in the viewer first", record)

    def _motion_bound(self, t0: float, t1: float) -> tuple[float, float]:
        """Bounds of the motion between @t0 and @t1.

        Returns:
            The largest distance any arm or task point can have moved [m] and
            the largest single joint change [rad].
        """
        omega = abs(self.viewer._omega)
        reach = np.linalg.norm(self.viewer.placement.base_xy) + TASK_REACH
        moved, joint = 0.0, 0.0
        q_prev = self._q_at(t0)
        q_ref = q_prev
        t = t0
        while t < t1:
            t_next = min(t + MONITOR_SUBSTEP, t1)
            q = self._q_at(t_next)
            moved += ARM_REACH * np.abs(q - q_prev).sum() + omega * reach * (t_next - t)
            joint = max(joint, float(np.abs(q - q_ref).max()))
            q_prev, t = q, t_next
        return moved, joint

    def _next_check(self, t: float, clearance: float, horizon: float) -> float:
        """Next time to check: the arm and the task cannot have moved more than
        @clearance (but at least MONITOR_MIN_MOVE), no joint more than
        MONITOR_JOINT_STEP.

        Args:
            t: Time of the last check.
            clearance: Clearance found there [m].
            horizon: Latest time to check up to.

        Returns:
            The next time.
        """
        limit = max(clearance, MONITOR_MIN_MOVE)
        t_next = min(t + MONITOR_MAX_STEP, horizon)
        lo, hi = t, t_next
        moved, joint = self._motion_bound(t, hi)
        if moved <= limit and joint <= MONITOR_JOINT_STEP:
            return hi
        for _ in range(12):  # bisection on the bound
            mid = (lo + hi) / 2
            moved, joint = self._motion_bound(t, mid)
            if moved <= limit and joint <= MONITOR_JOINT_STEP:
                lo = mid
            else:
                hi = mid
        return max(lo, t + 1e-4)

    def _monitor_loop(self) -> None:
        """Checks the scene for hard collisions, slightly ahead of the clock.

        Steps through the motion so that between two checks nothing can move
        further than the clearance found (or than MONITOR_JOINT_STEP in a joint,
        for self-collisions). The exact distance to the cell and to the task
        objects is recomputed only when its lower bound (last distance minus
        the motion since) drops below MONITOR_REFRESH.
        """
        last_state = None
        bounds = {}  # obstacle group -> lower bound of the clearance [m]
        q_self = None  # configuration of the last self-collision check
        while self._running:
            try:
                if self._collision is not None:
                    time.sleep(0.02)
                    continue
                now = self._now()
                horizon = now + MONITOR_LOOKAHEAD
                t = self._checked
                if t >= horizon:
                    time.sleep(0.01)
                    continue
                q = self._q_at(t)
                angle = self.viewer.turntable_angle(t)
                state = (q.tobytes(), round(angle, 9), id(self.viewer.placement))
                if self._dirty:
                    self._dirty = False
                    bounds.clear()
                    q_self = None
                elif state == last_state:
                    # Nothing moved since the last check.
                    moving = self.viewer._omega != 0.0 or self._in_motion_at(t)
                    self._checked = t + (MONITOR_SUBSTEP if moving else horizon - t)
                    continue
                last_state = state
                collider = self._collider
                collider.place(q, self.viewer.placement, angle)
                hits = []
                if q_self is None or not np.array_equal(q, q_self):
                    hits += collider.placed_self_hits()
                    q_self = q
                table_hits, clearance = collider.placed_table()
                hits += table_hits
                for group in ("cell", "task"):
                    if bounds.get(group, -1.0) < MONITOR_REFRESH:
                        group_hits, bounds[group] = collider.placed_group(group)
                        hits += group_hits
                    clearance = min(clearance, bounds[group])
                if hits:
                    self._on_collision(t, sorted(set(hits)))
                    continue
                self._last_free = t
                if self._dirty:  # changed meanwhile: check again from there
                    continue
                # The step may reach beyond the horizon (a change of the motion
                # or the scene rewinds the check anyway); the horizon only
                # decides when to wait for the clock.
                t_next = self._next_check(t, clearance, t + MONITOR_MAX_STEP)
                moved, _ = self._motion_bound(t, t_next)
                for group in bounds:
                    bounds[group] -= moved
                self._checked = t_next
            except Exception as e:  # keep monitoring whatever happens
                print(f"collision monitor: {e!r}")
                time.sleep(0.1)

    def _on_collision(self, t: float, hits: list) -> None:
        """Stops everything at the last collision-free time before @t.

        Args:
            t: Sim time of the first colliding check.
            hits: Colliding pairs.
        """
        t_free = min(self._last_free, t)
        q_stop = self._q_at(t_free)
        q_start = q_stop
        with self._lock:
            for seg in self._segments:
                if seg.t0 <= t < seg.t_end:
                    q_start = seg.q_from.copy()
            self._collision = _Collision(t, t_free, hits, q_stop, q_start)
            self._collision_reported = False
        self.viewer.stop_spin(t_free)
        self._show_collision(hits)
        pairs = ", ".join(f"{a}-{b}" for a, b in hits)
        print(f"COLLISION at t = {t:.2f} s: {pairs}; robot stopped, press Release")

    def _show_collision(self, hits: list) -> None:
        """Draws the colliding bodies red and enables Release."""
        server = self.viewer.server
        names = sorted({name for pair in hits for name in pair})
        with server.atomic():
            for name in names:
                mesh, T = self._collider.body(name)
                vertices = trimesh_points(mesh.vertices + 0.001 * mesh.vertex_normals, T)
                self._red.append(server.scene.add_mesh_simple(
                    f"/collision/{name}", vertices, mesh.faces,
                    color=(0.9, 0.1, 0.1), opacity=0.75))
        self._collision_text.content = (
            "**COLLISION** at t = {:.2f} s  \n".format(self._collision.t)
            + "  \n".join(f"{a} - {b}" for a, b in hits)
            + "  \nrobot and turntable stopped")
        self._release_button.disabled = False

    def _release(self) -> None:
        """Release button: clears the stop, robot back to the colliding move's start."""
        record = self._collision
        if record is None:
            return
        with self._lock:
            self._segments.clear()
            self._q_rest = record.q_start.copy()
            self._collision = None
            self._collision_reported = False
        for handle in self._red:
            handle.remove()
        self._red.clear()
        self._collision_text.content = "no collision"
        self._release_button.disabled = True
        self._recheck()

    # --- CRSRobot interface ---

    def grab_image(self, timeout=10000):
        """Renders the camera image of the simulated scene.

        Timing as measured on the station: the call blocks GRAB_TIME
        (FIRST_GRAB_TIME on the first call, which opens the camera) and the image
        shows the scene (robot, turntable angle) as it was camera_delay before
        the call returns.

        Args:
            timeout: Ignored (grab timeout of the real camera [ms]).

        Returns:
            Image (1200, 1920, 3), BGR uint8, like the real grab_image.
        """
        self._guard(motion=False)
        t_call = self._now()
        first = threading.get_ident() not in self._camera_opened
        self._camera_opened.add(threading.get_ident())
        renderer = self._renderer
        renderer.camera = self.camera
        renderer.prepare()  # OpenGL context of this thread, on its first call
        t_return = t_call + (FIRST_GRAB_TIME if first else GRAB_TIME)
        t_image = t_return - self.camera_delay
        self._sleep(t_image - self._now())
        self._await_checked(t_image)
        image = renderer.render(self._q_at(t_image), self.viewer.placement,
                                self.viewer.turntable_angle(t_image))
        self._camera_view.image = image[::4, ::4, ::-1]  # RGB, 480 x 300
        self._sleep(t_return - self._now())
        return image

    def release(self):
        """Does nothing (the real one releases errors of the control unit)."""

    def reset_motors(self):
        """Does nothing (the real one resets the motors)."""

    def close(self):
        """Does nothing (the real one closes the serial connection).

        The simulated robot and its viewer keep running (see stop()).
        """

    def stop(self) -> None:
        """Ends the simulation: its threads and the viewer (simulation only).

        For scripts and tests that make several CRS93Sim in one process.
        """
        self._running = False
        self.viewer.server.stop()

    def initialize(self, home: bool = True):
        """Stops any motion, sets the default speed and acceleration, homes.

        Args:
            home: Hard home and soft home (to q_home) afterwards, as the real
                robot does.
        """
        self._guard(motion=True)
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
        self._guard(motion=True)
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
        self._guard(motion=True)
        self._enqueue(self.q_home)
        self.wait_for_motion_stop()

    def move_to_q(self, q: ArrayLike):
        """Queues a coordinated move to @q; blocks only for send_delay.

        The move blends into the queued one if it continues in exactly the same
        joint-space direction (from that move's target), else it starts from
        standstill when that one ends.

        Args:
            q: Target configuration [rad].
        """
        assert self._initialized, "You need to initialize the robot before moving it."
        assert self.in_limits(q), "Joint limits violated."
        self._guard(motion=True)
        self._enqueue(q)
        self._sleep(self.send_delay)

    def get_q(self) -> np.ndarray:
        """Reads the current configuration; blocks for query_delay.

        Returns:
            The configuration [rad] at the middle of the round trip.
        """
        self._guard(motion=False)
        t = self._now() + self.query_delay / 2  # the unit answers here
        self._sleep(self.query_delay)
        self._await_checked(t)
        self._guard(motion=False)
        return self._q_at(t)

    def in_motion(self) -> bool:
        """Reads whether the robot moves; blocks for status_delay.

        Returns:
            Whether a move runs or is queued at the middle of the round trip.
        """
        self._guard(motion=False)
        t = self._now() + self.status_delay / 2
        self._sleep(self.status_delay)
        self._await_checked(t)
        self._guard(motion=False)
        return self._in_motion_at(t)

    def wait_for_motion_stop(self):
        """Blocks until the queued motion ends (+ status_delay / 2).

        Blocks at least status_delay, and also waits for moves queued meanwhile
        by other threads.
        """
        t_return = self._now() + self.status_delay
        while True:
            self._guard(motion=False)  # raises if a collision stops the motion
            end = max(t_return, self._motion_end() + self.status_delay / 2)
            dt = end - self._now()
            if dt <= 0:
                self._await_checked(end)
                self._guard(motion=False)
                if self._motion_end() + self.status_delay / 2 <= end:
                    break
                continue
            self._sleep(min(dt, 0.05))

    # --- viewer ---

    def _add_status_gui(self) -> None:
        """Adds the "CRS93 simulation" status panel to the viewer."""
        with self.viewer.server.gui.add_folder("CRS93 simulation"):
            self._status = self.viewer.server.gui.add_markdown("")
        with self.viewer.server.gui.add_folder("Collision"):
            self._collision_text = self.viewer.server.gui.add_markdown("no collision")
            self._release_button = self.viewer.server.gui.add_button(
                "Release", disabled=True,
                hint="Clears the stop; the robot goes back to the start of the "
                     "colliding move")
            self._release_button.on_click(lambda _: self._release())
        with self.viewer.server.gui.add_folder("Camera"):
            self._camera_view = self.viewer.server.gui.add_image(
                np.full((300, 480, 3), 200, np.uint8), label="last grab_image",
                format="jpeg")

    def _render_loop(self) -> None:
        """Shows the whole scene at the current sim time, RENDER_FPS times a second."""
        shown = None
        while self._running:
            now = self._now()
            q = self._q_at(now)
            moved = shown is None or not np.array_equal(q, shown)
            self.viewer.show(now, q if moved else None)  # whole scene at one time
            shown = q
            with self._lock:
                queued = sum(seg.t_end > now for seg in self._segments)
            flange = np.array2string(self.fk(q)[:3, 3], precision=3, suppress_small=True)
            self._status.content = (
                f"q [deg]: {np.array2string(np.rad2deg(q), precision=1)}  \n"
                f"flange xyz [m]: {flange}  \n"
                f"initialized: {self._initialized}, "
                f"in motion: {self._in_motion_at(now)}, queued moves: {queued}")
            time.sleep(1 / RENDER_FPS)

    def _keep_alive(self) -> None:
        """At exit: lets the queued motion finish, then keeps the scene open.

        The real arm also finishes its queued moves after the script ends. Runs
        until Ctrl+C.
        """
        try:
            try:
                self.wait_for_motion_stop()
            except CollisionError as e:
                print(f"CollisionError: {e}")
            print("Script finished, simulation still open (Ctrl+C to quit)")
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
