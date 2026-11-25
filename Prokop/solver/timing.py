"""Task P3.2: timing of the CRS93's moves, as measured on the robot (E5, E6).

The same model as CRS93Sim (Prokop/simulation.py), usable without the
simulator, for planning (P2.6) and execution (P3.3):

* A move_to_q is a coordinated move: the slowest joint makes its shortest
  trapezoidal move (speed and acceleration from the ctu_crs settings), the
  other joints are scaled to the same profile.
* move_to_q blocks SEND_DELAY (the serial write); the joints start moving
  COMMAND_DELAY after the call.
* Queued moves run one after another. A move continuing in exactly the same
  joint-space direction blends into the previous one (one longer move, no stop)
  if its command arrives before the previous move starts slowing down; any other
  move starts from standstill (the real robot stops, or nearly stops, there).

Example:
    ::

        timing = RobotTiming(CRS93(tty_dev=None), speed=0.2)
        T = timing.move_duration(q0, q1)
        arrivals = timing.schedule(q0, [q1, q2, q3])   # all sent at once at t = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured on the CRS93 (docs/MEASUREMENTS.md); the same values as CRS93Sim.
COMMAND_DELAY = 0.12  # s, move_to_q call -> the joints start moving
SEND_DELAY = 0.025  # s, move_to_q blocks (serial write)
BLEND_TOL = 1e-3  # unit joint-space directions closer than this blend
STOPPED_STEP_LEAD = 0.20  # s at rest to check and prepare the next command


def same_direction(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two joint-space moves point the same way (the blending rule).

    Args:
        a: First move (joint displacement) [rad].
        b: Second move [rad].

    Returns:
        True if both are nonzero and their unit vectors differ by less than
        BLEND_TOL.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return bool(na > 0 and nb > 0 and np.linalg.norm(a / na - b / nb) < BLEND_TOL)


@dataclass
class TimedMove:
    """One move of a schedule (several queued targets may share one move).

    Attributes:
        t0: Start of the motion [s].
        T: Duration [s].
        tau: Speed-up (and slow-down) ramp [s].
        q_from: Start configuration [rad].
        q_to: End configuration [rad].
        hold_from: Start of a preceding hold to include in collision validation [s].
    """

    t0: float
    T: float
    tau: float
    q_from: np.ndarray
    q_to: np.ndarray
    hold_from: float | None = None

    @property
    def t_end(self) -> float:
        """End of the motion [s]."""
        return self.t0 + self.T

    def progress(self, t: float) -> float:
        """Normalized progress s(t) in [0, 1] along the trapezoidal profile."""
        if self.T <= 0:
            return 1.0
        t = float(np.clip(t - self.t0, 0.0, self.T))
        v = 1.0 / (self.T - self.tau)
        if t < self.tau:
            return 0.5 * v / self.tau * t**2
        if t <= self.T - self.tau:
            return v * (t - self.tau / 2)
        return 1.0 - 0.5 * v / self.tau * (self.T - t) ** 2

    def time_at(self, s: float) -> float:
        """Inverse of progress: when the move reaches the fraction @s [0, 1]."""
        if self.T <= 0:
            return self.t0
        tau, T = self.tau, self.T
        v = 1.0 / (T - tau)
        s_ramp = 0.5 * v * tau
        if s <= s_ramp:
            return self.t0 + float(np.sqrt(2 * s * tau / v))
        if s <= 1.0 - s_ramp:
            return self.t0 + s / v + tau / 2
        return self.t0 + T - float(np.sqrt(max(0.0, 2 * (1.0 - s) * tau / v)))

    def q_at(self, t: float) -> np.ndarray:
        """Configuration at time @t [rad]."""
        return self.q_from + self.progress(t) * (self.q_to - self.q_from)


class RobotTiming:
    """Timing model of the robot at one speed setting (implements the contracts'
    RobotTiming protocol).

    Attributes:
        speed: Relative speed setting (as robot.set_speed_relative).
        v_max: Max joint speeds at that setting [rad/s].
        a_max: Joint accelerations (default setting) [rad/s^2].
        command_delay: Call -> motion start [s].
        send_delay: How long move_to_q blocks [s].
    """

    def __init__(self, robot, speed: float, command_delay: float = COMMAND_DELAY,
                 send_delay: float = SEND_DELAY) -> None:
        """Converts the robot's settings into joint units.

        Args:
            robot: A ctu_crs CRS93 (offline is fine: CRS93(tty_dev=None)) or
                CRS93Sim; only its parameters are read.
            speed: Relative speed in [0, 1].
            command_delay: Call -> motion start [s].
            send_delay: How long move_to_q blocks [s].
        """
        assert 0.0 <= speed <= 1.0, "speed must be in [0, 1]"
        self.speed = speed
        irc = np.rint(robot._min_speed_irc256_per_ms + speed
                      * (robot._max_speed_irc256_per_ms - robot._min_speed_irc256_per_ms))
        acc = np.rint(robot._default_acceleration_irc_per_ms)
        irc_per_rad = np.abs(robot._deg_to_irc) * 180 / np.pi
        self.v_max = irc * 1000 / 256 / irc_per_rad
        self.a_max = acc * 1e6 / 256 / irc_per_rad
        self.command_delay = command_delay
        self.send_delay = send_delay

    def profile(self, dq: np.ndarray) -> tuple[float, float]:
        """Duration and ramp of a coordinated move.

        Args:
            dq: Joint displacement [rad].

        Returns:
            Duration T and ramp tau [s].
        """
        d = np.abs(np.asarray(dq, dtype=float))
        ramp = self.v_max / self.a_max
        T_joint = np.where(d > self.v_max * ramp, d / self.v_max + ramp,
                           2 * np.sqrt(d / self.a_max))
        k = int(np.argmax(T_joint))
        T = float(T_joint[k])
        return T, min(float(ramp[k]), T / 2)

    def move_duration(self, q0: np.ndarray, q1: np.ndarray, speed: float | None = None
                      ) -> float:
        """Duration of one coordinated move from standstill to standstill.

        Args:
            q0: Start [rad].
            q1: End [rad].
            speed: Must equal self.speed if given (the protocol passes it).

        Returns:
            Duration [s], without the command delay.
        """
        assert speed is None or speed == self.speed, "build a RobotTiming per speed"
        return self.profile(np.asarray(q1, float) - np.asarray(q0, float))[0]

    def blends(self, dq_prev: np.ndarray, dq_next: np.ndarray) -> bool:
        """Whether the next queued move can blend into the previous one.

        Only the direction rule; schedule() also checks that the command arrives
        in time.
        """
        return same_direction(np.asarray(dq_prev, float), np.asarray(dq_next, float))

    def stopped_move(self, q_from: np.ndarray, q_to: np.ndarray, t_call: float) -> TimedMove:
        """Profile for a command sent while the robot is at rest."""
        q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
        duration, ramp = self.profile(q_to - q_from)
        return TimedMove(t_call + self.command_delay, duration, ramp, q_from, q_to)

    def schedule_stopped(self, q_start: np.ndarray, targets: list[np.ndarray],
                         t_call: float = 0.0) -> tuple[list[float], list[TimedMove]]:
        """Time moves sent after the previous move stops and a check interval passes."""
        moves = []
        q_from = q_start
        for q in targets:
            move = self.stopped_move(q_from, q, t_call)
            moves.append(move)
            t_call = max(move.t_end, t_call + self.send_delay) + STOPPED_STEP_LEAD
            q_from = q
        return [move.t_end for move in moves], moves

    def schedule(self, q_start: np.ndarray, targets: list[np.ndarray],
                 t_call: float = 0.0, call_times: list[float] | None = None
                 ) -> tuple[list[float], list[TimedMove]]:
        """Times a sequence of queued move_to_q calls, as the robot executes them.

        Args:
            q_start: Configuration at rest when the first command is sent [rad].
            targets: The targets in order [rad].
            t_call: Time of the first call [s]; by default the calls follow each
                other back to back (send_delay apart).
            call_times: Explicit call time of every target instead [s].

        Returns:
            The arrival time at every target [s] and the resulting moves (blended
            targets share one move).
        """
        if call_times is None:
            call_times = [t_call + k * self.send_delay for k in range(len(targets))]
        moves: list[TimedMove] = []
        members: list[list[int]] = []  # target indices ending inside each move
        q_rest = np.asarray(q_start, dtype=float)
        for k, (q, t) in enumerate(zip(targets, call_times)):
            q = np.asarray(q, dtype=float)
            arrival = t + self.command_delay
            last = moves[-1] if moves else None
            q_from = last.q_to if last else q_rest
            if last and arrival < last.t_end - last.tau and same_direction(
                    last.q_to - last.q_from, q - q_from):
                T, tau = self.profile(q - last.q_from)
                moves[-1] = TimedMove(last.t0, T, tau, last.q_from, q)
                members[-1].append(k)
                continue
            t0 = max(arrival, last.t_end) if last else arrival
            T, tau = self.profile(q - q_from)
            moves.append(TimedMove(t0, T, tau, q_from.copy(), q))
            members.append([k])
        arrivals = [0.0] * len(targets)
        for move, ks in zip(moves, members):
            total = np.linalg.norm(move.q_to - move.q_from)
            for k in ks:
                done = np.linalg.norm(np.asarray(targets[k], float) - move.q_from)
                arrivals[k] = move.time_at(done / total if total > 0 else 1.0)
        return arrivals, moves
