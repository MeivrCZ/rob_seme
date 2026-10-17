"""Shared robot interface. Real CRS93 or a *kinematic* CRS93, never both.

Virtual mode uses ctu_crs offline FK/IK. It does NOT simulate motors, dynamics,
collisions, control limits, calibration bias or the physical robot.
"""
from __future__ import annotations

import numpy as np
from threading import RLock
from functools import wraps


def _with_robot_lock(method):
    """Prevent sync polling from racing with serial commands, homing or close."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._io_lock:
            return method(self, *args, **kwargs)
    return wrapper



class RobotManager:
    def __init__(self, tty_dev: str = "/dev/mars", *,
                 real_motion_enabled: bool = False,
                 joint_limits_deg=None,
                 max_step_deg: float = 3.0):
        self._io_lock = RLock()
        if not isinstance(real_motion_enabled, bool):
            raise ValueError("real_motion_enabled must be true/false")
        if not np.isfinite(max_step_deg) or not 0 < max_step_deg <= 10:
            raise ValueError("max_step_deg must be in (0, 10] degrees")
        self.real_motion_enabled = real_motion_enabled
        self.max_step_rad = float(np.deg2rad(max_step_deg))
        self._limits_rad = None
        if joint_limits_deg is not None:
            limits = np.asarray(joint_limits_deg, dtype=float)
            if limits.shape != (6, 2) or not np.all(np.isfinite(limits)) or not np.all(limits[:, 0] < limits[:, 1]):
                raise ValueError("joint_limits_deg requires six [minimum,maximum] pairs")
            self._limits_rad = np.deg2rad(limits)
        self.tty_dev = tty_dev
        self._robot = None
        self.mode: str | None = None  # None, "real", "virtual"
        self._virtual_q: np.ndarray | None = None
        self._virtual_q_home: np.ndarray | None = None
        self._released = False  # REAL only: release requested; feedback may be off
        self._release_confirmed = False
        self._motion_uncertain = False

    @property
    def connected(self) -> bool:
        return self._robot is not None

    @property
    def released(self) -> bool:
        """True once REAL release was requested; remains locked until reconnect."""
        return self._released

    @property
    def release_confirmed(self) -> bool:
        """Whether the release API returned without error (hardware remains unverified)."""
        return self._release_confirmed

    @_with_robot_lock
    def connect(self, *, home: bool) -> None:
        """Connect to the real robot ONLY. Homing may physically move the robot."""
        if self.connected:
            raise RuntimeError("Robot already connected; disconnect first.")
        from ctu_crs import CRS93
        robot = CRS93(tty_dev=self.tty_dev)
        try:
            robot.initialize(home=home)
        except Exception:
            robot.close()
            raise
        self._robot = robot
        self.mode = "real"
        self._released = False
        self._release_confirmed = False
        self._motion_uncertain = False

    @_with_robot_lock
    def connect_virtual(self) -> None:
        """Use authentic offline CRS93 FK/IK without touching /dev/mars."""
        if self.connected:
            raise RuntimeError("Robot already connected; disconnect first.")
        from ctu_crs import CRS93
        robot = CRS93(tty_dev=None)  # documented OFFLINE mode
        q_home = np.asarray(robot.q_home, dtype=float).reshape(-1)
        if q_home.shape != (6,) or not np.all(np.isfinite(q_home)):
            raise RuntimeError("Offline CRS93 did not supply 6 valid home joints.")
        self._virtual_q = q_home.copy()
        self._virtual_q_home = q_home.copy()
        self._robot = robot
        self.mode = "virtual"
        self._released = False
        self._release_confirmed = False
        self._motion_uncertain = False

    def require(self):
        if self._robot is None:
            raise RuntimeError("Robot disconnected. Enter: robot connect virtual|home|nohome")
        return self._robot

    @_with_robot_lock
    def get_q(self) -> np.ndarray:
        robot = self.require()
        if self._released:
            raise RuntimeError("Robot released: joint feedback is disconnected; pose is not reliable.")
        if self.mode == "virtual":
            return self._virtual_q.copy()
        return np.asarray(robot.get_q(), dtype=float).copy()

    @_with_robot_lock
    def flange_pose(self) -> np.ndarray:
        return np.asarray(self.require().fk(self.get_q()), dtype=float).reshape(4, 4)

    @_with_robot_lock
    def virtual_set_q(self, joints: np.ndarray) -> None:
        """Teleport *kinematic state*, NOT a robot command. No limit checking."""
        self.require()
        if self.mode != "virtual":
            raise RuntimeError("Setting joints is allowed only in VIRTUAL mode.")
        q = np.asarray(joints, dtype=float).reshape(-1)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("Expected six finite joint angles.")
        self._virtual_q = q.copy()

    @staticmethod
    def _joint_vector(value, *, label: str) -> np.ndarray:
        q = np.asarray(value, dtype=float).reshape(-1)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{label} must contain exactly six finite joint angles")
        return q

    def _validate_motion(self, current: np.ndarray, target: np.ndarray) -> None:
        """Software guardrails, NOT a collision checker or safety-rated interlock."""
        if self.mode != "real":
            return
        if self._released:
            raise RuntimeError("Robot released: motion forbidden until station is restored and reconnected")
        if self._motion_uncertain:
            raise RuntimeError("Previous motion outcome uncertain: check robot physically and restart safely")
        if not self.real_motion_enabled:
            raise RuntimeError("REAL motion disabled in config.json (motion.enable_real_motion=false)")
        if self._limits_rad is None:
            raise RuntimeError("REAL motion blocked: configure verified motion.joint_limits_deg in config.json")
        lo, hi = self._limits_rad[:, 0], self._limits_rad[:, 1]
        if np.any(current < lo) or np.any(current > hi):
            raise RuntimeError("Current joints outside configured joint limits; refusing motion")
        if np.any(target < lo) or np.any(target > hi):
            raise ValueError("Requested target outside configured joint limits")
        if np.max(np.abs(target - current)) > self.max_step_rad + 1e-10:
            raise ValueError(f"Requested real move exceeds configured {np.rad2deg(self.max_step_rad):g} deg per joint; split into small moves")

    @_with_robot_lock
    def preview_move(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate move and return (current,target) for console preview.

        Real mode requires a stationary robot and explicit configuration.
        """
        self.require()
        target = self._joint_vector(joints, label="Target")
        current = self._joint_vector(self.get_q(), label="Current joints")
        if self.mode == "real":
            self.assert_stationary()
            self._validate_motion(current, target)
        return current.copy(), target.copy()

    @_with_robot_lock
    def move_to_q(self, joints: np.ndarray, *, expected_q: np.ndarray | None = None) -> None:
        """Move virtual state or send ONE guarded real joint-space command.

        Caller must obtain a new explicit human confirmation in real mode.
        No implicit homing. No interpolation or collision checking in our app.
        """
        robot = self.require()
        target = self._joint_vector(joints, label="Target")
        if self.mode == "virtual":
            self.virtual_set_q(target)
            return
        if self.mode != "real":
            raise RuntimeError("Robot is not connected")
        self.assert_stationary()
        current = self._joint_vector(self.get_q(), label="Current joints")
        self._validate_motion(current, target)
        if expected_q is not None:
            expected = self._joint_vector(expected_q, label="Previewed start")
            if np.max(np.abs(current - expected)) > np.deg2rad(0.25):
                raise RuntimeError("Robot moved since preview; cancel command and preview again")
        send = getattr(robot, "move_to_q", None)
        wait = getattr(robot, "wait_for_motion_stop", None)
        if not callable(send) or not callable(wait):
            raise RuntimeError("Installed ctu_crs does not expose move_to_q/wait_for_motion_stop")
        # Keep the serial-interface lock until motion completion to avoid
        # interleaved driver queries from the viewer sync thread.
        self._motion_uncertain = True
        send(target.copy())
        wait()
        if robot.in_motion():
            raise RuntimeError("Driver returned from wait but robot still reports motion")
        self._motion_uncertain = False

    @_with_robot_lock
    def ik(self, pose: np.ndarray):
        """Offline/online kinematics via ctu_crs; not an action command."""
        return self.require().ik(np.asarray(pose, dtype=float))

    @_with_robot_lock
    def soft_home(self) -> None:
        robot = self.require()
        if self.mode == "virtual":
            self._virtual_q = self._virtual_q_home.copy()
        else:
            if self._released:
                raise RuntimeError("Robot released: do not move until the station has been re-enabled and the connection reinitialized.")
            robot.soft_home()
            robot.wait_for_motion_stop()

    @_with_robot_lock
    def release(self) -> None:
        """Engage brakes and disconnect feedback on a stopped REAL CRS93.

        Caller must obtain explicit operator confirmation before calling.
        This is NOT an E-STOP, a homing routine or power-off.
        """
        if self.mode != "real":
            raise RuntimeError("robot release is available only with a connected REAL robot.")
        if self._released:
            raise RuntimeError("Robot has already been released.")
        robot = self.require()
        self.assert_stationary()  # fail closed if in_motion() is unavailable
        release_command = getattr(robot, "release", None)
        if not callable(release_command):
            raise RuntimeError("The installed ctu_crs version does not provide robot.release().")
        # If the command fails mid-flight, the physical state is unknown.
        # Lock out subsequent motion on this connection even on exception.
        self._released = True
        release_command()
        self._release_confirmed = True

    @_with_robot_lock
    def assert_stationary(self) -> None:
        """Refuse a runtime mode switch while real arm is moving.

        This check is not a substitute for a safety-rated stop.
        """
        if self.mode != "real":
            return
        robot = self.require()
        if self._released:
            # Feedback was disconnected when brakes engaged. Do not query motion
            # from an interface that can no longer provide reliable feedback.
            return
        check = getattr(robot, "in_motion", None)
        if not callable(check):
            raise RuntimeError("Cannot check robot motion; refusing mode switch.")
        if check():
            raise RuntimeError("Robot is moving. Wait for a complete stop before switching modes.")

    @_with_robot_lock
    def close(self) -> None:
        if self._robot is not None:
            try:
                # close() is necessary on hardware; offline models own no port.
                if self.mode == "real":
                    self._robot.close()
            finally:
                self._robot = None
                self._virtual_q = None
                self._virtual_q_home = None
                self.mode = None
                self._released = False
                self._release_confirmed = False
                self._motion_uncertain = False
