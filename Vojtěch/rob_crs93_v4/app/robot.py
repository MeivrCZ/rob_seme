"""Shared robot interface. Real CRS93 or a *kinematic* CRS93, never both.

Virtual mode uses ctu_crs offline FK/IK. It does NOT simulate motors, dynamics,
collisions, control limits, calibration bias or the physical robot.
"""
from __future__ import annotations

import numpy as np


class RobotManager:
    def __init__(self, tty_dev: str = "/dev/mars"):
        self.tty_dev = tty_dev
        self._robot = None
        self.mode: str | None = None  # None, "real", "virtual"
        self._virtual_q: np.ndarray | None = None
        self._virtual_q_home: np.ndarray | None = None
        self._released = False  # REAL only: release requested; feedback may be off
        self._release_confirmed = False

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

    def require(self):
        if self._robot is None:
            raise RuntimeError("Robot disconnected. Enter: robot connect virtual|home|nohome")
        return self._robot

    def get_q(self) -> np.ndarray:
        robot = self.require()
        if self._released:
            raise RuntimeError("Robot released: joint feedback is disconnected; pose is not reliable.")
        if self.mode == "virtual":
            return self._virtual_q.copy()
        return np.asarray(robot.get_q(), dtype=float).copy()

    def flange_pose(self) -> np.ndarray:
        return np.asarray(self.require().fk(self.get_q()), dtype=float).reshape(4, 4)

    def virtual_set_q(self, joints: np.ndarray) -> None:
        """Teleport *kinematic state*, NOT a robot command. No limit checking."""
        self.require()
        if self.mode != "virtual":
            raise RuntimeError("Setting joints is allowed only in VIRTUAL mode.")
        q = np.asarray(joints, dtype=float).reshape(-1)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("Expected six finite joint angles.")
        self._virtual_q = q.copy()

    def ik(self, pose: np.ndarray):
        """Offline/online kinematics via ctu_crs; not an action command."""
        return self.require().ik(np.asarray(pose, dtype=float))

    def soft_home(self) -> None:
        robot = self.require()
        if self.mode == "virtual":
            self._virtual_q = self._virtual_q_home.copy()
        else:
            if self._released:
                raise RuntimeError("Robot released: do not move until the station has been re-enabled and the connection reinitialized.")
            robot.soft_home()
            robot.wait_for_motion_stop()

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
