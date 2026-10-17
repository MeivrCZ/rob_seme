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

    @property
    def connected(self) -> bool:
        return self._robot is not None

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

    def require(self):
        if self._robot is None:
            raise RuntimeError("Robot disconnected. Enter: robot connect virtual|home|nohome")
        return self._robot

    def get_q(self) -> np.ndarray:
        robot = self.require()
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
            robot.soft_home()
            robot.wait_for_motion_stop()

    def assert_stationary(self) -> None:
        """Refuse a runtime mode switch while real arm is moving.

        This check is not a substitute for a safety-rated stop.
        """
        if self.mode != "real":
            return
        robot = self.require()
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
