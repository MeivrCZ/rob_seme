"""Offline mode integration tests: hardware APIs replaced with isolated fakes."""
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.camera import CameraManager
from app.context import Context
from app.extensions import ExtensionManager
from app.modes import ModeManager
from app.robot import RobotManager
from app.store import CalibrationStore
from main import dispatch


class FakeRobot:
    events = []
    moving = False
    fail_initialize = False

    def __init__(self, tty_dev=None):
        self.tty_dev = tty_dev
        self.q_home = np.zeros(6)
        self._q = np.ones(6) * .01
        self.events.append(("new", tty_dev))

    def initialize(self, home=True):
        self.events.append(("initialize", home))
        if self.fail_initialize:
            raise RuntimeError("Fake robot not ready")

    def in_motion(self):
        return self.moving

    def get_q(self):
        return self._q.copy()

    def fk(self, q):
        t = np.eye(4)
        t[:3, 3] = np.asarray(q)[:3]
        return t

    def ik(self, t):
        return [self.q_home.copy()]

    def close(self):
        self.events.append(("robot_close", self.tty_dev))


class FakeCamera:
    events = []
    fail_open = False

    def connect_by_name(self, name):
        self.events.append(("camera_connect", name))

    def open(self):
        self.events.append(("camera_open",))
        if self.fail_open:
            raise RuntimeError("Fake camera is busy")

    def close(self):
        self.events.append(("camera_close",))

    def grab_image(self):
        return np.zeros((8, 12), np.uint8)


class SwitchingTests(unittest.TestCase):
    def setUp(self):
        FakeRobot.events = []
        FakeRobot.moving = False
        FakeRobot.fail_initialize = False
        FakeCamera.events = []
        FakeCamera.fail_open = False
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.frames = root / "recordings"
        self.frames.mkdir()
        assert cv2.imwrite(str(self.frames / "frame.png"), np.ones((24, 16), np.uint8))
        self.ctx = Context(root, RobotManager("/dev/fake"), CameraManager("camera-crs93"),
                           CalibrationStore(root / "calibration"))
        self.modes = ModeManager(self.ctx)
        self.ext = ExtensionManager(self.ctx)
        package = types.ModuleType("ctu_crs")
        package.__path__ = []
        package.CRS93 = FakeRobot
        basler = types.ModuleType("ctu_crs.basler_camera")
        basler.BaslerCamera = FakeCamera
        mock = patch.dict(sys.modules, {"ctu_crs": package, "ctu_crs.basler_camera": basler})
        mock.start()
        self.addCleanup(mock.stop)
        self.addCleanup(self.ext.close)
        self.addCleanup(self.ctx.camera.close)
        self.addCleanup(self.ctx.robot.close)

    def test_roundtrip_keep_calibration_extensions_virtual_state(self):
        self.ctx.calibration.save("camera_intrinsics", {"focal": 555})
        with redirect_stdout(io.StringIO()):
            self.ext.load("diagnostics")
        instance = self.ext.instances["diagnostics"]

        self.modes.switch_virtual(self.frames)
        self.assertEqual(self.modes.mode, "VIRTUAL")
        self.assertEqual(self.ctx.camera.mode, "replay")
        q = np.array([.1, .2, .3, .4, .5, .6])
        self.ctx.robot.virtual_set_q(q)
        self.assertFalse(any(e[0] == "initialize" for e in FakeRobot.events))

        self.modes.switch_real(home=False)
        self.assertEqual(self.modes.mode, "REAL")
        self.assertEqual([e for e in FakeRobot.events if e[0] == "initialize"], [("initialize", False)])
        self.modes.switch_real(home=True)  # should NOT re-home an already live real robot
        self.assertEqual([e for e in FakeRobot.events if e[0] == "initialize"], [("initialize", False)])
        self.modes.switch_virtual()
        self.assertEqual(self.modes.mode, "VIRTUAL")
        self.assertTrue(np.allclose(self.ctx.robot.get_q(), q))
        self.assertEqual(self.ctx.camera.mode, "replay")
        self.assertIs(instance, self.ext.instances["diagnostics"])
        self.assertEqual(self.ctx.calibration.load("camera_intrinsics")["focal"], 555)

        self.modes.disconnect()
        self.assertEqual(self.modes.mode, "DISCONNECTED")
        self.assertEqual(self.ctx.calibration.load("camera_intrinsics")["focal"], 555)

    def test_bad_replay_directory_does_not_disrupt_real_mode(self):
        self.modes.switch_real(home=False)
        with self.assertRaisesRegex(ValueError, "Replay directory"):
            self.modes.switch_virtual(self.ctx.root / "nonexistent")
        self.assertEqual(self.modes.mode, "REAL")

    def test_busy_real_robot_refuses_disconnection(self):
        self.modes.switch_real(home=False)
        FakeRobot.moving = True
        with self.assertRaisesRegex(RuntimeError, "moving"):
            self.modes.switch_virtual(self.frames)
        self.assertEqual(self.modes.mode, "REAL")
        with self.assertRaisesRegex(RuntimeError, "moving"):
            self.modes.disconnect()
        self.assertEqual(self.modes.mode, "REAL")
        FakeRobot.moving = False

    def test_camera_busy_preserves_virtual(self):
        self.modes.switch_virtual(self.frames)
        q = [0, 0, 0, .1, .2, .3]
        self.ctx.robot.virtual_set_q(q)
        FakeCamera.fail_open = True
        with self.assertRaisesRegex(RuntimeError, "camera is busy"):
            self.modes.switch_real(home=True)
        self.assertEqual(self.modes.mode, "VIRTUAL")
        self.assertTrue(np.allclose(self.ctx.robot.get_q(), q))
        self.assertEqual(self.ctx.camera.mode, "replay")
        self.assertFalse(any(e[0] == "initialize" for e in FakeRobot.events))

    def test_robot_connection_failure_restores_virtual(self):
        self.modes.switch_virtual(self.frames)
        q = [0, .1, .2, .3, .4, .5]
        self.ctx.robot.virtual_set_q(q)
        FakeRobot.fail_initialize = True
        with self.assertRaisesRegex(RuntimeError, "robot not ready"):
            self.modes.switch_real(home=False)
        self.assertEqual(self.modes.mode, "VIRTUAL")
        self.assertEqual(self.ctx.camera.mode, "replay")
        self.assertTrue(np.allclose(self.ctx.robot.get_q(), q))

    def test_cli_refuses_implicit_real_connection(self):
        with redirect_stdout(io.StringIO()):
            dispatch(self.ctx, self.ext, self.modes, ["mode", "real"])
        self.assertEqual(self.modes.mode, "DISCONNECTED")
        with patch("builtins.input", return_value="wrong"), redirect_stdout(io.StringIO()):
            dispatch(self.ctx, self.ext, self.modes, ["mode", "real", "home"])
        self.assertFalse(any(e[0] == "initialize" for e in FakeRobot.events))

    def test_switch_cli_commands(self):
        with redirect_stdout(io.StringIO()):
            dispatch(self.ctx, self.ext, self.modes, ["mode", "virtual", str(self.frames)])
            dispatch(self.ctx, self.ext, self.modes, ["mode", "status"])
        self.assertEqual(self.modes.mode, "VIRTUAL")
        with patch("builtins.input", return_value="REFERENCE"), redirect_stdout(io.StringIO()):
            dispatch(self.ctx, self.ext, self.modes, ["mode", "real", "nohome"])
        self.assertEqual(self.modes.mode, "REAL")


if __name__ == "__main__":
    unittest.main()
