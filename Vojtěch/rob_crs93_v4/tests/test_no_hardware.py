"""Smoke tests: no robot or camera hardware needed."""
import tempfile
import unittest
from pathlib import Path

from app.camera import CameraManager
from app.robot import RobotManager
from app.store import CalibrationStore
from app.context import Context
from app.extensions import ExtensionManager


class OfflineTests(unittest.TestCase):
    def test_persistent_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(Path(tmp))
            store.save("calibration_example", {"metadata": [1, 2, 3]})
            self.assertEqual(store.names(), ["calibration_example"])
            self.assertEqual(store.load("calibration_example")["metadata"][0], 1)

    def test_extensions_do_not_connect_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp), RobotManager(), CameraManager(), CalibrationStore(Path(tmp) / "calib"))
            mgr = ExtensionManager(ctx)
            mgr.load("diagnostics")
            mgr.call("diagnostics", ["status"])
            self.assertFalse(ctx.robot.connected)
            self.assertFalse(ctx.camera.connected)
            mgr.reload("diagnostics")
            mgr.unload("diagnostics")
            self.assertEqual(mgr.instances, {})

    def test_homography_help_without_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp), RobotManager(), CameraManager(), CalibrationStore(Path(tmp) / "calib"))
            mgr = ExtensionManager(ctx)
            mgr.load("homography_calibration")
            mgr.call("homography_calibration", ["help"])
            mgr.unload("homography_calibration")


if __name__ == "__main__":
    unittest.main()
