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
            store.save("camera_intrinsics", {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
            self.assertEqual(store.names(), ["camera_intrinsics"])
            self.assertEqual(store.load("camera_intrinsics")["camera_matrix"][0][0], 1)

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

    def test_intrinsics_help_without_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp), RobotManager(), CameraManager(), CalibrationStore(Path(tmp) / "calib"))
            mgr = ExtensionManager(ctx)
            mgr.load("camera_intrinsics")
            mgr.call("camera_intrinsics", ["help"])
            mgr.unload("camera_intrinsics")


if __name__ == "__main__":
    unittest.main()
