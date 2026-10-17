"""Offline tests for continuous capture and rejected-calibration diagnostics."""
import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.store import CalibrationStore
from extensions.charuco_calibration import CharucoCalibration, _board


class TestLoopAndDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.ctx = SimpleNamespace(
            root=root,
            camera=SimpleNamespace(mode=None, name="camera-crs93"),
            calibration=CalibrationStore(root / "data" / "calibration"),
        )
        self.ext = CharucoCalibration(self.ctx)
        self.addCleanup(self.ext.close)

    def test_start_requires_open_camera(self):
        with self.assertRaisesRegex(RuntimeError, "Open camera"):
            self.ext.start_loopcapture()

    def test_loop_saves_once_and_skips_duplicate(self):
        board = _board(20, 15)
        frame = board.generateImage((700, 900))
        self.ctx.camera.mode = "replay"
        self.ctx.camera.capture = lambda: frame.copy()
        with redirect_stdout(io.StringIO()):
            self.ext.start_loopcapture(interval=0.5, max_frames=2)
            time.sleep(0.85)
            self.ext.stop_loopcapture()
        self.assertEqual(len(self.ext.images()), 1)
        self.assertGreaterEqual(self.ext._loop_stats["skipped"], 1)
        self.assertFalse(self.ext.loopcapture_running)

    def test_archive_keeps_images_but_excludes_from_next_calibration(self):
        path = self.ext.capture_dir / "test.png"
        cv2.imwrite(str(path), np.zeros((10, 10), np.uint8))
        with redirect_stdout(io.StringIO()):
            archived = self.ext.archive_images()
        self.assertTrue((archived / "test.png").is_file())
        self.assertEqual(self.ext.images(), [])

    def test_rejected_calibration_saves_full_diagnostics_not_intrinsics(self):
        for i in range(12):
            (self.ext.capture_dir / f"{i:02d}.png").touch()
        image = np.zeros((1000, 1500), np.uint8)
        class Detector:
            count = 0
            def detectBoard(self, frame):
                self.count += 1
                i = self.count
                pts = np.column_stack((
                    np.linspace(200, 400, 25) * (1+i*0.012) + 18*i,
                    np.linspace(100, 430, 25) * (1+i*0.012) + 11*i,
                )).astype(np.float32).reshape(-1, 1, 2)
                return pts, np.arange(25, dtype=np.int32).reshape(-1, 1), [], None
        fake_detector = Detector()
        K = np.array([[15213.6, 0, 9000], [0, 14106.5, -3000], [0, 0, 1]], float)
        with (patch("extensions.charuco_calibration._read_gray", return_value=(image, image)),
              patch.object(cv2.aruco, "CharucoDetector", return_value=fake_detector),
              patch.object(cv2, "calibrateCamera", return_value=(
                  0.23, K, np.zeros((1, 5)), [np.zeros((3, 1))] * 12,
                  [np.array([[0.], [0.], [2.]])] * 12))):
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "REJECTED"):
                    self.ext.calibrate(20, 15)
        diag = self.ctx.calibration.load("camera_intrinsics_diagnostics")
        self.assertEqual(len(diag["per_view"]), 12)
        self.assertEqual(diag["camera_matrix"], K.tolist())
        self.assertIn("cx=9000.0", "\n".join(diag["rejection_reasons"]))
        self.assertIn("cy=-3000.0", "\n".join(diag["rejection_reasons"]))
        self.assertIn("Full K", output.getvalue())
        self.assertIn("total=0.0", output.getvalue())
        self.assertIsNone(self.ctx.calibration.load("camera_intrinsics"))

    def test_early_failure_saves_diagnostics_without_matrix(self):
        (self.ext.capture_dir / "empty.png").touch()
        image = np.zeros((1000, 1500), np.uint8)
        class Detector:
            def detectBoard(self, img):
                return None, None, [], None
        with (patch("extensions.charuco_calibration._read_gray", return_value=(image, image)),
              patch.object(cv2.aruco, "CharucoDetector", return_value=Detector()),
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "Only 0 usable views"):
                self.ext.calibrate(20, 15)
        d = self.ctx.calibration.load("camera_intrinsics_diagnostics")
        self.assertIsNone(d["camera_matrix"])
        self.assertFalse(d["accepted"])


if __name__ == "__main__":
    unittest.main()
