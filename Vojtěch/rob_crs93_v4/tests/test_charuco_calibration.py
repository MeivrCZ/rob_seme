"""Offline sanity tests, no robot or camera hardware necessary."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from extensions.charuco_calibration import (
    CharucoCalibration, _board, detect_image, _safe_charuco_points,
)


class TestCharucoCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ctx = SimpleNamespace(root=root,
                                   camera=SimpleNamespace(name="camera-crs93"),
                                   calibration=SimpleNamespace(save=lambda *_: None))
        self.ext = CharucoCalibration(self.ctx)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_board_image(self):
        board = _board(20, 15)
        png = self.ctx.root / "board.png"
        image = board.generateImage((1000, 1400), marginSize=0)
        self.assertTrue(cv2.imwrite(str(png), image))
        return png

    def test_detect_all_corners_on_generated_board(self):
        image = self._make_board_image()
        _, _, _, _, ci, _, mi, nm, nc = detect_image(image)
        self.assertEqual(nm, 70)
        self.assertEqual(nc, 117)
        self.assertEqual(len(ci), 117)
        self.assertEqual(len(mi), 70)

    def test_import_deduplicates(self):
        png = self._make_board_image()
        self.ext.import_images(png)
        self.ext.import_images(png)
        self.assertEqual(len(self.ext.images()), 1)

    def test_inspect_overlay(self):
        png = self._make_board_image()
        output = self.ext.inspect(png)
        self.assertTrue(output.is_file())
        self.assertEqual(self.ext.last_details["corners"], 117)

    def test_calibration_blocks_single_image(self):
        self.ext.import_images(self._make_board_image())
        with self.assertRaisesRegex(RuntimeError, "Only 1 usable views"):
            self.ext.calibrate(20, 15)

    def test_inconsistent_corner_ids_detected(self):
        points = np.zeros((10, 1, 2), np.float32)
        ids = np.arange(9, dtype=np.int32).reshape(-1, 1)
        cc, ci, warning = _safe_charuco_points(points, ids)
        self.assertIsNone(cc)
        self.assertIsNone(ci)
        self.assertIn("mismatch", warning)

    def test_inspect_mismatched_arrays_does_not_crash(self):
        png = self._make_board_image()
        gray = np.zeros((120, 160), dtype=np.uint8)
        result = (gray, gray, None, np.zeros((10,1,2), np.float32),
                  np.arange(9, dtype=np.int32).reshape(-1,1), [], None, 0, 9)
        with patch("extensions.charuco_calibration.detect_image", return_value=result):
            output = self.ext.inspect(png)
        self.assertTrue(output.exists())
        self.assertEqual(self.ext.last_details["corners"], 0)
        self.assertIn("mismatch", self.ext.last_details["detection_warning"])

    def test_capture_keeps_original_even_if_preview_fails(self):
        self.ctx.camera.capture = lambda: np.zeros((120, 160, 3), dtype=np.uint8)
        with patch.object(self.ext, "inspect", side_effect=RuntimeError("preview failed")):
            png = self.ext.capture()
        self.assertTrue(png.exists())

    def test_invalid_marker_dimension(self):
        with self.assertRaises(ValueError):
            self.ext.calibrate(20, 21)


if __name__ == "__main__":
    unittest.main()
