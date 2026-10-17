"""Tests needing only the Python standard library (not ctu_crs/viser)."""
from __future__ import annotations

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.prokop_visualizer import ProkopVisualizer


class BridgeTests(unittest.TestCase):
    def test_missing_viewer(self):
        with tempfile.TemporaryDirectory() as td:
            ext = ProkopVisualizer(SimpleNamespace(root=Path(td)))
            with self.assertRaises(FileNotFoundError):
                ext.start()

    def test_repo_sibling_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            viewer = root / 'Prokop' / 'view_models.py'
            viewer.parent.mkdir()
            viewer.write_text('pass\n')
            host = root / 'Vojtech' / 'our_app'
            host.mkdir(parents=True)
            ext = ProkopVisualizer(SimpleNamespace(root=host))
            self.assertEqual(ext._locate_viewer(), (root, viewer))

    def test_process_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            viewer = root / 'Prokop' / 'view_models.py'
            viewer.parent.mkdir()
            viewer.write_text('import time\ntime.sleep(30)\n')
            host = root / 'OurFolder'
            host.mkdir()
            ext = ProkopVisualizer(SimpleNamespace(root=host))
            with contextlib.redirect_stdout(io.StringIO()):
                ext.start(8099)
                self.assertTrue(ext._running())
                self.assertEqual(ext.port, 8099)
                with self.assertRaises(RuntimeError):
                    ext.start(8099)
                ext.stop()
            self.assertFalse(ext._running())
            self.assertIsNone(ext.process)
            self.assertIsNone(ext.port)

    def test_invalid_port(self):
        ext = ProkopVisualizer(SimpleNamespace(root=Path.cwd()))
        with self.assertRaises(ValueError):
            ext.start(70000)

    def test_unload_idempotent(self):
        ext = ProkopVisualizer(SimpleNamespace(root=Path.cwd()))
        with contextlib.redirect_stdout(io.StringIO()):
            ext.close()
            ext.close()


if __name__ == '__main__':
    unittest.main()
