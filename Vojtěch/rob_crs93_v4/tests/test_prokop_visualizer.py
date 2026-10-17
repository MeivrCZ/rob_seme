"""Offline read-only end-to-end integration tests; no robot or Viser required."""
from __future__ import annotations
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.prokop_visualizer import ProkopVisualizer
from extensions.prokop_bridge_runner import instrument_viewer

VIEWER_TEST_CODE = '''from pathlib import Path
import json
import time

def main():
    def set_q(q):
        Path(__file__).with_name("captured.json").write_text(json.dumps(q))
    time.sleep(20)

if __name__ == "__main__":
    main()
'''

class FakeRobot:
    def __init__(self):
        self.mode = 'virtual'
        self.connected = True
        self.released = False
        self.q = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    def get_q(self):
        return self.q.copy()

class ProkopBridgeTests(unittest.TestCase):
    def test_ast_rejects_viewer_without_set_q(self):
        with self.assertRaisesRegex(RuntimeError,'Expected one def set_q'):
            instrument_viewer('def main(): pass', 'viewer.py')

    def test_missing_viewer(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(root=Path(td), robot=FakeRobot())
            ext = ProkopVisualizer(ctx)
            with self.assertRaises(FileNotFoundError):
                ext.start()

    def test_read_only_sync_and_virtual_update(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            viewer = root/'Prokop'/'view_models.py'
            viewer.parent.mkdir()
            viewer.write_text(VIEWER_TEST_CODE)
            host = root/'Vojtech'/'rob_crs93_v4'
            host.mkdir(parents=True)
            robot = FakeRobot()
            ext = ProkopVisualizer(SimpleNamespace(root=host, robot=robot))
            self.assertEqual(ext._locate_viewer(), (root, viewer))
            try:
                ext.start(port=18765)
                self.assertTrue(ext._running())
                captured = viewer.with_name('captured.json')
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not captured.is_file():
                    time.sleep(0.05)
                self.assertTrue(captured.is_file(), (host/'data/logs/prokop_visualizer.log').read_text())
                self.assertEqual(json.loads(captured.read_text()), robot.q)
                robot.q = [0.4] * 6
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    if json.loads(captured.read_text()) == robot.q:
                        break
                    time.sleep(0.05)
                self.assertEqual(json.loads(captured.read_text()), robot.q)
                ext.sync_off()
                self.assertFalse(ext._thread is not None and ext._thread.is_alive())
            finally:
                ext.stop()
            self.assertFalse(ext._running())

    def test_disconnected_and_released_blocks_sync(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            viewer = root/'Prokop'/'view_models.py'
            viewer.parent.mkdir()
            viewer.write_text(VIEWER_TEST_CODE)
            host = root/'team'/'host'
            host.mkdir(parents=True)
            r = FakeRobot()
            ext = ProkopVisualizer(SimpleNamespace(root=host, robot=r))
            try:
                ext.start(18766, auto_sync=False)
                r.released = True
                with self.assertRaisesRegex(RuntimeError, 'released'):
                    ext.sync_on()
                with self.assertRaisesRegex(RuntimeError, 'released'):
                    ext.sync_once()
                r.released = False
                r.connected = False
                with self.assertRaisesRegex(RuntimeError, 'connect robot'):
                    ext.sync_on()
            finally:
                ext.stop()

if __name__ == '__main__':
    unittest.main()
