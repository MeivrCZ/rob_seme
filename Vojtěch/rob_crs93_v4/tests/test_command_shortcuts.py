"""Tests for extension discovery and Prokop viewer shortcuts; no physical robot."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.extensions import ExtensionManager
from main import dispatch


class CommandsTests(unittest.TestCase):
    def test_discovery_ignores_helper_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            extension_dir = Path(tmp) / "extensions"
            extension_dir.mkdir()
            (extension_dir / "diagnostics.py").write_text("def setup(ctx): return None\n")
            (extension_dir / "my_extension.py").write_text("def setup(ctx): return None\n")
            (extension_dir / "prokop_bridge_runner.py").write_text("def run(): return None\n")
            (extension_dir / "__init__.py").write_text("")
            (extension_dir / "_internal.py").write_text("def setup(ctx): return None\n")
            ctx = SimpleNamespace(root=Path(tmp))
            manager = ExtensionManager(ctx)
            self.assertEqual(manager.available(), ["diagnostics", "my_extension"])

    def test_load_all_skips_existing_and_continues_after_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ext_dir = Path(tmp) / "extensions"
            ext_dir.mkdir()
            for name in ["already", "bad", "good"]:
                (ext_dir / f"{name}.py").write_text("def setup(ctx): return None\n")
            ctx = SimpleNamespace(root=Path(tmp))
            manager = ExtensionManager(ctx)
            manager.instances["already"] = object()
            calls = []
            def fake_load(name):
                calls.append(name)
                if name == "bad":
                    raise ImportError("missing dependency")
                manager.instances[name] = object()
            manager.load = fake_load
            with contextlib.redirect_stdout(io.StringIO()):
                result = manager.load_all()
            self.assertEqual(calls, ["bad", "good"])
            self.assertEqual(result["loaded"], ["good"])
            self.assertEqual(result["skipped"], ["already"])
            self.assertIn("bad", result["failed"])

    def test_pview_start_autoload_and_is_idempotent(self):
        class Viewer:
            running = False
            def _running(self):
                return self.running
        manager = SimpleNamespace(instances={}, call=Mock())
        viewer = Viewer()
        def load(name):
            self.assertEqual(name, "prokop_visualizer")
            manager.instances[name] = viewer
        manager.load = Mock(side_effect=load)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(dispatch(None, manager, None, ["pview", "start"]))
        manager.load.assert_called_once_with("prokop_visualizer")
        manager.call.assert_called_once_with("prokop_visualizer", ["start"])
        viewer.running = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(dispatch(None, manager, None, ["pview", "start"]))
        manager.load.assert_called_once()
        self.assertEqual(manager.call.call_args.args, ("prokop_visualizer", ["status"]))

    def test_pview_port_and_stop(self):
        viewer = SimpleNamespace(_running=lambda: False)
        manager = SimpleNamespace(instances={"prokop_visualizer": viewer}, call=Mock())
        with contextlib.redirect_stdout(io.StringIO()):
            dispatch(None, manager, None, ["pview", "start", "8082"])
            dispatch(None, manager, None, ["pview", "stop"])
        self.assertEqual(manager.call.call_args_list[0].args,
                         ("prokop_visualizer", ["start", "8082"]))
        self.assertEqual(manager.call.call_args_list[1].args,
                         ("prokop_visualizer", ["stop"]))

    def test_load_all_dispatch(self):
        manager = SimpleNamespace(load_all=Mock())
        with contextlib.redirect_stdout(io.StringIO()):
            dispatch(None, manager, None, ["load", "all"])
        manager.load_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
