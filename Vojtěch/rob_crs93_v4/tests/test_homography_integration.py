"""Native console integration without talking to a physical robot or camera."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from app.extensions import ExtensionManager
from extensions.homography_calibration import HomographyExtension
from main import dispatch


class FakeRobot:
    mode = "real"
    released = False

    def __init__(self):
        self.q = np.array([.42, -.11, .07, 0., 0., 0.])
        self.wait_count = 0

    def assert_stationary(self):
        self.wait_count += 1

    def get_q(self):
        return self.q.copy()

    def require(self):
        return self

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q)[:3]
        return T

    def move_to_q(self, *args):
        raise AssertionError("Homography extension must never command motion")


class FakeCamera:
    mode = "real"

    def capture(self):
        return np.zeros((1200, 1920, 3), dtype=np.uint8)


def test_new_homo_shortcut_auto_loads_without_hardware(tmp_path, capsys):
    ctx = SimpleNamespace(root=tmp_path, robot=FakeRobot(), camera=FakeCamera())
    manager = ExtensionManager(ctx)
    assert dispatch(ctx, manager, None, ["homo", "help"])
    assert list(manager.instances) == ["homography_calibration"]
    assert "Homo" in capsys.readouterr().out
    assert dispatch(ctx, manager, None, ["homo", "status"])
    assert len(manager.instances) == 1


def test_capture_and_offset_persist_without_moving(tmp_path, monkeypatch):
    robot = FakeRobot()
    ctx = SimpleNamespace(root=tmp_path, robot=robot, camera=FakeCamera())
    ext = HomographyExtension(ctx)
    ext.run(["offset", "0.01", "0.02", "-0.01"])
    empty_restart = HomographyExtension(ctx)
    assert empty_restart.session.target_offset_tcp_m == pytest.approx((.01, .02, -.01))
    assert empty_restart.session.samples == []
    monkeypatch.setattr("homography_capture.select_point", lambda image, method="click": (700., 500.))
    ext.run(["capture"])
    assert robot.wait_count == 2
    assert ext.session.samples[0]["xyz_m"] == pytest.approx([.43, -.09, .06])
    assert (tmp_path / "data/homography/sample_001.png").is_file()
    assert ext.session.is_dirty
    restored = HomographyExtension(ctx)
    assert restored.session.target_offset_tcp_m == pytest.approx((.01,.02,-.01))
    assert restored.session.samples == ext.session.samples
    with pytest.raises(RuntimeError, match="solve"):
        restored.session.pixel_to_world(700, 500)
    with pytest.raises(RuntimeError, match="smaz body"):
        restored.run(["offset", "0", "0", "0"])
    restored.run(["undo"])
    assert not (tmp_path / "data/homography/sample_001.png").exists()


def test_no_virtual_or_replay_capture(tmp_path, monkeypatch):
    ctx = SimpleNamespace(root=tmp_path, robot=FakeRobot(), camera=FakeCamera())
    ext = HomographyExtension(ctx)
    ctx.robot.mode = "virtual"
    with pytest.raises(RuntimeError, match="REAL"):
        ext.run(["capture"])
    ctx.robot.mode = "real"
    ctx.camera.mode = "replay"
    with pytest.raises(RuntimeError, match="REAL"):
        ext.run(["capture"])
    ctx.camera.mode = "real"
    ctx.robot.released = True
    with pytest.raises(RuntimeError, match="released"):
        ext.run(["capture"])
