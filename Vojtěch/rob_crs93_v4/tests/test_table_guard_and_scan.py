"""No physical hardware used in these safety/preflight tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from app.plane_guard import TableClearanceGuard
from app.robot import RobotManager
from extensions.homography_calibration import HomographyExtension

MODEL = Path(__file__).resolve().parents[3] / 'Prokop'


def test_clearance_checks_complete_urdf_and_hoop():
    guard = TableClearanceGuard(MODEL, .0)
    q = np.zeros(6)
    fk = lambda joints: guard.transforms(joints)['flange']
    assert guard.check(q, fk, [.135, 0, 0])[2] > .01
    with pytest.raises(ValueError, match='Measured hoop centre'):
        guard.check(q, fk, [0, 0, -.99])


def test_fk_mismatch_aborts_without_motion():
    guard = TableClearanceGuard(MODEL, 0.)
    with pytest.raises(RuntimeError, match='does not agree'):
        guard.check(np.zeros(6), lambda q: np.eye(4), [.135, 0, 0])


def test_joint_mesh_surface_clearance():
    guard = TableClearanceGuard(MODEL, .2)
    with pytest.raises(ValueError, match='link1'):
        guard.check(np.zeros(6), lambda q: guard.transforms(q)['flange'], [.135, 0, 0])


def test_missing_table_blocks_real_moves():
    from test_motion import DummyCRS93
    from unittest.mock import patch
    import sys, types
    module = types.ModuleType('ctu_crs')
    module.CRS93 = DummyCRS93
    with patch.dict(sys.modules, {'ctu_crs': module}):
        robot = RobotManager('/dev/mock', real_motion_enabled=True)
        robot.connect(home=False)
        with pytest.raises(RuntimeError, match='robot table Z'):
            robot.move_to_q(np.zeros(6))
        assert not robot.require().moves
        robot.close()
