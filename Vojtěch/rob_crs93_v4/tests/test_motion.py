"""Offline safety and CLI tests: none of these commands access a physical robot."""
from __future__ import annotations

import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.robot import RobotManager
from app.context import Context
from app.camera import CameraManager
from app.store import CalibrationStore
from app.extensions import ExtensionManager
from app.modes import ModeManager
from main import dispatch


class DummyCRS93:
    instances = []
    def __init__(self, tty_dev=None):
        self.tty_dev = tty_dev
        self.q_home = np.zeros(6)
        self.q = np.zeros(6)
        self.moves = []
        self.moving = False
        self.instances.append(self)

    def initialize(self, home=True): pass
    def close(self): pass
    def in_motion(self): return self.moving
    def get_q(self): return self.q.copy()
    def move_to_q(self, q):
        self.moves.append(np.array(q).copy())
        self.q = np.array(q).copy()
    def wait_for_motion_stop(self): pass
    def fk(self, q): return np.eye(4)
    def ik(self, pose): return [self.q_home]
    def release(self): pass


class MotionTests(unittest.TestCase):
    def setUp(self):
        DummyCRS93.instances = []
        pkg = types.ModuleType('ctu_crs')
        pkg.__path__ = []
        pkg.CRS93 = DummyCRS93
        self.fake = patch.dict(sys.modules, {'ctu_crs': pkg})
        self.fake.start()
        self.addCleanup(self.fake.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.limits = [[-90, 90] for _ in range(6)]  # TEST-ONLY, NOT physical CRS93 limits.

    def mk(self, **options):
        robot = RobotManager('/dev/mock', **options)
        self.addCleanup(robot.close)
        return robot

    def run_cli(self, robot, args, confirm=None):
        ctx = Context(self.root, robot, CameraManager('none'), CalibrationStore(self.root / 'calib'))
        ext = ExtensionManager(ctx)
        self.addCleanup(ext.close)
        modes = ModeManager(ctx)
        if confirm is None:
            with redirect_stdout(io.StringIO()):
                return dispatch(ctx, ext, modes, args)
        with patch('builtins.input', return_value=confirm), redirect_stdout(io.StringIO()):
            return dispatch(ctx, ext, modes, args)

    def test_no_real_motion_by_default(self):
        bot = self.mk()
        bot.connect(home=False)
        self.assertRaisesRegex(RuntimeError, 'disabled', bot.preview_move, np.deg2rad([1, 0, 0, 0, 0, 0]))
        self.assertEqual(bot.require().moves, [])

    def test_real_without_joint_limits_always_blocked(self):
        bot = self.mk(real_motion_enabled=True)
        bot.connect(home=False)
        with self.assertRaisesRegex(RuntimeError, 'joint_limits_deg'):
            bot.move_to_q(np.deg2rad([1, 0, 0, 0, 0, 0]))

    def test_small_relative_joint_motion_requires_human_confirm(self):
        bot = self.mk(real_motion_enabled=True, joint_limits_deg=self.limits, max_step_deg=3)
        bot.connect(home=False)
        self.run_cli(bot, ['robot', 'jog', 'q0', '2'], 'CANCEL')
        self.assertEqual(len(bot.require().moves), 0)
        self.run_cli(bot, ['robot', 'jog', 'q0', '2'], 'MOVE')
        self.assertEqual(len(bot.require().moves), 1)
        self.assertAlmostEqual(np.rad2deg(bot.get_q()[0]), 2)

    def test_absolute_goal_degree_and_virtual_mode(self):
        bot = self.mk()
        bot.connect_virtual()
        self.run_cli(bot, ['robot', 'goto', 'deg', '2', '3', '4', '5', '6', '7'])
        np.testing.assert_allclose(np.rad2deg(bot.get_q()), [2, 3, 4, 5, 6, 7])
        self.run_cli(bot, ['robot', 'jog', 'q5', '-2'])
        self.assertAlmostEqual(np.rad2deg(bot.get_q()[5]), 5)
        self.assertEqual(bot.require().moves, [])

    def test_real_limits_step_moving_and_release_guard(self):
        bot = self.mk(real_motion_enabled=True, joint_limits_deg=self.limits, max_step_deg=3)
        bot.connect(home=False)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            bot.preview_move(np.deg2rad([4, 0, 0, 0, 0, 0]))
        bot.require().q[0] = np.deg2rad(89)
        with self.assertRaisesRegex(ValueError, 'outside'):
            bot.preview_move(np.deg2rad([91, 0, 0, 0, 0, 0]))
        bot.require().q[0] = 0
        bot.require().moving = True
        with self.assertRaisesRegex(RuntimeError, 'moving'):
            bot.preview_move(np.zeros(6))
        bot.require().moving = False
        bot.release()
        with self.assertRaisesRegex(RuntimeError, 'released'):
            bot.move_to_q(np.zeros(6))

    def test_stale_state_abort(self):
        bot = self.mk(real_motion_enabled=True, joint_limits_deg=self.limits)
        bot.connect(home=False)
        now, goal = bot.preview_move(np.deg2rad([1, 0, 0, 0, 0, 0]))
        bot.require().q[0] = np.deg2rad(1)
        with self.assertRaisesRegex(RuntimeError, 'since preview'):
            bot.move_to_q(goal, expected_q=now)
        self.assertEqual(bot.require().moves, [])

    def test_invalid_config_and_values(self):
        with self.assertRaises(ValueError): self.mk(real_motion_enabled=True, joint_limits_deg=[[0, 10]])
        with self.assertRaises(ValueError): self.mk(max_step_deg=999)
        bot = self.mk()
        bot.connect_virtual()
        with self.assertRaises(ValueError): bot.preview_move([np.nan] * 6)
        with self.assertRaises(ValueError): self.run_cli(bot, ['robot', 'jog', 'q6', '2'])


if __name__ == '__main__':
    unittest.main()
