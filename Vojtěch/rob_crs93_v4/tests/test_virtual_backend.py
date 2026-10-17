"""Virtual CRS93 backend and camera replay: no robot required."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from app.camera import CameraManager
from app.robot import RobotManager


class FakeCRS93:
    """ONLY a tiny stand-in for testing backend management without ctu_crs installed."""
    def __init__(self, tty_dev=None):
        assert tty_dev is None, "Fake hardware was accidentally requested!"
        self.q_home = np.array([.1, -.2, .3, -.4, .5, -.6])
        self._tty = tty_dev

    def fk(self, q):
        t = np.eye(4)
        t[:3, :3] = cv2.Rodrigues(np.asarray(q)[3:])[0]
        t[:3, 3] = q[:3]
        return t

    def ik(self, pose):
        return [self.q_home]


class VirtualTests(unittest.TestCase):
    def test_virtual_robot_not_hardware(self):
        fake_package = types.ModuleType("ctu_crs")
        fake_package.CRS93 = FakeCRS93
        with patch.dict(sys.modules, {"ctu_crs": fake_package}):
            robot = RobotManager("/this/path/must/not/be/opened")
            robot.connect_virtual()
            self.assertEqual(robot.mode, "virtual")
            self.assertTrue(np.allclose(robot.get_q(), FakeCRS93().q_home))
            q = np.array([0., 1., 2., .1, .2, .3])
            robot.virtual_set_q(q)
            self.assertTrue(np.allclose(robot.flange_pose()[:3, 3], [0,1,2]))
            self.assertEqual(len(robot.ik(np.eye(4))), 1)
            with self.assertRaises(RuntimeError):
                robot.connect(home=True)
            robot.soft_home()
            self.assertTrue(np.allclose(robot.get_q(), FakeCRS93().q_home))
            robot.close()
            self.assertFalse(robot.connected)

    def test_replay_loops_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            self.assertTrue(cv2.imwrite(str(source/'1.png'), np.full((12, 16), 8, dtype=np.uint8)))
            self.assertTrue(cv2.imwrite(str(source/'2.png'), np.full((12, 16), 16, dtype=np.uint8)))
            cam = CameraManager()
            cam.open_replay(source)
            self.assertEqual(cam.mode, 'replay')
            self.assertEqual(int(cam.capture()[0,0]), 8)
            self.assertEqual(int(cam.capture()[0,0]), 16)
            self.assertEqual(int(cam.capture()[0,0]), 8)
            cam.close()
            self.assertFalse(cam.connected)



if __name__ == "__main__":
    unittest.main()
