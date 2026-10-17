"""Virtual backend, camera replay and eye-to-hand solver: no robot required."""
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
from extensions.eye_to_hand import make_transform, solve_eye_to_hand


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


class EyeToHandMathTests(unittest.TestCase):
    def test_solve_recovers_ground_truth(self):
        rng = np.random.default_rng(42)
        def random_pose():
            r = cv2.Rodrigues(rng.normal(0., .75, size=3))[0]
            return make_transform(r, rng.uniform(-.25,.25,size=3))
        true_T_B_C = random_pose()
        true_T_F_T = random_pose()
        measurements = []
        for _ in range(20):
            T_B_F = random_pose()
            T_C_T = np.linalg.inv(true_T_B_C) @ T_B_F @ true_T_F_T
            measurements.append({'T_base_flange': T_B_F.tolist(), 'T_camera_target': T_C_T.tolist()})
        estimate = solve_eye_to_hand(measurements)
        self.assertTrue(np.allclose(estimate['T_base_camera'], true_T_B_C, atol=1e-7))
        self.assertTrue(np.allclose(estimate['T_flange_target'], true_T_F_T, atol=1e-7))
        self.assertLess(estimate['translation_residual_rms_mm'], 1e-5)
        self.assertLess(estimate['rotation_residual_rms_deg'], 1e-5)

    def test_requires_diverse_poses(self):
        g = np.eye(4).tolist()
        samples = [{'T_base_flange':g,'T_camera_target':g} for _ in range(8)]
        with self.assertRaisesRegex(ValueError, "diversity"):
            solve_eye_to_hand(samples)
        with self.assertRaisesRegex(ValueError, "8 diverse"):
            solve_eye_to_hand(samples[:7])


if __name__ == '__main__':
    unittest.main()
