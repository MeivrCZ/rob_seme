from pathlib import Path
import math

import cv2
import numpy as np
import pytest

from homography import Homography
from homography_capture import HomographySession


def make_points(seed=1, noise=0.35, outliers=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(low=(.25, -.24), high=(.54, .24), size=(60, 2))
    H = np.array([[2200., -120., -480.], [150., 1900., 580.], [.15, -.1, 1.]], dtype=float)
    uv = cv2.perspectiveTransform(points.reshape(-1, 1, 2), H).reshape(-1, 2)
    uv += rng.normal(0, noise, uv.shape)
    for index in range(outliers):
        uv[index] = np.array([400., 400.]) + index * 7
    return points, uv, H


def test_homography_roundtrip(tmp_path):
    xy, px, true_H = make_points()
    fitted = Homography.fit(px, xy, (1920, 1200), z_m=.05)
    assert fitted.rms_px < 1.0
    assert fitted.inliers > 55
    true_px = cv2.perspectiveTransform(xy[15:20].reshape(-1, 1, 2), true_H).reshape(-1, 2)
    for (x, y), (u, v) in zip(xy[15:20], true_px):
        assert np.linalg.norm(np.asarray(fitted.pixel_to_world(u, v, strict=False)) - (x, y)) < .001
    path = tmp_path / "H.npz"
    fitted.save(path)
    read = Homography.load(path)
    np.testing.assert_allclose(read.H, fitted.H)
    assert read.image_size == (1920, 1200)
    with pytest.raises(ValueError, match="Rozliseni"):
        read.pixel_to_world(600, 700, image_size=(1280, 720))


def test_outliers_handled(tmp_path):
    xy, px, _ = make_points(outliers=10)
    fitted = Homography.fit(px, xy, (1920, 1200), z_m=.05)
    assert 47 <= fitted.inliers <= 52
    assert fitted.rms_px < 1


def test_z_plane_and_samples(tmp_path):
    s = HomographySession(folder=tmp_path)
    s.add((120, 200), (.2, .1, .05), (1920, 1200))
    with pytest.raises(ValueError, match="jedne rovine"):
        s.add((130, 200), (.3, .1, .07), (1920, 1200))
    s2 = HomographySession(folder=tmp_path)
    assert s2.load_samples() == 1
    s2.undo()
    assert not s2.samples


def test_reject_uncovered_frame():
    xy = np.array([[.2, .1], [.3, .1], [.3, .2], [.2, .2]])
    px = np.array([[10., 10.], [20., 10.], [20., 20.], [10., 20.]])
    with pytest.raises(ValueError, match="nepokryvaji"):
        Homography.fit(px, xy, (1920, 1200))


def test_prokop_visual_offset_is_renderer_only():
    """The viewer must feed an offset to URDF, never to FK or GUI joint state."""
    import ast
    path = Path(__file__).resolve().parents[3] / "Prokop" / "view_models.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Avoid importing Viser or hardware: check the exact expression passed to renderer.
    assert "robot_vis.update_cfg(display_q(q_now))" in source
    assert "T = robot.fk(q_now)" in source
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "display_q")
    assert "shown[5] += JOINT6_VIEW_OFFSET_RAD" in ast.get_source_segment(source, method)
    assert "shown = np.asarray(q, dtype=float).copy()" in ast.get_source_segment(source, method)
    assert "q_now = np.array(q, dtype=float)" in source


def test_live_capture_never_moves_robot(tmp_path, monkeypatch):
    from homography_capture import select_point

    class FakeRobot:
        def __init__(self):
            self.q = np.zeros(6)
        def in_motion(self):
            return False
        def get_q(self):
            return self.q.copy()
        def fk(self, q):
            T = np.eye(4)
            T[:3, 3] = [.4, -.1, .05]
            return T
        def grab_image(self):
            return np.zeros((1200, 1920, 3), np.uint8)
        def move_to_q(self, *args):
            raise AssertionError("Kalibrace nesmi hybat robotem")

    monkeypatch.setattr('homography_capture.select_point', lambda img, method='click': (100., 200.))
    s = HomographySession(folder=tmp_path, target_offset_tcp_m=(.01, -.02, 0.))
    assert s.capture(FakeRobot())
    assert s.samples[0]['xyz_m'] == pytest.approx([.41, -.12, .05])
    assert (tmp_path / 'sample_001.png').is_file()


