"""Pure OpenCV tests - no CRS93 hardware or physical motion."""
from pathlib import Path
import cv2
import numpy as np
import pytest

from hoop_detection import detect_hoop_circle
from homography_capture import detect_single_circle, HomographySession
from hoop_auto_scan import HoopAutoScanner


def image_with_hoops(centres, radius=168, stem=True):
    frame = np.full((1200, 1920, 3), 224, dtype=np.uint8)
    for cx, cy in centres:
        cv2.circle(frame, (cx, cy), radius, (52, 83, 72), 14)
        if stem:
            cv2.line(frame, (cx, cy-radius), (cx, 250), (52, 83, 72), 13)
    # A smaller circular part of the robot in the same field of view.
    cv2.circle(frame, (110, 140), 20, (55, 55, 55), 5)
    return frame


def test_hoop_with_attached_stem_and_two_edges():
    image = image_with_hoops([(260, 805)])
    result = detect_hoop_circle(image)
    assert result.center == pytest.approx((260, 805), abs=3)
    assert result.radius_px > 120
    assert detect_single_circle(image) == pytest.approx((260, 805), abs=3)
    assert result.method == 'contour'
    assert result.annotate(image).shape == image.shape


def test_radius_over_old_150_limit():
    image = image_with_hoops([(310, 810)], radius=220)
    assert detect_single_circle(image) == pytest.approx((310, 810), abs=3)


def test_two_real_rings_do_not_produce_false_sample():
    image = image_with_hoops([(290, 800), (1430, 800)], stem=False)
    result = detect_hoop_circle(image)
    assert result.center is None


def test_partial_ring_does_not_produce_false_sample():
    image = image_with_hoops([(65, 800)], stem=False)
    assert detect_single_circle(image) is None


def test_empty_camera_image():
    frame = np.full((1200, 1920, 3), 220, dtype=np.uint8)
    assert detect_single_circle(frame) is None


def test_offline_debug_of_saved_failure(tmp_path):
    from types import SimpleNamespace
    session = HomographySession(folder=tmp_path)
    cv2.imwrite(str(tmp_path / 'auto_failed_001.png'), image_with_hoops([(270, 810)]))
    scan = HoopAutoScanner(SimpleNamespace(), SimpleNamespace(), session)
    result = scan.diagnose_saved_failure()
    assert result.center == pytest.approx((270, 810), abs=3)
    assert (tmp_path / 'auto_failed_001_debug.png').exists()
    assert not session.samples


def test_test_command_never_moves(tmp_path):
    from types import SimpleNamespace
    class Stationary:
        mode = 'real'
        released = False
        real_motion_enabled = True
        def assert_stationary(self): pass
        def move_to_q(self, *args, **kwargs): raise AssertionError('NO MOTION')
    class Camera:
        mode = 'real'
        def capture(self): return image_with_hoops([(250, 805)])
    session = HomographySession(folder=tmp_path)
    scan = HoopAutoScanner(Stationary(), Camera(), session)
    result = scan.test_detection()
    assert result.center == pytest.approx((250, 805), abs=3)
    assert (tmp_path / 'hoop_test_raw.png').exists()
    assert (tmp_path / 'hoop_test_debug.png').exists()
    assert not session.samples


def test_solid_round_object_not_a_hoop():
    image = np.full((1200, 1920, 3), 230, np.uint8)
    cv2.circle(image, (720, 700), 154, (60, 60, 60), -1)
    assert detect_single_circle(image) is None
