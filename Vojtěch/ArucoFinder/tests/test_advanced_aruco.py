import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ARUCO_FINDER = Path(__file__).resolve().parents[1]
VOJTECH = ARUCO_FINDER.parent
with pytest.MonkeyPatch.context() as imports:
    for directory in (VOJTECH.parent / "Prokop", VOJTECH / "Homografy", ARUCO_FINDER):
        imports.syspath_prepend(str(directory))
    # rob_crs93_v4 has its own module with the same top-level name.
    imports.delitem(sys.modules, "homography", raising=False)
    try:
        from advanced_aruco import AdvancedArucoFinder, ArucoResult, TimedFrame
        from aruco_block_detector import ArucoBlockDetector, BlockPose
        from camera_phase_tracker import CameraPhaseTracker
        from homography import Homography
        from solver.contracts import BaseMeasurement, TurntableModel
    finally:
        sys.modules.pop("homography", None)


@pytest.fixture
def homography():
    calibration = Homography()
    # An overhead camera with 1 mm per pixel in the marker plane.
    calibration.P = np.array([
        [1000.0, 0.0, 0.0, 0.0],
        [0.0, 1000.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return calibration


def frame(t_capture, *, x_shift=None):
    image = np.full((320, 320), 255, dtype=np.uint8)
    if x_shift is not None:
        dictionary = cv2.aruco.getPredefinedDictionary(ArucoBlockDetector.DICTIONARY)
        for marker_id, x, y in ((1, 40, 60), (2, 184, 204)):
            x += x_shift
            image[y:y + 64, x:x + 64] = cv2.aruco.generateImageMarker(dictionary, marker_id, 64)
    return TimedFrame(image, t_capture)


def model(direction):
    # The fitted offset and yaw intentionally differ from the observations.
    return TurntableModel(
        axis_xy=np.array([0.18, 0.12]),
        base_offset=np.array([0.01, 0.02]),
        base_yaw=0.5,
        direction=direction,
        omega=np.pi / 2,
        t_ref=2.0,
    )


@pytest.mark.parametrize("direction", (1, -1))
def test_missing_markers_follow_latest_measured_pose(homography, direction):
    turntable = model(direction)
    finder = AdvancedArucoFinder(ArucoBlockDetector(), homography, turntable)
    measured = finder.advAruco(frame(10.0, x_shift=0))
    assert measured is not None
    assert measured.pose is not None
    assert measured.pose.visible_ids == (1, 2)
    assert measured.xy == pytest.approx([0.1435, 0.1635], abs=0.0005)
    assert measured.t_last_valid == 10.0

    estimated = finder.advAruco(frame(11.0))
    assert estimated is not None
    assert estimated.pose is None
    assert estimated.t_capture == 11.0
    assert estimated.t_last_valid == 10.0
    x, y = measured.xy - turntable.axis_xy
    assert estimated.xy == pytest.approx(turntable.axis_xy + [-direction * y, direction * x])
    assert estimated.yaw == pytest.approx(measured.yaw + direction * np.pi / 2)

    later_estimate = finder.advAruco(frame(12.0))
    assert later_estimate is not None
    assert later_estimate.t_last_valid == 10.0
    assert later_estimate.xy == pytest.approx(2 * turntable.axis_xy - measured.xy)

    recovered = finder.advAruco(frame(15.0, x_shift=40))
    assert recovered is not None
    assert recovered.pose is not None
    assert recovered.xy == pytest.approx([0.1835, 0.1635], abs=0.0005)
    estimated_after_recovery = finder.advAruco(frame(17.0))
    assert estimated_after_recovery is not None
    assert estimated_after_recovery.t_last_valid == 15.0
    assert estimated_after_recovery.xy == pytest.approx(2 * turntable.axis_xy - recovered.xy)


def test_fallback_requires_a_model_and_a_valid_measurement(homography):
    detector = ArucoBlockDetector()
    without_measurement = AdvancedArucoFinder(detector, homography, model(1))
    assert without_measurement.advAruco(frame(10.0)) is None

    without_model = AdvancedArucoFinder(detector, homography)
    measured = without_model.advAruco(frame(10.0, x_shift=0))
    assert measured is not None
    assert measured.pose is not None
    assert without_model.advAruco(frame(11.0)) is None


def test_orbit_seed_is_preserved_when_older_frames_arrive(homography):
    turntable = model(1)
    seed = BaseMeasurement(10.0, np.array([0.21, 0.12]), np.deg2rad(170), 2)
    finder = AdvancedArucoFinder(
        ArucoBlockDetector(), homography, turntable, last_measurement=seed,
    )

    finder.advAruco(frame(9.0, x_shift=0))
    assert finder.advAruco(frame(9.5)) is None
    estimated = finder.advAruco(frame(11.0))
    assert estimated is not None
    assert estimated.t_last_valid == seed.t_capture
    assert estimated.xy == pytest.approx([0.18, 0.15])
    assert estimated.yaw == pytest.approx(np.deg2rad(-100))


def phase_observation(turntable, capture_time, *, measured_at=None):
    measurement_time = capture_time if measured_at is None else measured_at
    theta = turntable.theta_at(measurement_time)
    offset = complex(*turntable.base_offset) * np.exp(1j * theta)
    xy = turntable.axis_xy + [offset.real, offset.imag]
    yaw = float(np.angle(np.exp(1j * (theta + turntable.base_yaw))))
    pose = BlockPose((0, 0), 0.0, (1, 2), {})
    return ArucoResult(capture_time, xy, yaw, pose, capture_time)


@pytest.mark.parametrize("direction", (1, -1))
def test_camera_phase_tracker_ignores_fallback_and_outlier(homography, direction):
    turntable = model(direction)
    turntable.omega = np.deg2rad(12.0)
    turntable.theta_ref = np.deg2rad(179 * direction) - turntable.base_yaw
    finder = AdvancedArucoFinder(ArucoBlockDetector(), homography, turntable)
    tracker = CameraPhaseTracker(finder, lambda: frame(0.0).image, camera_delay=0.11)

    for timestamp in (2.0, 2.2, 2.5):
        tracker._observe(phase_observation(turntable, timestamp))
    estimate = tracker.estimate()
    assert estimate.t == 2.5
    assert estimate.theta == pytest.approx(turntable.theta_at(estimate.t))

    observed = phase_observation(turntable, 2.7)
    tracker._observe(ArucoResult(2.7, observed.xy, observed.yaw, None, 2.5))
    tracker._observe(ArucoResult(2.7, observed.xy + 0.1, observed.yaw, observed.pose, 2.7))
    assert tracker.estimate().t == 2.5
    assert estimate.predict(3.1)[1] > estimate.predict(2.7)[1]

    tracker._observe(phase_observation(turntable, 3.1))
    assert tracker.estimate().t == 3.1
    assert tracker.estimate().predict(3.5)[0] == pytest.approx(turntable.theta_at(3.5))


def test_camera_phase_tracker_checks_that_turntable_moves(homography):
    turntable = model(1)
    turntable.omega = np.deg2rad(12.0)
    finder = AdvancedArucoFinder(ArucoBlockDetector(), homography, turntable)
    tracker = CameraPhaseTracker(finder, lambda: frame(0.0).image, camera_delay=0.11)

    for timestamp in (2.0, 2.2, 2.4, 2.6, 2.8):
        tracker._observe(phase_observation(turntable, timestamp, measured_at=2.0))
    assert not tracker.wait_ready(timeout=0)
