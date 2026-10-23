"""camera.py: the rendered image agrees with the camera model (where things are
in the image), and has the format of the real camera."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from camera import Camera, SceneRenderer
from cell import CAMERA_RESOLUTION
from placement import Placement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_survey import find_ring  # noqa: E402

# Survey pose of experiment E8: hoop face-on, 49 cm above the work area.
SURVEY_Q = np.deg2rad([-7.5, -4.04, -91.71, 0.0, -84.26, 0.0])
OUT_OF_VIEW = Placement(np.array([-0.6, 0.0]), np.zeros(2), 0.0)
IN_VIEW = Placement(np.array([0.48, 0.02]), np.array([0.01, -0.02]), 0.4)


@pytest.fixture(scope="module")
def renderer(task):
    return SceneRenderer(task)


def markers(image) -> dict[int, np.ndarray]:
    """ArUco 4x4_50 markers: id -> corners (4, 2)."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(image)
    return {} if ids is None else {int(i): c.reshape(4, 2)
                                   for i, c in zip(ids.ravel(), corners)}


def test_format(renderer, robot):
    image = renderer.render(robot.q_home, IN_VIEW, 0.0)
    w, h = CAMERA_RESOLUTION
    assert image.shape == (h, w, 3) and image.dtype == np.uint8


def test_deterministic(renderer, robot):
    a = renderer.render(robot.q_home, IN_VIEW, 0.3)
    b = renderer.render(robot.q_home, IN_VIEW, 0.3)
    assert np.array_equal(a, b)


def test_empty_table_colour(renderer, robot):
    """The table renders light grey, like the real one (~215)."""
    image = renderer.render(robot.q_home, OUT_OF_VIEW, 0.0)
    assert np.allclose(np.median(image.reshape(-1, 3), axis=0), 215, atol=8)


@pytest.mark.parametrize("angle", [0.0, 0.7, 2.5])
def test_markers_where_projected(renderer, robot, task, angle):
    """Both base markers (ids 1, 2 of DICT_4X4_50) are detected at the
    projection of their centres (base.urdf guide_1 / guide_2)."""
    found = markers(renderer.render(robot.q_home, IN_VIEW, angle))
    assert set(found) == {1, 2}
    T_base = renderer.poses(robot.q_home, IN_VIEW, angle)["base"]
    for marker_id, link in ((1, "guide_1"), (2, "guide_2")):
        centre = (T_base @ task.base.get_transform(link, task.base.base_link))[:3, 3]
        assert np.linalg.norm(found[marker_id].mean(axis=0)
                              - renderer.camera.project(centre)) < 2.0


def test_turntable_rotates_markers(renderer, robot):
    """Turning the turntable top by a turns the markers in the image by a (the
    camera looks straight down; image y points down, so the sense flips)."""
    def marker_angle(angle):
        c = markers(renderer.render(robot.q_home, IN_VIEW, angle))[1]
        edge = c[1] - c[0]
        return np.arctan2(edge[1], edge[0])

    turn = marker_angle(0.5) - marker_angle(0.0)
    assert abs(np.angle(np.exp(1j * (turn + 0.5)))) < np.deg2rad(1)


def test_hoop_where_projected(renderer, robot):
    """In the survey pose of E8 the hoop ring is found at its projection, with
    the size the real image showed (430.6 px)."""
    from collision import ArmChecker
    image = renderer.render(SURVEY_Q, OUT_OF_VIEW, 0.0)
    predicted = renderer.camera.project(ArmChecker().hoop_center(SURVEY_Q))
    centre, diameter = find_ring(image, predicted)
    assert np.linalg.norm(centre - predicted) < 3.0
    assert diameter == pytest.approx(430.6, rel=0.03)


def test_camera_projection_matches_survey_tool(robot):
    """camera.Camera.project and tools/camera_survey.project agree."""
    from camera_survey import project
    p = np.array([0.5, -0.06, 0.4])
    assert np.allclose(Camera().project(p), project(p))


def test_other_camera(task, robot):
    """A different camera model (e.g. shifted 1 cm) moves the image."""
    shifted = Camera()
    shifted.T = shifted.T.copy()
    shifted.T[0, 3] += 0.01
    a = markers(SceneRenderer(task).render(robot.q_home, IN_VIEW, 0.0))[1]
    b = markers(SceneRenderer(task, shifted).render(robot.q_home, IN_VIEW, 0.0))[1]
    shift = (b - a).mean(axis=0)  # camera +x (away from the robot) = image -v
    assert shift[1] < -10 and abs(shift[0]) < 2
