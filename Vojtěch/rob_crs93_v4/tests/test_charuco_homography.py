"""Offline tests on an actual generated legacy ChArUco image, no camera/robot."""
from types import SimpleNamespace
import numpy as np
import cv2
import pytest

from charuco_homography import (
    BOARD_SQUARES, CharucoHomographySession, detect_board, make_board, rigid_pose,
)
from extensions.homography_calibration import HomographyExtension


class BoardCamera:
    mode = 'replay'

    def __init__(self, frame):
        self.frame = frame

    def capture(self):
        return self.frame.copy()


def _frame():
    board = make_board(20, 15)
    assert BOARD_SQUARES == (10, 14)
    original = board.generateImage((1000, 1400), marginSize=0)
    # Put white margins all around, so ChArUco can also see edge markers.
    frame = np.full((1550, 1150), 255, np.uint8)
    frame[75:1475, 75:1075] = original
    return frame


def _world(local, z=.05):
    angle = np.deg2rad(19)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return np.asarray(local) @ R + [0.27, -.16]


def test_legacy_board_has_117_corners():
    idx, pixels, local, preview = detect_board(_frame(), 20, 15)
    assert len(idx) == 117
    assert set(idx) == set(range(117))
    assert pixels.shape == (117, 2)
    assert local.shape == (117, 2)
    assert preview.shape == (1550, 1150, 3)


def test_board_one_capture_three_anchors_maps_to_robot(tmp_path):
    session = CharucoHomographySession(tmp_path)
    session.set_dimensions(20, 15)
    assert session.capture(_frame()) == 117
    idx = [0, 8, 116, 108]
    local = np.array(session.local_xy)
    id_to_index = {id_: i for i, id_ in enumerate(session.ids)}
    for id_ in idx:
        xy = _world(local[id_to_index[id_]])
        session.add_anchor(id_, (*xy, .05))
    H, rms_m = session.solve()
    assert rms_m < 1e-9
    assert H.rms_px < 1.0
    for id_ in (40, 60, 75):
        i = id_to_index[id_]
        xy = H.pixel_to_world(*session.pixels[i])
        assert np.linalg.norm(np.asarray(xy) - _world(local[i])) < .001
    restored = CharucoHomographySession(tmp_path)
    restored.load()
    assert restored.anchors == session.anchors
    assert restored.square_mm == 20.


def test_bad_geometry_and_bad_ref_point_rejected(tmp_path):
    s = CharucoHomographySession(tmp_path)
    s.set_dimensions(20, 15)
    s.capture(_frame())
    local = np.array(s.local_xy)
    look = {id_: i for i, id_ in enumerate(s.ids)}
    for i in [0, 1, 2]:
        s.add_anchor(i, (*_world(local[look[i]]), .05))
    with pytest.raises(ValueError, match='nekolinearni|primce|daleko'):
        s.solve()
    s.anchors.clear()
    for i in (0, 8, 116):
        xy = _world(local[look[i]])
        if i == 116:
            xy = xy + [.04, 0]
        s.add_anchor(i, (*xy, .05))
    with pytest.raises(ValueError, match='referencni body nesedi'):
        s.solve()


def test_board_cli_works_with_replay_but_teach_needs_real(tmp_path):
    frame = _frame()
    ctx = SimpleNamespace(
        root=tmp_path,
        camera=BoardCamera(frame),
        robot=SimpleNamespace(mode='virtual', released=False),
    )
    ext = HomographyExtension(ctx)
    ext.run(['board', 'config', '20', '15'])
    ext.run(['board', 'capture'])
    assert len(ext.board.ids) == 117
    assert ext.board.preview_path.is_file()
    assert ext.session.is_dirty
    with pytest.raises(RuntimeError, match='REAL'):
        ext.run(['board', 'teach', '0'])
    local = np.asarray(ext.board.local_xy)
    look = {id_: i for i, id_ in enumerate(ext.board.ids)}
    for i in (0, 8, 116):
        x, y = _world(local[look[i]])
        ext.run(['board', 'anchor', str(i), str(x), str(y), '0.05'])
    ext.run(['board', 'solve'])
    assert not ext.session.is_dirty
    uv = ext.board.pixels[look[60]]
    got = ext.session.pixel_to_world(*uv)
    assert np.linalg.norm(np.asarray(got) - _world(local[look[60]])) < 0.001
    reloaded = HomographyExtension(ctx)
    assert not reloaded.session.is_dirty
    assert len(reloaded.board.anchors) == 3
    ext.run(['board', 'clear'])
    assert ext.session.is_dirty
    with pytest.raises(RuntimeError, match='solve'):
        ext.session.pixel_to_world(*uv)
