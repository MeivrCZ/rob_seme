"""Tests of solver.wire (checkpoints by tier) and the pieces of solver.planner."""

import numpy as np
import pytest

from solver.planner import _q_at, flange_target, ring_in_flange, select_closest
from solver.timing import RobotTiming
from solver.wire import SEGMENTS, segment_of, wire_path, wire_points


@pytest.mark.parametrize("tier", range(5))
def test_tier_counts_and_endpoints(tier):
    path = wire_path(tier)
    assert len(path.frames) == len(SEGMENTS) * 2 ** tier + 1
    assert np.allclose(path.positions[0], SEGMENTS[0].a)
    assert np.allclose(path.positions[-1], SEGMENTS[-1].b)
    assert path.final_index == len(path.frames) - 1


def test_each_tier_keeps_the_previous_checkpoints():
    for tier in range(4):
        coarse, fine = wire_path(tier).positions, wire_path(tier + 1).positions
        assert np.allclose(fine[::2], coarse)


def test_frames_follow_the_convention():
    for T in wire_path(3).frames:
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(T[:3, 0], [0, 1, 0])  # x = puzzle +y
        assert abs(T[1, 3]) < 1e-12  # in the x-z plane


def test_bend_points_lie_on_the_arc_and_tangents_are_continuous():
    points, travel, segment = wire_points(3)
    arc = SEGMENTS[1]
    on_arc = points[segment == 1]
    assert np.allclose(np.linalg.norm(on_arc - arc.centre, axis=1), arc.radius)
    # The bend meets the top bar and the vertical run smoothly: no bisected corner.
    assert np.allclose(travel[segment == 0][-1], [-1, 0, 0])
    assert np.allclose(travel[segment == 1][-1], [0, 0, -1])


def test_segment_of_moves():
    seg = segment_of(1)
    assert seg[0] == -1 and list(seg[1:3]) == [0, 0] and list(seg[-2:]) == [4, 4]


def test_select_closest_minimises_total_travel():
    a = np.zeros((2, 6)); a[1, 0] = 1.0
    b = np.zeros((2, 6)); b[0, 0] = 3.0; b[1, 0] = 0.9
    c = np.zeros((1, 6)); c[0, 0] = 1.0
    assert select_closest([a, b, c]) == [1, 1, 0]
    assert select_closest([a, np.zeros((0, 6))]) is None


def test_flange_target_puts_the_ring_on_the_checkpoint(robot):
    T = wire_path(0).frames[2].copy()
    T[:3, 3] += [0.45, 0.0, 0.1]
    F = flange_target(T, roll=0.3)
    ring = F @ ring_in_flange()
    assert np.allclose(ring[:3, 3], T[:3, 3])
    assert np.allclose(ring[:3, 2], -T[:3, 2])  # side B leads: hoop -z along travel


def test_q_at_follows_the_schedule(robot):
    timing = RobotTiming(robot, 0.2)
    q0 = np.zeros(6); q1 = np.full(6, 0.2); q2 = np.array([0.2, 0.0, 0.2, 0, 0, 0])
    arrivals, moves = timing.schedule(q0, [q1, q2])
    assert np.allclose(_q_at(moves, q0, 0.0), q0)
    assert np.allclose(_q_at(moves, q0, arrivals[0]), q1, atol=1e-9)
    assert np.allclose(_q_at(moves, q0, arrivals[1] + 1.0), q2)


def test_side_a_puts_the_hoop_z_along_travel():
    T = wire_path(0).frames[2].copy()
    T[:3, 3] += [0.45, 0.0, 0.1]
    ring = flange_target(T, roll=-0.7, face=1) @ ring_in_flange()
    assert np.allclose(ring[:3, 3], T[:3, 3])
    assert np.allclose(ring[:3, 2], T[:3, 2])


def test_orientation_grid_size():
    from solver.hoop_pose import orientation_grid
    grid = orientation_grid(np.deg2rad(15), np.deg2rad([0, 30]), np.deg2rad(45))
    assert len(grid) == 24 * (1 + 8)


@pytest.mark.parametrize("face", (-1, 1))
def test_tilted_target_keeps_the_ring_centre_and_tilts_the_axis(face):
    T = wire_path(0).frames[3].copy()
    T[:3, 3] += [0.45, 0.0, 0.1]
    ring = flange_target(T, roll=0.4, face=face, tilt=np.deg2rad(30), direction=1.0) \
        @ ring_in_flange()
    assert np.allclose(ring[:3, 3], T[:3, 3])
    angle = np.arccos(np.clip(face * ring[:3, 2] @ T[:3, 2], -1, 1))
    assert np.isclose(angle, np.deg2rad(30))


def test_path_ends_just_under_the_jog_and_the_ring_height_is_checked():
    from solver.hoop_pose import _rx
    from solver.wire import JOG_BOTTOM, END_Z, PAD_TOP_Z, RING_OUTER_RADIUS, ring_top_above_pad
    path = wire_path(0)
    assert END_Z < JOG_BOTTOM[2] and np.isclose(path.positions[-1][2], END_Z)
    level = path.frames[-1] @ _rx(np.pi)  # side B, no tilt: the ring lies flat
    assert np.isclose(ring_top_above_pad(level), END_Z - PAD_TOP_Z)
    upright = level @ _rx(np.pi / 2)  # axis horizontal: the rim reaches the outer radius
    assert np.isclose(ring_top_above_pad(upright), END_Z + RING_OUTER_RADIUS - PAD_TOP_Z)


def test_select_closest_avoids_forbidden_transitions():
    a = np.zeros((2, 6)); a[1, 0] = 1.0
    b = np.zeros((2, 6)); b[0, 0] = 3.0; b[1, 0] = 0.9
    c = np.zeros((1, 6)); c[0, 0] = 1.0
    assert select_closest([a, b, c], forbidden={(2, 1, 0)}) == [1, 0, 0]  # via b0 = 3.0
    assert select_closest([a, b, c], forbidden={(2, 0, 0), (2, 1, 0)}) is None


def test_wire_frames_along_the_wire_match_the_checkpoints():
    from solver.wire import checkpoint_arclengths, wire_frame_at
    for bend in (3, 5):
        path = wire_path(0, bend_points=bend)
        s = checkpoint_arclengths(0, bend_points=bend)
        assert np.all(np.diff(s) > 0)
        for frame, s_k in zip(path.frames[1:-1], s[1:-1]):
            assert np.allclose(wire_frame_at(s_k)[:3, 3], frame[:3, 3], atol=1e-9)


def test_end_pose_keeps_the_ring_off_the_base():
    from solver.hoop_pose import _rx
    from solver.wire import (END_MIN_BOTTOM, END_Z, PAD_TOP_Z, RING_THICKNESS, end_pose_ok,
                             ring_bottom_above_pad)
    level = wire_path(0).frames[-1] @ _rx(np.pi)  # side B flat: the ring below hoop_center
    assert np.isclose(ring_bottom_above_pad(level), END_Z - RING_THICKNESS - PAD_TOP_Z)
    assert end_pose_ok(level)
    tilted = level @ _rx(np.deg2rad(30))  # the rim dips ~17 mm: too close to the base
    assert ring_bottom_above_pad(tilted) < END_MIN_BOTTOM and not end_pose_ok(tilted)
