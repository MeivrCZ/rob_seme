"""solver/feasibility.py (P2.8): the feasibility funnel, and solver/ik_batch.py."""

import numpy as np
import pytest
from collision import ArmChecker
from ctu_crs import CRS93
from solver.contracts import OMEGA, TurntableModel, example_model, example_path
from solver.feasibility import (FACES, MAX_STEP_CHANGE, THETA_STEP, build_map,
                                conditioning, hoop_offset, ring_frames, screen)
from solver.ik_batch import batch_fk, batch_ik, expand_turns, in_limits

D = np.deg2rad


@pytest.fixture(scope="module")
def crs():
    return CRS93(tty_dev=None)


@pytest.fixture(scope="module")
def result(crs):
    return screen(crs, example_model(), example_path(), 0.2, directions=(1, -1))


# --- ik_batch ---

def test_batch_fk_equals_ctu(crs, rng):
    Q = rng.uniform(crs.q_min, crs.q_max, (20, 6))
    assert np.allclose(batch_fk(crs, Q), np.array([crs.fk(q) for q in Q]), atol=1e-12)


def test_batch_ik_solutions_reach_their_targets(crs, rng):
    Q0 = rng.uniform(crs.q_min, crs.q_max, (500, 6))
    T = batch_fk(crs, Q0)
    Q, valid = batch_ik(crs, T)
    assert np.allclose(batch_fk(crs, Q[valid]), np.repeat(T[:, None], 8, 1)[valid], atol=1e-9)
    wrapped = np.abs(np.angle(np.exp(1j * (Q - Q0[:, None, :])))).max(-1)
    assert np.all(np.nanmin(wrapped, axis=1) < 1e-6)  # the original is among them


def test_batch_ik_has_ctus_solutions(crs, rng):
    """Every solution of ctu_crs's ik is found (away from the stretched-arm case)."""
    Q0 = rng.uniform(crs.q_min, crs.q_max, (100, 6))
    Q0[:, 2] = np.clip(Q0[:, 2], D(5), None)  # not fully stretched
    T = batch_fk(crs, Q0)
    Q, valid = batch_ik(crs, T)
    for i in range(len(T)):
        for s in crs.ik(T[i]):
            d = np.abs(np.angle(np.exp(1j * (Q[i][valid[i]] - s)))).max(-1)
            assert d.min() < 1e-6


def test_expand_turns_adds_in_limit_variants(crs):
    Q = np.zeros((1, 1, 6))
    Q[0, 0, 3] = D(179)  # joint 4 at +179: -181 is within its limits too
    E = expand_turns(crs, Q)
    ok = E[0][in_limits(crs, E[0])]
    assert {round(np.rad2deg(v), 6) for v in ok[:, 3]} == {179.0, -181.0}


# --- targets and conditioning ---

def test_hoop_offset_matches_the_collision_model(crs):
    q = crs.q_home + D([10, -5, 20, 30, -15, 40])
    assert np.allclose((crs.fk(q) @ hoop_offset())[:3, 3], ArmChecker().hoop_center(q),
                       atol=1e-9)


def test_ring_frames(crs):
    C = example_model().puzzle_T(0.3) @ example_path().frames[4]
    for face in FACES:
        F = ring_frames(C, np.array([0.0, 1.0, 2.5]), face)
        assert np.allclose(F[:, :3, 3], C[:3, 3])  # centre on the checkpoint
        assert np.allclose(F[:, :3, 2], face * C[:3, 2])  # normal along / against travel
    F = ring_frames(C, np.array([0.0, np.pi / 2]), 1)
    assert np.allclose(F[1, :3, 0], C[:3, 1])  # roll rotates about the ring axis


def test_conditioning(crs):
    q = crs.q_home.copy()
    assert 0 < conditioning(crs, q) <= 1
    assert conditioning(crs, q * np.array([1, 1, 1, 1, 0, 1])) == 0  # wrist singular
    assert conditioning(crs, np.array([0, 0, 0, 0, D(-45), 0])) == 0  # arm stretched
    at_limit = q.copy()
    at_limit[0] = crs.q_max[0]
    assert conditioning(crs, at_limit) == 0


# --- the funnel ---

def test_example_placement_is_feasible(result):
    assert result.feasible and result.candidates and result.stage == ""
    conds = [c.conditioning for c in result.candidates]
    assert conds == sorted(conds, reverse=True) and conds[0] > 0
    assert {c.direction for c in result.candidates} == {1, -1}


def test_candidate_paths_are_consistent(crs, result):
    """Every configuration puts the ring centre on its checkpoint at the grid angle
    it was looked up at, within the limits, with no branch switch between
    checkpoints; that angle is within half a grid step of the angle the
    turntable has at the path's own nominal arrival."""
    model, path = example_model(), example_path()
    offset = hoop_offset()
    for c in result.candidates[:5]:
        assert np.all(np.diff(c.times) > 0)
        assert np.all(np.abs(np.diff(c.q, axis=0)) <= MAX_STEP_CHANGE + 1e-12)
        at_arrival = c.theta0 + c.direction * OMEGA * c.times
        snap = np.abs(np.angle(np.exp(1j * (at_arrival - c.thetas))))
        assert snap.max() <= 0.5 * THETA_STEP + 1e-9
        for i, (q, theta) in enumerate(zip(c.q, c.thetas)):
            assert crs.in_limits(q)
            want = model.checkpoint_T(path, i, theta)
            got = crs.fk(q) @ offset
            assert np.allclose(got[:3, 3], want[:3, 3], atol=1e-6)
            assert np.allclose(got[:3, 2], c.face[i] * want[:3, 2], atol=1e-6)


def test_out_of_reach_is_reported_at_stage_a(crs):
    far = TurntableModel(axis_xy=np.array([1.2, 0.0]), base_offset=np.zeros(2),
                         base_yaw=0.0, direction=1)
    res = screen(crs, far, example_path(), 0.2)
    assert not res.feasible and res.stage == "A" and "out of the arm's reach" in res.reason


def test_map_is_reused(crs, result):
    res = screen(crs, example_model(), example_path(), 0.2, fmap=result.map)
    assert res.feasible and res.map is result.map
