from types import SimpleNamespace

import numpy as np
import pytest

from collision import PATH_STEP
from solver.puzzle_plan import PHASE_STEP, PuzzleCollisionChecker
from solver.timing import TimedMove


def test_merged_samples_keep_joint_and_phase_spacing():
    start = np.zeros(6)
    target = np.deg2rad([30.0, -10.0, 0.0, 0.0, 0.0, 0.0])
    move = TimedMove(0.2, 1.0, 0.2, start, target, hold_from=0.0)
    checker = SimpleNamespace(command_delay=0.12)
    omega = np.deg2rad(22.0)
    samples = list(PuzzleCollisionChecker._move_samples(checker, move, lambda t: omega * t, None))
    joints, phases = zip(*samples)

    assert joints[0] == pytest.approx(start)
    assert joints[-1] == pytest.approx(target)
    assert phases[0] == pytest.approx(0.0)
    assert phases[-1] == pytest.approx(omega * move.t_end)
    assert np.abs(np.diff(joints, axis=0)).max() <= PATH_STEP + 1e-9
    assert np.abs(np.diff(phases)).max() <= PHASE_STEP + 1e-9
    separate_checks = np.ceil(np.abs(target - start).max() / PATH_STEP) + np.ceil(omega * move.t_end / PHASE_STEP)
    assert len(samples) < separate_checks
