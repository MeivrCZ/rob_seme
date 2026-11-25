"""solver/turntable_chain.py (P1.1): the URDF chain against the contracts' reference."""

import numpy as np
import pytest
from solver.contracts import TurntableModel, example_model
from solver.turntable_chain import (BASE, JOINT, MARKERS, PUZZLE, TurntableChain,
                                    chain_urdf, checkpoint_name)


def random_model(rng) -> TurntableModel:
    return TurntableModel(axis_xy=rng.uniform([0.3, -0.25], [0.65, 0.25]),
                          base_offset=rng.uniform(-0.08, 0.08, 2),
                          base_yaw=rng.uniform(-np.pi, np.pi),
                          direction=int(rng.choice([-1, 1])))


@pytest.fixture(scope="module")
def path(puzzle_path):
    return puzzle_path


def test_fk_equals_the_reference(path, rng):
    for _ in range(5):
        model = random_model(rng)
        chain = TurntableChain(model, path)
        for theta in rng.uniform(-10, 10, 4):
            assert np.allclose(chain.puzzle_T(theta), model.puzzle_T(theta), atol=1e-9)
            assert np.allclose(chain.link_pose(BASE, theta), model.base_T(theta), atol=1e-9)
            for i in (0, 7, len(path.frames) - 1):
                assert np.allclose(chain.checkpoint_T(i, theta),
                                   model.checkpoint_T(path, i, theta), atol=1e-9)


def test_theta_is_periodic(path):
    chain = TurntableChain(example_model(), path)
    assert np.allclose(chain.checkpoint_poses(0.4), chain.checkpoint_poses(0.4 + 2 * np.pi),
                       atol=1e-9)


def test_one_joint_and_all_frames(path):
    chain = TurntableChain(example_model(), path)
    assert chain.urdf.actuated_joint_names == [JOINT]
    poses = chain.link_poses(0.0)
    for link in (PUZZLE, BASE, *MARKERS, checkpoint_name(0), checkpoint_name(len(path.frames) - 1)):
        assert link in poses
    # the marker centres lie 37.5 mm off the base centre on both axes, 5 mm up
    T_base = poses[BASE]
    for link in MARKERS:
        local = np.linalg.solve(T_base, poses[link])[:3, 3]
        assert np.allclose(np.abs(local), [0.0375, 0.0375, 0.005], atol=1e-9)


def test_without_path():
    chain = TurntableChain(example_model())
    assert chain.checkpoint_poses(0.0).shape == (0,)
    with pytest.raises(ValueError):
        chain.checkpoint_T(0, 0.0)


def test_meshes_load(path):
    chain = TurntableChain(example_model(), path, load_meshes=True)
    assert chain.urdf.scene is not None and len(chain.urdf.scene.geometry) >= 5
    assert "<joint name=\"theta\" type=\"continuous\">" in chain_urdf(example_model())
