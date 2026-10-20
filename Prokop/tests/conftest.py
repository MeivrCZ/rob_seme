"""Shared fixtures. Run from the repo root:  uv run pytest Prokop/tests"""

import socket
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Prokop/

from ctu_crs import CRS93  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def robot() -> CRS93:
    """Offline ctu_crs robot (kinematics, limits, q_home)."""
    return CRS93(tty_dev=None)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def task():
    from placement import TaskScene
    return TaskScene()
