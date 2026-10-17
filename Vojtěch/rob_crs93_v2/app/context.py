"""Shared resources survive loading and unloading extensions."""
from dataclasses import dataclass
from pathlib import Path

from app.camera import CameraManager
from app.robot import RobotManager
from app.store import CalibrationStore


@dataclass
class Context:
    root: Path
    robot: RobotManager
    camera: CameraManager
    calibration: CalibrationStore
