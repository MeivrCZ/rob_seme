"""Localize the ArUco base, using the turntable model when markers disappear."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from aruco_block_detector import ArucoBlockDetector, BlockPose
from solver.contracts import BaseMeasurement, TurntableModel, rot_z
from turntable_orbit import MARKER_HEIGHT_ABOVE_TABLE_M, measure_base_pose

if TYPE_CHECKING:
    from homography import Homography


@dataclass(frozen=True)
class TimedFrame:
    """Camera pixels and their exposure time in monotonic seconds."""

    image: np.ndarray
    t_capture: float


@dataclass(frozen=True)
class ArucoResult:
    """Base XY (m) and yaw (rad) in robot coordinates at t_capture.

    pose contains the observed image-space markers. It is None for a prediction.
    t_last_valid is the capture time of the detection behind this result.
    """

    t_capture: float
    xy: np.ndarray
    yaw: float
    pose: BlockPose | None
    t_last_valid: float


class AdvancedArucoFinder:
    def __init__(self, detector: ArucoBlockDetector, homography: Homography,
                 turntable: TurntableModel | None = None, *,
                 last_measurement: BaseMeasurement | None = None) -> None:
        self.detector = detector
        self.homography = homography
        self.turntable = turntable
        self._last_measurement = (
            replace(last_measurement, xy=last_measurement.xy.copy())
            if last_measurement is not None else None
        )

    def advAruco(self, frame: TimedFrame) -> ArucoResult | None:
        """Measure the base or propagate the latest detection to this frame."""
        timestamp = frame.t_capture
        if not np.isfinite(timestamp):
            raise ValueError("Čas snímku musí být konečný.")

        pose = self.detector.detect(frame.image)
        last = self._last_measurement
        if pose is not None:
            marker_z = self.homography.table_z_m + MARKER_HEIGHT_ABOVE_TABLE_M
            sample = measure_base_pose(pose, timestamp, self.homography, marker_z)
            if last is None or timestamp >= last.t_capture:
                self._last_measurement = sample
            return ArucoResult(timestamp, sample.xy.copy(), sample.yaw, pose, timestamp)

        if last is None or self.turntable is None or timestamp < last.t_capture:
            return None

        elapsed = timestamp - last.t_capture
        delta = self.turntable.direction * self.turntable.omega * elapsed
        axis = self.turntable.axis_xy
        xy = axis + rot_z(delta)[:2, :2] @ (last.xy - axis)
        yaw = (last.yaw + delta + np.pi) % (2 * np.pi) - np.pi
        return ArucoResult(timestamp, xy, float(yaw), None, last.t_capture)
