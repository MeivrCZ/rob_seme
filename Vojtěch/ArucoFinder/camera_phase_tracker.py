"""Update the fitted turntable phase from time-stamped ArUco detections."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from threading import Condition, Event, Thread

import numpy as np

from advanced_aruco import AdvancedArucoFinder, ArucoResult, TimedFrame
from solver.contracts import PhaseEstimate, rot_z


PHASE_GATE = np.deg2rad(6.0)
SPEED_CHECK_WINDOW = 0.8  # seconds of accepted observations
MIN_SPEED_CHECK_TRAVEL = np.deg2rad(5.0)
SPEED_CHECK_PHASE_TOLERANCE = np.deg2rad(2.5)
SPEED_SD = np.deg2rad(0.8)  # rad/s, until the phase model is calibrated on hardware


class CameraPhaseTracker:
    """Continuously observe a fitted, rotating turntable in a camera worker.

    The phase reference stays at the most recent accepted exposure. During a
    gap, PhaseEstimate.predict propagates it using the signed fitted speed and
    increases the uncertainty with the time since that exposure.
    """

    def __init__(self, finder: AdvancedArucoFinder, grab_image: Callable[[], np.ndarray],
                 camera_delay: float, *, speed_sd: float = SPEED_SD,
                 clock: Callable[[], float] = time.monotonic) -> None:
        model = finder.turntable
        if model is None or model.direction not in (-1, 1) or not np.isfinite(model.omega) or model.omega <= 0:
            raise ValueError("Pro sledování je potřeba model otáčející se točny.")
        if not np.isfinite(camera_delay) or camera_delay < 0:
            raise ValueError("Zpoždění kamery musí být konečné a nezáporné.")
        if not np.isfinite(speed_sd) or speed_sd <= 0:
            raise ValueError("Nejistota rychlosti musí být kladná a konečná.")

        self._model = model
        self._omega = model.direction * model.omega
        self._speed_variance = speed_sd**2
        self._speed_window = max(SPEED_CHECK_WINDOW, MIN_SPEED_CHECK_TRAVEL / model.omega)
        self._finder = AdvancedArucoFinder(finder.detector, finder.homography, model)
        self._grab_image = grab_image
        self._camera_delay = camera_delay
        self._clock = clock
        self._changed = Condition()
        self._stop = Event()
        self._estimate: PhaseEstimate | None = None
        self._error: Exception | None = None
        self._observations: deque[tuple[float, float]] = deque()
        self._thread = Thread(target=self._run, name="aruco-phase-tracker", daemon=True)

    def __enter__(self) -> CameraPhaseTracker:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        with self._changed:
            self._changed.notify_all()
        self._thread.join()

    def wait_ready(self, timeout: float) -> bool:
        """Wait until real observations confirm the fitted rotation."""
        with self._changed:
            self._changed.wait_for(
                lambda: self._estimate is not None or self._error is not None or self._stop.is_set(), timeout,
            )
            if self._error is not None:
                raise RuntimeError(f"Sledování točny selhalo: {self._error}") from self._error
            return self._estimate is not None and not self._stop.is_set()

    def estimate(self) -> PhaseEstimate:
        with self._changed:
            if self._error is not None:
                raise RuntimeError(f"Sledování točny selhalo: {self._error}") from self._error
            if self._stop.is_set():
                raise RuntimeError("Sledování točny už skončilo.")
            if self._estimate is None:
                raise RuntimeError("Kamera zatím nepotvrdila fázi otáčející se točny.")
            return replace(self._estimate, covariance=self._estimate.covariance.copy())

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                image = self._grab_image()
                if self._stop.is_set():
                    break
                frame = TimedFrame(image, self._clock() - self._camera_delay)
                result = self._finder.advAruco(frame)
                if result is not None:
                    self._observe(result)
        except Exception as error:
            with self._changed:
                self._error = error
                self._changed.notify_all()

    def _observe(self, result: ArucoResult) -> None:
        timestamp = result.t_capture
        pose = result.pose
        if pose is None or not np.isfinite(result.yaw) or not np.isfinite(result.xy).all():
            return

        observations = self._observations
        predicted = self._model.theta_at(timestamp)
        if observations:
            previous_time, previous_theta = observations[-1]
            if timestamp <= previous_time:
                return
            predicted = previous_theta + self._omega * (timestamp - previous_time)

        residual = (result.yaw - self._model.base_yaw - predicted + np.pi) % (2 * np.pi) - np.pi
        if observations and abs(residual) > PHASE_GATE:
            return

        theta = predicted + residual
        expected_xy = self._model.axis_xy + rot_z(theta)[:2, :2] @ self._model.base_offset
        two_markers = len(pose.visible_ids) == 2
        if np.linalg.norm(result.xy - expected_xy) > (0.010 if two_markers else 0.016):
            return

        # A stopped or reversed turntable can pass the frame-to-frame angle gate.
        # Compare with an older observation before refreshing the reference.
        confirmed = False
        if observations:
            reference_time, reference_theta = observations[0]
            travel = self._omega * (timestamp - reference_time)
            confirmed = abs(travel) >= MIN_SPEED_CHECK_TRAVEL
            if confirmed and abs(theta - reference_theta - travel) > SPEED_CHECK_PHASE_TOLERANCE + 0.12 * abs(travel):
                return

        observations.append((timestamp, theta))
        while len(observations) > 1 and timestamp - observations[1][0] >= self._speed_window:
            observations.popleft()

        phase_sd = np.deg2rad(0.4 if two_markers else 0.8)
        with self._changed:
            if confirmed or self._estimate is not None:
                self._estimate = PhaseEstimate(timestamp, theta, self._omega,
                                               np.diag([phase_sd**2, self._speed_variance]))
                self._changed.notify_all()
