"""Estimate the turntable orbit from timestamped ArUco camera images."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from aruco_block_detector import ArucoBlockDetector
from homography import Homography
from solver.contracts import CAMERA_DELAY, BaseMeasurement, TurntableModel

MARKER_HEIGHT_ABOVE_TABLE_M = 0.0646
MIN_SAMPLES = 4


@dataclass(frozen=True)
class OrbitFit:
    """Turntable model and residual errors in robot-frame meters and radians."""

    model: TurntableModel
    xy_rmse_m: float
    yaw_rmse_rad: float

    @property
    def radius_m(self) -> float:
        return float(np.linalg.norm(self.model.base_offset))

    @property
    def signed_omega(self) -> float:
        return self.model.direction * self.model.omega

    @property
    def period_s(self) -> float:
        return float(2 * np.pi / self.model.omega)


def capture_series(
    grab_image: Callable[[], np.ndarray],
    count: int = 10,
    interval_s: float = 1.0,
    camera_delay_s: float = CAMERA_DELAY,
) -> Iterator[tuple[float, np.ndarray]]:
    """Warm up the camera, then yield frames with monotonic exposure times."""
    if count < MIN_SAMPLES:
        raise ValueError(f"Pro měření dráhy jsou potřeba alespoň {MIN_SAMPLES} snímky.")
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("Interval mezi snímky musí být kladný a konečný.")
    if not np.isfinite(camera_delay_s) or camera_delay_s < 0:
        raise ValueError("Zpoždění kamery musí být konečné a nezáporné.")

    grab_image()  # První snímek má delší dobu pořízení.
    start = time.monotonic()
    for index in range(count):
        time.sleep(max(0.0, start + index * interval_s - time.monotonic()))
        frame = grab_image()
        yield time.monotonic() - camera_delay_s, frame


def measure_base(image: np.ndarray, t_capture: float, detector: ArucoBlockDetector,
                 homography: Homography, marker_z: float) -> BaseMeasurement | None:
    """Measure the base center and the direction of its x axis in robot XY."""
    pose = detector.detect(image)
    if pose is None:
        return None

    markers = np.array([
        [homography.pixel_to_robot(float(u), float(v), marker_z) for u, v in corners]
        for corners in pose.corners_by_id.values()
    ])
    if len(markers) == 2:
        xy = markers.mean(axis=(0, 1))
    else:
        xy = np.array(homography.pixel_to_robot(*pose.center_px, marker_z))
    # ArUco corners are ordered in the marker frame. Transform the edges before
    # measuring yaw so a tilted camera does not skew the angle.
    direction = (markers[:, 1] - markers[:, 0] + markers[:, 2] - markers[:, 3]).mean(axis=0)
    yaw = float(np.arctan2(direction[1], direction[0]))
    return BaseMeasurement(t_capture, xy, yaw, len(markers))


def _robust_fit(design: np.ndarray, values: np.ndarray, weights: np.ndarray,
                noise_floor: float) -> np.ndarray:
    weights_sqrt = np.sqrt(weights)
    for _ in range(5):
        result = np.linalg.lstsq(design * weights_sqrt[:, None], values * weights_sqrt,
                                rcond=None)[0]
        residuals = np.abs(values - design @ result)
        scale = max(noise_floor, 1.4826 * np.median(residuals))
        weights_sqrt = np.sqrt(weights * np.minimum(1.0, 2.5 * scale / np.maximum(scale, residuals)))
    return result


def fit_orbit(measurements: Sequence[BaseMeasurement]) -> OrbitFit:
    """Fit constant angular speed and a circle through the timed base positions.

    Motion is resolved when consecutive valid samples differ by less than half
    a revolution. The last exposure defines zero phase for the returned model.
    """
    if len(measurements) < MIN_SAMPLES:
        raise ValueError(f"Pro odhad dráhy jsou potřeba alespoň {MIN_SAMPLES} detekce.")

    ordered = sorted(measurements, key=lambda sample: sample.t_capture)
    t_ref = ordered[-1].t_capture
    dt = np.array([sample.t_capture - t_ref for sample in ordered])
    xy = np.array([sample.xy for sample in ordered])
    yaw = np.array([sample.yaw for sample in ordered])
    if (xy.shape != (len(dt), 2) or not np.isfinite(xy).all()
            or not np.isfinite(yaw).all() or not np.isfinite(dt).all()
            or np.any(np.diff(dt) <= 0)):
        raise ValueError("Měření vyžadují konečné souřadnice a různé časy pořízení.")
    weights = np.array([1.0 if sample.n_markers == 2 else 0.25 for sample in ordered])

    # A provisional rate aligns the yaw observations before removing angular
    # wrapping. The median is insensitive to an isolated bad observation.
    yaw_steps = np.angle(np.exp(1j * np.diff(yaw)))
    rate = np.median(yaw_steps / np.diff(dt))
    residual_phase = np.exp(1j * (yaw - rate * dt))
    anchor = np.angle(weights @ residual_phase)
    yaw_continuous = anchor + np.angle(residual_phase * np.exp(-1j * anchor)) + rate * dt
    yaw_design = np.column_stack((np.ones(len(dt)), dt))
    yaw_ref, omega = _robust_fit(yaw_design, yaw_continuous, weights, np.deg2rad(0.15))
    if abs(omega * dt[0]) < np.deg2rad(20):
        raise ValueError("Točna se otočila o méně než 20°. Zapni Spin nebo prodluž měření.")

    # Complex numbers represent XY pairs. A unit complex exponential rotates
    # the fitted offset about the fixed axis in the plane of the turntable.
    positions = xy[:, 0] + 1j * xy[:, 1]
    design = np.column_stack((np.ones(len(dt)), np.exp(1j * omega * dt)))
    axis, offset = _robust_fit(design, positions, weights, 0.0002)
    residuals = np.abs(positions - design @ [axis, offset])
    if abs(offset) <= max(0.0005, 2.5 * np.median(residuals)):
        # With a centered base, its yaw is observable even when the orbit is
        # smaller than the uncertainty of the measured XY positions.
        axis = _robust_fit(design[:, :1], positions, weights, 0.0002)[0]
        offset = 0j

    model = TurntableModel(
        axis_xy=np.array([axis.real, axis.imag]),
        base_offset=np.array([offset.real, offset.imag]),
        base_yaw=float(np.angle(np.exp(1j * yaw_ref))),
        direction=1 if omega > 0 else -1,
        omega=float(abs(omega)),
        theta_ref=0.0,
        t_ref=t_ref,
    )
    xy_error = np.abs(positions - design @ [axis, offset])
    yaw_error = np.angle(np.exp(1j * (yaw - (yaw_ref + omega * dt))))
    return OrbitFit(model, float(np.sqrt(np.mean(xy_error ** 2))),
                    float(np.sqrt(np.mean(yaw_error ** 2))))


def draw_orbit(image: np.ndarray, fit: OrbitFit, measurements: Sequence[BaseMeasurement],
               homography: Homography, marker_z: float) -> np.ndarray:
    """Overlay the fitted circle, measured centers and axis on a camera frame."""
    def project(xy: np.ndarray) -> tuple[int, int]:
        u, v = homography.robot_to_pixel(*xy, marker_z)
        return round(u), round(v)

    overlay = image.copy()
    model = fit.model
    axis = project(model.axis_xy)
    curve = np.array([
        project(model.axis_xy + fit.radius_m * np.array([np.cos(theta), np.sin(theta)]))
        for theta in np.linspace(0.0, 2 * np.pi, 180, endpoint=False)
    ], dtype=np.int32)
    cv2.polylines(overlay, [curve], True, (25, 25, 25), 7, cv2.LINE_AA)
    cv2.polylines(overlay, [curve], True, (235, 210, 25), 4, cv2.LINE_AA)
    for sample in measurements:
        point = project(sample.xy)
        cv2.circle(overlay, point, 8, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(overlay, point, 5, (0, 165, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(overlay, axis, (25, 25, 25), cv2.MARKER_CROSS, 33, 7, cv2.LINE_AA)
    cv2.drawMarker(overlay, axis, (220, 40, 255), cv2.MARKER_CROSS, 33, 4, cv2.LINE_AA)

    direction = "CCW" if model.direction > 0 else "CW"
    lines = (
        f"Orbit: R={1000 * fit.radius_m:.1f} mm    {direction}    {fit.period_s:.2f} s/rev",
        "Cyan: fitted orbit    Orange: observed centers    Magenta: fitted axis",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = max(cv2.getTextSize(line, font, 0.83, 2)[0][0] for line in lines)
    cv2.rectangle(overlay, (16, 18), (48 + width, 110), (35, 35, 35), -1)
    for row, line in enumerate(lines):
        origin = (32, 50 + 44 * row)
        cv2.putText(overlay, line, origin, font, 0.83, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay
