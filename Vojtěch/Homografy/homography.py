"""3D camera-to-robot calibration using the CRS93 tool hoop.

This module DOES NOT move the robot. It provides target positions for your robot
controller, finds the center of the hoop in a camera frame, collects measurements,
and estimates one 3x4 projective camera matrix.

The planned positions describe the HOOP CENTER. To generate executable target
poses (homogeneous 4x4 matrices for CRS93 IK), choose ONE fixed TCP orientation
using set_reference_pose() or configure_level_forward(). Every target pose then
uses this same rotation, so the hoop does not tilt or turn while scanning.

Coordinates:
    robot x, y, z  ... meters in ONE fixed robot coordinate frame
    camera u, v     ... pixels, origin at top-left

Important:
    Measured 3D position must correspond to the physical center of the hoop,
    NOT merely the TCP, unless these coincide. The detected ellipse center is
    an approximation to the projection of the 3D circle center.
    Lens distortion is not modeled; ideally undistort frames first.
    Robot base Z must be vertical if configure_level_forward() is used with its
    defaults. Robot TCP/local tool axes are robot-setup dependent: verify
    their directions before sending generated poses to IK / the robot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union
import warnings

import cv2
import numpy as np


@dataclass(frozen=True)
class CircleDetection:
    center_px: tuple[float, float]
    axes_px: tuple[float, float]
    angle_deg: float
    score: float


class Homography:
    """3D projection calibration (despite the historical name, NOT a 3x3 homography)."""

    def __init__(
        self,
        x_range_m: tuple[float, float] = (0.28, 0.46),
        y_range_m: tuple[float, float] = (-0.15, 0.15),
        z_levels_m: tuple[float, ...] = (0.10, 0.15, 0.20),
        table_z_m: float = 0.0,
        nx: int = 4,
        ny: int = 3,
        max_circle_diameter_px: float = 1500,
        min_circle_diameter_px: float = 25,
        hoop_center_in_tcp_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        # X/Y defaults are illustrative only; verify reachability and visibility
        # before commanding any robot motion!
        self.max_circle_diameter_px = float(max_circle_diameter_px)
        self.min_circle_diameter_px = float(min_circle_diameter_px)
        if not (0 < self.min_circle_diameter_px < self.max_circle_diameter_px):
            raise ValueError("Invalid circle diameter limits")

        # Fixed orientation of TCP expressed in robot BASE coordinates.
        # Not guessed automatically: the tool axis conventions are unknown.
        # A reference pose or verified axis convention must be provided
        # before planning target 4x4 poses.
        self.fixed_rotation: Optional[np.ndarray] = None
        self.orientation_source: Optional[str] = None
        self.hoop_center_in_tcp_m = self._check_vector(
            hoop_center_in_tcp_m, "hoop_center_in_tcp_m"
        )
        self.target_poses: Optional[np.ndarray] = None

        self.last_detection: Optional[CircleDetection] = None
        self.samples: dict[int, dict] = {}
        self.P: Optional[np.ndarray] = None
        self.train_ids: list[int] = []
        self.validation_ids: list[int] = []
        self.fit_info: dict = {}
        self.generate_points(x_range_m, y_range_m, z_levels_m, nx, ny, table_z_m)

    def generate_points(
        self,
        x_range_m: tuple[float, float] = (0.28, 0.46),
        y_range_m: tuple[float, float] = (-0.15, 0.15),
        z_levels_m: tuple[float, ...] = (0.10, 0.15, 0.20),
        nx: int = 4,
        ny: int = 3,
        table_z_m: float = 0.0,
        *,
        reference_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Make nx*ny targets on each Z plane. Units are meters.

        z_levels_m are HEIGHTS ABOVE TABLE; table_z_m is robot-frame table Z.

        Returns shape (N,3), e.g. (36,3). Calls never move the robot.
        Regenerating points intentionally resets collected measurements.

        reference_pose: optional known good 4x4 BASE->TCP pose whose rotation
        places the hoop parallel to the table and facing in the chosen fixed
        direction. Its translation is ignored (XYZ come from the grid).
        If fixed orientation is set, also prepares self.target_poses (N,4,4).
        Planned targets are not IK/collision/safety checked.
        """
        if reference_pose is not None:
            # Only the rotation of the demonstrated pose is copied.
            self.set_reference_pose(reference_pose)
        if nx < 2 or ny < 2 or len(z_levels_m) < 2:
            raise ValueError("Need nx>=2, ny>=2 and at least two different Z levels")
        if not (x_range_m[0] < x_range_m[1] and y_range_m[0] < y_range_m[1]):
            raise ValueError("X and Y ranges must be increasing")
        if len(set(z_levels_m)) != len(z_levels_m):
            raise ValueError("Z levels must be distinct")

        self.x_range_m = tuple(map(float, x_range_m))
        self.y_range_m = tuple(map(float, y_range_m))
        self.table_z_m = float(table_z_m)
        self.heights_above_table_m = np.asarray(z_levels_m, dtype=float)
        self.z_levels_m = self.heights_above_table_m + self.table_z_m
        self.nx, self.ny = int(nx), int(ny)
        self.x_positions_m = np.linspace(*self.x_range_m, self.nx)
        self.y_positions_m = np.linspace(*self.y_range_m, self.ny)

        # Snake scan to avoid unnecessarily large travels between grid rows.
        xy = []
        for row, y in enumerate(self.y_positions_m):
            xs = self.x_positions_m if row % 2 == 0 else self.x_positions_m[::-1]
            for x in xs:
                xy.append((float(x), float(y)))
        self.xy_points_m = np.array(xy, dtype=float)  # (12,2) with default settings
        self.points_m = np.array(
            [[x, y, z] for z in self.z_levels_m for x, y in self.xy_points_m],
            dtype=float,
        )  # (36,3)

        self.samples = {}
        self.P = None
        self.train_ids = []
        self.validation_ids = []
        self.fit_info = {}
        self._update_target_poses()
        return self.points_m.copy()

    @staticmethod
    def _check_vector(values, name: str) -> np.ndarray:
        vec = np.asarray(values, dtype=float)
        if vec.shape != (3,) or not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} must be a finite 3-vector")
        return vec.copy()

    @staticmethod
    def _check_rotation(rotation: np.ndarray) -> np.ndarray:
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            raise ValueError("fixed rotation must be a finite 3x3 matrix")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(R), 1, atol=1e-5
        ):
            raise ValueError("fixed rotation must be a right-handed SO(3) matrix")
        return R.copy()

    def _update_target_poses(self) -> None:
        """Convert planned hoop-center XYZ to TCP 4x4 targets."""
        if self.fixed_rotation is None:
            self.target_poses = None
            return
        N = len(self.points_m)
        poses = np.tile(np.eye(4, dtype=float), (N, 1, 1))
        poses[:, :3, :3] = self.fixed_rotation
        # desired hoop position = desired TCP position + R @ hoop_offset
        correction = self.fixed_rotation @ self.hoop_center_in_tcp_m
        poses[:, :3, 3] = self.points_m - correction
        self.target_poses = poses

    def set_fixed_orientation(
        self, rotation: np.ndarray, *, source: str = "provided rotation"
    ) -> None:
        """Lock one TCP-to-base rotation for all targets (never commands motion)."""
        self.fixed_rotation = self._check_rotation(rotation)
        self.orientation_source = str(source)
        self._update_target_poses()

    def set_reference_pose(self, reference_pose: np.ndarray) -> None:
        """Use the orientation of a previously checked 'level, forward' pose.

        reference_pose must be 4x4 homogeneous BASE->TCP transform, e.g.
        robot.fk(robot.get_q()) when the hoop is physically level with the
        table and correctly facing the chosen forward direction. Position of
        this pose is ignored; only its 3x3 rotation is reused.
        """
        T = np.asarray(reference_pose, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("reference_pose must be a finite 4x4 matrix")
        if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("reference_pose must be a homogeneous transform")
        self.set_fixed_orientation(T[:3, :3], source="taught reference pose")

    @staticmethod
    def _orthonormal_axes(normal, forward, *, context: str) -> np.ndarray:
        """Right-handed orthonormal frame: X forward in hoop plane, Z normal."""
        z = Homography._check_vector(normal, f"{context} normal")
        zn = np.linalg.norm(z)
        if zn < 1e-10:
            raise ValueError(f"{context} normal cannot be zero")
        z /= zn
        x = Homography._check_vector(forward, f"{context} forward")
        x -= np.dot(x, z) * z
        xn = np.linalg.norm(x)
        if xn < 1e-10:
            raise ValueError(f"{context} forward cannot be parallel to normal")
        x /= xn
        y = np.cross(z, x)
        return np.column_stack((x, y, z))

    def configure_level_forward(
        self, *,
        hoop_normal_tool: tuple[float, float, float] = (0.0, 0.0, 1.0),
        hoop_forward_tool: tuple[float, float, float] = (1.0, 0.0, 0.0),
        up_robot: tuple[float, float, float] = (0.0, 0.0, 1.0),
        forward_robot: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Compute one fixed TCP orientation with hoop plane horizontal.

        The hoop-plane normal (in TCP coordinates) is mapped to robot up,
        and a direction IN the hoop plane (in TCP coords) is mapped to robot
        forward, fixing the remaining yaw. Defaults ASSUME tool hoop-normal
        is local +Z, its forward is local +X, robot +Z is up and +X forward.
        These are NOT verified for a real CRS93 hoop! Check the physical axis
        directions or prefer set_reference_pose() with a demonstrated pose.
        """
        tool_frame = self._orthonormal_axes(
            hoop_normal_tool, hoop_forward_tool, context="tool"
        )
        robot_frame = self._orthonormal_axes(
            up_robot, forward_robot, context="robot"
        )
        R = robot_frame @ tool_frame.T
        self.set_fixed_orientation(R, source="level/forward axis alignment")
        return R.copy()

    def set_hoop_offset(self, hoop_center_in_tcp_m: tuple[float, float, float]) -> None:
        """Offset TCP->physical hoop center in LOCAL TCP coordinates [meters]."""
        self.hoop_center_in_tcp_m = self._check_vector(
            hoop_center_in_tcp_m, "hoop_center_in_tcp_m"
        )
        self._update_target_poses()

    def generate_poses(self, *, reference_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """Return N homogeneous 4x4 target TCP poses with ONE orientation.

        Intended for CRS93 robot.ik(pose). Does NOT check IK/limits/collisions
        or move the robot. XYZ points are hoop centers; the TCP target is
        shifted by the configured TCP->hoop offset.
        """
        if reference_pose is not None:
            self.set_reference_pose(reference_pose)
        if self.fixed_rotation is None:
            raise RuntimeError(
                "Fixed orientation not configured. First call set_reference_pose("
                "robot.fk(robot.get_q())) with the hoop level and facing "
                "forward, or configure_level_forward() with verified axes."
            )
        return self.target_poses.copy()

    @property
    def poses(self) -> np.ndarray:
        """Copy of all planned fixed-orientation 4x4 TCP poses."""
        return self.generate_poses()

    @property
    def points(self) -> np.ndarray:
        """Planned tool-hoop CENTER positions, shape (N,3), in meters."""
        return self.points_m.copy()

    def capture_circle(
        self,
        frame: Optional[Union[np.ndarray, str, Path]] = None,
        *,
        camera: Optional[object] = None,
        camera_index: int = 0,
        roi: Optional[tuple[int, int, int, int]] = None,
        expected_center: Optional[tuple[float, float]] = None,
        debug_path: Optional[Union[str, Path]] = None,
    ) -> tuple[float, float]:
        """Acquire/detect hoop and return center (u,v) in ORIGINAL image pixels.

        frame: existing OpenCV image or path to image (preferred for testing).
        camera: existing cv2.VideoCapture or callable returning an image.
        camera_index: used only if frame and camera are None.
        roi: optional (x,y,width,height) rectangle in full image.
        expected_center: optional approximate pixel location to prefer a nearby ring.
        debug_path: if supplied, write detection visualization to disk.

        Raises RuntimeError if camera acquisition or circle detection fails.
        Never returns a guessed (0,0) position.
        """
        if frame is None:
            if camera is None:
                cap = cv2.VideoCapture(camera_index)
                try:
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot open camera #{camera_index}")
                    ok, image = cap.read()
                finally:
                    cap.release()
                if not ok:
                    raise RuntimeError("Camera returned no frame")
            else:
                output = camera() if callable(camera) else camera.read()
                if isinstance(output, tuple) and len(output) == 2:
                    ok, image = output
                    if not ok:
                        raise RuntimeError("Camera returned no frame")
                else:
                    image = output
        elif isinstance(frame, (str, Path)):
            image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {frame}")
        else:
            image = frame

        if image is None or not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("Expected a valid OpenCV image")
        if image.ndim == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.dtype != np.uint8:
            raise ValueError("Expected 8-bit image data")

        full_h, full_w = gray.shape
        offset_x = offset_y = 0
        if roi is not None:
            offset_x, offset_y, width, height = map(int, roi)
            if (offset_x < 0 or offset_y < 0 or width <= 0 or height <= 0
                    or offset_x + width > full_w or offset_y + height > full_h):
                raise ValueError("ROI must be entirely within image bounds")
            gray = gray[offset_y:offset_y + height, offset_x:offset_x + width]

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        # Adaptive enough for a moderately varying scene without expensive
        # Hough-transform scans over all possible 750px radii.
        median = float(np.median(blurred))
        lower = max(20, int(0.65 * median))
        upper = min(255, max(lower + 35, int(1.35 * median)))
        edges = cv2.Canny(blurred, lower, upper)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        candidates: list[CircleDetection] = []

        for contour in contours:
            if len(contour) < 30:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < np.pi * self.min_circle_diameter_px * 0.55:
                continue
            if len(contour) < 5:
                continue
            (cx, cy), (a, b), theta = cv2.fitEllipse(contour)
            diameter = max(a, b)
            small_diameter = min(a, b)
            if not (self.min_circle_diameter_px <= diameter <= self.max_circle_diameter_px):
                continue
            if small_diameter / diameter < 0.42:
                continue

            # Check that the contour actually follows a fitted ellipse, and
            # covers most of its circumference (not just a small arc).
            pts = contour[:, 0, :].astype(float)
            dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
            ct = np.cos(np.deg2rad(theta))
            st = np.sin(np.deg2rad(theta))
            xr = ct * dx + st * dy
            yr = -st * dx + ct * dy
            ex, ey = xr / (a / 2), yr / (b / 2)
            radial_error = float(np.median(np.abs(np.hypot(ex, ey) - 1.0)))
            if radial_error > 0.15:
                continue
            angles = (np.arctan2(ey, ex) + 2 * np.pi) % (2 * np.pi)
            occupied = np.unique(np.floor(angles * 24 / (2 * np.pi)).astype(int)).size
            coverage = occupied / 24
            if coverage < 0.70:
                continue

            full_center = (float(cx + offset_x), float(cy + offset_y))
            size_score = min(1.0, np.sqrt(diameter / self.max_circle_diameter_px))
            score = (4.0 * coverage + 2.0 * max(0.0, 1 - 6 * radial_error)
                     + 1.8 * size_score)
            if expected_center is not None:
                distance = np.hypot(full_center[0] - expected_center[0],
                                    full_center[1] - expected_center[1])
                score += 5.0 * np.exp(-distance / max(40.0, diameter * 0.5))
            candidates.append(CircleDetection(
                center_px=full_center, axes_px=(float(a), float(b)),
                angle_deg=float(theta), score=float(score),
            ))

        if not candidates:
            self.last_detection = None
            if debug_path is not None:
                self._save_debug(image, debug_path, None, roi)
            raise RuntimeError("Hoop not detected; inspect image or adjust ROI/lighting")

        # Prefer pairs of concentric ellipses (the two edges of the same hoop).
        for i, candidate in enumerate(candidates):
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                center_sep = np.hypot(*(np.subtract(candidate.center_px, other.center_px)))
                if center_sep < 0.035 * max(candidate.axes_px) and (
                    0.65 < max(other.axes_px) / max(candidate.axes_px) < 1.35
                ):
                    candidates[i] = CircleDetection(
                        candidate.center_px, candidate.axes_px,
                        candidate.angle_deg, candidate.score + 0.6,
                    )
                    break

        best = max(candidates, key=lambda d: d.score)
        self.last_detection = best
        if debug_path is not None:
            self._save_debug(image, debug_path, best, roi)
        return best.center_px

    @staticmethod
    def _save_debug(image, path, detection, roi):
        debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        if debug.shape[2] == 4:
            debug = cv2.cvtColor(debug, cv2.COLOR_BGRA2BGR)
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 180, 0), 2)
        if detection is not None:
            ellipse = (detection.center_px, detection.axes_px, detection.angle_deg)
            cv2.ellipse(debug, ellipse, (0, 255, 0), 3)
            u, v = detection.center_px
            cv2.drawMarker(debug, (round(u), round(v)), (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.putText(debug, f"({u:.1f}, {v:.1f})", (round(u) + 10, round(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(debug, "HOOP NOT FOUND", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), debug):
            raise RuntimeError(f"Cannot write debug image: {path}")

    def record(
        self,
        point_index: int,
        pixel_uv: tuple[float, float],
        actual_xyz_m: Optional[tuple[float, float, float]] = None,
        *,
        replace: bool = False,
    ) -> None:
        """Add the measurement for one planned point.

        Read actual_xyz_m from robot forward kinematics AFTER robot has stopped.
        It must be the position of the physical hoop center (not TCP if offset).
        If omitted, the nominal target XYZ is used as a LESS ACCURATE fallback.
        """
        idx = int(point_index)
        if idx < 0 or idx >= len(self.points_m):
            raise IndexError("Invalid planned point index")
        if idx in self.samples and not replace:
            raise ValueError(f"Point {idx} already measured; pass replace=True")
        if actual_xyz_m is None:
            warnings.warn("Using TARGET rather than measured 3D position", RuntimeWarning,
                          stacklevel=2)
            xyz = self.points_m[idx].copy()
        else:
            xyz = np.asarray(actual_xyz_m, dtype=float)
        uv = np.asarray(pixel_uv, dtype=float)
        if xyz.shape != (3,) or uv.shape != (2,) or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(uv)):
            raise ValueError("Expected finite actual_xyz_m=(x,y,z) and pixel_uv=(u,v)")
        self.samples[idx] = {"xyz_m": xyz.tolist(), "pixel_uv": uv.tolist()}
        # Data changed: existing calibration is now stale.
        self.P = None
        self.train_ids.clear()
        self.validation_ids.clear()
        self.fit_info = {}

    @staticmethod
    def _normalize_2d(uv: np.ndarray):
        centroid = uv.mean(axis=0)
        mean_distance = np.linalg.norm(uv - centroid, axis=1).mean()
        if mean_distance < 1e-12:
            raise ValueError("All pixel points coincide")
        s = np.sqrt(2) / mean_distance
        T = np.array([[s, 0, -s * centroid[0]],
                      [0, s, -s * centroid[1]], [0, 0, 1]], dtype=float)
        uv_h = np.column_stack((uv, np.ones(len(uv))))
        return T, (T @ uv_h.T).T

    @staticmethod
    def _normalize_3d(xyz: np.ndarray):
        centroid = xyz.mean(axis=0)
        mean_distance = np.linalg.norm(xyz - centroid, axis=1).mean()
        if mean_distance < 1e-12:
            raise ValueError("All robot points coincide")
        s = np.sqrt(3) / mean_distance
        T = np.eye(4)
        T[:3, :3] *= s
        T[:3, 3] = -s * centroid
        xyz_h = np.column_stack((xyz, np.ones(len(xyz))))
        return T, (T @ xyz_h.T).T

    @classmethod
    def _fit_dlt(cls, xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
        if len(xyz) < 6:
            raise ValueError("At least six 3D/2D correspondences are needed")
        if np.linalg.matrix_rank(np.column_stack((xyz, np.ones(len(xyz))))) < 4:
            raise ValueError("Calibration points must not all lie in one plane")
        T3, Xn = cls._normalize_3d(xyz)
        T2, xn = cls._normalize_2d(uv)
        A = np.zeros((len(xyz) * 2, 12), dtype=float)
        for i, (X, q) in enumerate(zip(Xn, xn)):
            u, v = q[0], q[1]
            A[2 * i, 0:4] = X
            A[2 * i, 8:12] = -u * X
            A[2 * i + 1, 4:8] = X
            A[2 * i + 1, 8:12] = -v * X
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        P_norm = vt[-1].reshape(3, 4)
        P = np.linalg.inv(T2) @ P_norm @ T3
        P /= np.linalg.norm(P)
        return P

    @staticmethod
    def _project_many(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        points_h = np.column_stack((xyz, np.ones(len(xyz))))
        q = (P @ points_h.T).T
        uv = np.full((len(xyz), 2), np.nan)
        valid = np.abs(q[:, 2]) > 1e-12
        uv[valid] = q[valid, :2] / q[valid, 2, None]
        return uv

    def calculate(
        self,
        validation_fraction: float = 0.25,
        *,
        ransac: bool = True,
        ransac_threshold_px: float = 8.0,
        ransac_iterations: int = 400,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Estimate 3x4 projection matrix from recorded 3D/2D pairs.

        Default: 25% of measurements held out for INDEPENDENT validation,
        approximately the same proportion at each calibrated height.
        Robust RANSAC is applied to the remaining training points.
        """
        if not (0.0 <= validation_fraction < 0.8):
            raise ValueError("validation_fraction must be in [0,0.8)")
        ids = sorted(self.samples)
        if len(ids) < 6:
            raise ValueError("Need at least six recorded 3D/2D measurements")
        rng = np.random.default_rng(random_seed)
        validation: list[int] = []
        if validation_fraction:
            groups: dict[int, list[int]] = {}
            for idx in ids:
                z = self.samples[idx]["xyz_m"][2]
                group = int(np.argmin(abs(self.z_levels_m - z)))
                groups.setdefault(group, []).append(idx)
            for group_ids in groups.values():
                n_val = int(round(len(group_ids) * validation_fraction))
                if n_val:
                    validation.extend(rng.choice(group_ids, n_val, replace=False).tolist())
        validation_set = set(validation)
        train = [i for i in ids if i not in validation_set]
        if len(train) < 6:
            raise ValueError("Too few training measurements; lower validation_fraction")
        xyz = np.asarray([self.samples[i]["xyz_m"] for i in train], dtype=float)
        uv = np.asarray([self.samples[i]["pixel_uv"] for i in train], dtype=float)

        if ransac and len(train) >= 8:
            best = np.zeros(len(train), dtype=bool)
            best_median = float("inf")
            for _ in range(ransac_iterations):
                subset = rng.choice(len(train), 8, replace=False)
                try:
                    trial = self._fit_dlt(xyz[subset], uv[subset])
                except (np.linalg.LinAlgError, ValueError):
                    continue
                err = np.linalg.norm(self._project_many(trial, xyz) - uv, axis=1)
                inliers = np.isfinite(err) & (err < ransac_threshold_px)
                n_inliers = int(inliers.sum())
                median = float(np.median(err[inliers])) if n_inliers else float("inf")
                if n_inliers > int(best.sum()) or (n_inliers == int(best.sum()) and median < best_median):
                    best, best_median = inliers, median
            if best.sum() < 6:
                raise RuntimeError("RANSAC could not find enough consistent calibration points")
            # One additional iteration refines inliers after fitting to best subset.
            P = self._fit_dlt(xyz[best], uv[best])
            err = np.linalg.norm(self._project_many(P, xyz) - uv, axis=1)
            refined = np.isfinite(err) & (err < ransac_threshold_px)
            if refined.sum() >= 6:
                P = self._fit_dlt(xyz[refined], uv[refined])
                best = refined
            inlier_ids = [i for i, good in zip(train, best) if good]
        else:
            P = self._fit_dlt(xyz, uv)
            inlier_ids = train[:]

        self.P = P
        self.train_ids = train
        self.validation_ids = sorted(validation)
        self.fit_info = {
            "training_count": len(train),
            "inlier_count": len(inlier_ids),
            "rejected_ids": [i for i in train if i not in inlier_ids],
            "validation_count": len(validation),
            "model": "3x4 pinhole projective matrix (normalized DLT)",
        }
        return P.copy()

    def robot_to_pixel(self, x: float, y: float, z: float) -> tuple[float, float]:
        """Project robot-frame 3D point to camera pixel coordinates."""
        if self.P is None:
            raise RuntimeError("Calibration not calculated or loaded")
        uv_h = self.P @ np.array([x, y, z, 1.0], dtype=float)
        if abs(uv_h[2]) < 1e-12:
            raise ValueError("Point projects to infinity")
        return float(uv_h[0] / uv_h[2]), float(uv_h[1] / uv_h[2])

    def pixel_to_robot(self, u: float, v: float, z: float) -> tuple[float, float]:
        """Intersect the camera ray at pixel (u,v) with robot-plane Z=z.

        Requires known Z! A single pixel alone cannot determine 3D position.
        """
        if self.P is None:
            raise RuntimeError("Calibration not calculated or loaded")
        p = self.P
        M = np.array([
            [u * p[2, 0] - p[0, 0], u * p[2, 1] - p[0, 1]],
            [v * p[2, 0] - p[1, 0], v * p[2, 1] - p[1, 1]],
        ], dtype=float)
        b = np.array([
            p[0, 2] * z + p[0, 3] - u * (p[2, 2] * z + p[2, 3]),
            p[1, 2] * z + p[1, 3] - v * (p[2, 2] * z + p[2, 3]),
        ])
        if np.linalg.cond(M) > 1e10:
            raise ValueError("Camera ray is almost parallel to the requested Z-plane")
        x, y = np.linalg.solve(M, b)
        return float(x), float(y)

    def pixel_to_robot_at_height(self, u: float, v: float, height_above_table_m: float) -> tuple[float, float]:
        """Convenience conversion using a height ABOVE the calibrated table."""
        return self.pixel_to_robot(u, v, self.table_z_m + height_above_table_m)

    def validate(self, use_training: bool = False) -> dict:
        """Return pixel reprojection and XY reconstruction errors.

        Default uses held-out points (not seen by calculate). If none exist,
        falls back to training points and explicitly labels those errors as
        non-independent. use_training=True forces training-set evaluation.
        """
        if self.P is None:
            raise RuntimeError("Call calculate() before validate()")
        if self.validation_ids and not use_training:
            ids = self.validation_ids
            subset_label = "holdout (independent)"
        else:
            ids = self.train_ids if self.train_ids else sorted(self.samples)
            subset_label = "training (NOT independent)"
        px_errors = []
        mm_errors = []
        per_point = []
        for idx in ids:
            sample = self.samples[idx]
            x, y, z = sample["xyz_m"]
            u, v = sample["pixel_uv"]
            pu, pv = self.robot_to_pixel(x, y, z)
            px_error = float(np.hypot(pu - u, pv - v))
            rx, ry = self.pixel_to_robot(u, v, z)
            mm_error = float(1000 * np.hypot(rx - x, ry - y))
            px_errors.append(px_error)
            mm_errors.append(mm_error)
            per_point.append({"point_index": idx, "error_px": px_error,
                              "error_xy_mm": mm_error})
        if not px_errors:
            raise ValueError("No points available to validate")
        return {
            "subset": subset_label,
            "count": len(ids),
            "pixel_rmse": float(np.sqrt(np.mean(np.square(px_errors)))),
            "pixel_mean": float(np.mean(px_errors)),
            "pixel_max": float(np.max(px_errors)),
            "xy_rmse_mm": float(np.sqrt(np.mean(np.square(mm_errors)))),
            "xy_mean_mm": float(np.mean(mm_errors)),
            "xy_max_mm": float(np.max(mm_errors)),
            "per_point": per_point,
        }

    def save(self, path: Union[str, Path] = "homography_calibration.json") -> None:
        """Save settings, planned points, recorded samples and fitted matrix."""
        data = {
            "version": 2,
            "units": {"robot": "m", "camera": "px"},
            "config": {
                "x_range_m": list(self.x_range_m),
                "y_range_m": list(self.y_range_m),
                "z_levels_m": self.heights_above_table_m.tolist(),
                "table_z_m": self.table_z_m,
                "nx": self.nx, "ny": self.ny,
                "min_circle_diameter_px": self.min_circle_diameter_px,
                "max_circle_diameter_px": self.max_circle_diameter_px,
                "hoop_center_in_tcp_m": self.hoop_center_in_tcp_m.tolist(),
                "fixed_rotation": (None if self.fixed_rotation is None
                                   else self.fixed_rotation.tolist()),
                "orientation_source": self.orientation_source,
            },
            "points_m": self.points_m.tolist(),
            "samples": {str(i): s for i, s in sorted(self.samples.items())},
            "P": None if self.P is None else self.P.tolist(),
            "train_ids": self.train_ids,
            "validation_ids": self.validation_ids,
            "fit_info": self.fit_info,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path] = "homography_calibration.json") -> "Homography":
        """Restore class instance from JSON produced by save()."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") not in (1, 2):
            raise ValueError("Unsupported calibration file version")
        c = data["config"]
        obj = cls(
            x_range_m=tuple(c["x_range_m"]), y_range_m=tuple(c["y_range_m"]),
            z_levels_m=tuple(c["z_levels_m"]), table_z_m=c.get("table_z_m", 0.0),
            nx=c["nx"], ny=c["ny"],
            max_circle_diameter_px=c["max_circle_diameter_px"],
            min_circle_diameter_px=c["min_circle_diameter_px"],
            hoop_center_in_tcp_m=c.get("hoop_center_in_tcp_m", (0, 0, 0)),
        )
        obj.points_m = np.asarray(data["points_m"], dtype=float)
        if c.get("fixed_rotation") is not None:
            obj.set_fixed_orientation(
                np.asarray(c["fixed_rotation"], dtype=float),
                source=c.get("orientation_source") or "loaded",
            )
        obj._update_target_poses()
        obj.samples = {int(i): s for i, s in data["samples"].items()}
        obj.P = None if data["P"] is None else np.asarray(data["P"], dtype=float)
        obj.train_ids = list(data.get("train_ids", []))
        obj.validation_ids = list(data.get("validation_ids", []))
        obj.fit_info = dict(data.get("fit_info", {}))
        return obj


if __name__ == "__main__":
    homography = Homography()
    print("Planned (not safety-checked!) 3D points in meters:")
    for i, xyz in enumerate(homography.points):
        print(f"{i:2d}: X={xyz[0]:.3f}  Y={xyz[1]:.3f}  Z={xyz[2]:.3f}")
    print(f"\nTotal points: {len(homography.points)}")
    print("No robot motion is performed by this module.")
    print("Before planning full poses, set_reference_pose(T_base_tcp) "
          "or configure_level_forward() with verified tool axes.")
