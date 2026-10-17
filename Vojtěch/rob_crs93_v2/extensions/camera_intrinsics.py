"""First real calibration extension: chessboard camera intrinsics."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


class CameraIntrinsics:
    def __init__(self, context):
        self.ctx = context
        self.capture_dir = context.root / "data" / "captures" / "intrinsics"
        self.capture_dir.mkdir(parents=True, exist_ok=True)

    def run(self, argv: list[str]) -> None:
        if not argv or argv[0] == "help":
            print("Commands: call camera_intrinsics capture")
            print("          call camera_intrinsics list")
            print("          call camera_intrinsics calibrate <cols> <rows> <square_m>")
            print("cols/rows = INNER checkerboard corners; square_m = square size in meters")
        elif argv[0] == "capture" and len(argv) == 1:
            self.capture()
        elif argv[0] == "list" and len(argv) == 1:
            files = self.files()
            print(f"Saved raw images: {len(files)}")
            for file in files:
                print(" ", file.name)
        elif argv[0] == "calibrate" and len(argv) == 4:
            cols, rows = int(argv[1]), int(argv[2])
            square_size = float(argv[3])
            self.calibrate(cols, rows, square_size)
        else:
            raise ValueError("Try: call camera_intrinsics help")

    def files(self) -> list[Path]:
        return sorted(self.capture_dir.glob("*.png"))

    def capture(self) -> None:
        import cv2
        image = self.ctx.camera.capture()
        path = self.capture_dir / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")
            + uuid4().hex[:8] + ".png"
        )
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")
        print(f"Saved camera frame: {path}")
        print("Move/tilt the checkerboard before taking the next calibration image.")

    def calibrate(self, cols: int, rows: int, square_size: float) -> None:
        import cv2
        if cols < 3 or rows < 3 or not (np.isfinite(square_size) and square_size > 0):
            raise ValueError("At least 3x3 inner corners and positive square size are required.")
        object_template = np.zeros((cols * rows, 3), dtype=np.float32)
        object_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        object_template *= square_size

        object_points = []
        image_points = []
        image_size = None
        valid_files = []
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001,
        )
        for file in self.files():
            gray = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"[skip] unreadable image: {file.name}")
                continue
            size = (gray.shape[1], gray.shape[0])
            if image_size is not None and size != image_size:
                print(f"[skip] different resolution: {file.name}")
                continue
            found, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not found:
                print(f"[skip] board not found: {file.name}")
                continue
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(refined)
            object_points.append(object_template.copy())
            valid_files.append(file.name)
            image_size = size
        if len(valid_files) < 10:
            raise RuntimeError(
                f"Only {len(valid_files)} valid views. Capture at least 10 varied "
                "checkerboard poses (ideally 15-25). Nothing was overwritten."
            )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )
        # Mean reprojection error gives an additional diagnostic alongside OpenCV RMS.
        pixel_errors = []
        for object_pts, observed, rv, tv in zip(
            object_points, image_points, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(
                object_pts, rv, tv, camera_matrix, dist_coeffs
            )
            pixel_errors.extend(np.linalg.norm(
                observed.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
            ).tolist())
        mean_error = float(np.mean(pixel_errors))
        if not np.isfinite(rms) or not np.all(np.isfinite(camera_matrix)):
            raise RuntimeError("Invalid numeric calibration result; no data saved.")
        print(f"Views: {len(valid_files)} | RMS: {rms:.3f} px | "
              f"mean reprojection error: {mean_error:.3f} px")
        if rms > 1.0:
            print("WARNING: Large reprojection RMS. Review image sharpness, board "
                  "geometry and viewpoint diversity before trusting this calibration.")
            print("Calibration NOT saved; retake problematic images and retry.")
            return
        data = {
            "schema_version": 1,
            "kind": "camera_intrinsics",
            "camera_name": self.ctx.camera.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "image_size_wh": list(image_size),
            "checkerboard_inner_corners_wh": [cols, rows],
            "square_size_m": square_size,
            "rms_px": float(rms),
            "mean_reprojection_error_px": mean_error,
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
            "source_images": valid_files,
            "note": "Intrinsics only. Robot/camera extrinsic transform NOT calibrated.",
        }
        path = self.ctx.calibration.save("camera_intrinsics", data)
        print(f"Intrinsic calibration saved: {path}")
        print("Keep the camera resolution, focus and zoom unchanged for reuse.")

    def close(self) -> None:
        pass


def setup(ctx):
    return CameraIntrinsics(ctx)
