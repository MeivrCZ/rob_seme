"""Fixed-camera eye-to-hand calibration with checkerboard on the ROBOT FLANGE.

Frames: T_A_B maps coordinates expressed in B into A.
Each observation: T_B_F_i (CRS93 FK), T_C_T_i (checkerboard PnP).
Constancy: T_B_F_i @ T_F_T == T_B_C @ T_C_T_i.

For OpenCV calibrateHandEye, use inv(T_B_F_i) as gripper2base inputs,
and T_C_T_i as target2cam inputs: the returned transform is T_B_C.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import os
import tempfile
import numpy as np


def make_transform(rotation, translation) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    t[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return t


def checker_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    if cols < 3 or rows < 3 or not np.isfinite(square_m) or square_m <= 0:
        raise ValueError("Need >=3x3 inner corners and positive square size [m].")
    pts = np.zeros((cols * rows, 3), dtype=np.float64)
    pts[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_m
    return pts


def solve_eye_to_hand(samples: list[dict]) -> dict:
    """Estimate T_base_camera using measured matrices, no hardware required.

    Each dict needs 'T_base_flange', 'T_camera_target' (homogeneous 4x4).
    """
    import cv2
    if len(samples) < 8:
        raise ValueError("At least 8 diverse observations are needed (recommend 15-25).")
    g_list, c_list = [], []
    for obs in samples:
        g = np.asarray(obs["T_base_flange"], dtype=float)
        c = np.asarray(obs["T_camera_target"], dtype=float)
        if g.shape != (4, 4) or c.shape != (4, 4) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(c)):
            raise ValueError("Invalid 4x4 transforms in observations.")
        for t in (g, c):
            if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-3) or np.linalg.det(t[:3,:3]) < 0.99:
                raise ValueError("Transforms must be rigid SE(3) matrices.")
        g_list.append(g)
        c_list.append(c)
    inv_g = [np.linalg.inv(g) for g in g_list]
    rotations = np.array([g[:3, :3] for g in g_list])
    if np.linalg.norm(np.std(rotations, axis=0)) < 0.08:
        raise ValueError("Robot orientation diversity too low; tilt/rotate the target across poses.")

    rotation, translation = cv2.calibrateHandEye(
        [g[:3, :3] for g in inv_g], [g[:3, 3] for g in inv_g],
        [c[:3, :3] for c in c_list], [c[:3, 3] for c in c_list],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    T_B_C = make_transform(rotation, translation)
    if not np.all(np.isfinite(T_B_C)):
        raise RuntimeError("Hand-eye solver produced invalid values; add diverse poses.")

    # Implied checkerboard->flange must be constant across all captures.
    inferred = [np.linalg.inv(g) @ T_B_C @ c for g, c in zip(g_list, c_list)]
    R_mean = sum((t[:3, :3] for t in inferred), np.zeros((3, 3)))
    u, _, vh = np.linalg.svd(R_mean)
    R_mean = u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh
    trans_mean = np.mean([t[:3, 3] for t in inferred], axis=0)
    T_F_T = make_transform(R_mean, trans_mean)
    pos_errors = [float(np.linalg.norm(t[:3, 3] - trans_mean) * 1000) for t in inferred]
    angle_errors = [float(np.degrees(np.arccos(np.clip((np.trace(R_mean.T @ t[:3, :3]) - 1) / 2, -1, 1)))) for t in inferred]
    return {
        "T_base_camera": T_B_C.tolist(),
        "T_flange_target": T_F_T.tolist(),
        "translation_residual_rms_mm": float(np.sqrt(np.mean(np.square(pos_errors)))),
        "rotation_residual_rms_deg": float(np.sqrt(np.mean(np.square(angle_errors)))),
        "n_observations": len(samples),
    }


def atomic_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as f:
            temp = f.name
            json.dump(content, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


class EyeToHand:
    def __init__(self, ctx):
        self.ctx = ctx
        self.dataset = ctx.root / "data" / "captures" / "eye_to_hand"
        self.dataset.mkdir(parents=True, exist_ok=True)

    def run(self, argv: list[str]):
        if not argv or argv == ["help"]:
            print("call eye_to_hand capture <cols> <rows> <square_m>   (REAL robot + REAL camera only)")
            print("call eye_to_hand list | solve | show")
            print("Fix the checkerboard RIGIDLY to the flange. Record 15-25 diverse poses.")
            return
        if argv[0] == "capture" and len(argv) == 4:
            self.capture(int(argv[1]), int(argv[2]), float(argv[3]))
        elif argv == ["list"]:
            for file in self.files():
                print(file.name)
            print(f"Observations: {len(self.files())}")
        elif argv == ["solve"]:
            self.solve()
        elif argv == ["show"]:
            saved = self.ctx.calibration.load("eye_to_hand")
            print(json.dumps(saved, indent=2) if saved else "Eye-to-hand not calibrated.")
        else:
            raise ValueError("Try: call eye_to_hand help")

    def files(self) -> list[Path]:
        return sorted(self.dataset.glob("*.json"))

    def intrinsics(self):
        data = self.ctx.calibration.load("camera_intrinsics")
        if data is None:
            raise RuntimeError("First calibrate camera intrinsics.")
        if data.get("camera_name") != self.ctx.camera.name:
            raise RuntimeError("Camera name differs from stored intrinsics.")
        K = np.asarray(data["camera_matrix"], dtype=float)
        dist = np.asarray(data["dist_coeffs"], dtype=float)
        return data, K, dist

    def capture(self, cols: int, rows: int, square_m: float):
        import cv2
        if self.ctx.robot.mode != "real" or self.ctx.camera.mode != "real":
            raise RuntimeError("Physical calibration requires REAL CRS93 and REAL Basler. Simulation/replay only for offline tests.")
        robot = self.ctx.robot.require()
        if robot.in_motion():
            raise RuntimeError("Robot is moving. Wait until the robot has stopped.")
        intr, K, dist = self.intrinsics()
        pts = checker_points(cols, rows, square_m)
        frame = self.ctx.camera.capture()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        wh = [int(gray.shape[1]), int(gray.shape[0])]
        if wh != intr["image_size_wh"]:
            raise RuntimeError(f"Camera resolution {wh} differs from calibrated {intr['image_size_wh']}.")
        found, corners = cv2.findChessboardCorners(gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            raise RuntimeError("Checkerboard not found; no observation recorded.")
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS, 50, .001))
        ok, rvec, tvec = cv2.solvePnP(pts, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("PnP failed; no observation recorded.")
        R, _ = cv2.Rodrigues(rvec)
        T_C_T = make_transform(R, tvec)
        projected, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
        error = float(np.sqrt(np.mean(np.sum((corners.reshape(-1,2)-projected.reshape(-1,2))**2, axis=1))))
        if error > 1.5:
            raise RuntimeError(f"PnP reprojection RMS {error:.2f} px too high; observation rejected.")
        # Recheck stationary state at time of robot readback (no auto motion).
        q = self.ctx.robot.get_q()
        T_B_F = self.ctx.robot.flange_pose()
        if robot.in_motion():
            raise RuntimeError("Robot moved while capturing; observation rejected.")
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8]
        image_name = identifier + ".png"
        image_path = self.dataset / image_name
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError("Failed to save calibration frame.")
        metadata = {
            "kind": "eye_to_hand_observation", "created_utc": datetime.now(timezone.utc).isoformat(),
            "image": image_name, "cols": cols, "rows": rows, "square_m": square_m,
            "image_size_wh": wh, "camera_name": self.ctx.camera.name,
            "q_rad": q.tolist(), "T_base_flange": T_B_F.tolist(),
            "T_camera_target": T_C_T.tolist(), "pnp_rms_px": error,
        }
        atomic_json(self.dataset / (identifier + ".json"), metadata)
        print(f"Recorded #{len(self.files())}: {identifier}, PnP RMS {error:.3f} px")

    def solve(self):
        intr, _, _ = self.intrinsics()
        obs = [json.loads(f.read_text(encoding="utf-8")) for f in self.files()]
        if not obs:
            raise RuntimeError("No observations. Use 'capture' on the real station.")
        board_defs = {(o["cols"], o["rows"], o["square_m"], tuple(o["image_size_wh"]), o["camera_name"]) for o in obs}
        if len(board_defs) != 1:
            raise RuntimeError("Mixed chessboard patterns/resolutions/cameras; split datasets.")
        if next(iter(board_defs))[3] != tuple(intr["image_size_wh"]):
            raise RuntimeError("Intrinsics resolution does not match observation images.")
        result = solve_eye_to_hand(obs)
        result.update({
            "schema_version": 1, "kind": "eye_to_hand", "camera_name": self.ctx.camera.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "convention": "T_A_B maps B-coordinate points into A; translations in meters",
            "input_files": [p.name for p in self.files()],
            "warning": "Validate with independent physical target; this is not a guarantee of absolute accuracy.",
        })
        print(f"Estimated T_base_camera:\n{np.array(result['T_base_camera'])}")
        print(f"Checkerboard attachment consistency RMS: {result['translation_residual_rms_mm']:.2f} mm / {result['rotation_residual_rms_deg']:.2f} deg")
        if result["translation_residual_rms_mm"] > 10 or result["rotation_residual_rms_deg"] > 2:
            raise RuntimeError("Inconsistent observations! Calibration NOT saved. Review mount and frames.")
        path = self.ctx.calibration.save("eye_to_hand", result)
        print(f"Saved {path}. Now verify it against an INDEPENDENT known point/target.")

    def close(self):
        pass


def setup(ctx):
    return EyeToHand(ctx)
