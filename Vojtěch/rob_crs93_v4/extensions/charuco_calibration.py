"""ChArUco intrinsics for the fixed Basler camera at CRS93 station.

Board determined from real camera image (2026-10-17):
  10 x 14 chess squares, 70 DICT_4X4_100 markers,
  legacy OpenCV ChArUco pattern, 117 internal corners.

Use the same extension in REAL mode or with replay camera. No motor commands.
Physical square and marker size (in millimeters) MUST be measured for
metric downstream pose estimates. Image diversity is necessary for intrinsics.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from threading import Event, Lock, Thread
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

BOARD_SIZE = (10, 14)  # columns x rows; rotated on the supplied photograph
DICTIONARY_NAME = "DICT_4X4_100"
# This approximates observed marker/square size; inspect only.
# Calibration itself requires measured dimensions.
INSPECT_MARKER_TO_SQUARE_RATIO = 0.75
MIN_CORNERS = 16
MIN_VIEWS = 12
RECOMMENDED_VIEWS = 20


def _cv2():
    import cv2
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        raise RuntimeError("Required: opencv-contrib-python>=4.8 (cv2.aruco.CharucoBoard)")
    return cv2


def _board(square_mm: float, marker_mm: float):
    cv2 = _cv2()
    if (not np.isfinite(square_mm) or not np.isfinite(marker_mm) or
            square_mm <= 0 or marker_mm <= 0 or marker_mm >= square_mm):
        raise ValueError("Enter positive measured dimensions: 0 < marker_mm < square_mm")
    if not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("Your OpenCV needs the CharucoDetector API (opencv-contrib-python>=4.8)")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard(
        BOARD_SIZE, float(square_mm) / 1000, float(marker_mm) / 1000,
        dictionary,
    )
    if not hasattr(board, "setLegacyPattern"):
        raise RuntimeError("OpenCV lacks setLegacyPattern; upgrade opencv-contrib-python")
    board.setLegacyPattern(True)
    return board


def _read_gray(filepath: Path):
    cv2 = _cv2()
    image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {filepath}")
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        if gray.dtype == np.uint16:
            gray = (gray / 256).astype(np.uint8)
        else:
            raise ValueError("Unsupported image depth, need 8-bit PNG/JPG")
    return image, gray


def _safe_charuco_points(corners, ids):
    """Normalize OpenCV 4.x/5.x arrays; reject inconsistent detections.

    A mismatched corner/ID array must never be handed to OpenCV drawing
    or calibration. A few OpenCV builds have returned inconsistent pairs.
    """
    if corners is None or ids is None:
        return None, None, "no ChArUco corners"
    try:
        raw_corners = np.asarray(corners)
        raw_ids = np.asarray(ids)
        if raw_corners.size % 2:
            return None, None, f"odd corner coordinate count: {raw_corners.size}"
        points = np.asarray(raw_corners, dtype=np.float32).reshape(-1, 1, 2)
        corner_ids = np.asarray(raw_ids, dtype=np.int32).reshape(-1, 1)
    except (ValueError, TypeError) as exc:
        return None, None, f"invalid corner/ID arrays: {exc}"
    if len(points) != len(corner_ids):
        return None, None, (f"corner/ID mismatch: {len(points)} corner positions, "
                             f"{len(corner_ids)} IDs")
    if not len(points):
        return None, None, "no ChArUco corners"
    if not np.all(np.isfinite(points)):
        return None, None, "non-finite corner coordinates"
    if len(np.unique(corner_ids)) != len(corner_ids):
        return None, None, "duplicate ChArUco corner IDs"
    return np.ascontiguousarray(points), np.ascontiguousarray(corner_ids), None


def detect_image(path: Path, square_mm: float = 20.0,
                 marker_mm: float = 15.0):
    """Return camera image and ChArUco observations; mm defaults ONLY for detection."""
    cv2 = _cv2()
    image, gray = _read_gray(path)
    board = _board(square_mm, marker_mm)
    charuco, corner_ids, marker_corners, marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    n_markers = 0 if marker_ids is None else int(len(marker_ids))
    n_corners = 0 if corner_ids is None else int(len(corner_ids))
    return image, gray, board, charuco, corner_ids, marker_corners, marker_ids, n_markers, n_corners


class CharucoCalibration:
    def __init__(self, context):
        self.ctx = context
        self.capture_dir = context.root / "data" / "captures" / "charuco"
        self.overlay_dir = self.capture_dir / "overlays"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.last_details = None
        self._loop_thread: Thread | None = None
        self._loop_stop = Event()
        self._loop_lock = Lock()
        self._loop_stats = {"saved": 0, "skipped": 0, "attempts": 0, "error": None}
        self._loop_interval = 2.0
        self._loop_limit = 40

    def run(self, argv: list[str]) -> None:
        cmd = argv[0].lower() if argv else "help"
        if cmd == "help" and len(argv) <= 1:
            print("""ChArUco board: 10x14 squares; DICT_4X4_100; legacy pattern.
  call charuco_calibration capture                capture one frame from active camera/replay
  call charuco_calibration loopcapture [start [SECONDS [MAX_FRAMES]]]
      background capture every 2s by default, stop automatically after 40 SAVED frames
      skips poorly detected or near-identical board views; move and tilt the board
  call charuco_calibration loopcapture stop        stop background capture (keeps all files)
  call charuco_calibration loopcapture status      show worker and saved/skipped counts
  call charuco_calibration import PATH            copy existing PNG or a folder of PNGs
  call charuco_calibration inspect [PATH]         detect, annotate and print statistics
  call charuco_calibration list                   list source images
  call charuco_calibration archive                move previous frames into timestamped backup
  call charuco_calibration calibrate SQUARE_MM MARKER_MM
      values must be measured physically (marker_mm < square_mm)
  call charuco_calibration report                 show saved VALID camera intrinsics
  call charuco_calibration diagnostics            show last calibration attempt (valid or rejected)
About 20-30 distinct, tilted, sharp views recommended. Camera remains fixed.
This does not calibrate camera-to-robot transform (eye-to-hand).""")
        elif cmd == "capture" and len(argv) == 1:
            self.capture()
        elif cmd == "loopcapture":
            if len(argv) == 1 or (len(argv) >= 2 and argv[1] == "start"):
                interval = float(argv[2]) if len(argv) >= 3 else 2.0
                count = int(argv[3]) if len(argv) >= 4 else 40
                if len(argv) > 4:
                    raise ValueError("loopcapture start [SECONDS [MAX_FRAMES]]")
                self.start_loopcapture(interval, count)
            elif argv[1:] == ["stop"]:
                self.stop_loopcapture()
            elif argv[1:] == ["status"]:
                self.loopcapture_status()
            else:
                raise ValueError("loopcapture [start [SECONDS [MAX_FRAMES]]] | stop | status")
        elif cmd == "import" and len(argv) == 2:
            self.import_images(self._path(argv[1]))
        elif cmd == "inspect" and len(argv) in (1, 2):
            if len(argv) == 2:
                path = self._path(argv[1])
            else:
                files = self.images()
                if not files:
                    raise RuntimeError("No images; use capture or import PATH first.")
                path = files[-1]
            self.inspect(path)
        elif cmd == "list" and len(argv) == 1:
            print(f"ChArUco images: {len(self.images())}")
            for path in self.images():
                print(" ", path.name)
        elif cmd == "archive" and len(argv) == 1:
            self.archive_images()
        elif cmd == "calibrate" and len(argv) == 3:
            if self.loopcapture_running:
                raise RuntimeError("Stop loopcapture before calibrate, so input images won't change.")
            self.calibrate(float(argv[1]), float(argv[2]))
        elif cmd == "report" and len(argv) == 1:
            self.report()
        elif cmd == "diagnostics" and len(argv) == 1:
            data = self.ctx.calibration.load("camera_intrinsics_diagnostics")
            if data is None:
                print("No calibration attempt diagnostics saved yet.")
            else:
                self._print_diagnostics(data)
        else:
            raise ValueError("Try: call charuco_calibration help")

    def _path(self, name: str) -> Path:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = (self.ctx.root / path)
        return path.resolve()

    def images(self) -> list[Path]:
        return sorted(p for p in self.capture_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    def capture(self, image=None, *, make_preview: bool = True) -> Path:
        cv2 = _cv2()
        if image is None:
            image = self.ctx.camera.capture()  # camera is owned by host; never reopen here
        if image is None:
            raise RuntimeError("Camera returned no image")
        dest = self.capture_dir / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")
            + uuid4().hex[:8] + ".png")
        # Write to a temporary file and atomically rename so calibration never
        # reads a half-written PNG while a capture is in progress.
        temp = dest.with_name(dest.stem + ".partial.png")
        try:
            if not cv2.imwrite(str(temp), image):
                raise RuntimeError(f"Could not save image: {temp}")
            temp.replace(dest)
        finally:
            temp.unlink(missing_ok=True)
        print(f"Saved frame: {dest}")
        # A calibration preview must not kill the host or disconnect hardware.
        # Even if detection fails, keep the original frame for later inspection.
        if make_preview:
            try:
                self.inspect(dest)
            except Exception as exc:
                print(f"[ChArUco warning] Snapshot saved, preview failed: "
                      f"{type(exc).__name__}: {exc}")
        return dest

    @property
    def loopcapture_running(self) -> bool:
        return self._loop_thread is not None and self._loop_thread.is_alive()

    def start_loopcapture(self, interval: float = 2.0, max_frames: int = 40) -> None:
        if not np.isfinite(interval) or not 0.5 <= interval <= 60:
            raise ValueError("Interval must be between 0.5 and 60 seconds.")
        if not 1 <= max_frames <= 200:
            raise ValueError("MAX_FRAMES must be 1..200 (prevents accidental disk fill).")
        if getattr(self.ctx.camera, "mode", None) is None:
            raise RuntimeError("Open camera first: camera open (or camera replay DIR).")
        if self.loopcapture_running:
            raise RuntimeError("loopcapture already running; use loopcapture status/stop.")
        self._loop_stop.clear()
        self._loop_interval = interval
        self._loop_limit = max_frames
        with self._loop_lock:
            self._loop_stats = {"saved": 0, "skipped": 0, "attempts": 0, "error": None}
        self._loop_thread = Thread(target=self._loop_worker, daemon=True, name="charuco-loopcapture")
        self._loop_thread.start()
        print(f"loopcapture started: {interval:g}s, maximum {max_frames} saved images. "
              "Tilt/move board between images; type 'call charuco_calibration loopcapture stop' to stop.")

    def _loop_worker(self) -> None:
        previous_observations = None
        initial_camera_mode = self.ctx.camera.mode
        try:
            cv2 = _cv2()
            detector = cv2.aruco.CharucoDetector(_board(20, 15))
            while not self._loop_stop.is_set():
                if self.ctx.camera.mode != initial_camera_mode:
                    raise RuntimeError("Camera mode changed or camera was closed.")
                image = self.ctx.camera.capture()
                if image is None:
                    raise RuntimeError("Camera did not return an image.")
                if image.ndim == 2:
                    gray = image
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4
                                        else cv2.COLOR_BGR2GRAY)
                if gray.dtype == np.uint16:
                    gray = (gray / 256).astype(np.uint8)
                corners, ids, _, _ = detector.detectBoard(gray)
                points, ids, _ = _safe_charuco_points(corners, ids)
                with self._loop_lock:
                    self._loop_stats["attempts"] += 1
                reason = None
                if points is None or len(points) < MIN_CORNERS:
                    reason = f"fewer than {MIN_CORNERS} valid ChArUco corners"
                elif previous_observations is not None:
                    last_pts, last_ids = previous_observations
                    current_by_id = dict(zip(ids.ravel().tolist(), points.reshape(-1, 2)))
                    old_by_id = dict(zip(last_ids.ravel().tolist(), last_pts.reshape(-1, 2)))
                    common = current_by_id.keys() & old_by_id.keys()
                    if len(common) >= MIN_CORNERS:
                        displacement = [np.linalg.norm(current_by_id[i] - old_by_id[i]) for i in common]
                        if float(np.median(displacement)) < 12:
                            reason = "too similar to previous saved view (move or tilt board)"
                if reason is None:
                    self.capture(image, make_preview=False)
                    previous_observations = points.copy(), ids.copy()
                    with self._loop_lock:
                        self._loop_stats["saved"] += 1
                        saved = self._loop_stats["saved"]
                    print(f"[loopcapture] saved {saved}/{self._loop_limit} ({len(points)} corners)")
                    if saved >= self._loop_limit:
                        print("[loopcapture] requested image count reached; worker stopped.")
                        break
                else:
                    with self._loop_lock:
                        self._loop_stats["skipped"] += 1
                    print(f"[loopcapture] skipped: {reason}")
                if self._loop_stop.wait(self._loop_interval):
                    break
        except Exception as exc:
            with self._loop_lock:
                self._loop_stats["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[loopcapture] stopped after error: {type(exc).__name__}: {exc}")
        finally:
            self._loop_stop.set()
            print("[loopcapture] stopped. Captured files remain on disk.")

    def stop_loopcapture(self) -> None:
        self._loop_stop.set()
        thread = self._loop_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=15.0)
            if thread.is_alive():
                raise RuntimeError("Camera capture is still running; do not close camera/unload until it stops.")
        self.loopcapture_status()

    def loopcapture_status(self) -> None:
        with self._loop_lock:
            stats = dict(self._loop_stats)
        print(f"loopcapture: {'RUNNING' if self.loopcapture_running else 'STOPPED'}, "
              f"saved={stats['saved']}/{self._loop_limit}, skipped={stats['skipped']}, "
              f"attempts={stats['attempts']}, interval={self._loop_interval:g}s")
        if stats["error"]:
            print(f"  Last error: {stats['error']}")

    def archive_images(self) -> Path | None:
        if self.loopcapture_running:
            raise RuntimeError("Stop loopcapture before archiving.")
        frames = self.images()
        if not frames:
            print("No images to archive.")
            return None
        dest = (self.capture_dir / "archive" /
                (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:6]))
        dest.mkdir(parents=True, exist_ok=False)
        for image in frames:
            shutil.move(str(image), str(dest / image.name))
        print(f"Archived {len(frames)} frames into {dest}; no files deleted.")
        return dest

    def import_images(self, path: Path) -> None:
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.is_file() and
                           p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError(f"Image or directory does not exist: {path}")
        if not files:
            raise RuntimeError("No PNG/JPG images in the source directory")
        existing = {hashlib.sha256(p.read_bytes()).hexdigest() for p in self.images()}
        added = 0
        for source in files:
            content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if content_hash in existing:
                continue
            dest = self.capture_dir / (source.stem + "_" + content_hash[:8] + source.suffix.lower())
            shutil.copy2(source, dest)
            existing.add(content_hash)
            added += 1
        print(f"Imported {added} new images (out of {len(files)}; duplicates skipped).")

    def inspect(self, path: Path) -> Path:
        cv2 = _cv2()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        (image, gray, _, cc, ci, mc, mi, n_markers, n_corners) = detect_image(path)
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if mi is not None and len(mi):
            try:
                cv2.aruco.drawDetectedMarkers(preview, mc, mi)
            except cv2.error as exc:
                print(f"[ChArUco warning] Cannot draw markers: {exc}")
        points, ids, error = _safe_charuco_points(cc, ci)
        if error and n_corners:
            print(f"[ChArUco warning] {error}; invalid corners not drawn.")
        if points is not None:
            # Draw manually: OpenCV 5.0 can reject otherwise valid arrays in
            # drawDetectedCornersCharuco with an assertion. This also avoids
            # a preview error becoming a control-app shutdown.
            for xy in points.reshape(-1, 2):
                x, y = (int(round(float(v))) for v in xy)
                cv2.circle(preview, (x, y), 3, (0, 255, 0), -1)
        label = f"Markers {n_markers}/70  Corners {len(points) if points is not None else 0}/117"
        cv2.rectangle(preview, (8, 8), (565, 54), (30, 30, 30), -1)
        cv2.putText(preview, label, (20, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.86, (255, 255, 255), 2, cv2.LINE_AA)
        output = self.overlay_dir / f"{path.stem}_detected.png"
        if not cv2.imwrite(str(output), preview):
            raise RuntimeError(f"Failed to write overlay: {output}")
        valid_count = len(points) if points is not None else 0
        self.last_details = {"path": str(path), "markers": n_markers, "corners": valid_count,
                             "detection_warning": error if n_corners else None}
        print(f"{path.name}: {n_markers}/70 markers, {valid_count}/117 valid corners. Overlay: {output}")
        if valid_count < MIN_CORNERS:
            print("Not enough detectable ChArUco corners; move/tilt board and improve lighting.")
        return output

    def calibrate(self, square_mm: float, marker_mm: float):
        cv2 = _cv2()
        board = _board(square_mm, marker_mm)
        detector = cv2.aruco.CharucoDetector(board)
        world_corners = np.array(board.getChessboardCorners(), dtype=np.float32)
        obj_list = []
        img_list = []
        used_files = []
        image_size = None
        coverages = []
        # One image cannot constrain intrinsics; record different tilted views.
        for path in self.images():
            try:
                _, gray = _read_gray(path)
            except ValueError as exc:
                print(f"[skip] {path.name}: {exc}")
                continue
            size = (gray.shape[1], gray.shape[0])
            if image_size is not None and size != image_size:
                print(f"[skip] different resolution: {path.name}")
                continue
            cc, ci, _, _ = detector.detectBoard(gray)
            cc, ci, warning = _safe_charuco_points(cc, ci)
            if warning and ci is None:
                print(f"[skip] {path.name}: {warning}")
            count = 0 if ci is None else len(ci)
            if count < MIN_CORNERS:
                print(f"[skip] only {count} corners: {path.name}")
                continue
            ids = ci.reshape(-1).astype(int)
            if len(np.unique(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(world_corners):
                print(f"[skip] invalid corner IDs: {path.name}")
                continue
            obj_list.append(world_corners[ids].copy())
            img_pts = np.asarray(cc, dtype=np.float32).reshape(-1, 1, 2)
            img_list.append(img_pts)
            coverages.append(float(len(ids))/len(world_corners))
            used_files.append(path.name)
            image_size = size
            print(f"[OK] {path.name}: {len(ids)}/117 corners")
        if len(used_files) < MIN_VIEWS:
            reason = (f"Only {len(used_files)} usable views; need >= {MIN_VIEWS} "
                      f"(recommended {RECOMMENDED_VIEWS}-30).")
            self._save_pre_fit_diagnostics(reason, used_files, image_size)
            raise RuntimeError(reason + " No calibration saved.")
        # Different viewpoints are necessary: do not trust identical stationary views.
        all_centers = np.asarray([p.reshape(-1, 2).mean(axis=0) for p in img_list])
        all_scales = np.asarray([np.linalg.norm(np.ptp(p.reshape(-1, 2), axis=0)) for p in img_list])
        center_span_px = np.ptp(all_centers, axis=0)
        scale_ratio = float(all_scales.max() / max(1., all_scales.min()))
        if (max(center_span_px[0] / image_size[0], center_span_px[1] / image_size[1]) < .1
                and scale_ratio < 1.2):
            reason = "Almost identical views: move AND tilt the board over the image."
            self._save_pre_fit_diagnostics(reason, used_files, image_size)
            raise RuntimeError(reason + " No calibration saved.")
        try:
            rms, K, distortion, rotation_vecs, translation_vecs = cv2.calibrateCamera(
                obj_list, img_list, image_size, None, None,
            )
        except cv2.error as exc:
            reason = f"OpenCV calibrateCamera failed: {exc}"
            self._save_pre_fit_diagnostics(reason, used_files, image_size)
            raise RuntimeError(reason) from exc
        # Per-view RMS in pixels; identify bad views.
        view_rms = []
        all_errors = []
        for object_pts, observed, rv, tv in zip(obj_list, img_list, rotation_vecs, translation_vecs):
            projected, _ = cv2.projectPoints(object_pts, rv, tv, K, distortion)
            norms = np.linalg.norm(observed.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
            view_rms.append(float(np.sqrt(np.mean(norms**2))))
            all_errors.extend(norms.tolist())
        mean_error = float(np.mean(all_errors))
        cx, cy = float(K[0, 2]), float(K[1, 2])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        # Even a rejected solution is useful for diagnosing poor viewing geometry.
        tilts_deg = []
        tilt_x_deg = []
        tilt_y_deg = []
        for rv in rotation_vecs:
            R, _ = cv2.Rodrigues(rv)
            n = R[:, 2]  # calibration plane normal in camera coordinates
            tilts_deg.append(float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1)))))
            tilt_x_deg.append(float(np.degrees(np.arctan2(abs(n[1]), abs(n[2])))))
            tilt_y_deg.append(float(np.degrees(np.arctan2(abs(n[0]), abs(n[2])))))
        reasons = []
        if not np.isfinite(rms) or not np.all(np.isfinite(K)) or not np.all(np.isfinite(distortion)):
            reasons.append("Non-finite calibration values (K, distortion or RMS).")
        if not (0.2*image_size[0] < fx < 20*image_size[0]):
            reasons.append(f"fx={fx:.1f}px outside plausible range "
                           f"({0.2*image_size[0]:.0f}, {20*image_size[0]:.0f})px")
        if not (0.2*image_size[1] < fy < 20*image_size[1]):
            reasons.append(f"fy={fy:.1f}px outside plausible range "
                           f"({0.2*image_size[1]:.0f}, {20*image_size[1]:.0f})px")
        if not (-0.5*image_size[0] < cx < 1.5*image_size[0]):
            reasons.append(f"cx={cx:.1f}px outside plausible range "
                           f"({-0.5*image_size[0]:.0f}, {1.5*image_size[0]:.0f})px")
        if not (-0.5*image_size[1] < cy < 1.5*image_size[1]):
            reasons.append(f"cy={cy:.1f}px outside plausible range "
                           f"({-0.5*image_size[1]:.0f}, {1.5*image_size[1]:.0f})px")
        if tilts_deg and (max(tilts_deg) < 12 or max(tilts_deg) - min(tilts_deg) < 8):
            reasons.append(f"Insufficient tilt diversity: min={min(tilts_deg):.1f}°, "
                           f"max={max(tilts_deg):.1f}°; need strongly and differently tilted views")
        if rms > 1.0:
            reasons.append(f"Reprojection RMS {rms:.3f}px exceeds 1.0px")
        diagnostics = {
            "schema_version": 1,
            "kind": "camera_intrinsics_diagnostics",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "image_size_wh": list(image_size),
            "camera_matrix": K.tolist(),
            "dist_coeffs": distortion.ravel().tolist(),
            "rms_px": float(rms),
            "mean_reprojection_error_px": mean_error,
            "fx_fy_ratio": float(fx / fy) if fy else None,
            "principal_point_px": [cx, cy],
            "source_images": used_files,
            "per_view": [
                {"filename": name, "reprojection_rms_px": float(err),
                 "tilt_total_deg": float(tilt), "tilt_x_deg": float(tx), "tilt_y_deg": float(ty),
                 "corner_coverage": float(coverage)}
                for name, err, tilt, tx, ty, coverage in zip(
                    used_files, view_rms, tilts_deg, tilt_x_deg, tilt_y_deg, coverages)
            ],
            "note": "Pose tilt angles are estimated using the fitted intrinsics; "
                    "do not trust their absolute values if the camera matrix was rejected.",
        }
        out_diag = self.ctx.calibration.save("camera_intrinsics_diagnostics", diagnostics)
        self._print_diagnostics(diagnostics)
        print(f"Saved attempt diagnostics: {out_diag}")
        if reasons:
            raise RuntimeError("Calibration REJECTED (see reasons and full K above). "
                               "Previous valid camera_intrinsics was NOT overwritten.")
        data = {
            "schema_version": 1,
            "kind": "camera_intrinsics",
            "camera_name": self.ctx.camera.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "OpenCV ChArUco calibrateCamera / legacy pattern",
            "image_size_wh": list(image_size),
            "board_squares_wh": list(BOARD_SIZE),
            "aruco_dictionary": DICTIONARY_NAME,
            "legacy_pattern": True,
            "square_size_m": float(square_mm/1000),
            "marker_size_m": float(marker_mm/1000),
            "rms_px": float(rms),
            "mean_reprojection_error_px": mean_error,
            "view_rms_px": dict(zip(used_files, view_rms)),
            "camera_matrix": K.tolist(),
            "dist_coeffs": distortion.ravel().tolist(),
            "source_images": used_files,
            "mean_corner_coverage": float(np.mean(coverages)),
            "board_tilt_degrees": dict(zip(used_files, tilts_deg)),
            "caution": "Intrinsics only: camera-to-robot pose not yet calibrated. "
                       "Check on independent images before use.",
        }
        out = self.ctx.calibration.save("camera_intrinsics", data)
        print(f"Saved camera intrinsics: {out}")
        print("Keep resolution/focus/zoom unchanged; validate on new images.")
        return data

    def _save_pre_fit_diagnostics(self, reason, files, size) -> None:
        diagnostics = {
            "schema_version": 1,
            "kind": "camera_intrinsics_diagnostics",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "accepted": False,
            "rejection_reasons": [reason],
            "image_size_wh": list(size) if size else None,
            "camera_matrix": None,
            "dist_coeffs": None,
            "rms_px": None,
            "mean_reprojection_error_px": None,
            "principal_point_px": None,
            "fx_fy_ratio": None,
            "source_images": files,
            "per_view": [],
            "note": "No camera matrix was estimated; insufficient data or OpenCV error.",
        }
        path = self.ctx.calibration.save("camera_intrinsics_diagnostics", diagnostics)
        self._print_diagnostics(diagnostics)
        print(f"Saved attempt diagnostics: {path}")

    @staticmethod
    def _print_diagnostics(diagnostics: dict) -> None:
        print("\n===== ChArUco intrinsics diagnostics =====")
        print("Accepted:", diagnostics.get("accepted"))
        print("Views:", len(diagnostics.get("per_view", [])))
        print("Resolution:", diagnostics.get("image_size_wh"))
        print("RMS [px]:", diagnostics.get("rms_px"))
        print("Mean reprojection error [px]:", diagnostics.get("mean_reprojection_error_px"))
        matrix = diagnostics.get("camera_matrix")
        print("Full K [px]:\n", "not estimated" if matrix is None else np.array2string(
            np.asarray(matrix), precision=5, suppress_small=True))
        print("Distortion coefficients:", diagnostics.get("dist_coeffs"))
        print("Principal point [cx, cy] px:", diagnostics.get("principal_point_px"))
        print("fx/fy:", diagnostics.get("fx_fy_ratio"))
        print("Per-image tilt and reprojection RMS (degrees/px):")
        for item in diagnostics.get("per_view", []):
            print(f"  {item['filename']}: total={item['tilt_total_deg']:.1f}°, "
                  f"x={item['tilt_x_deg']:.1f}°, y={item['tilt_y_deg']:.1f}°, "
                  f"RMS={item['reprojection_rms_px']:.3f}px")
        print("NOTE:", diagnostics.get("note", ""))
        for reason in diagnostics.get("rejection_reasons", []):
            print("REJECTED:", reason)
        print("========================================\n")

    def report(self):
        result = self.ctx.calibration.load("camera_intrinsics")
        if result is None:
            print("No saved camera_intrinsics calibration yet.")
        else:
            print(f"Saved calibration {result.get('created_at_utc')}")
            print(f"  RMS: {result.get('rms_px')} px")
            print(f"  K: {np.array(result.get('camera_matrix'))}")
            print(f"  distortion: {result.get('dist_coeffs')}")

    def close(self):
        self.stop_loopcapture()


def setup(ctx):
    return CharucoCalibration(ctx)
