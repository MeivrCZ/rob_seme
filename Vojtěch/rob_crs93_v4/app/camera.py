"""One camera owner: actual Basler hardware OR repeatable image replay."""
from __future__ import annotations
from pathlib import Path
from threading import RLock


class CameraManager:
    def __init__(self, name: str = "camera-crs93"):
        self.name = name
        # Shared by CLI snapshots, calibration worker and mode switches.
        self._lock = RLock()
        self._camera = None
        self._replay_files: list[Path] = []
        self._replay_idx = 0
        self._replay_dir: Path | None = None

    @property
    def mode(self) -> str | None:
        with self._lock:
            if self._camera is not None:
                return "real"
            if self._replay_files:
                return "replay"
            return None

    @property
    def replay_directory(self) -> Path | None:
        return self._replay_dir

    @property
    def connected(self) -> bool:
        return self.mode is not None

    def open(self) -> None:
        with self._lock:
            if self.mode == "real":
                return
            if self.mode == "replay":
                raise RuntimeError("Close replay before opening Basler camera.")
            from ctu_crs.basler_camera import BaslerCamera
            camera = BaslerCamera()
            try:
                camera.connect_by_name(self.name)
                camera.open()
            except Exception:
                camera.close()
                raise
            self._camera = camera

    @staticmethod
    def validate_replay_directory(directory: Path) -> list[Path]:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Replay directory not found: {directory}")
        files = sorted(p for p in directory.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        if not files:
            raise ValueError(f"No image files found: {directory}")
        return files

    def open_replay(self, directory: Path) -> None:
        with self._lock:
            if self.connected:
                raise RuntimeError("Camera already open; close it before changing mode.")
            files = self.validate_replay_directory(directory)
            self._replay_files = files
            self._replay_idx = 0
            self._replay_dir = Path(directory).resolve()

    def capture(self):
        with self._lock:
            if self.mode == "replay":
                import cv2
                path = self._replay_files[self._replay_idx % len(self._replay_files)]
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Unreadable replay image: {path}")
                self._replay_idx += 1
                return image
            if self._camera is None:
                raise RuntimeError("Camera not open. Enter: camera open OR camera replay DIR")
            image = self._camera.grab_image()
            if image is None:
                raise RuntimeError("Camera did not return an image.")
            return image

    def close(self) -> None:
        with self._lock:
            if self._camera is not None:
                try:
                    self._camera.close()
                finally:
                    self._camera = None
            self._replay_files = []
            self._replay_idx = 0
            self._replay_dir = None
