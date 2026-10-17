"""Interaktivni kalibrace homografie CRS93 v jedne rovine, bez camera intrinsics.

Existujiciho robota predavejte do HomographySession - modul ho sam nehomi
ani nesaha na motory. Terce na TCP musi byt fyzicky spravne pripevneny.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from homography import Homography

DEFAULT_DIR = Path(__file__).resolve().parent / "data" / "homography"


def _bgr_frame(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return np.ascontiguousarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError("Obraz z kamery ma nepodporovany format")


def detect_single_circle(frame: np.ndarray) -> tuple[float, float] | None:
    """Automaticky stred kruhoveho otvoru hoopu, jen pokud je jednoznacny."""
    from hoop_detection import detect_hoop_circle
    return detect_hoop_circle(frame).center


def select_point(frame: np.ndarray, *, method: str = "click") -> tuple[float, float] | None:
    """Klik na stred terce, ENTER potvrdi, ESC zrusi, a = auto kruh.

    Vyber neni mozny v bezhlavem prostredi bez OpenCV GUI.
    """
    frame = _bgr_frame(frame)
    point: list[tuple[float, float] | None] = [None]
    suggested = detect_single_circle(frame) if method == "auto" else None
    if suggested:
        point[0] = suggested

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point[0] = (float(x), float(y))

    title = "Homografie - klik na stred terce, ENTER potvrdit, a auto, ESC zrusit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while True:
            preview = frame.copy()
            if point[0] is not None:
                u, v = map(int, point[0])
                cv2.drawMarker(preview, (u, v), (20, 220, 20), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(preview, f"({u}, {v})", (u + 12, v - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 220, 20), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(35) & 0xFF
            if key in (13, 10, 32) and point[0] is not None:
                return point[0]
            if key in (27, ord("q")):
                return None
            if key == ord("a"):
                point[0] = detect_single_circle(frame)
                if point[0] is None:
                    print("Nebylo nalezeno prave jedno kolecko. Klikni rucne.")
    finally:
        cv2.destroyWindow(title)


@dataclass
class HomographySession:
    """Kalibracni data jsou ukladana v metrech v soustave zakladny robota.

    Podporuje volitelny offset stredu terce od TCP, v souradnicich TCP.
    Dulezite: robot.fk(q) a robot.grab_image() jsou z ctu_crs (CRS93).
    """
    folder: Path = field(default_factory=lambda: DEFAULT_DIR)
    target_offset_tcp_m: tuple[float, float, float] = (0., 0., 0.)
    z_tolerance_m: float = .003
    samples: list[dict] = field(default_factory=list)
    image_size: tuple[int, int] | None = None
    homography: Homography | None = None
    is_dirty: bool = False  # prevent mapping with stale H after sample/offset changes

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        o = np.asarray(self.target_offset_tcp_m, dtype=float)
        if o.shape != (3,) or not np.all(np.isfinite(o)):
            raise ValueError("target_offset_tcp_m musi byt 3 cisla v metrech")
        if self.z_tolerance_m <= 0:
            raise ValueError("z_tolerance_m musi byt kladne")

    def add(self, pixel: tuple[float, float], world_xyz: tuple[float, float, float],
            image_size: tuple[int, int]) -> None:
        """Prida primou dvojici vygenerovanou ze stejne fotografie a 3D pose."""
        p = np.asarray(pixel, dtype=float)
        xyz = np.asarray(world_xyz, dtype=float)
        width, height = map(int, image_size)
        if p.shape != (2,) or xyz.shape != (3,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(xyz)):
            raise ValueError("Chybne souradnice pixelu nebo polohy terce")
        if not (0 <= p[0] < width and 0 <= p[1] < height):
            raise ValueError("Bod musi byt na obrazku")
        if self.image_size is not None and (width, height) != self.image_size:
            raise ValueError("Behem kalibrace se zmenilo rozliseni kamery")
        if self.samples:
            current_z = float(np.median([s["xyz_m"][2] for s in self.samples]))
            if abs(xyz[2] - current_z) > self.z_tolerance_m:
                raise ValueError(f"Terc musi zustat v jedne rovine Z ~ {current_z:.4f} m; "
                                 f"tento ma Z={xyz[2]:.4f} m")
        self.image_size = (width, height)
        self.samples.append({"uv_px": p.tolist(), "xyz_m": xyz.tolist()})
        self.homography = None
        self.is_dirty = True
        self.save_samples()

    def capture(self, robot: Any, *, method: str = "click") -> bool:
        """Robot nema byt v pohybu. Vezme frame i pose a vybere stred terce.

        Neprovadi ZADNY pohyb ani homing; fyzicky stred terce musi byt znamy.
        """
        if robot.in_motion():
            raise RuntimeError("Robot se hybe! Capture lze provest jen po zastaveni")
        frame = robot.grab_image()
        if frame is None:
            raise RuntimeError("Kamera nevratila obraz")
        frame = _bgr_frame(frame)
        q = np.asarray(robot.get_q(), dtype=float)
        T = np.asarray(robot.fk(q), dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError("FK robota nevratila validni 4x4 transformaci")
        # Offset je v lokalnich souradnicich nastroje, jeho pozici spocitame predem.
        offset = np.asarray([*self.target_offset_tcp_m, 1.], dtype=float)
        xyz = (T @ offset)[:3]
        uv = select_point(frame, method=method)
        if uv is None:
            print("Capture zrusen")
            return False
        if robot.in_motion():
            raise RuntimeError("Robot se behem vyberu bodu pohnul; snimek zahazuji")
        if not np.allclose(np.asarray(robot.get_q(), dtype=float), q, atol=1e-3):
            raise RuntimeError("Poloha robota se behem vyberu zmenila; snimek zahazuji")
        self.add(uv, tuple(xyz), (frame.shape[1], frame.shape[0]))
        self.folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.folder / f"sample_{len(self.samples):03d}.png"), frame)
        print(f"Bod #{len(self.samples)} pixel={uv}, svet XY={xyz[:2].round(4)}, Z={xyz[2]:.4f} m")
        return True

    def solve(self, *, min_samples: int = 8, max_rms_px: float = 3.0) -> Homography:
        if self.image_size is None or len(self.samples) < min_samples:
            raise ValueError(f"Je potreba alespon {min_samples} kalibracnich bodu")
        px = np.array([s["uv_px"] for s in self.samples], dtype=float)
        xy = np.array([s["xyz_m"][:2] for s in self.samples], dtype=float)
        z = float(np.median([s["xyz_m"][2] for s in self.samples]))
        calib = Homography.fit(
            px, xy, self.image_size, z_m=z, max_rms_px=max_rms_px
        )
        self.homography = calib
        self.folder.mkdir(parents=True, exist_ok=True)
        calib.save(self.folder / "homography.npz")
        self.is_dirty = False
        self.save_samples()
        print(f"Homografie ulozena: {self.folder / 'homography.npz'}")
        print(f"RMS {calib.rms_px:.2f} px, RANSAC {calib.inliers}/{calib.total} bodu, Z={z:.4f} m")
        return calib

    def pixel_to_world(self, u: float, v: float, strict: bool = True) -> tuple[float, float]:
        if self.is_dirty:
            raise RuntimeError("Od posledniho vypoctu se zmenila data: nejdrive homo solve")
        if self.homography is None:
            self.homography = Homography.load(self.folder / "homography.npz")
        return self.homography.pixel_to_world(u, v, strict=strict)

    def save_samples(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / "samples.json"
        payload = {"version": 1, "image_size": self.image_size,
                   "target_offset_tcp_m": self.target_offset_tcp_m,
                   "z_tolerance_m": self.z_tolerance_m, "samples": self.samples,
                   "is_dirty": self.is_dirty}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_samples(self) -> int:
        path = self.folder / "samples.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["version"] != 1:
            raise ValueError("Nekompatibilni kalibracni data")
        old_offset = np.asarray(payload["target_offset_tcp_m"], dtype=float)
        if not np.allclose(old_offset, self.target_offset_tcp_m):
            raise ValueError("Zmenil se offset terce vuci TCP; nelze smichat data")
        size = payload["image_size"]
        loaded_samples = list(payload["samples"])
        # Offset lze ulozit i pred prvnim snimkem. Takovy soubor je platny.
        if size is None:
            if loaded_samples:
                raise ValueError("Kalibracni data maji body, ale chybi rozmery obrazku")
            self.samples = []
            self.image_size = None
            self.is_dirty = bool(payload.get("is_dirty", False))
            self.homography = None
            return 0
        self.samples = []
        self.image_size = None
        for s in loaded_samples:
            # add() zapisuje po kazdem bodu; opetovne nahrani je zbytecne,
            # data zachovame po validaci v pameti a zapisu potom jednou.
            self._validate_loaded_sample(s, size)
        if loaded_samples:
            zs = np.array([row["xyz_m"][2] for row in loaded_samples], dtype=float)
            if float(np.max(np.abs(zs - np.median(zs)))) > self.z_tolerance_m:
                raise ValueError("Ulozene body nemaji konstantni vysku Z")
        self.samples = loaded_samples
        self.image_size = tuple(map(int, size))
        self.is_dirty = bool(payload.get("is_dirty", True))
        self.homography = None
        if not self.is_dirty and (self.folder / "homography.npz").exists():
            self.homography = Homography.load(self.folder / "homography.npz")
        return len(self.samples)

    def _validate_loaded_sample(self, sample: dict, size: list[int]) -> None:
        p = np.asarray(sample["uv_px"], dtype=float)
        xyz = np.asarray(sample["xyz_m"], dtype=float)
        if p.shape != (2,) or xyz.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(xyz).all():
            raise ValueError("Neplatna ulozena kalibracni korespondence")
        if not 0 <= p[0] < size[0] or not 0 <= p[1] < size[1]:
            raise ValueError("Pixel ulozeneho bodu je mimo obraz")

    def undo(self) -> None:
        if not self.samples:
            raise ValueError("Zadne body")
        last_index = len(self.samples)
        self.samples.pop()
        (self.folder / f"sample_{last_index:03d}.png").unlink(missing_ok=True)
        self.homography = None
        self.is_dirty = True
        self.save_samples()

    def clear(self) -> None:
        self.samples.clear()
        for image in self.folder.glob("sample_*.png"):
            image.unlink()
        self.image_size = None
        self.homography = None
        self.is_dirty = True
        self.save_samples()

    def status(self) -> str:
        z = (np.median([s["xyz_m"][2] for s in self.samples]) if self.samples
             else self.homography.z_m if self.homography and not self.is_dirty else None)
        info = f"Homografie: {len(self.samples)} bodu, obraz {self.image_size}, Z={z}"
        if self.homography and not self.is_dirty:
            info += f", RMS={self.homography.rms_px:.2f} px"
        if self.is_dirty:
            info += ", data changed - run homo solve"
        return info
