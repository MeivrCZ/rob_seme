"""ChArUco -> fixed-plane homography with *robot-frame* anchors (no intrinsics).

The printed board defines local metric XY. At least three measured non-collinear
ChArUco intersections define its 2D rigid pose relative to the CRS93 base.
The board is assumed stationary, flat and parallel to the robot's XY plane.
No hardware is imported or moved here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np

from homography import Homography

BOARD_SQUARES = (10, 14)
BOARD_DICTIONARY = "DICT_4X4_100"
LEGACY_PATTERN = True


def make_board(square_mm: float, marker_mm: float):
    if not (np.isfinite(square_mm) and np.isfinite(marker_mm)
            and 0 < marker_mm < square_mm):
        raise ValueError("Zmer rozmery: 0 < MARKER_MM < SQUARE_MM")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("Vyžadován opencv-contrib-python>=4.8 s cv2.aruco.CharucoDetector")
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, BOARD_DICTIONARY))
    board = aruco.CharucoBoard(
        BOARD_SQUARES, float(square_mm) / 1000., float(marker_mm) / 1000., dictionary
    )
    if not hasattr(board, "setLegacyPattern"):
        raise RuntimeError("OpenCV postrádá setLegacyPattern (nutné pro naši vytištěnou desku)")
    board.setLegacyPattern(LEGACY_PATTERN)
    return board


def _gray8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        gray = frame[..., 0]
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[-1] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Nepodporovany obrazovy format")
    if gray.dtype == np.uint16:
        gray = (gray >> 8).astype(np.uint8)
    if gray.dtype != np.uint8:
        raise ValueError("Vyžadován 8- nebo 16-bitový obraz kamery")
    return gray


def detect_board(frame: np.ndarray, square_mm: float, marker_mm: float):
    """Returns: ids, pixels Nx2, board-local metric XY Nx2, overlay."""
    board = make_board(square_mm, marker_mm)
    gray = _gray8(frame)
    corners, ids, marker_corners, marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if corners is None or ids is None:
        raise ValueError("ChArUco rohy nenalezeny. Zkontroluj slovnik/pattern a ostrost.")
    pixels = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    indices = np.asarray(ids, dtype=np.int32).reshape(-1)
    if len(indices) != len(pixels) or len(np.unique(indices)) != len(indices):
        raise ValueError("Nekonzistentni detekce ID rohu")
    chess = np.asarray(board.getChessboardCorners(), dtype=float)
    if len(indices) < 12 or np.any(indices < 0) or np.any(indices >= len(chess)):
        raise ValueError(f"Jen {len(indices)} ChArUco rohu; potreba alespon 12 a validni ID")
    local_xy = chess[indices, :2].copy()
    image_size = (gray.shape[1], gray.shape[0])
    hull = cv2.convexHull(pixels.astype(np.float32))
    if cv2.contourArea(hull) / (image_size[0] * image_size[1]) < .01:
        raise ValueError("Sachovnice je v obrazu prilis mala (<1 % plochy obrazu)")
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedCornersCharuco(
        preview, pixels.astype(np.float32).reshape(-1, 1, 2),
        indices.astype(np.int32).reshape(-1, 1), (0, 220, 0)
    )
    for (u, v), index in zip(pixels, indices):
        cv2.putText(preview, str(int(index)), (int(u) + 7, int(v) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, .47, (0, 30, 255), 1, cv2.LINE_AA)
    return indices, pixels, local_xy, preview


def rigid_pose(local_xy: np.ndarray, world_xy: np.ndarray, *, max_error_m: float = .005):
    """Estimate rotation+translation, never fit a fictitious XY scale/shear."""
    local = np.asarray(local_xy, dtype=float)
    world = np.asarray(world_xy, dtype=float)
    if local.shape != world.shape or local.ndim != 2 or local.shape[1] != 2 or len(local) < 3:
        raise ValueError("Potreba alespon 3 dvojice rohu ID -> robot XY")
    if not np.isfinite(local).all() or not np.isfinite(world).all():
        raise ValueError("Neplatne souradnice")
    if cv2.contourArea(cv2.convexHull(local.astype(np.float32))) < 0.0001:
        raise ValueError("Vyber 3 rohy dostatecne daleko od sebe, ne v jedne primce")
    A = local - local.mean(axis=0)
    B = world - world.mean(axis=0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    # For row vector points: world = local @ R + t, det(R)=+1.
    reflection = np.eye(2)
    reflection[-1, -1] = np.linalg.det(U @ Vt)
    R = U @ reflection @ Vt
    t = world.mean(axis=0) - local.mean(axis=0) @ R
    estimates = local @ R + t
    errors = np.linalg.norm(estimates - world, axis=1)
    worst = float(errors.max())
    if worst > max_error_m:
        raise ValueError(f"Roboticke referencni body nesedi: max chyba {worst*1000:.1f} mm "
                         f"> {max_error_m*1000:.1f} mm. Over oznacene ID, rozmery a TCP.")
    return R, t, float(np.sqrt(np.mean(errors ** 2)))


@dataclass
class CharucoHomographySession:
    folder: Path
    square_mm: float | None = None
    marker_mm: float | None = None
    image_size: tuple[int, int] | None = None
    ids: list[int] = field(default_factory=list)
    pixels: list[list[float]] = field(default_factory=list)
    local_xy: list[list[float]] = field(default_factory=list)
    anchors: dict[int, list[float]] = field(default_factory=dict)  # ID: robot XYZ in m

    @property
    def config_path(self) -> Path:
        return self.folder / "charuco_board.json"

    @property
    def image_path(self) -> Path:
        return self.folder / "charuco_frame.png"

    @property
    def preview_path(self) -> Path:
        return self.folder / "charuco_ids.png"

    def set_dimensions(self, square_mm: float, marker_mm: float):
        make_board(square_mm, marker_mm)
        self.square_mm = float(square_mm)
        self.marker_mm = float(marker_mm)
        self.clear_observations()
        self.save()

    def clear_observations(self):
        self.image_size = None
        self.ids.clear()
        self.pixels.clear()
        self.local_xy.clear()
        self.anchors.clear()
        self.image_path.unlink(missing_ok=True)
        self.preview_path.unlink(missing_ok=True)
        self.save()

    def capture(self, frame: np.ndarray):
        if self.square_mm is None or self.marker_mm is None:
            raise ValueError("Nejprve zmer desku a zadej: homo board config SQUARE_MM MARKER_MM")
        if self.anchors:
            raise ValueError("Deska ma jiz referencni body. Pred novym snimkem: homo board clear")
        ids, pixels, local, overlay = detect_board(frame, self.square_mm, self.marker_mm)
        self.folder.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.image_path), frame):
            raise RuntimeError("Nepodarilo se ulozit fotku")
        if not cv2.imwrite(str(self.preview_path), overlay):
            raise RuntimeError("Nepodarilo se ulozit obrazek s ID")
        self.image_size = (frame.shape[1], frame.shape[0])
        self.ids = [int(i) for i in ids]
        self.pixels = pixels.tolist()
        self.local_xy = local.tolist()
        self.save()
        return len(self.ids)

    def add_anchor(self, corner_id: int, xyz_m: tuple[float, float, float]):
        if corner_id not in self.ids:
            raise ValueError(f"ID {corner_id} neni na snimku; najdi IDs v {self.preview_path.name}")
        xyz = np.asarray(xyz_m, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Robot XYZ musi byt tri konecne hodnoty v metrech")
        self.anchors[int(corner_id)] = xyz.tolist()
        self.save()

    def solve(self) -> tuple[Homography, float]:
        if self.image_size is None or len(self.ids) < 12:
            raise ValueError("Nejprve homo board capture")
        if len(self.anchors) < 3:
            raise ValueError("Potreba alespon 3 nekolinearni reference: homo board teach ID / anchor ID X Y Z")
        id_to_index = {id_: i for i, id_ in enumerate(self.ids)}
        local = np.asarray(self.local_xy)
        anchors_indices = [id_to_index[i] for i in self.anchors]
        anchors = np.asarray(list(self.anchors.values()), dtype=float)
        if float(np.max(np.abs(anchors[:, 2] - np.median(anchors[:, 2])))) > .003:
            raise ValueError("Referencni body nejsou v jedne rovine Z (tolerance 3 mm)")
        R, t, rms_m = rigid_pose(local[anchors_indices], anchors[:, :2])
        robot_xy = local @ R + t
        z = float(np.median(anchors[:, 2]))
        H = Homography.fit(np.asarray(self.pixels), robot_xy, self.image_size,
                           z_m=z, min_coverage=.01, max_rms_px=3.0)
        return H, rms_m

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1, "board_squares": list(BOARD_SQUARES),
            "dictionary": BOARD_DICTIONARY, "legacy_pattern": LEGACY_PATTERN,
            "square_mm": self.square_mm, "marker_mm": self.marker_mm,
            "image_size": self.image_size, "ids": self.ids, "pixels": self.pixels,
            "local_xy": self.local_xy, "anchors": {str(k): v for k, v in self.anchors.items()},
        }
        self.config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self):
        if not self.config_path.is_file():
            return
        d = json.loads(self.config_path.read_text(encoding="utf-8"))
        if (d.get("version") != 1 or d["board_squares"] != list(BOARD_SQUARES) or
                d["dictionary"] != BOARD_DICTIONARY or d["legacy_pattern"] != LEGACY_PATTERN):
            raise ValueError("Neplatna nebo nekompatibilni ChArUco deska")
        self.square_mm = d["square_mm"]
        self.marker_mm = d["marker_mm"]
        if self.square_mm is not None:
            make_board(self.square_mm, self.marker_mm)
        self.image_size = tuple(d["image_size"]) if d["image_size"] else None
        self.ids = [int(i) for i in d["ids"]]
        self.pixels = d["pixels"]
        self.local_xy = d["local_xy"]
        self.anchors = {int(i): list(map(float, xyz)) for i, xyz in d["anchors"].items()}
        if len(self.ids) != len(self.pixels) or len(self.ids) != len(self.local_xy):
            raise ValueError("Poskozena data ChArUco pozorovani")

    def status(self) -> str:
        return (f"ChArUco {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]}, {BOARD_DICTIONARY}, legacy; "
                f"ctverec={self.square_mm} mm, marker={self.marker_mm} mm; "
                f"detekovano {len(self.ids)} rohu, kotvy {len(self.anchors)}/3; "
                f"viz {self.preview_path}")
