"""Lokalizace stredu a natoceni bloku pomoci jedne nebo dvou ArUco znacek.

Pouziti:
    python -m pip install opencv-contrib-python numpy
    python aruco_block_detector.py "snimek.png"

Znacky: DICT_4X4_50, ID 1 (nahore vlevo), ID 2 (dole vpravo).
Vychozi geometrie je zkalibrovana z referencni fotografie 1920 x 1200.
Souradnice jsou pixely (x vpravo, y dolu). Uhel je ve stupnich,
kladny po smeru hodinovych rucicek v obrazu.

Neni nutne znat kalibracni matici kamery, pokud chceme pouze obrazove
souradnice. Pro vyssi presnost doporucujeme predem undistort snimku.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BlockPose:
    """Vysledek jedne detekce."""

    center_px: tuple[int, int]
    angle_deg: float
    visible_ids: tuple[int, ...]
    corners_by_id: dict[int, np.ndarray] = field(repr=False)


class ArucoBlockDetector:
    """Detektor bloku s moznosti spolehliveho provozu s jedinou znackou.

    Posun od znacky ke stredu bloku je v lokalnich jednotkach znacky:
    (u, v) = (0,0) v levem hornim rohu znacky a (1,1) v pravem dolnim.
    Homografie pro kazdou znacku prenese tento posun na aktualni obraz.

    Vychozi posuny byly vypocteny z dodane ukazkove fotografie.
    Pri zmene rozmisteni / velikosti znacek zavolejte calibrate(...)
    na snimku s obema viditelnymi znackami.
    """

    DICTIONARY = cv2.aruco.DICT_4X4_50

    # Pozice stredu bloku v soustave jednotlivych znacek.
    # Hodnoty jsou relativni k velikosti znacky, nikoli pixely.
    DEFAULT_OFFSETS_UV = {
        1: (1.4329242, 1.4250765),
        2: (-0.4343172, -0.4452172),
    }

    _UNIT_SQUARE = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
    )

    def __init__(self, calibration_file: str | Path | None = None):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.DICTIONARY)
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.offsets_uv = dict(self.DEFAULT_OFFSETS_UV)

        if calibration_file is not None:
            self.load_calibration(calibration_file)

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Obrazek je prazdny.")
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError("Ocekavan obrazek BGR, BGRA nebo grayscale.")

    def _markers(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        corners_list, ids, _ = self._detector.detectMarkers(self._gray(frame))
        if ids is None:
            return {}

        markers = {}
        for marker_id, corners in zip(ids.ravel(), corners_list):
            marker_id = int(marker_id)
            if marker_id in self.offsets_uv:
                # Poradi rohu: horni levy, horni pravy, dolni pravy, dolni levy.
                markers[marker_id] = corners.reshape(4, 2).astype(np.float32)
        return markers

    @classmethod
    def _homography(cls, corners: np.ndarray) -> np.ndarray:
        return cv2.getPerspectiveTransform(cls._UNIT_SQUARE, corners)

    @staticmethod
    def _transform_point(H: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
        point = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
        return cv2.perspectiveTransform(point, H)[0, 0].astype(np.float64)

    def detect(self, frame: np.ndarray) -> BlockPose | None:
        """Vrati polohu stredu a uhel, nebo None bez rozpoznanych znacek.

        Pri dvou znackach vyuzije jejich stredy; pri jedne odhadne
        polohu stredu z kalibrovaneho posunu v lokalni rovine znacky.
        """
        markers = self._markers(frame)
        if not markers:
            return None

        if 1 in markers and 2 in markers:
            # Znacky jsou rozmistene symetricky kolem stredu bloku.
            center = 0.5 * (
                markers[1].mean(axis=0).astype(np.float64)
                + markers[2].mean(axis=0).astype(np.float64)
            )
        else:
            marker_id, corners = next(iter(markers.items()))
            H = self._homography(corners)
            center = self._transform_point(H, self.offsets_uv[marker_id])

        # Smer osy X znacek, orientace vuci vodorovne ose obrazu.
        # Obe znacky jsou na bloku nalepeny ve stejne orientaci.
        directions = []
        for corners in markers.values():
            # Prumer hornich a dolnich hran potlaci sum detekce rohu.
            vx = 0.5 * (corners[1] - corners[0] + corners[2] - corners[3])
            vx /= max(float(np.linalg.norm(vx)), 1e-9)
            directions.append(vx)
        direction = np.mean(directions, axis=0)
        angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))

        return BlockPose(
            center_px=(int(round(center[0])), int(round(center[1]))),
            angle_deg=angle_deg,
            visible_ids=tuple(sorted(markers)),
            corners_by_id=markers,
        )

    def calibrate(
        self,
        frame: np.ndarray,
        center_px: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Prekalibruje posuny z fotografie, kde jsou videt obe znacky.

        center_px lze predat rucne (skutecny stred bloku v referencnim
        snimku). Pokud chybi, pouzije stred mezi stredy znacek.
        Vraci referencni stred v obrazovych pixelech.
        """
        markers = self._markers(frame)
        if 1 not in markers or 2 not in markers:
            raise ValueError("Kalibrace vyzaduje obe znacky (ID 1 a 2).")

        if center_px is None:
            center = 0.5 * (markers[1].mean(axis=0) + markers[2].mean(axis=0))
        else:
            center = np.asarray(center_px, dtype=np.float32)
            if center.shape != (2,):
                raise ValueError("center_px musi byt dvojice (x, y).")

        for marker_id, corners in markers.items():
            inv_H = np.linalg.inv(self._homography(corners))
            u, v = self._transform_point(inv_H, (float(center[0]), float(center[1])))
            self.offsets_uv[marker_id] = (float(u), float(v))
        return float(center[0]), float(center[1])

    def save_calibration(self, path: str | Path) -> None:
        content = {str(k): list(v) for k, v in self.offsets_uv.items()}
        Path(path).write_text(json.dumps(content, indent=2), encoding="utf-8")

    def load_calibration(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for marker_id in (1, 2):
            key = str(marker_id)
            if key not in data or len(data[key]) != 2:
                raise ValueError(f"Chybi kalibrace znacky ID {marker_id}.")
        self.offsets_uv = {
            marker_id: (float(data[str(marker_id)][0]), float(data[str(marker_id)][1]))
            for marker_id in (1, 2)
        }

    @staticmethod
    def draw(frame: np.ndarray, pose: BlockPose | None) -> np.ndarray:
        """Do kopie obrazku vykresli znacky, stred a smer natoceni."""
        if frame.ndim == 2:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            vis = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            vis = frame.copy()
        if pose is None:
            cv2.putText(vis, "Blok nenalezen", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return vis

        for marker_id, corners in pose.corners_by_id.items():
            pts = np.round(corners).astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 190, 255), 3, cv2.LINE_AA)
            c = pts.mean(axis=0).astype(int)
            cv2.putText(vis, f"ID {marker_id}", tuple(c + [0, -25]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 80, 255), 2, cv2.LINE_AA)

        x, y = pose.center_px
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 60, 4)
        a = np.radians(pose.angle_deg)
        endpoint = (int(x + 135 * np.cos(a)), int(y + 135 * np.sin(a)))
        cv2.arrowedLine(vis, (x, y), endpoint, (0, 0, 255), 4,
                        cv2.LINE_AA, tipLength=0.2)
        cv2.putText(vis, f"Stred: ({x}, {y}) | Uhel: {pose.angle_deg:+.2f} deg",
                    (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (10, 40, 220), 3, cv2.LINE_AA)
        cv2.putText(vis, f"Viditelne znacky: {pose.visible_ids}",
                    (30, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (10, 40, 220), 2, cv2.LINE_AA)
        return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Detekce stredu a natoceni ArUco bloku")
    parser.add_argument("image", help="Cesta ke snimku")
    parser.add_argument("--calibrate", action="store_true",
                        help="Zkalibrovat z tohoto snimku (musi byt videt obe znacky)")
    parser.add_argument("--config", default=None,
                        help="Soubor JSON s kalibraci, k nacteni nebo ulozeni")
    parser.add_argument("--output", default="detekce_bloku.png",
                        help="Cesta k obrazku s vykreslenou detekci")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        parser.error(f"Nelze otevrit obrazek: {args.image}")

    detector = ArucoBlockDetector(
        calibration_file=args.config if args.config and not args.calibrate else None
    )
    if args.calibrate:
        center = detector.calibrate(frame)
        print(f"Kalibracni stred: ({center[0]:.1f}, {center[1]:.1f})")
        if args.config:
            detector.save_calibration(args.config)
            print(f"Kalibrace ulozena: {args.config}")

    pose = detector.detect(frame)
    if pose:
        print(f"Stred: {pose.center_px}, uhel: {pose.angle_deg:+.2f} stupnu, "
              f"viditelne ID: {pose.visible_ids}")
    else:
        print("Blok nenalezen.")

    if not cv2.imwrite(args.output, detector.draw(frame, pose)):
        raise OSError(f"Nepodarilo se ulozit soubor: {args.output}")
    print(f"Vizualizace: {args.output}")


if __name__ == "__main__":
    main()
