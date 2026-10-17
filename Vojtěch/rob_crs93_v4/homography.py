"""Homografie pro statickou kameru CRS93: (u, v) <-> (x, y) v jedne rovine.

Bez kalibrace intrinsics/extrinsics. Souradnice x,y jsou v metrech
v soustave, ktera byla pouzita pri porizeni kalibracnich dat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Homography:
    """Prevadi pixely na X/Y v kalibrovane rovine.

    H: matice pixel -> world. Povoluje pouze stejny rozmer obrazu
    jako pri kalibraci. `strict=True` zakazuje extrapolaci mimo
    konvexni obal kalibracnich bodu.
    """

    H: np.ndarray
    image_size: tuple[int, int]  # (sirka_px, vyska_px)
    z_m: float
    world_hull: np.ndarray
    rms_px: float
    inliers: int
    total: int

    @classmethod
    def fit(
        cls,
        pixels: np.ndarray,
        world_xy: np.ndarray,
        image_size: tuple[int, int],
        *,
        z_m: float = 0.0,
        ransac_px: float = 3.0,
        min_coverage: float = 0.03,
        max_rms_px: float = 3.0,
    ) -> "Homography":
        """Najde projekci z XY do obrazu (RANSAC ma prah v pixelech),
        pak ji invertuje pro pixel -> XY.

        Minimum 4 nekolinearni korespondence; doporuceno 12+.
        """
        px = np.asarray(pixels, dtype=np.float64)
        xy = np.asarray(world_xy, dtype=np.float64)
        if px.ndim != 2 or xy.ndim != 2 or px.shape != xy.shape or px.shape[1] != 2:
            raise ValueError("pixels a world_xy musi mit stejny tvar (N,2)")
        n = len(px)
        if n < 4:
            raise ValueError("Pro homografii jsou potreba alespon 4 dvojice bodu")
        if not np.isfinite(px).all() or not np.isfinite(xy).all():
            raise ValueError("Souradnice obsahuji NaN/Inf")
        w, h = int(image_size[0]), int(image_size[1])
        if w <= 0 or h <= 0:
            raise ValueError("Neplatne rozmery obrazu")
        if np.any(px < 0) or np.any(px[:, 0] >= w) or np.any(px[:, 1] >= h):
            raise ValueError("Nektere body lezi mimo obraz")
        if ransac_px <= 0 or max_rms_px <= 0:
            raise ValueError("Prahy chyb museji byt kladne")

        px_hull = cv2.convexHull(px.astype(np.float32))
        xy_hull = cv2.convexHull(xy.astype(np.float32))
        if cv2.contourArea(xy_hull) <= 1e-9:
            raise ValueError("Roboticke body jsou kolinearni nebo skoro stejne")
        coverage = cv2.contourArea(px_hull) / (w * h)
        if coverage < min_coverage:
            raise ValueError(
                f"Body nepokryvaji obraz dostatecne ({coverage:.1%} < {min_coverage:.1%}). "
                "Pohybuj tercem pres vetsi cast obrazu."
            )

        # RANSAC prah musi byt v jednotkach CILOVEHO systemu: pixely!
        # Proto hledame svet -> kamera a na konci matici invertujeme.
        method = cv2.RANSAC if n > 4 else 0
        H_xy_to_px, mask = cv2.findHomography(
            xy, px, method=method, ransacReprojThreshold=float(ransac_px)
        )
        if H_xy_to_px is None or not np.isfinite(H_xy_to_px).all():
            raise ValueError("Homografii nelze spocitat. Over geometrii bodu.")
        inlier_mask = np.ones(n, dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
        n_inliers = int(inlier_mask.sum())
        if n_inliers < 4:
            raise ValueError("Mene nez 4 platne korespondence po RANSAC")
        projected = cv2.perspectiveTransform(
            xy[inlier_mask].reshape(-1, 1, 2).astype(np.float64), H_xy_to_px
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - px[inlier_mask], axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        if rms > max_rms_px:
            raise ValueError(f"Velka reprojekcni chyba {rms:.2f} px > {max_rms_px:.2f} px")
        H_px_to_xy = np.linalg.inv(H_xy_to_px)
        if not np.isfinite(H_px_to_xy).all():
            raise ValueError("Nespravna homografie")
        H_px_to_xy /= H_px_to_xy[2, 2]
        inlier_hull = cv2.convexHull(xy[inlier_mask].astype(np.float32)).reshape(-1, 2)
        return cls(
            H=H_px_to_xy, image_size=(w, h), z_m=float(z_m),
            world_hull=inlier_hull, rms_px=rms, inliers=n_inliers, total=n
        )

    def _check_resolution(self, image_size: tuple[int, int] | None) -> None:
        if image_size is not None and tuple(image_size) != self.image_size:
            raise ValueError(
                f"Rozliseni obrazu {tuple(image_size)} se lisi od kalibrace {self.image_size}. "
                "Zmen rozliseni kamery zpet nebo kalibruj znovu."
            )

    def pixel_to_world(
        self, u: float, v: float, *, image_size: tuple[int, int] | None = None,
        strict: bool = True
    ) -> tuple[float, float]:
        self._check_resolution(image_size)
        w, h = self.image_size
        if not (0 <= u < w and 0 <= v < h):
            raise ValueError("Pixel je mimo obraz")
        point = np.array([float(u), float(v), 1.0])
        mapped = self.H @ point
        if abs(mapped[2]) < 1e-12:
            raise ValueError("Nevlastni bod homografie")
        x, y = (mapped[:2] / mapped[2]).tolist()
        if strict and cv2.pointPolygonTest(
            np.asarray(self.world_hull, np.float32).reshape(-1, 1, 2),
            (float(x), float(y)), False
        ) < 0:
            raise ValueError("Vysledek lezi mimo zkalibrovanou oblast - nelze bezpecne extrapolovat")
        return float(x), float(y)

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        mapped = np.linalg.inv(self.H) @ np.array([x, y, 1.0], dtype=float)
        if abs(mapped[2]) < 1e-12:
            raise ValueError("Nevlastni bod homografie")
        u, v = mapped[:2] / mapped[2]
        return float(u), float(v)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, version=np.array(1), H=self.H, image_size=np.asarray(self.image_size),
            z_m=np.array(self.z_m), world_hull=self.world_hull,
            rms_px=np.array(self.rms_px), inliers=np.array(self.inliers),
            total=np.array(self.total)
        )

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("Nepodporovana verze kalibrace")
            H = np.array(data["H"], dtype=float)
            image_size = tuple(map(int, data["image_size"].tolist()))
            hull = np.array(data["world_hull"], dtype=np.float32)
            if H.shape != (3, 3) or not np.isfinite(H).all() or hull.ndim != 2 or hull.shape[1] != 2:
                raise ValueError("Poskozeny kalibracni soubor")
            return cls(
                H=H, image_size=image_size, z_m=float(data["z_m"]),
                world_hull=hull, rms_px=float(data["rms_px"]),
                inliers=int(data["inliers"]), total=int(data["total"])
            )


def pixel_to_world(u: float, v: float, h_matrix: np.ndarray) -> tuple[float, float]:
    """Pomocna kompatibilni funkce (jako v kodu kolegy), bez kontroly oblasti."""
    p = np.asarray(h_matrix, dtype=float) @ np.array([u, v, 1.0])
    if abs(p[2]) < 1e-12:
        raise ValueError("Nevlastni bod homografie")
    return float(p[0] / p[2]), float(p[1] / p[2])
