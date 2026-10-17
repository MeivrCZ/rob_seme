"""Conservative detection of a complete, nearly face-on calibration hoop.

The centre is accepted only when there is one plausible closed ring in the
camera image. The auto scan must NEVER silently select one of several objects.
No robot commands or hardware access in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class HoopDetection:
    center: tuple[float, float] | None
    radius_px: float | None
    method: str
    reason: str
    candidates: tuple[tuple[float, float, float], ...]

    def annotate(self, image: np.ndarray) -> np.ndarray:
        preview = np.asarray(image).copy()
        if preview.ndim == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        if preview.ndim == 3 and preview.shape[2] == 4:
            preview = cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)
        for u, v, r in self.candidates:
            cv2.circle(preview, (round(u), round(v)), round(r), (0, 130, 255), 2)
            cv2.circle(preview, (round(u), round(v)), 5, (0, 130, 255), -1)
        if self.center is not None:
            u, v = self.center
            cv2.drawMarker(preview, (round(u), round(v)), (0, 255, 0),
                           cv2.MARKER_CROSS, 36, 3)
        label = ('OK' if self.center is not None else 'STOP') + ': ' + self.reason
        # A dark label background makes the diagnosis legible on a white table.
        cv2.rectangle(preview, (0, 0), (min(preview.shape[1], 1100), 48), (0, 0, 0), -1)
        cv2.putText(preview, label[:100], (12, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (0, 255, 0) if self.center is not None else (0, 150, 255), 2)
        return preview


def _cluster(candidates: list[tuple[float, float, float]]) -> list[list[tuple[float, float, float]]]:
    """Concentric inner/outer rim detections represent ONE hoop, not two."""
    clusters: list[list[tuple[float, float, float]]] = []
    for u, v, radius in sorted(candidates, key=lambda item: -item[2]):
        group = next((g for g in clusters if
                      np.hypot(u - np.median([a[0] for a in g]),
                               v - np.median([a[1] for a in g])) <=
                      max(12., .09 * max(radius, max(a[2] for a in g)))), None)
        if group is None:
            clusters.append([(u, v, radius)])
        else:
            group.append((u, v, radius))
    return clusters


def _valid_circle(u: float, v: float, r: float, shape: tuple[int, int]) -> bool:
    h, w = shape
    # Reject tiny circular hardware details and partial rings touching frame.
    r_min = max(35., min(h, w) * .035)
    r_max = min(h, w) * .29
    return r_min <= r <= r_max and r + 5 < u < w - r - 5 and r + 5 < v < h - r - 5


def _contour_candidates(gray: np.ndarray) -> list[tuple[float, float, float]]:
    h, w = gray.shape
    # Inner opening of the hoop is a closed contour even though its outer
    # perimeter joins the stem. Try global and local thresholding for lighting.
    masks = [cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]]
    masks.append(cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 81, 9))
    items: list[tuple[float, float, float]] = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for contour, hierarchy_row in zip(contours, hierarchy[0]):
            # The light aperture is a HOLE within the dark ring/stem.
            if hierarchy_row[3] < 0:
                continue
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            if area < 1000 or length <= 0:
                continue
            circularity = 4 * np.pi * area / (length * length)
            if circularity < .76:
                continue
            (u, v), r = cv2.minEnclosingCircle(contour)
            if not _valid_circle(u, v, r, (h, w)):
                continue
            x, y, ww, hh = cv2.boundingRect(contour)
            if min(ww, hh) == 0 or max(ww, hh) / min(ww, hh) > 1.30:
                continue
            if area / (np.pi * r * r) < .70:
                continue
            items.append((float(u), float(v), float(r)))
    return items


def _ring_contrast_ok(gray: np.ndarray, u: float, v: float, r: float) -> bool:
    """A true hoop has a light aperture, dark annulus, and light surroundings.

    Reject a single solid circular object (e.g. robot joint housing). Used
    for Hough fallback, where edge geometry alone is too permissive.
    """
    h, w = gray.shape
    contrasts = []
    for theta in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        x, y = np.cos(theta), np.sin(theta)
        def lum(rr):
            xx = round(u + rr * x)
            yy = round(v + rr * y)
            return float(gray[yy, xx]) if 0 <= xx < w and 0 <= yy < h else None
        inner = lum(.47 * r)
        outer = lum(1.23 * r)
        annulus = [lum(frac * r) for frac in (.85, .93, 1.00, 1.08, 1.14)]
        if inner is None or outer is None or any(z is None for z in annulus):
            continue
        contrasts.append(min(inner, outer) - min(annulus))
    return len(contrasts) >= 26 and np.median(contrasts) > 22 and \
        np.mean(np.asarray(contrasts) > 16) >= .65


def _hough_candidates(gray: np.ndarray) -> list[tuple[float, float, float]]:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 1.2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.3, minDist=60,
        param1=120, param2=65,
        minRadius=int(max(35, min(h, w) * .035)),
        maxRadius=int(min(h, w) * .29),
    )
    if circles is None:
        return []
    return [(float(u), float(v), float(r)) for u, v, r in circles[0]
            if _valid_circle(float(u), float(v), float(r), (h, w))
            and _ring_contrast_ok(gray, float(u), float(v), float(r))]


def detect_hoop_circle(frame: np.ndarray) -> HoopDetection:
    """Find ring centre conservatively. Never guess between separate circles.

    Prioritises the nearly circular HOLE of the hoop; Hough is fallback only.
    Includes diagnostic candidates for a saved annotated image when rejected.
    """
    frame = np.asarray(frame)
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError('Nepodporovany obraz kamery')
    gray = np.asarray(gray, dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    contours = [candidate for candidate in _contour_candidates(gray)
                if _ring_contrast_ok(gray, *candidate)]
    groups = _cluster(contours)
    if len(groups) == 1:
        group = groups[0]
        u, v = np.median([(c[0], c[1]) for c in group], axis=0)
        r = float(np.median([c[2] for c in group]))
        return HoopDetection((float(u), float(v)), r, 'contour',
                             f'jeden uzavreny kruhovy otvor, r={r:.0f}px', tuple(contours))
    if len(groups) > 1:
        return HoopDetection(None, None, 'contour',
                             f'{len(groups)} oddelenych kruhovych otvoru', tuple(contours))

    hough = _hough_candidates(gray)
    groups = _cluster(hough)
    if len(groups) == 1:
        group = groups[0]
        u, v = np.median([(c[0], c[1]) for c in group], axis=0)
        r = float(np.median([c[2] for c in group]))
        return HoopDetection((float(u), float(v)), r, 'hough',
                             f'jeden kruhovy kandidat, r={r:.0f}px', tuple(hough))
    why = 'zadny uzavreny kruh' if not groups else f'{len(groups)} ruznych kandidatu Hough'
    return HoopDetection(None, None, 'hough', why, tuple(hough))
