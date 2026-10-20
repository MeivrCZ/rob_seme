"""Hromadny test ArUcoBlockDetector na obrazcich ve slozce testpics.

Adresarova struktura:
    projekt/
      aruco_block_detector.py
      debug_testpics.py
      testpics/
        obrazek1.png
        obrazek2.jpg

Spusteni:
    python debug_testpics.py            # zobrazuje jednotlive vysledky
    python debug_testpics.py --no-gui   # jen vypocita a ulozi vysledky

Vysledky: testpics_debug/ (oznacene obrazky + results.csv)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from aruco_block_detector import ArucoBlockDetector


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: Path):
    """Nacteni, ktere funguje i s ceskou diakritikou v ceste na Windows."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def save_image(path: Path, image: np.ndarray) -> None:
    ext = path.suffix.lower()
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"Nepodarilo se zakodovat obrazek: {path}")
    encoded.tofile(str(path))


def fit_to_screen(image: np.ndarray, max_width: int = 1280,
                  max_height: int = 800) -> np.ndarray:
    """Zmensi nahled pri zachovani pomeru stran; ulozeny obrazek je plny."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ArUco detekce na cele slozce")
    parser.add_argument("--no-gui", action="store_true",
                        help="Neotevirat okna s obrazky (jen ulozit vysledky)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Volitelny JSON s kalibraci pro ArucoBlockDetector")
    args = parser.parse_args()

    source_dir = ROOT / "testpics"
    output_dir = ROOT / "testpics_debug"

    if not source_dir.is_dir():
        raise SystemExit(f"Slozka s obrazky neexistuje: {source_dir}")

    images = sorted(
        (p for p in source_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
    if not images:
        raise SystemExit(f"Ve slozce nejsou podporovane obrazky: {source_dir}")

    output_dir.mkdir(exist_ok=True)
    detector = ArucoBlockDetector(calibration_file=args.config)
    rows = []
    found = 0

    print(f"Nalezeno {len(images)} obrazku. Vysledky: {output_dir}")
    print("GUI: mezernik / libovolna klavesa = dalsi, Q nebo Esc = konec\n")

    for i, path in enumerate(images, start=1):
        try:
            frame = read_image(path)
            if frame is None:
                raise ValueError("Soubor nelze nacist jako obrazek")

            pose = detector.detect(frame)
            preview = detector.draw(frame, pose)
            output_path = output_dir / f"{path.stem}_debug.png"
            save_image(output_path, preview)

            if pose is None:
                status, center_x, center_y, angle, visible_ids = (
                    "NENALEZENO", "", "", "", ""
                )
                print(f"[{i}/{len(images)}] {path.name} -> NENALEZENO")
            else:
                found += 1
                status = "OK" if len(pose.visible_ids) == 2 else "JEDNA_ZNACKA"
                center_x, center_y = pose.center_px
                angle = round(pose.angle_deg, 3)
                visible_ids = ",".join(map(str, pose.visible_ids))
                print(f"[{i}/{len(images)}] {path.name} -> {status} | "
                      f"stred=({center_x}, {center_y}) px | "
                      f"uhel={angle:+.3f} deg | ID={visible_ids}")

            rows.append({
                "soubor": path.name,
                "stav": status,
                "x_px": center_x,
                "y_px": center_y,
                "uhel_deg": angle,
                "viditelne_id": visible_ids,
                "nahled": output_path.name,
            })

            if not args.no_gui:
                cv2.imshow("ArUco debug | klavesa = dalsi | Q / Esc = konec",
                           fit_to_screen(preview))
                key = cv2.waitKey(0) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("Zobrazeni ukonceno uzivatelem.")
                    break

        except Exception as exc:
            print(f"[{i}/{len(images)}] {path.name} -> CHYBA: {exc}")
            rows.append({
                "soubor": path.name,
                "stav": f"CHYBA: {exc}",
                "x_px": "", "y_px": "", "uhel_deg": "",
                "viditelne_id": "", "nahled": "",
            })

    if not args.no_gui:
        cv2.destroyAllWindows()

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["soubor", "stav", "x_px", "y_px",
                        "uhel_deg", "viditelne_id", "nahled"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetekovano: {found}/{len(rows)} zpracovanych obrazku")
    print(f"CSV: {csv_path}")
    print(f"Oznacene obrazky: {output_dir}")


if __name__ == "__main__":
    main()
