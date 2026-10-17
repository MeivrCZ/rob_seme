"""Pixel -> base XY calibration using fixed-plane homography for CRS93.

No motion/homing commands. Uses Context.robot and Context.camera, not a
separate CRS93 or Basler instance. This extension can be loaded while OFFLINE.
"""
from __future__ import annotations

import json
import numpy as np

from homography_capture import HomographySession


class _CaptureBridge:
    """Adapter for session's legacy capture(robot) API, with shared managers."""

    def __init__(self, ctx):
        self.ctx = ctx

    def in_motion(self) -> bool:
        # RobotManager.assert_stationary fails closed if driver lacks feedback.
        if self.ctx.robot.released:
            raise RuntimeError("Robot released: cannot trust joint feedback")
        if self.ctx.robot.mode == "real":
            self.ctx.robot.assert_stationary()
        return False

    def grab_image(self):
        return self.ctx.camera.capture()

    def get_q(self):
        return self.ctx.robot.get_q()

    def fk(self, q):
        # FK is pure kinematics; no motion commands.
        return self.ctx.robot.require().fk(np.asarray(q, dtype=float))


class HomographyExtension:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = HomographySession(folder=ctx.root / "data" / "homography")
        # Load samples for status and continuation, not robot/camera hardware.
        if (self.session.folder / "samples.json").is_file():
            saved = json.loads((self.session.folder / "samples.json").read_text(encoding="utf-8"))
            self.session.target_offset_tcp_m = tuple(saved["target_offset_tcp_m"])
            self.session.load_samples()

    def run(self, argv: list[str]) -> None:
        if not argv or argv == ["help"]:
            print("""Homografie: jedna rovina Z, pixel (u,v) -> robot (X,Y) [m].
  homo capture           vyber stred terce kliknutim (ENTER potvrdi, ESC zrusi)
  homo auto              pokus o kruh (ENTER potvrdi, klik opravi, ESC zrusi)
  homo offset X Y Z      nastav presny posun stredu terce v TCP [m] PRED snimanim
  homo solve             vypocet H, RANSAC, ulozeni do data/homography/homography.npz
  homo map U V           prevede pixel na (X,Y) ve vysce kalibrace
  homo status            vypise pocet bodu, offset, chybu a ulozeni
  homo load|undo|clear   obnov / odeber posledni / vymaz vzorky

Na TCP musi byt pevne pridelany viditelny terc. Polohy v prostoru
zachycuji STRED terce, ne automaticky TCP, pokud je terc posunuty.
Robotem hybej POUZE oddelenymi schvalenymi prikazy; tato extension nehybe.
Kalibruj alespon 8-20 poloh po plose, se stejnou vyskou Z.
POZOR: pro skutecnou kalibraci potrebujes REAL robot + REAL kameru.""")
            return
        cmd = argv[0].lower()
        if cmd in ("capture", "auto") and len(argv) == 1:
            # Do not silently generate physically false results from replay + virtual FK.
            if self.ctx.robot.mode != "real" or self.ctx.camera.mode != "real":
                raise RuntimeError("Capture vyzaduje REAL robot + REAL kameru (mode real home|nohome).")
            if self.ctx.robot.released:
                raise RuntimeError("Robot is released, feedback invalid")
            self.session.capture(_CaptureBridge(self.ctx), method="auto" if cmd == "auto" else "click")
        elif cmd == "offset" and len(argv) == 4:
            if self.session.samples:
                raise RuntimeError("Pred zmenou offsetu smaz body: homo clear")
            offset = np.array([float(x) for x in argv[1:]], dtype=float)
            if not np.isfinite(offset).all():
                raise ValueError("Offset musi obsahovat konecna cisla")
            self.session.target_offset_tcp_m = tuple(float(x) for x in offset)
            self.session.is_dirty = True
            self.session.homography = None
            self.session.save_samples()
            print(f"Offset terce od TCP: {self.session.target_offset_tcp_m} m")
        elif cmd == "solve" and len(argv) == 1:
            self.session.solve()
        elif cmd == "map" and len(argv) == 3:
            u, v = float(argv[1]), float(argv[2])
            result = self.session.pixel_to_world(u, v)
            z = self.session.homography.z_m
            print(f"pixel ({u:g}, {v:g}) -> robot X={result[0]:.5f} m, Y={result[1]:.5f} m at Z={z:.5f} m")
        elif cmd == "load" and len(argv) == 1:
            print(f"Loaded {self.session.load_samples()} samples")
        elif cmd == "undo" and len(argv) == 1:
            self.session.undo()
            print(self.session.status())
        elif cmd == "clear" and len(argv) == 1:
            self.session.clear()
            print(self.session.status())
        elif cmd == "status" and len(argv) == 1:
            print(self.session.status())
            print(f"Target offset in TCP: {self.session.target_offset_tcp_m} m")
            print(f"File: {self.session.folder / 'homography.npz'}")
        else:
            raise ValueError("Try: homo help | capture | auto | offset X Y Z | solve | map U V | status | load | undo | clear")


def setup(ctx):
    return HomographyExtension(ctx)
