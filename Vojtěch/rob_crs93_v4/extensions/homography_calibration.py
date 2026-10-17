"""Pixel -> base XY calibration using fixed-plane homography for CRS93.

No motion/homing commands. Uses Context.robot and Context.camera, not a
separate CRS93 or Basler instance. This extension can be loaded while OFFLINE.
"""
from __future__ import annotations

import json
import numpy as np

from hoop_auto_scan import HoopAutoScanner
from hoop_guided_scan import GuidedHoopScanner
from homography_capture import HomographySession
from charuco_homography import CharucoHomographySession


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
        self.board = CharucoHomographySession(folder=ctx.root / "data" / "homography")
        self.scanner = HoopAutoScanner(ctx.robot, ctx.camera, self.session)
        self.guided = GuidedHoopScanner(ctx.robot, ctx.camera, self.session)
        self.board.load()
        # Load samples for status and continuation, not robot/camera hardware.
        if (self.session.folder / "samples.json").is_file():
            saved = json.loads((self.session.folder / "samples.json").read_text(encoding="utf-8"))
            self.session.target_offset_tcp_m = tuple(saved["target_offset_tcp_m"])
            self.session.load_samples()
        if 0.05 < np.linalg.norm(self.session.target_offset_tcp_m) < 0.25:
            self.ctx.robot.set_hoop_offset(self.session.target_offset_tcp_m)

    def run(self, argv: list[str]) -> None:
        if not argv or argv == ["help"]:
            print("""Homografie: jedna rovina Z, pixel (u,v) -> robot (X,Y) [m].
  homo board config SQUARE_MM MARKER_MM     rozmery tiskle ChArUco desky
  homo board capture           nafot a detekuj ChArUco rohy (kamera REAL i replay)
  homo board status            vypis detekovane ID a cestu k obrazku s ID
  homo board teach ID          uloz referencni XYZ z TCP skutecneho stojiciho robota
  homo board anchor ID X Y Z  referencni XYZ zadane rucne [m]
  homo board solve             vypocti H z rohu a alespon 3 referenci
  homo board clear             novy snimek a nove reference
  homo capture           vyber stred terce kliknutim (ENTER potvrdi, ESC zrusi)
  homo auto              pokus o kruh (ENTER potvrdi, klik opravi, ESC zrusi)
  homo offset X Y Z      presny offset stredu hoopu od flange TCP [m]
  homo scan plan XMIN XMAX YMIN YMAX TABLE_Z HEIGHT_CM [NX NY]
                         kontrola 5x5 bodu (bez pohybu), vse v BASE metrech
  homo scan test         vyfoti hoop a otestuje detekci BEZ POHYBU
  homo scan debug        vyhodnoti ulozeny auto_failed_*.png BEZ POHYBU
  homo scan run          puvodni prujezd s BASE mrizkou
  homo scan guided plan TABLE_Z HEIGHT_CM [NX NY]
                         test viditelnosti, vysky a bezpecnosti BEZ POHYBU
  homo scan guided run   adaptivni kalibrace podle obrazu, kontrolovane kroky
  homo scan status       vypis planu
  homo solve             vypocet H, RANSAC, ulozeni do data/homography/homography.npz
  homo map U V           prevede pixel na (X,Y) ve vysce kalibrace
  homo status            vypise pocet bodu, offset, chybu a ulozeni
  homo load|undo|clear   obnov / odeber posledni / vymaz vzorky

ChArUco: nase deska je 10x14 DICT_4X4_100 legacy. Polož JI NAPLOCHO
a NEHYBEJ s ni mezi capture a teach! Pro prevod do robot XYZ jsou potreba
alespon 3 nekolinearni rohy, ktere presne zmeris/ukazes TCP nebo zadas.
Rohy i fyzicke referencni body musi byt ve stejne rovine Z. Puvodni
homo capture/auto alternativa pouziva terc na TCP, 8-20 pozic.
POZOR: homo scan run HYBE REAL ROBOTEM; dodrz 1 cm minimalni vzdalenost
celeho ramene a hoopu od stolu, kterou overuje geometricky model. Pred
prujezdem over fyzicky table Z, offset hoopu, prekazky a E-STOP.""")
            return
        cmd = argv[0].lower()
        if cmd == "scan":
            self._scan_command(argv[1:])
        elif cmd == "board":
            self._board_command(argv[1:])
        elif cmd in ("capture", "auto") and len(argv) == 1:
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
            if hasattr(self.ctx.robot, 'set_hoop_offset'):
                if .05 < np.linalg.norm(offset) < .25:
                    self.ctx.robot.set_hoop_offset(offset)
                elif hasattr(self.ctx.robot, 'clear_hoop_offset'):
                    self.ctx.robot.clear_hoop_offset()
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
            print(self.board.status())
        else:
            raise ValueError("Try: homo help | board ... | capture | auto | offset X Y Z | solve | map U V | status | load | undo | clear")


    def _scan_command(self, argv: list[str]):
        if not argv or argv == ["help"]:
            print("""Hoop auto kalibrace (REAL robot+REAL kamera), bez potvrzovani pohybu:
  homo offset X Y Z               zmereny offset stredu hoopu v TCP [m]
  homo clear                      smaz stare body pred novou kalibraci
  homo scan plan X0 X1 Y0 Y1 TABLE_Z HEIGHT_CM [NX NY]
                                  souradnice BASE CRS93 [m], HEIGHT nad stolem [cm]
  homo scan status                zobrazi plan; zadny pohyb
  homo scan test                  otestuje detekci na aktualnim obrazu, bez pohybu
  homo scan debug                 zpracuje posledni auto_failed_*.png, bez pohybu
  homo scan run                   puvodni scan s pevnou BASE mrizkou
  homo scan guided plan TABLE_Z HEIGHT_CM [NX NY]
                                  overi pocatecni viditelny vodorovny hoop, NEHYBE
  homo scan guided run            snima posuny X/Y, koriguje pohyb podle obrazu

PLAN vyzaduje fyzicky zmerenou vysku stolu v CRS93 BASE a zmereny offset.
Nehybe robotem. RUN posila fyzicke prikazy BEZ interaktivniho potvrzeni,
a vyzaduje provozni dozor, volny prostor a pristupny E-STOP.
Kontrola 1 cm je jen geometricka preflight kontrola a nenahrazuje
bezpecnostni funkce regulatoru/roboticke stanice.""")
            return
        if argv[0] == "guided":
            if len(argv) in (4, 6) and argv[1] == "plan":
                table_z, height_cm = map(float, argv[2:4])
                nx, ny = (5, 5) if len(argv) == 4 else (int(argv[4]), int(argv[5]))
                self.guided.prepare(table_z, height_cm, nx, ny)
                return
            if argv == ["guided", "run"]:
                self.guided.run()
                return
            if argv == ["guided", "status"]:
                setup = self.guided.setup
                if setup is None:
                    print("GUIDED: neni pripraven; homo scan guided plan TABLE_Z HEIGHT_CM [NX NY]")
                else:
                    print(f"GUIDED: obraz={setup.image_size}, start uv={setup.initial_uv}, "
                          f"Z={setup.center_z:.4f} m, cilova mrizka={setup.nx}x{setup.ny}; "
                          "neprovede zadny pohyb dokud nezadas guided run")
                return
            raise ValueError("homo scan guided plan TABLE_Z HEIGHT_CM [NX NY] | guided status | guided run")
        if argv[0] == "plan" and len(argv) in (7, 9):
            pars = [float(x) for x in argv[1:7]]
            nx, ny = (5, 5) if len(argv) == 7 else (int(argv[7]), int(argv[8]))
            plan = self.scanner.make_plan(*pars, nx=nx, ny=ny,
                                          offset_verified=self.ctx.robot._hoop_offset_tcp is not None)
            print(f"Plan: {len(plan.points)} bodu, {len(plan.segments)} kontrolovanych kloubovych mezikroku; "
                  f"hoop Z={plan.points[0].center_xyz[2]:.4f} m")
            return
        if argv == ["status"]:
            plan = self.scanner.plan
            if plan is None:
                print("Plan neexistuje; homo scan plan ...")
            else:
                print(f"Plan: {len(plan.points)} bodu, {len(plan.segments)} kloubovych mezikroku, "
                      f"table Z={plan.table_z:.4f} m; minimalni clearance=0.010 m")
            return
        if argv == ["test"]:
            self.scanner.test_detection()
            return
        if argv == ["debug"]:
            self.scanner.diagnose_saved_failure()
            return
        if argv == ["run"]:
            self.scanner.run()
            return
        raise ValueError("homo scan help | guided plan TABLE_Z HEIGHT_CM [NX NY] | guided run | plan X0 X1 Y0 Y1 TABLE_Z HEIGHT_CM [NX NY] | status | test | debug | run")

    def _invalidate_calibration(self):
        """Any edit to ChArUco observations/anchors makes the saved H unusable."""
        self.session.homography = None
        self.session.is_dirty = True
        self.session.save_samples()

    def _board_command(self, argv: list[str]):
        if not argv or argv == ["help"]:
            print("""ChArUco homografie, 10x14 squares / DICT_4X4_100 / legacy:
  homo board config SQUARE_MM MARKER_MM    zmerit skutecne rozmery (neodhadovat)
  homo board capture                       detekuje a ulozi fotografii a ID rohu
  homo board status                        seznam ID; obrazek charuco_ids.png
  homo board anchor ID X Y Z               souradnice daneho rohu v baze robota [m]
  homo board teach ID                      robota rucne nastav NAD dany roh; sejme
                                           TCP XYZ (nutny REAL stav, bez pohybu)
  homo board solve                         vypocet H (>=3 nekolinearni reference)
  homo board clear                         smaz detekci a reference (rozmery ponecha)

Kamera ani sachovnice se mezi capture a anchor/teach nesmi hybat.
Pro teach musi poloha spicky TCP presne odpovidat rohu v XY a Z musi
odpovidat rovine sachovnice. Opravnene posouvej robota POUZE jinymi prikazy.
Nejdriv vyber 3 od sebe vzdalene rohy z charuco_ids.png.
""")
            return
        cmd = argv[0].lower()
        if cmd == "config" and len(argv) == 3:
            self.board.set_dimensions(float(argv[1]), float(argv[2]))
            self._invalidate_calibration()
            print(self.board.status())
        elif cmd == "capture" and len(argv) == 1:
            if self.ctx.camera.mode not in ("real", "replay"):
                raise RuntimeError("Otevri kameru: camera open | camera replay DIR")
            n = self.board.capture(self.ctx.camera.capture())
            self._invalidate_calibration()
            print(f"Detekovano {n} ChArUco rohu, ulozena fotka s ID: {self.board.preview_path}")
            print("Vyber alespon 3 vzdalené ID rohu a zadej homo board teach/anchor")
        elif cmd == "anchor" and len(argv) == 5:
            self.board.add_anchor(int(argv[1]), tuple(float(v) for v in argv[2:]))
            self._invalidate_calibration()
            print(f"Ulozen roh {int(argv[1])}, celkem referenci {len(self.board.anchors)}")
        elif cmd == "teach" and len(argv) == 2:
            if self.ctx.robot.mode != "real" or self.ctx.robot.released:
                raise RuntimeError("Teach vyzaduje REAL referencovany robot s aktivnim feedbackem")
            target_id = int(argv[1])
            if target_id not in self.board.ids:
                raise ValueError(f"ID {target_id} neni na desce; viz {self.board.preview_path.name}")
            bridge = _CaptureBridge(self.ctx)
            bridge.in_motion()
            q = np.asarray(bridge.get_q(), dtype=float)
            T = np.asarray(bridge.fk(q), dtype=float)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                raise RuntimeError("Neplatna FK pro teach")
            offset = np.asarray([*self.session.target_offset_tcp_m, 1.0])
            xyz = (T @ offset)[:3]
            bridge.in_motion()
            if not np.allclose(bridge.get_q(), q, atol=1e-3):
                raise RuntimeError("Robot se behem teach pohnul; nic neulozeno")
            self.board.add_anchor(target_id, tuple(xyz))
            self._invalidate_calibration()
            print(f"Kotva ChArUco ID {target_id}: robot XYZ={xyz.round(5)} m")
        elif cmd == "solve" and len(argv) == 1:
            calib, rms_m = self.board.solve()
            # Commit only a successfully solved board mapping, and do not mix
            # robot-target samples with ChArUco-based correspondences.
            self.session.samples.clear()
            self.session.image_size = calib.image_size
            self.session.homography = calib
            self.session.is_dirty = False
            self.session.save_samples()
            calib.save(self.session.folder / "homography.npz")
            print(f"ChArUco H ulozena. Pixel RMS {calib.rms_px:.2f} px, "
                  f"RANSAC {calib.inliers}/{calib.total}; "
                  f"referencni XY RMS {rms_m*1000:.2f} mm, Z={calib.z_m:.4f} m")
            print("POZOR: H plati pouze v rovine desky; map mimo detekovanou oblast je blokovan.")
        elif cmd == "clear" and len(argv) == 1:
            self.board.clear_observations()
            self._invalidate_calibration()
            print("ChArUco pozorovani a kotvy smazany; rozmery zachovany")
        elif cmd == "status" and len(argv) == 1:
            print(self.board.status())
            print("Detekovana ID:", self.board.ids)
            print("Kotvy:", self.board.anchors)
        else:
            raise ValueError("homo board help|config SQUARE_MM MARKER_MM|capture|status|anchor ID X Y Z|teach ID|solve|clear")


def setup(ctx):
    return HomographyExtension(ctx)
