"""3D hoop-grid scan with optional camera captures and nonblocking ellipse detection.

Place in rob_crs93_v4/extensions/; its Homography class lives in
Vojtech/Homografy/homography.py (sibling of rob_crs93_v4).

Commands (via main.py's homo3d alias):
    homo3d plan [XMIN XMAX YMIN YMAX [NX NY]]
    homo3d run [SECONDS_PER_MOVE]    VIRTUAL: slow animated travel
    homo3d status
    homo3d offset X Y Z      configure RobotManager offset explicitly

Each reached position saves a raw camera frame plus a debug overlay and an
XYZ<->UV record when a camera is available. Detection and image-save failures
are reported but never mask/override robot-motion safety errors. No projection
matrix is calculated during the scan.
For each point, wrist yaw is FREE: the hoop stays horizontal while the planner
searches for a reachable orientation. The sixth joint is bounded to +/-120 deg
from joint zero, also on interpolated paths. All other joint, path, and tabletop
checks remain enabled. Software geometry is not certified collision avoidance.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import sys
import time
import numpy as np


# Verified level pose from the previous virtual test. Its yaw is no longer fixed:
# all target poses are Rz(yaw) @ LEVEL_REFERENCE_ROTATION. Rotation around the
# robot BASE Z keeps the tool plane horizontal while allowing its heading to vary.
# The actual hoop mounting must have the same horizontal-plane convention.
LEVEL_REFERENCE_ROTATION = np.diag([1.0, -1.0, -1.0])
YAW_STEP_DEG = 15  # samples 0, +/-15, +/-30, ... up to 180 degrees
LAST_JOINT_ABS_LIMIT_DEG = 120.0  # q5 in [-120,+120] deg, measured from joint zero
LAST_JOINT_ABS_LIMIT_RAD = float(np.deg2rad(LAST_JOINT_ABS_LIMIT_DEG))


def _rotation_about_vertical(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_candidates(reference_rad: float) -> list[float]:
    """Try previous yaw first; then gradually larger turns, modulo 360 deg."""
    shifts = [0]
    for d in range(YAW_STEP_DEG, 180, YAW_STEP_DEG):
        shifts.extend([d, -d])
    shifts.append(180)
    seen: set[float] = set()
    choices = []
    for delta in shifts:
        angle = float(np.arctan2(np.sin(reference_rad + np.deg2rad(delta)),
                                 np.cos(reference_rad + np.deg2rad(delta))))
        key = round(angle % (2*np.pi), 8)
        if key not in seen:
            seen.add(key)
            choices.append(angle)
    return choices


# Smooth VIRTUAL-only playback. Real hardware speed is untouched.
VIRTUAL_SECONDS_PER_MOVE = 2.0
VIRTUAL_SECONDS_AT_POINT = 0.45
VIRTUAL_FPS = 25.0
HEIGHTS_ABOVE_TABLE_M = (0.10, 0.15, 0.20)
MAX_JOINT_INCREMENT_RAD = float(np.deg2rad(2.0))
# Prevent switching between very different IK branches between scan positions.
# Does not replace mechanical limits, table clearance or a real collision planner.
MAX_INTERPOINT_JOINT_STEP_DEG = 90.0
GOOD_ENOUGH_NORM_DEG = 28.0  # Early exit for a genuinely small motion.
YAW_CHANGE_COST = 0.12       # Equivalent degrees of joints per yaw degree.

MAX_POSITION_ERROR_M = 0.004
MAX_ORIENTATION_FROBENIUS = 0.08


@dataclass(frozen=True)
class ScanTarget:
    index: int
    hoop_xyz: np.ndarray
    flange_pose: np.ndarray
    q: np.ndarray
    yaw_rad: float


def _load_homography_class(root: Path):
    path = root.parent / "Homografy" / "homography.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Nenalezen modul: {path}. Ocekavam slozku Homografy vedle rob_crs93_v4."
        )
    module_name = "_crs93_hoop3d_homography"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Nelze nacist {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return sys.modules[module_name].Homography


class Hoop3DMotionExtension:
    def __init__(self, ctx):
        self.ctx = ctx
        self._targets: list[ScanTarget] = []
        self._errors: list[tuple[int, str]] = []
        self._plan_start_q: np.ndarray | None = None
        self._plan_offset: np.ndarray | None = None
        self._plan_table_z: float | None = None
        self._plan_mode: str | None = None
        self._plan_limits: np.ndarray | None = None
        self._last_run: list[dict] = []
        self._report_dir = Path(ctx.root) / "data" / "homography3d_motion"
        self._report_warnings: set[str] = set()
        self._last_log_path: Path | None = None
        self._last_capture_dir: Path | None = None
        self._last_capture_summary: dict | None = None

    def run(self, argv: list[str]):
        if not argv or argv == ["help"]:
            print("""3D TEST: 12 bodu x 3 vysky (10, 15, 20 cm nad stolem).
  Ocko zustava vodorovne, smer (yaw) se meni podle dosazitelne IK.
  Posledni kloub q5: maximalne -120 az +120 stupnu od nuly.
  Bez prepnutI IK vetve: mezi body max 90 deg na libovolnem kloubu.
  robot table Z              nejdrive nastav zmerenou vysku stolu v BASE [m]
  homo3d offset X Y Z        nastav offset pro tento test (5-25 cm od priruby)
  homo offset X Y Z          prikaz puvodni kalibrace muze pouzivat vlastni stav
  homo3d plan                otestuje IK/limity/cestu pro 36 bodu BEZ pohybu
  homo3d plan X0 X1 Y0 Y1 [NX NY]
                             vlastni XY rozsah stREDU OCKA [m], vychozi NX=4, NY=3
  homo3d status              vypis vysledek planu/testu
  homo3d run [SEKUND]        projede body, kazdy bod vyfoti a zkusi detekovat ocko; ve VIRTUAL modu
                             plynuly pohyb (vychozi 2 s/bod, 0.45 s stani)
                             napr. homo3d run 4 = pomalejsi virtualni pohyb
                             REAL: rychlost motoru se timto NEMENI
Camera: REAL musi mit otevrenou Basler kameru; VIRTUAL podporuje replay.
Detekce ocka je nezavisla na pohybu a nikdy jej nezastavi.
Projekcni matici zatim automaticky NEPOCITAME.
Bezpecnostni kontroly IK, q-limit, geometrie stolu zustavaji povinne.
Pro REAL test zustava nutne povoleni motion.real_motion_enabled.
""")
            return
        sub = argv[0].lower()
        if sub == "plan":
            if len(argv) == 1:
                self.plan(0.28, 0.46, -0.15, 0.15, 4, 3)
            elif len(argv) in (5, 7):
                x0, x1, y0, y1 = map(float, argv[1:5])
                nx, ny = (4, 3) if len(argv) == 5 else (int(argv[5]), int(argv[6]))
                self.plan(x0, x1, y0, y1, nx, ny)
            else:
                raise ValueError("homo3d plan [XMIN XMAX YMIN YMAX [NX NY]]")
        elif sub == "offset" and len(argv) == 4:
            offset = np.array([float(v) for v in argv[1:]], dtype=float)
            if not np.isfinite(offset).all():
                raise ValueError("Offset musi byt konecna cisla v metrech")
            length = float(np.linalg.norm(offset))
            if not 0.05 < length < 0.25:
                raise ValueError(
                    f"Offset ma delku {length:.3f} m; pro kontrolu geometrie musi byt "
                    "5-25 cm od priruby. Pro skutecne ocko jsme pouzivali "
                    "homo3d offset 0.135 0 -0.03. Nezadavej odhad jen kvuli IK."
                )
            self.ctx.robot.set_hoop_offset(offset)
            self._targets = []
            self._errors = []
            self._plan_start_q = None
            self._plan_offset = None
            print(f"HOMO3D offset nastaven: {tuple(offset)} m (delka {length:.3f} m).")
            print("Pred dalsim run je nutne znovu provest homo3d plan.")
        elif sub == "status" and len(argv) == 1:
            self.status()
        elif sub == "run" and len(argv) in (1, 2):
            seconds = VIRTUAL_SECONDS_PER_MOVE if len(argv) == 1 else float(argv[1])
            if not np.isfinite(seconds) or not (0.2 <= seconds <= 30.0):
                raise ValueError("Doba virtualniho pohybu musi byt 0.2 az 30 s")
            self.execute(virtual_seconds_per_move=seconds, custom_timing=len(argv) == 2)
        else:
            raise ValueError("homo3d help | offset X Y Z | plan [X0 X1 Y0 Y1 [NX NY]] | status | run [SEKUND]")

    def _require(self):
        robot = self.ctx.robot
        if robot.mode not in ("virtual", "real"):
            raise RuntimeError("Nejdrive pripoj robota: mode virtual nebo mode real ...")
        if robot.released:
            raise RuntimeError("Robot je uvolneny, zpetna vazba neni spolehliva")
        if robot.mode == "real" and not robot.real_motion_enabled:
            raise RuntimeError("REAL pohyb je vypnuty v motion.real_motion_enabled")
        table_guard = getattr(robot, "_table_guard", None)
        offset = getattr(robot, "_hoop_offset_tcp", None)
        if table_guard is None or offset is None:
            missing = []
            if table_guard is None:
                missing.append("stul: 'robot table Z'")
            if offset is None:
                missing.append("offset v RobotManager: 'homo3d offset X Y Z'")
            raise RuntimeError(
                "HOMO3D nema nastaveno: " + ", ".join(missing) + ". "
                "Prikaz 'homo offset' muze nastavovat pouze puvodni kalibracni "
                "modul. Pro kontrolu modelu je vyzadovana delka offsetu 5-25 cm."
            )
        table_z = float(table_guard.table_z_m)
        offset = np.asarray(offset, dtype=float)
        if (offset.shape != (3,) or not np.isfinite(offset).all()
                or not 0.05 < np.linalg.norm(offset) < 0.25):
            raise RuntimeError(
                f"Offset ocka ma delku {np.linalg.norm(offset):.3f} m, "
                "musi byt 5-25 cm. Nastav ho pomoci 'homo3d offset X Y Z'."
            )
        if not np.isfinite(table_z):
            raise RuntimeError("Neplatna vyska stolu")
        robot.assert_stationary()
        q = np.asarray(robot.get_q(), dtype=float).reshape(-1)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError("Neplatna aktualni poloha kloubu")
        _, effective_limits = robot.get_joint_limits_deg()
        if effective_limits is None:
            raise RuntimeError("Robot nema dostupne efektivni limity kloubu")
        limits = np.deg2rad(np.asarray(effective_limits, dtype=float))
        if limits.shape != (6, 2) or not np.isfinite(limits).all():
            raise RuntimeError("Neplatne q limity")
        # HARD USER-REQUESTED LIMIT ON THE SIXTH JOINT (zero-referenced angle).
        # Never widen mechanical/driver limits: intersect them with +/-120 deg.
        limits[5, 0] = max(limits[5, 0], -LAST_JOINT_ABS_LIMIT_RAD)
        limits[5, 1] = min(limits[5, 1], LAST_JOINT_ABS_LIMIT_RAD)
        if not np.all(limits[:, 0] < limits[:, 1]):
            raise RuntimeError("Posledni kloub nema pouzitelny rozsah pri +/-120 deg")
        if np.any(q < limits[:, 0] - 1e-8) or np.any(q > limits[:, 1] + 1e-8):
            raise RuntimeError(
                "Vychozi konfigurace je mimo povolene limity (q5 musi byt v +/-120 deg). "
                "Nejdrive vrat robota do bezpecneho vychoziho stavu."
            )
        return q, limits, table_z, offset

    def _pose_error(self, q, expected_xyz, offset, expected_rotation):
        T = np.asarray(self.ctx.robot.require().fk(q), dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            return float("inf"), float("inf")
        measured = (T @ np.array([*offset, 1.0]))[:3]
        return (float(np.linalg.norm(measured - expected_xyz)),
                float(np.linalg.norm(T[:3, :3] - expected_rotation)))

    def _path_clear(self, start, goal, limits):
        """Mimic the robot manager's 2-degree sampled joint-space table guard.

        Does NOT replace RobotManager.preview_move() / move_to_q() checks.
        """
        delta = goal - start
        steps = max(1, int(np.ceil(np.max(np.abs(delta)) / MAX_JOINT_INCREMENT_RAD)))
        for k in range(steps + 1):
            q = start + delta * (k / steps)
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                raise ValueError("Mezikrok lezi mimo kloubove limity")
            self.ctx.robot.check_table_clearance(q)

    def plan(self, xmin, xmax, ymin, ymax, nx=4, ny=3):
        self._targets = []
        self._errors = []
        self._plan_start_q = None
        self._plan_offset = None
        self._plan_table_z = None
        self._plan_mode = None
        self._plan_limits = None
        # No motors move during planning.
        q0, limits, table_z, offset = self._require()
        numbers = np.array([xmin, xmax, ymin, ymax], dtype=float)
        if not np.isfinite(numbers).all() or not xmin < xmax or not ymin < ymax:
            raise ValueError("X/Y rozsahy musi byt konecne a rostouci")
        if not (2 <= nx <= 8 and 2 <= ny <= 8):
            raise ValueError("NX/NY musi byt 2..8")
        robot = self.ctx.robot
        # Explicit preflight for current configuration and permission to move.
        robot.preview_move(q0)
        robot.check_table_clearance(q0)
        Homography = _load_homography_class(Path(self.ctx.root))
        h = Homography(
            x_range_m=(xmin, xmax), y_range_m=(ymin, ymax),
            z_levels_m=HEIGHTS_ABOVE_TABLE_M, table_z_m=table_z,
            nx=nx, ny=ny, hoop_center_in_tcp_m=tuple(offset),
        )
        # Only generate 3D centres here. There is no longer ONE global fixed
        # rotation: each target gets its own horizontal yaw chosen by IK.
        points = h.points_m
        last_q = q0.copy()
        last_yaw = 0.0
        for idx, p in enumerate(points, start=1):
            try:
                q_goal = None
                chosen_pose = None
                chosen_yaw = None
                reasons = []
                solution_count = 0
                # Search across yaw AND all IK branches. The former implementation
                # accepted the first reachable yaw, which could select a 180/330
                # degree joint jump although another yaw was much smoother.
                ranked: list[tuple[float, float, np.ndarray, np.ndarray, float]] = []
                previous_yaw = float(last_yaw)
                for yaw in _yaw_candidates(previous_yaw):
                    rotation = _rotation_about_vertical(yaw) @ LEVEL_REFERENCE_ROTATION
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    # The hoop centre remains fixed as the flange rotates.
                    pose[:3, 3] = p - rotation @ offset
                    try:
                        solutions = robot.ik(pose)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        reasons.append(str(exc))
                        continue
                    for candidate in ([] if solutions is None else solutions):
                        q = np.asarray(candidate, dtype=float).reshape(-1)
                        if q.shape != (6,) or not np.isfinite(q).all():
                            continue
                        if np.any(q < limits[:, 0] - 1e-8) or np.any(q > limits[:, 1] + 1e-8):
                            continue
                        err_m, err_rot = self._pose_error(q, p, offset, rotation)
                        if err_m > MAX_POSITION_ERROR_M or err_rot > MAX_ORIENTATION_FROBENIUS:
                            continue
                        solution_count += 1
                        joint_step_deg = np.abs(np.rad2deg(q - last_q))
                        largest_step = float(joint_step_deg.max())
                        # The first move from home is setup; other scan moves
                        # must not switch to an opposite elbow/wrist/waist branch.
                        if idx > 1 and largest_step > MAX_INTERPOINT_JOINT_STEP_DEG:
                            continue
                        yaw_delta = float(abs(np.rad2deg(np.arctan2(
                            np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw)))))
                        travel = float(np.linalg.norm(joint_step_deg))
                        score = travel + YAW_CHANGE_COST * yaw_delta
                        ranked.append((score, largest_step, q.copy(), pose.copy(), float(yaw)))
                    # At the same/nearest yaw, a genuinely tiny movement is
                    # already good; no need to solve every 15-degree candidate.
                    if ranked:
                        score_best, max_best, *_ = min(ranked, key=lambda x: x[0])
                        if (score_best <= GOOD_ENOUGH_NORM_DEG
                                and max_best <= GOOD_ENOUGH_NORM_DEG):
                            best_entry = min(ranked, key=lambda x: x[0])
                            try:
                                self._path_clear(last_q, best_entry[2], limits)
                            except (ValueError, RuntimeError) as exc:
                                reasons.append(str(exc))
                            else:
                                break

                # Only now check swept table clearance, starting with the
                # lowest global cost across yaw and IK branches.
                for _, _, q, pose, yaw in sorted(ranked, key=lambda x: x[0]):
                    try:
                        self._path_clear(last_q, q, limits)
                    except (ValueError, RuntimeError) as exc:
                        reasons.append(str(exc))
                        continue
                    q_goal, chosen_pose, chosen_yaw = q, pose, yaw
                    break
                if q_goal is None:
                    detail = '; '.join(dict.fromkeys(reasons))[:180]
                    raise ValueError(
                        'Zadne plynule IK/yaw reseni s limity vcetne q5 +/-120 deg, '
                        f'max {MAX_INTERPOINT_JOINT_STEP_DEG:.0f} deg/kloub mezi body '
                        f'a bezpecnou cestou (IK kandidatu={solution_count}). {detail}'
                    )
                largest_step_deg = float(np.rad2deg(np.max(np.abs(q_goal - last_q))))
                self._targets.append(ScanTarget(idx, p.copy(), chosen_pose.copy(),
                                                q_goal.copy(), chosen_yaw))
                last_q = q_goal.copy()
                last_yaw = chosen_yaw
                print(f'  OK  {idx:02d}/{len(points)}  '
                      f'X={p[0]:.3f}, Y={p[1]:.3f}, Z={p[2]:.3f} m, '
                      f'yaw={np.rad2deg(chosen_yaw):+.0f} deg, '
                      f'q5={np.rad2deg(q_goal[5]):+.1f} deg, '
                      f'max dq={largest_step_deg:.1f} deg')
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                self._errors.append((idx, str(exc)))
                print(f'  NELZE {idx:02d}/{len(points)}  XYZ={np.round(p, 3)}: {exc}')
        self._plan_start_q = q0.copy()
        self._plan_offset = offset.copy()
        self._plan_table_z = table_z
        self._plan_mode = robot.mode
        self._plan_limits = limits.copy()
        count = len(points)
        print(f"\nPLAN: {len(self._targets)}/{count} dosazitelnych, {len(self._errors)} problemu.")
        if self._errors:
            print("RUN ZABLOKOVAN: uprav X/Y rozsah nebo referencni stav a opakuj homo3d plan.")
        else:
            print("Vsechny body prosly IK, limity i geometrickou kontrolou cest.")
            print("Pro pohyb pouzij 'homo3d run'.")
        self._write_report("last_plan.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": robot.mode,
            "table_z_m": table_z,
            "hoop_offset_tcp_m": offset.tolist(),
            "orientation": "level plane, variable yaw around base Z",
            "level_reference_rotation": LEVEL_REFERENCE_ROTATION.tolist(),
            "yaw_step_deg": YAW_STEP_DEG,
            "sixth_joint_limit_deg": [-LAST_JOINT_ABS_LIMIT_DEG, LAST_JOINT_ABS_LIMIT_DEG],
            "max_interpoint_joint_step_deg": MAX_INTERPOINT_JOINT_STEP_DEG,
            "points": [dict(index=i, xyz_m=p.tolist()) for i, p in enumerate(points, start=1)],
            "feasible": [dict(index=t.index, yaw_deg=float(np.rad2deg(t.yaw_rad)),
                              q5_deg=float(np.rad2deg(t.q[5]))) for t in self._targets],
            "failures": [dict(index=i, reason=why) for i, why in self._errors],
        })

    def _warn_report(self, key: str, message: str) -> None:
        """Logging must not accidentally stop the calibration motion test."""
        if key not in self._report_warnings:
            print(f"  [report warning] {message}")
            self._report_warnings.add(key)

    def _write_report(self, filename: str, data: dict) -> Path | None:
        """Best-effort JSON report; locked Windows destinations use a new name.

        Windows can reject os.replace() when another process has last_run.json
        open, or when the destination is read-only. The motion outcome must
        never be confused with a failure to write an optional report.
        """
        temporary = None
        try:
            self._report_dir.mkdir(parents=True, exist_ok=True)
            destination = self._report_dir / filename
            # Unique temporary file also avoids collisions with an orphan .tmp.
            temporary = self._report_dir / f".{destination.stem}_{uuid4().hex}.tmp"
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                temporary.replace(destination)
                return destination
            except PermissionError as exc:
                # Do not attempt to force-delete a locked file or alter ACLs.
                alternative = self._report_dir / (
                    f"{destination.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}.json"
                )
                temporary.replace(alternative)
                self._warn_report(filename, f"{destination.name} nelze prepsat ({exc}); ulozeno do {alternative.name}")
                return alternative
        except OSError as exc:
            self._warn_report(filename, f"Report {filename} nelze ulozit: {exc}. Test muze pokracovat.")
            return None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _append_run_event(self, entry: dict) -> None:
        """Append-only log, one entry per completed/failed point.

        Unlike repeatedly replacing last_run.json, append does not rename or
        overwrite an existing report that might be open in an editor.
        """
        if self._last_log_path is None:
            return
        try:
            self._report_dir.mkdir(parents=True, exist_ok=True)
            with self._last_log_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._warn_report("events", f"Prubezny log nelze zapsat: {exc}. Test pokracuje.")

    @staticmethod
    def _save_png_unicode_safe(path: Path, frame: np.ndarray) -> None:
        """Encode then write bytes: cv2.imwrite may fail on Windows Unicode paths."""
        import cv2
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
            raise ValueError("Camera returned an invalid/empty image")
        success, encoded = cv2.imencode(".png", frame)
        if not success:
            raise OSError("PNG encoding failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())

    @staticmethod
    def _debug_image(frame: np.ndarray, detection) -> np.ndarray:
        """Draw fitted ellipse and cross, or warning text, on a copy of frame."""
        import cv2
        if frame.ndim == 2:
            debug = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            debug = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            debug = frame.copy()
        if debug.dtype != np.uint8:
            # Only the overlay needs to be 8-bit; the RAW image is unmodified.
            debug = cv2.convertScaleAbs(debug, alpha=255.0 / max(255.0, float(np.max(debug))))
        if detection is not None:
            cv2.ellipse(debug, (detection.center_px, detection.axes_px,
                               detection.angle_deg), (0, 255, 0), 3)
            u, v = detection.center_px
            cx, cy = int(round(u)), int(round(v))
            cv2.drawMarker(debug, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 32, 2)
            cv2.putText(debug, f"HOOP ({u:.1f}, {v:.1f})", (max(10, cx - 150),
                         max(25, cy - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                         (0, 0, 255), 2)
        else:
            cv2.putText(debug, "HOOP NOT DETECTED", (24, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return debug

    def _capture_at_target(self, *, entry: dict, capture_dir: Path,
                           detector, offset: np.ndarray) -> None:
        """Optional IO stage AFTER the robot has safely stopped.

        Camera/detection/write errors are recorded, not raised. Genuine motion
        errors are handled by execute and still abort the remaining moves.
        """
        idx = int(entry["point_index"])
        entry["captured_utc"] = datetime.now(timezone.utc).isoformat()
        entry["capture_status"] = "not_started"
        entry["detection_status"] = "not_attempted"
        entry["pixel_uv"] = None

        # Actual 3D hoop centre, NOT commanded position and NOT flange centre.
        try:
            q = np.asarray(entry["actual_q_rad"], dtype=float)
            T = np.asarray(self.ctx.robot.require().fk(q), dtype=float).reshape(4, 4)
            actual_hoop = (T @ np.r_[offset, 1.0])[:3]
            entry["actual_hoop_xyz_m"] = actual_hoop.tolist()
        except Exception as exc:
            entry["actual_hoop_xyz_error"] = f"{type(exc).__name__}: {exc}"

        cam_mode = getattr(self.ctx.camera, "mode", None)
        if cam_mode not in ("real", "replay"):
            entry["capture_status"] = "camera_unavailable"
            print("  [foto] Kamera neni otevrena (ve VIRTUAL modu pouzij camera replay DIR).")
            return
        frame = None
        try:
            frame = self.ctx.camera.capture()
            if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
                raise ValueError("Camera returned empty/invalid image")
            entry["image_size_px"] = [int(frame.shape[1]), int(frame.shape[0])]
        except Exception as exc:
            entry["capture_status"] = "capture_failed"
            entry["capture_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [foto warning] Kamera selhala: {exc}. Pokracujeme.")
            return

        raw_file = capture_dir / f"point_{idx:03d}.png"
        try:
            self._save_png_unicode_safe(raw_file, frame)
            entry["image"] = raw_file.name
            entry["capture_status"] = "saved"
        except Exception as exc:
            entry["capture_status"] = "save_failed"
            entry["capture_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [foto warning] Nejde ulozit {raw_file.name}: {exc}. Pokracujeme.")

        detection = None
        try:
            uv = detector.capture_circle(frame=frame)
            detection = detector.last_detection
            entry["pixel_uv"] = [float(uv[0]), float(uv[1])]
            entry["detection_status"] = "detected"
            if detection is not None:
                entry["circle_axes_px"] = list(map(float, detection.axes_px))
                entry["circle_angle_deg"] = float(detection.angle_deg)
                entry["circle_score"] = float(detection.score)
            print(f"  [ocko] nalezeno ({uv[0]:.1f}, {uv[1]:.1f}) px")
        except Exception as exc:
            entry["detection_status"] = "not_found"
            entry["detection_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [ocko warning] Detekce nevysla: {exc}. Pokracujeme.")
        try:
            debug_file = capture_dir / f"point_{idx:03d}_debug.png"
            self._save_png_unicode_safe(debug_file, self._debug_image(frame, detection))
            entry["debug_image"] = debug_file.name
        except Exception as exc:
            entry["debug_save_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [debug warning] Nelze ulozit debug snimek: {exc}")

        if entry["capture_status"] == "saved":
            print(f"  [foto] {raw_file.name}")

    def _save_capture_summary(self, directory: Path, run_id: str) -> None:
        """Always attempt writing a human-readable capture manifest."""
        images = sum(e.get("capture_status") == "saved" for e in self._last_run)
        found = sum(e.get("detection_status") == "detected" for e in self._last_run)
        summary = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "camera_mode": getattr(self.ctx.camera, "mode", None),
            "coordinate_system": "robot BASE, metres; pixels from original frames",
            "samples": self._last_run,
            "captured_images": images,
            "detected_circles": found,
        }
        self._last_capture_summary = summary
        try:
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / "measurements.json"
            destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except OSError as exc:
            self._warn_report("capture_manifest", f"Nepodarilo se ulozit measurements.json: {exc}")
        print(f"FOTKY: {images}/{len(self._last_run)}, DETEKCE: {found}/{len(self._last_run)}")
        print(f"Adresar obrazku: {directory}")

    def status(self):
        offset = getattr(self.ctx.robot, "_hoop_offset_tcp", None)
        guard = getattr(self.ctx.robot, "_table_guard", None)
        print(f"RobotManager: table_Z={getattr(guard, 'table_z_m', None)} m, "
              f"hoop_offset={offset} m")
        print(f"Plan: {len(self._targets)} platnych bodu; {len(self._errors)} chyb")
        if self._plan_mode:
            print(f"  mode={self._plan_mode}, table_Z={self._plan_table_z:.3f} m")
        if self._last_run:
            complete = sum(s["status"] == "reached" for s in self._last_run)
            print(f"Posledni prujezd: {complete}/{len(self._last_run)} potvrzenych pozic")
        print(f"Reporty: {self._report_dir}")
        if self._last_log_path is not None:
            print(f"Prubezny log: {self._last_log_path}")
        if self._last_capture_dir is not None:
            print(f"Posledni fotky: {self._last_capture_dir}")

    def _animate_virtual_move(self, start_q: np.ndarray, goal_q: np.ndarray, seconds: float):
        """Interpolate ONLY offline joint state so pview can poll and display it.

        The full start->goal joint-space segment was already checked by plan().
        This is not a real robot trajectory generator or collision simulator.
        """
        steps = max(2, int(np.ceil(seconds * VIRTUAL_FPS)))
        interval = seconds / steps
        robot = self.ctx.robot
        for k in range(1, steps + 1):
            alpha = k / steps
            intermediate = start_q + alpha * (goal_q - start_q)
            robot.move_to_q(intermediate)
            # Let the viewer's polling/websocket task see intermediate frames.
            time.sleep(interval)

    def execute(self, *, virtual_seconds_per_move=VIRTUAL_SECONDS_PER_MOVE,
                custom_timing=False):
        if self._plan_start_q is None or not self._targets or self._errors:
            raise RuntimeError("Nejdrive priprav bezchybny plan: homo3d plan")
        robot = self.ctx.robot
        current_q, limits, table_z, offset = self._require()
        if (robot.mode != self._plan_mode
                or abs(table_z - self._plan_table_z) > 1e-6
                or not np.allclose(offset, self._plan_offset, atol=1e-6)
                or np.max(np.abs(current_q - self._plan_start_q)) > np.deg2rad(0.25)):
            raise RuntimeError("Zmenil se stav robota/stolu/offsetu; zopakuj homo3d plan")
        if robot.mode == "real":
            if custom_timing:
                raise RuntimeError(
                    "homo3d run SEKUND funguje pouze ve VIRTUAL modu; "
                    "rychlost realnych motoru zde neovladame. Pouzij 'homo3d run'."
                )
            print("POZOR: 3D scan fyzicky POHYBUJE robotem pres vsechny body.")
            print("Pred spustenim fyzicky over volny prostor a dosah E-STOP.")
            if input("Pro spusteni napis presne RUN36: ").strip() != "RUN36":
                print("Zruseno, robot se nepohnul.")
                return
        if robot.mode == "real":
            if getattr(self.ctx.camera, "mode", None) != "real":
                raise RuntimeError(
                    "REAL kalibracni prujezd vyzaduje otevrenou Basler kameru. "
                    "Pouzij 'camera open' nebo 'mode real ...', potom znovu plan/run."
                )
            # Do a camera preflight BEFORE moving a single joint.
            try:
                preflight = self.ctx.camera.capture()
                if not isinstance(preflight, np.ndarray) or preflight.size == 0:
                    raise ValueError("Prazdny obrazek")
            except Exception as exc:
                raise RuntimeError(
                    f"Kamera nedodala zkusebni snimek: {exc}. Robot se nepohnul."
                ) from exc
        if robot.mode == "virtual":
            print(f"VIRTUAL animace: {virtual_seconds_per_move:g} s na presun, "
                  f"{VIRTUAL_SECONDS_AT_POINT:g} s na zastaveni v kazdem bode, "
                  f"{VIRTUAL_FPS:g} FPS. Pro zpomaleni: homo3d run 4")
        self._last_run = []
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        self._last_log_path = self._report_dir / f"run_{run_id}.jsonl"
        self._last_capture_dir = Path(self.ctx.root) / "data" / "homography3d" / f"capture_{run_id}"
        # The detector is used only for vision; actual motion validation is unchanged.
        Homography = _load_homography_class(Path(self.ctx.root))
        detector = Homography(max_circle_diameter_px=1500, table_z_m=table_z,
                              hoop_center_in_tcp_m=tuple(offset))
        print(f"Fotky + debug + mereni: {self._last_capture_dir}")
        # Journal progress once per point; do NOT overwrite last_run.json dozens
        # of times mid-motion (that is fragile on Windows when the file is open).
        try:
            for order, target in enumerate(self._targets, start=1):
                entry = {"point_index": target.index,
                         "yaw_deg": float(np.rad2deg(target.yaw_rad)),
                         "target_xyz_m": target.hoop_xyz.tolist(),
                         "target_q_rad": target.q.tolist(),
                         "status": "pending"}
                self._last_run.append(entry)
                print(f"[{order:02d}/{len(self._targets)}] -> "
                      f"ocko XYZ={np.array2string(target.hoop_xyz, precision=3)}")
                try:
                    # Mandatory additional q5 +/-120 deg and swept path checks,
                    # also rechecked immediately before every real motion.
                    start_joints = np.asarray(robot.get_q(), dtype=float)
                    if order > 1:
                        max_step_deg = float(np.max(np.abs(np.rad2deg(target.q - start_joints))))
                        if max_step_deg > MAX_INTERPOINT_JOINT_STEP_DEG + 1e-7:
                            raise RuntimeError(
                                f"Necekany skok IK mezi body: {max_step_deg:.1f} deg; "
                                "pohyb odmitnut")
                    self._path_clear(start_joints, target.q, limits)
                    current, goal = robot.preview_move(target.q)
                    if robot.mode == "virtual":
                        self._animate_virtual_move(current, goal, virtual_seconds_per_move)
                        time.sleep(VIRTUAL_SECONDS_AT_POINT)
                    else:
                        # One guarded hardware move; speed configuration not changed.
                        robot.move_to_q(goal, expected_q=current)
                    robot.assert_stationary()
                    q_actual = np.asarray(robot.get_q(), dtype=float)
                    err_pos, err_rot = self._pose_error(
                        q_actual, target.hoop_xyz, offset, target.flange_pose[:3, :3]
                    )
                    entry["actual_q_rad"] = q_actual.tolist()
                    entry["error_mm"] = 1000.0 * err_pos
                    entry["status"] = "reached" if err_pos <= MAX_POSITION_ERROR_M and err_rot <= MAX_ORIENTATION_FROBENIUS else "pose_mismatch"
                    print(f"  {entry['status']}  chyba FK: {1000*err_pos:.2f} mm")
                    if entry["status"] != "reached":
                        raise RuntimeError("Robot nedosahl ocekavanou pozu, dalsi body se neprovedou")
                    # Capture is best-effort; no image/detection exception may
                    # interrupt motion. Only the robot's motion safety can stop it.
                    try:
                        self._capture_at_target(entry=entry,
                                                capture_dir=self._last_capture_dir,
                                                detector=detector, offset=offset)
                    except Exception as exc:
                        entry["vision_unexpected_error"] = f"{type(exc).__name__}: {exc}"
                        print(f"  [foto warning] Neocekavana chyba snimani: {exc}. Pokracujeme.")
                except BaseException as exc:
                    if entry["status"] == "pending":
                        entry["status"] = "failed"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    self._append_run_event(entry)
        finally:
            if self._last_capture_dir is not None:
                self._save_capture_summary(self._last_capture_dir, run_id)
            summary = {
                "started_from_mode": self._plan_mode,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "points": self._last_run,
                "capture_directory": (str(self._last_capture_dir)
                                      if self._last_capture_dir is not None else None),
            }
            # The uniquely named summary survives even if last_run.json is
            # locked in another application. Both writes are best effort.
            archive = self._write_report(f"run_{run_id}.json", summary)
            latest = self._write_report("last_run.json", summary)
            if archive is not None:
                print(f"Report prujezdu: {archive}")
            if latest is not None and latest != self._report_dir / "last_run.json":
                print(f"Nahradni last_run report: {latest}")
        print(f"HOTOVO: {len(self._targets)} pozic projetych v modu {robot.mode}.")
        print(f"Prubezny log: {self._last_log_path}")


def setup(ctx):
    """ExtensionManager entry point. Does not move or connect hardware."""
    return Hoop3DMotionExtension(ctx)
