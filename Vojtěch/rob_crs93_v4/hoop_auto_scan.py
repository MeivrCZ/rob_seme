"""Single-plane homography sweep for a real CRS93 + Basler camera.

No hardware objects are created here. The caller passes the app's shared managers.
Planning never moves motors. Execution does not ask for move confirmation;
RobotManager still checks the complete sampled arm-and-hoop tabletop clearance.

This is not a collision checker: the operator must validate the complete swept
volume (including all robot links and the entire hoop) on the physical station.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import cv2
import numpy as np

from homography_capture import HomographySession, _bgr_frame, detect_single_circle


@dataclass
class ScanPoint:
    center_xyz: np.ndarray
    joints: np.ndarray


@dataclass
class ScanPlan:
    points: list[ScanPoint]
    segments: list[np.ndarray]
    starting_q: np.ndarray
    offset_tcp: tuple[float, float, float]
    table_z: float
    nx: int
    ny: int
    joint_step_rad: float

    @property
    def total_moves(self):
        return len(self.segments)


class HoopAutoScanner:
    """One-plane sweep using a fixed flange orientation for every target."""

    def __init__(self, robot, camera, session: HomographySession):
        self.robot = robot
        self.camera = camera
        self.session = session
        self.plan: ScanPlan | None = None

    def _require_real(self):
        if self.robot.mode != "real" or self.robot.released:
            raise RuntimeError("Automaticky prujezd vyzaduje REAL robota s platnym feedbackem")
        if self.camera.mode != "real":
            raise RuntimeError("Automaticky prujezd vyzaduje REAL kameru (camera open)")
        if not self.robot.real_motion_enabled:
            raise RuntimeError("Realne pohyby jsou vypnute v motion.real_motion_enabled")
        self.robot.assert_stationary()

    def _fk(self, q):
        T = np.asarray(self.robot.require().fk(np.asarray(q, dtype=float)), dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError("Neplatna FK matice robota")
        return T

    def _center(self, q, offset):
        return (self._fk(q) @ np.array([*offset, 1.0]))[:3]

    def make_plan(self, xmin: float, xmax: float, ymin: float, ymax: float,
                  table_z: float, height_cm: float, nx: int = 5, ny: int = 5,
                  *, offset_verified: bool = False) -> ScanPlan:
        """Prepare a checkable *joint-space* path; never sends motor commands.

        X/Y are the permitted hoop-centre rectangle in CRS93 BASE coordinates.
        z=table_z+height_cm/100 is the hoop-centre height, not flange height.
        """
        self.plan = None
        self._require_real()
        if not offset_verified:
            raise RuntimeError("Nejprve zmer a potvrď offset stredu hoopu: homo offset X Y Z")
        if self.session.samples or (self.session.homography is not None and not self.session.is_dirty):
            raise RuntimeError("Pred novou automatickou kalibraci pouzij 'homo clear'")
        numbers = np.array([xmin, xmax, ymin, ymax, table_z, height_cm], dtype=float)
        if not np.isfinite(numbers).all():
            raise ValueError("Rozsahy a vyska musi byt konecna cisla")
        if not xmin < xmax or not ymin < ymax or not (3 <= nx <= 8 and 3 <= ny <= 8):
            raise ValueError("Potrebujeme XMIN<XMAX, YMIN<YMAX, NX/NY v intervalu 3 az 8")
        if not (0 < height_cm <= 100):
            raise ValueError("Zadej kladnou vysku stredu hoopu v cm")
        # Allow meaningful calibration coverage, but no guessed world-space X/Y bounds.
        if xmax - xmin < 0.12 or ymax - ymin < 0.10:
            raise ValueError("Rozsah je prilis maly: potreba alespon 12 cm X a 10 cm Y")
        offset = tuple(map(float, self.session.target_offset_tcp_m))
        offset_vec = np.asarray(offset)
        if not np.isfinite(offset_vec).all() or not 0.05 < np.linalg.norm(offset_vec) < 0.25:
            raise ValueError("Offset musi byt skutecny stred hoopu v TCP; predpokladana vzdalenost 5-25 cm")
        raw, effective_deg = self.robot.get_joint_limits_deg()
        if effective_deg is None:
            raise RuntimeError("Chybi overene limity kloubu v driveru CRS93")
        bounds = np.deg2rad(np.asarray(effective_deg, dtype=float))
        start_q = np.asarray(self.robot.get_q(), dtype=float)
        if start_q.shape != (6,) or not np.isfinite(start_q).all():
            raise ValueError("Invalidni aktualni poloha")
        # Configure measured table plane and offset before ANY preview. Guards
        # verify 1-cm clearance for all joint/link surfaces and hoop envelope.
        self.robot.set_table_z(table_z)
        self.robot.set_hoop_offset(offset)
        self.robot.check_table_clearance(start_q)
        # The hoop lies in the flange XY plane for the nominal Prokop mount:
        # adapter_to_hoop maps its ring normal to flange -Z.  A BASE-frame
        # flange orientation with local Z downward therefore puts the ring
        # HORIZONTAL (the Basler sees a circle, not a ring edge-on).
        # Local flange +X points back toward the robot, improving reach near
        # the front edge of the calibrated workspace.  Do NOT inherit the
        # home orientation: home aims flange +X downward and causes both
        # unreachable targets and a near-edge-on view of the hoop.
        # The actual physical mounting must be checked before any REAL run.
        R = np.diag([-1.0, 1.0, -1.0])
        z_target = table_z + height_cm / 100.0
        points: list[ScanPoint] = []
        segments: list[np.ndarray] = []
        previous = start_q.copy()
        step_limit = float(np.deg2rad(2.0))
        if not np.isfinite(step_limit) or step_limit <= 0:
            raise RuntimeError("Neplatny limit pohybu kloubu")
        ys = np.linspace(ymin, ymax, ny)
        xs = np.linspace(xmin, xmax, nx)
        for row, y in enumerate(ys):
            # snake traversal avoids long diagonal jumps
            x_line = xs if row % 2 == 0 else xs[::-1]
            for x in x_line:
                center = np.array([x, y, z_target], dtype=float)
                flange_pose = np.eye(4)
                flange_pose[:3, :3] = R
                flange_pose[:3, 3] = center - R @ offset_vec
                solutions = self.robot.ik(flange_pose)
                valid = []
                for solution in ([] if solutions is None else solutions):
                    q = np.asarray(solution, dtype=float).reshape(-1)
                    if q.shape != (6,) or not np.isfinite(q).all():
                        continue
                    if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
                        continue
                    fk = self._fk(q)
                    if np.linalg.norm((fk @ np.array([*offset, 1.0]))[:3] - center) > .004:
                        continue
                    if np.linalg.norm(fk[:3, :3] - R) > .08:
                        continue
                    valid.append(q)
                if not valid:
                    raise ValueError(
                        f"IK/limity: nedostupny bod stred hoopu X={x:.3f}, "
                        f"Y={y:.3f}, Z={z_target:.3f} m pri vodorovnem hoopu. "
                        "PLAN nicim nepohnul. Zmensi rozsah mrizky; "
                        "zachovej skutecne zmerenou vysku stolu a offset."
                    )
                target = min(valid, key=lambda q: np.linalg.norm(q - previous))
                steps = max(1, int(math.ceil(np.max(np.abs(target - previous)) / step_limit)))
                # Every intermediate joint-space pose is FK checked. This is NOT
                # a collision check for entire robot links or external equipment.
                for k in range(1, steps + 1):
                    q_way = previous + (target - previous) * (k / steps)
                    if np.any(q_way < bounds[:, 0]) or np.any(q_way > bounds[:, 1]):
                        raise ValueError("Mezikrok mimo roboticke limity")
                    self.robot.check_table_clearance(q_way)
                    segments.append(q_way.copy())
                points.append(ScanPoint(center, target.copy()))
                previous = target.copy()
        # Preflight robot availability and current move gate, no motion.
        self.robot.preview_move(start_q)
        self.plan = ScanPlan(points, segments, start_q, offset, table_z, nx, ny, step_limit)
        return self.plan

    def run(self) -> int:
        """Run the preplanned sweep without a per-move or batch confirmation.

        Operator must supervise with physical E-STOP accessible. A detected
        error stops sending *new* movements; only hardware E-STOP stops a
        currently executing driver command immediately.
        """
        plan = self.plan
        if plan is None:
            raise RuntimeError("Nejprve pouzij 'homo scan plan ...'")
        self._require_real()
        if tuple(self.session.target_offset_tcp_m) != plan.offset_tcp:
            raise RuntimeError("Od planovani byl zmenen offset; plan je neplatny")
        if self.robot._table_guard is None or self.robot._table_guard.table_z_m != plan.table_z:
            raise RuntimeError("Od planovani byla zmenena rovina stolu")
        if self.session.samples:
            raise RuntimeError("Session jiz obsahuje body; obnov nebo vymaz data a planuj znovu")
        if np.max(np.abs(self.robot.get_q() - plan.starting_q)) > np.deg2rad(.25):
            raise RuntimeError("Robot od planovani zmenil polohu; planuj znovu")
        count = len(plan.points)
        print(f"AUTO PRUJEZD: {count} poloh, {plan.total_moves} kontrolovanych mezikroku; "
              f"Z stredu hoopu = {plan.points[0].center_xyz[2]:.3f} m, mez nad stolem >= 10 mm.")
        print("BEZ POTVRZOVANI. SLEDUJ ROBOTA A MEJ PRISTUP K FYZICKEMU E-STOP.")
        # RobotManager checks geometry for each commanded segment again.
        idx_segment = 0
        collected = 0
        try:
            for target in plan.points:
                while idx_segment < len(plan.segments):
                    q_goal = plan.segments[idx_segment]
                    now, goal = self.robot.preview_move(q_goal)
                    self.robot.move_to_q(goal, expected_q=now)
                    actual = np.asarray(self.robot.get_q(), dtype=float)
                    if np.max(np.abs(actual - goal)) > np.deg2rad(.5):
                        raise RuntimeError("Robot nedosahl prikazaneho mezikroku")
                    idx_segment += 1
                    if np.max(np.abs(goal - target.joints)) <= 1e-10:
                        break
                self.robot.assert_stationary()
                q_before = np.asarray(self.robot.get_q(), dtype=float)
                xyz = self._center(q_before, plan.offset_tcp)
                if np.linalg.norm(xyz - target.center_xyz) > .005:
                    raise RuntimeError("Stred hoopu neodpovida planovanemu bodu (FK)")
                frame = _bgr_frame(self.camera.capture())
                uv = detect_single_circle(frame)
                self.robot.assert_stationary()
                if np.max(np.abs(self.robot.get_q() - q_before)) > np.deg2rad(.15):
                    raise RuntimeError("Robot se behem fotografovani pohnul")
                if uv is None:
                    self.session.folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(self.session.folder / f"auto_failed_{collected+1:03d}.png"), frame)
                    raise RuntimeError(f"Kruh se nenasel jednoznacne na bode {collected+1}; "
                                       "kontrolni fotka auto_failed_XXX.png. Prujezd ZASTAVEN.")
                self.session.add(uv, tuple(xyz), (frame.shape[1], frame.shape[0]))
                collected += 1
                self.session.folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.session.folder / f"sample_{len(self.session.samples):03d}.png"), frame)
                print(f"[{collected}/{count}] uv=({uv[0]:.1f},{uv[1]:.1f}) "
                      f"XYZ=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) m")
            calib = self.session.solve()
            print(f"Auto homografie hotova: RMS={calib.rms_px:.2f} px; "
                  "zkontroluj nezavisle testovaci body pred manipulaci.")
            return collected
        finally:
            # If an error occurs, preserve all valid samples and invalidate plan.
            # Do not attempt to return home automatically after faults.
            self.plan = None
