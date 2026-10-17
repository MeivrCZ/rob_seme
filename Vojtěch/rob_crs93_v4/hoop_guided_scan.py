"""Image-guided, one-height hoop homography calibration for CRS93.

If the hoop is not yet visible, preflight a known, previously photographed
station-specific seed pose. Execution makes a fully checked move to that pose,
verifies the hoop, and only then starts camera-feedback corrections. The robot
NEVER sweeps blindly to hunt for a lost hoop. Visual feedback is NOT collision
protection.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from homography_capture import _bgr_frame, HomographySession
from hoop_detection import detect_hoop_circle


FLANGE_ROTATION = np.diag([-1., 1., -1.])
# First point of the previous real CRS93 scan: full hoop was detected at
# approximately (u,v)=(248,820). These numbers are STATION-SPECIFIC and are
# merely a photographed acquisition pose, NOT the image-to-robot homography.
# Once this seed is acquired the controller learns its actual image Jacobian.
VISIBLE_SEED_XY_M = (0.280, -0.170)
STAGING_MAX_JOINT_STEP_RAD = float(np.deg2rad(2.0))


@dataclass(frozen=True)
class GuidedSetup:
    table_z: float
    center_z: float
    offset: tuple[float, float, float]
    initial_q: np.ndarray
    image_size: tuple[int, int] | None
    initial_uv: tuple[float, float] | None
    initial_radius: float | None
    nx: int
    ny: int
    seed_q: np.ndarray


@dataclass(frozen=True)
class _Observation:
    q: np.ndarray
    xyz: np.ndarray
    uv: np.ndarray
    radius: float
    image_size: tuple[int, int]
    frame: np.ndarray


class GuidedHoopScanner:
    """Adaptive camera-space scanning, with mandatory safe rollback on loss."""

    def __init__(self, robot, camera, session: HomographySession):
        self.robot = robot
        self.camera = camera
        self.session = session
        self.setup: GuidedSetup | None = None
        self.jacobian: np.ndarray | None = None  # pixel per metre, 2x2
        self.total_steps = 0
        self.failed_moves = 0

    def _ready(self):
        if (self.robot.mode != 'real' or self.robot.released or
                self.camera.mode != 'real' or not self.robot.real_motion_enabled):
            raise RuntimeError('GUIDED potrebuje REAL robota, kameru a povoleny pohyb')
        self.robot.assert_stationary()

    def _center(self, q, offset):
        fk = np.asarray(self.robot.require().fk(np.asarray(q, float)), float)
        if fk.shape != (4, 4) or not np.isfinite(fk).all():
            raise RuntimeError('Neplatna FK')
        return (fk @ np.array([*offset, 1.]))[:3], fk[:3, :3]

    def _observe(self, offset, *, save_failure=False):
        self.robot.assert_stationary()
        q = np.asarray(self.robot.get_q(), float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError('Chybi zpetna vazba kloubu')
        xyz, rot = self._center(q, offset)
        frame = _bgr_frame(self.camera.capture())
        self.robot.assert_stationary()
        if np.max(np.abs(self.robot.get_q() - q)) > np.deg2rad(.15):
            raise RuntimeError('Poloha robota se behem snimku zmenila')
        found = detect_hoop_circle(frame)
        if found.center is None or found.radius_px is None:
            if save_failure:
                self.session.folder.mkdir(parents=True, exist_ok=True)
                path = self.session.folder / 'guided_lost_raw.png'
                cv2.imwrite(str(path), frame)
                cv2.imwrite(str(self.session.folder / 'guided_lost_debug.png'), found.annotate(frame))
            return None
        return _Observation(q.copy(), xyz, np.array(found.center, float),
                            float(found.radius_px), (frame.shape[1], frame.shape[0]), frame)

    def _make_seed(self, start_q, offset, center_z):
        """Plan a VERIFIED joint-space route to a previously photographed pose.

        No motion here. Check the direct path and all intermediate poses with
        exactly the same model/limits that actual RobotManager.move_to_q uses.
        If that path is not possible, STOP instead of searching blindly.
        """
        center = np.array([*VISIBLE_SEED_XY_M, center_z], float)
        pose = np.eye(4)
        pose[:3, :3] = FLANGE_ROTATION
        pose[:3, 3] = center - FLANGE_ROTATION @ np.asarray(offset, float)
        _, limits_deg = self.robot.get_joint_limits_deg()
        if limits_deg is None:
            raise RuntimeError('Nelze naplanovat pocatecni polohu: chybi kloubove limity')
        limits = np.deg2rad(np.asarray(limits_deg, float))
        candidates = []
        solutions = self.robot.ik(pose)
        for solution in ([] if solutions is None else solutions):
            q = np.asarray(solution, float).reshape(-1)
            if q.shape != (6,) or not np.isfinite(q).all():
                continue
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                continue
            xyz, rot = self._center(q, offset)
            if (np.linalg.norm(xyz - center) > .004 or
                    np.linalg.norm(rot - FLANGE_ROTATION) > .08):
                continue
            candidates.append(q)
        if not candidates:
            raise RuntimeError('IK: ani drive overena vychozi poloha hoopu '
                               'X=0.280 Y=-0.170 neni dosazitelna v teto vysce. '
                               'Robot se nepohnul.')
        errors = []
        for q_seed in sorted(candidates, key=lambda q: np.linalg.norm(q-start_q)):
            try:
                # Full start-to-seed motion must be cleared, including meshes.
                self.robot.preview_move(q_seed)
                # Check explicit staging increments also; no waypoint is
                # dispatched until the WHOLE direct route was approved.
                n = max(1, int(np.ceil(np.max(np.abs(q_seed - start_q)) /
                                       STAGING_MAX_JOINT_STEP_RAD)))
                waypoints = []
                for k in range(1, n + 1):
                    q = start_q + (q_seed-start_q) * (k/n)
                    self.robot.check_table_clearance(q)
                    waypoints.append(q.copy())
                return q_seed.copy(), tuple(waypoints)
            except (RuntimeError, ValueError) as exc:
                errors.append(str(exc))
        raise RuntimeError('Cesta z aktualni polohy do zorneho pole '
                           'nesplnuje IK/limity/odstup od stolu. '
                           'Robot se nepohnul. ' + '; '.join(errors[:2]))

    def prepare(self, table_z: float, height_cm: float, nx: int = 5, ny: int = 5):
        """NO MOTION: plan either from a visible hoop OR guarded acquisition."""
        self.setup = None
        self._ready()
        if not all(np.isfinite(v) for v in [table_z, height_cm]):
            raise ValueError('Vyska stolu a hoopu musi byt konecna')
        if not (3 <= nx <= 6 and 3 <= ny <= 6 and 2 <= height_cm <= 50):
            raise ValueError('Vyska 2..50 cm, mrizka 3..6 v kazde ose')
        if self.session.samples:
            raise RuntimeError('Nejdrive homo clear')
        offset = tuple(float(x) for x in self.session.target_offset_tcp_m)
        if not .05 < np.linalg.norm(offset) < .25:
            raise ValueError('Nejdrive homo offset X Y Z')
        if getattr(self.robot, '_hoop_offset_tcp', None) is None:
            raise ValueError('Chybi potvrzeny offset hoopu v robot manageru')
        self.robot.set_table_z(table_z)
        self.robot.set_hoop_offset(offset)
        q_start = np.asarray(self.robot.get_q(), float)
        self.robot.check_table_clearance(q_start)
        xyz, rot = self._center(q_start, offset)
        target_z = float(table_z + height_cm / 100.)
        image = None
        if (abs(xyz[2] - target_z) <= .007 and
                np.linalg.norm(rot - FLANGE_ROTATION) <= .12):
            image = self._observe(offset)
            if image is not None and not self._inside(image):
                image = None
        if image is not None:
            q_seed = q_start.copy()
            initial_uv = tuple(map(float, image.uv))
            initial_radius = image.radius
            print(f'GUIDED PLAN: hoop uz viditelny v uv={initial_uv}; Z={xyz[2]:.4f}m. '
                  'Zadny uvodni presun nebude nutny.')
        else:
            # NEVER move from an unknown state directly as part of plan.
            # Stage to the photographed station seed only if preflight passes.
            q_seed, _ = self._make_seed(q_start, offset, target_z)
            initial_uv, initial_radius = None, None
            print(f'GUIDED PLAN: start Z={xyz[2]:.4f}m, cil stred hoopu '
                  f'XYZ=({VISIBLE_SEED_XY_M[0]:.3f}, {VISIBLE_SEED_XY_M[1]:.3f}, '
                  f'{target_z:.4f})m. Cesta IK/limity/1cm proverena, BEZ POHYBU.')
            print('GUIDED RUN nejdrive automaticky dojede na viditelnou pozici, '
                  'overi kruh a nasledne pouzije korekce obrazu.')
        self.setup = GuidedSetup(float(table_z), target_z, offset, q_start.copy(),
                                 image.image_size if image is not None else None,
                                 initial_uv, initial_radius, nx, ny, q_seed.copy())
        self.jacobian = None
        self.total_steps = 0
        self.failed_moves = 0
        return self.setup

    def _stage_to_seed(self, setup):
        """Move to the preflighted acquisition pose. Never search in XY blind."""
        current = np.asarray(self.robot.get_q(), float)
        if np.max(np.abs(current - setup.initial_q)) > np.deg2rad(.25):
            raise RuntimeError('Robot se od pripravy vychozi polohy pohnul')
        # All checks are repeated immediately before sending any motion.
        goal, segments = self._make_seed(current, setup.offset, setup.center_z)
        if np.max(np.abs(goal - setup.seed_q)) > np.deg2rad(.1):
            raise RuntimeError('IK se od pripravy zmenila; zkus znovu guided plan')
        print(f'GUIDED START: automaticky presun na vychozi polohu '
              f'({len(segments)} mezikroku), s kontrolou IK/1cm/feedbacku.')
        for i, q in enumerate(segments, 1):
            start, target = self.robot.preview_move(q)
            self.robot.move_to_q(target, expected_q=start)
            self.total_steps += 1
            reached = np.asarray(self.robot.get_q(), float)
            if np.max(np.abs(reached - target)) > np.deg2rad(.5):
                raise RuntimeError('Nenavazala skutecna poloha: presun ZASTAVEN')
            if i % 20 == 0 or i == len(segments):
                print(f'GUIDED: uvodni presun {i}/{len(segments)}')
        obs = self._observe(setup.offset, save_failure=True)
        if obs is None or not self._inside(obs):
            raise RuntimeError('Na predem overene vychozi poloze neni cely hoop videt. '
                               'Dalsi automaticke pohyby ZASTAVENY. '
                               'Viz guided_lost_raw.png / guided_lost_debug.png.')
        if abs(obs.xyz[2] - setup.center_z) > .004:
            raise RuntimeError('Po automatickem presunu nesedi vyska hoopu')
        print(f'GUIDED ACQUIRED: kruh uv={obs.uv.round(1).tolist()}, '
              f'r={obs.radius:.1f}px, Z={obs.xyz[2]:.4f}m')
        return obs

    @staticmethod
    def _inside(obs: _Observation, extra_px=12):
        u, v = obs.uv
        w, h = obs.image_size
        margin = obs.radius + extra_px
        return margin < u < w - margin and margin < v < h - margin

    def _waypoint(self, current: _Observation, xy_desired):
        """Compute a small XY translation at fixed Z and fixed flange rotation."""
        setup = self.setup
        target = np.asarray(xy_desired, float).reshape(2)
        if not np.isfinite(target).all():
            return None
        world_center = np.array([target[0], target[1], setup.center_z])
        pose = np.eye(4)
        pose[:3, :3] = FLANGE_ROTATION
        pose[:3, 3] = world_center - FLANGE_ROTATION @ np.asarray(setup.offset)
        _, bounds_deg = self.robot.get_joint_limits_deg()
        if bounds_deg is None:
            raise RuntimeError('IK nelze overit: chybi limity')
        bounds = np.deg2rad(np.asarray(bounds_deg, float))
        solutions = self.robot.ik(pose)
        valid = []
        for sol in ([] if solutions is None else solutions):
            q = np.asarray(sol, float).reshape(-1)
            if q.shape != (6,) or not np.isfinite(q).all():
                continue
            if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
                continue
            xyz, rot = self._center(q, setup.offset)
            if np.linalg.norm(xyz - world_center) > .004 or np.linalg.norm(rot - FLANGE_ROTATION) > .08:
                continue
            # One small Cartesian step must not switch IK branch or swing links.
            if np.max(np.abs(q - current.q)) > np.deg2rad(7.):
                continue
            valid.append(q)
        if not valid:
            return None
        return min(valid, key=lambda q: np.linalg.norm(q - current.q))

    def _attempt(self, current: _Observation, delta_xy, *, can_backtrack=True):
        """One guarded movement, image check, and safe rollback on visual loss."""
        delta = np.asarray(delta_xy, float).reshape(2)
        if not np.isfinite(delta).all() or np.linalg.norm(delta) > .012001:
            raise ValueError('Maximalni XY krok 12 mm')
        if np.linalg.norm(delta) < 1e-4:
            return current
        q_target = self._waypoint(current, current.xyz[:2] + delta)
        if q_target is None:
            self.failed_moves += 1
            return None
        try:
            before, goal = self.robot.preview_move(q_target)
        except (ValueError, RuntimeError):
            self.failed_moves += 1
            return None
        if np.max(np.abs(before - current.q)) > np.deg2rad(.25):
            raise RuntimeError('Robot od posledniho bezpecneho snimku zmenil pozici')
        self.robot.move_to_q(goal, expected_q=before)
        self.total_steps += 1
        actual = np.asarray(self.robot.get_q(), float)
        if np.max(np.abs(actual - goal)) > np.deg2rad(.5):
            raise RuntimeError('Robot nedosahl koncovou polohu: vizualni rizeni ZASTAVENO')
        obs = self._observe(self.setup.offset, save_failure=True)
        good = (obs is not None and obs.image_size == current.image_size and
                self._inside(obs) and .65 < obs.radius / current.radius < 1.5 and
                np.linalg.norm(obs.uv - current.uv) < .28 * min(current.image_size))
        if good and self.jacobian is not None:
            displacement = self.jacobian @ (obs.xyz[:2] - current.xyz[:2])
            # A different circular object must not become a valid measurement
            # merely because the hoop went out of view. Compare with the
            # predicted *local* image displacement from recent real samples.
            good = (np.linalg.norm(obs.uv - current.uv - displacement) <
                    max(35., .40 * np.linalg.norm(displacement)))
        if good:
            if abs(obs.xyz[2] - self.setup.center_z) > .004:
                raise RuntimeError('Rovina Z se zmenila: zastavuji')
            return obs
        self.failed_moves += 1
        if not can_backtrack:
            raise RuntimeError('Hoop zmizel po zpetnem pohybu; fyzicky zkontroluj robota')
        # Revert the exact LAST VERIFIED joint-space segment, not a blind guess
        # at where the hoop might be. If rollback fails, stop the entire scan.
        back_start, back_goal = self.robot.preview_move(current.q)
        if np.max(np.abs(back_start - actual)) > np.deg2rad(.25):
            raise RuntimeError('Ztrata polohoveho feedbacku; nelze bezpecne vratit krok')
        self.robot.move_to_q(back_goal, expected_q=back_start)
        self.total_steps += 1
        if np.max(np.abs(self.robot.get_q() - current.q)) > np.deg2rad(.5):
            raise RuntimeError('Navrat na posledni overenou pozici selhal')
        recovered = self._observe(self.setup.offset)
        if recovered is None or np.linalg.norm(recovered.uv - current.uv) > 35:
            raise RuntimeError('Navrat nedal stejny obraz hoopu; sken ZASTAVEN')
        print('GUIDED: kruh ztracen/nespolehlivy, navrat do posledni viditelne polohy.')
        return None

    def _store(self, obs):
        # Persist well-separated samples; no overwrite of previous valid images.
        if self.session.samples:
            distance = min(np.linalg.norm(obs.uv - np.asarray(s['uv_px']))
                           for s in self.session.samples)
            if distance < 48:
                return False
        self.session.add(tuple(obs.uv), tuple(obs.xyz), obs.image_size)
        cv2.imwrite(str(self.session.folder / f'sample_{len(self.session.samples):03d}.png'), obs.frame)
        print(f'GUIDED bod {len(self.session.samples):02d}: uv={obs.uv.round(1).tolist()} '
              f'XY={obs.xyz[:2].round(4).tolist()}')
        return True

    def _probe(self, obs, direction):
        for sign in (1., -1.):
            for step in (.008, .004):
                delta = np.asarray(direction, float) * sign * step
                measured = self._attempt(obs, delta)
                if measured is not None:
                    return measured
        raise RuntimeError('Nelze provest bezpecne pohybove sondy X/Y. Neni misto pro kalibraci.')

    def _update_jacobian(self, old, new):
        dxy = new.xyz[:2] - old.xyz[:2]
        duv = new.uv - old.uv
        if np.linalg.norm(dxy) < .001:
            return
        predicted = self.jacobian @ dxy
        if np.linalg.norm(duv - predicted) > 180:
            raise RuntimeError('Nesouhlasi obrazovy a roboticky posun; kontroluj montaz/identitu hoopu')
        next_j = self.jacobian + .35 * np.outer(duv - predicted, dxy) / (dxy @ dxy)
        if np.linalg.cond(next_j) < 80 and 350 < np.linalg.norm(next_j, 2) < 22000:
            self.jacobian = next_j

    def _targets(self, obs):
        w, h = obs.image_size
        margin = max(obs.radius + 45, .17 * min(w, h))
        # Keep ample radius margin; never intentionally target the image edge.
        xlo, xhi = max(margin, .18 * w), min(w - margin, .82 * w)
        ylo, yhi = max(margin, .18 * h), min(h - margin, .82 * h)
        if xhi <= xlo or yhi <= ylo:
            raise RuntimeError('Hoop zabira prilis velkou cast obrazu')
        xs, ys = np.linspace(xlo, xhi, self.setup.nx), np.linspace(ylo, yhi, self.setup.ny)
        return [np.array([x, y]) for y in ys for x in xs]

    def run(self):
        # A physical run is always one-shot, including faults. Never allow a
        # stale start pose/visual mapping to be resumed without a new preview.
        try:
            return self._run_once()
        finally:
            self.setup = None

    def _run_once(self):
        setup = self.setup
        if setup is None:
            raise RuntimeError('Nejdrive homo scan guided plan TABLE_Z HEIGHT_CM [NX NY]')
        self._ready()
        if np.max(np.abs(self.robot.get_q() - setup.initial_q)) > np.deg2rad(.25):
            raise RuntimeError('Robot se od guided plan pohnul; vytvor novy plan')
        if self.session.samples or tuple(self.session.target_offset_tcp_m) != setup.offset:
            raise RuntimeError('Data nebo offset se od planovani zmenily')
        if setup.initial_uv is None:
            # Prepare did NOT touch hardware; now acquire a previously observed
            # safe station pose. If detection fails there, STOP immediately.
            obs = self._stage_to_seed(setup)
        else:
            obs = self._observe(setup.offset)
            if obs is None or not self._inside(obs):
                raise RuntimeError('Hoop od pripravy zmizel: zadny pohyb')
            if np.linalg.norm(obs.uv - np.asarray(setup.initial_uv)) > 28:
                raise RuntimeError('Kamera/hoop se od planovani posunuly')
        print('GUIDED RUN: bez predem zadane BASE mrizky; male XY kroky, korekce podle kamery.')
        print('POZOR: operator musi sledovat robota s dostupnym fyzickym E-STOP.')
        self._store(obs)
        # Two independent motor probes measure the *real* local 2x2 pixel Jacobian.
        a = obs
        b = self._probe(a, [1, 0])
        self._store(b)
        c = self._probe(b, [0, 1])
        self._store(c)
        movements = np.column_stack((b.xyz[:2] - a.xyz[:2], c.xyz[:2] - b.xyz[:2]))
        shifts = np.column_stack((b.uv - a.uv, c.uv - b.uv))
        if np.linalg.cond(movements) > 20:
            raise RuntimeError('Roboticke sondy nemaji dve nezavisle osy')
        j = shifts @ np.linalg.inv(movements)
        if (np.linalg.cond(j) > 35 or not 350 < np.linalg.norm(j, 2) < 22000):
            raise RuntimeError('Kamera nereaguje pouzitelne na zmeny X/Y; robot se zastavil')
        self.jacobian = j
        print('GUIDED: lokalni Jacobian px/m:\n' + np.array2string(j, precision=1))
        obs = c
        targets = self._targets(obs)
        # Approach the image centre first; subsequently always choose the
        # closest remaining image target to reduce physical path length.
        targets.insert(0, np.array([obs.image_size[0]/2, obs.image_size[1]/2]))
        steps_budget = 300
        skipped = 0
        index = 0
        while targets:
            if index == 0:
                target = targets.pop(0)
            else:
                closest = min(range(len(targets)), key=lambda i: np.linalg.norm(targets[i] - obs.uv))
                target = targets.pop(closest)
            index += 1
            # Avoid duplicate scan requests near an already visited area.
            if self.session.samples and min(np.linalg.norm(target - np.array(s['uv_px']))
                                            for s in self.session.samples) < 35:
                continue
            stuck = 0
            for _ in range(40):
                if self.total_steps >= steps_budget:
                    break
                error = target - obs.uv
                if np.linalg.norm(error) < 35:
                    self._store(obs)
                    break
                try:
                    delta = np.linalg.solve(self.jacobian, error)
                except np.linalg.LinAlgError:
                    raise RuntimeError('Jacobian je singularni; zastavuji')
                dist = np.linalg.norm(delta)
                if not np.isfinite(dist) or dist < 1e-9:
                    raise RuntimeError('Neplatna korekce pozice')
                delta *= min(1., .012 / dist)
                accepted = None
                for scale in (1., .5, .25):
                    accepted = self._attempt(obs, delta*scale)
                    if accepted is not None:
                        break
                if accepted is None:
                    stuck += 1
                    if stuck >= 2:
                        skipped += 1
                        print(f'GUIDED: pixelovy cil {index+1} nedosazitelny / mimo obraz; preskakuji')
                        break
                    continue
                stuck = 0
                self._update_jacobian(obs, accepted)
                obs = accepted
                self._store(obs)
            else:
                skipped += 1
            if self.total_steps >= steps_budget:
                print('GUIDED: prekrocen maximalni pocet kroku, zastavuji dalsi pohyb')
                break
        print(f'GUIDED: {len(self.session.samples)} bodu; kroku={self.total_steps}; '
              f'neuspesnych zkusebnich pohybu={self.failed_moves}; preskoceno={skipped}')
        if len(self.session.samples) < 8:
            raise RuntimeError('Prilis malo bodu, homografii neukladam')
        calib = self.session.solve()
        print(f'GUIDED HOMOGRAFIE: RMS={calib.rms_px:.2f}px. '
              'Over vysledek na nezavislych bodech.')
        return calib
