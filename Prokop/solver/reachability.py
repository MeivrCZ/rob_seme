"""The reachability map: which turntable phases leave puzzle D's checkpoints reachable.

Built offline by tools/build_reachability_map.py: per checkpoint, whether a free,
in-room pose of the ring exists with the puzzle at (rho, psi) relative to the robot
(rho = the puzzle origin's distance from the robot's base axis, psi = its heading
relative to the line from the robot to it). For a perceived TurntableModel every
turntable angle maps to one (rho, psi), so:

* phases(model): per checkpoint, the turntable angles (1 deg steps) at which it is
  reachable;
* start_phases(model, traversal): the start angles from which every checkpoint is
  reachable when the traversal gets there (checkpoint k reached at the start angle plus
  the rotation during traversal * its share of the wire);
* check(model): the summary at the start of a run: unreachable straight away (some
  checkpoint at no phase, or no start phase) or the reachable share.

The lookup takes the nearest cell (1 cm, 10 deg) and the map's fixed margins (6 mm,
5.5 deg): a quick screen, not a guarantee; the planner decides.

Example:
    ::

        rmap = ReachabilityMap.load()
        ok, report = rmap.check(model)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solver.contracts import TurntableModel

MAP_PATH = Path(__file__).resolve().parent / "data" / "reachability_map.npz"
PHASES = np.deg2rad(np.arange(0.0, 360.0, 1.0))
TRAVERSAL_GUESS = 7.0  # s, a typical traversal (the planner's are 4-8 s)


@dataclass
class ReachabilityMap:
    rho: np.ndarray
    psi: np.ndarray
    reachable: np.ndarray  # (checkpoints, rho, psi)
    share: np.ndarray  # (checkpoints,) where each checkpoint lies along the wire, 0 .. 1

    @staticmethod
    def load(path: Path = MAP_PATH) -> "ReachabilityMap":
        from solver.wire import checkpoint_arclengths
        d = np.load(path)
        s = checkpoint_arclengths(0, float(d["entry"]), int(d["bend_points"]))
        return ReachabilityMap(d["rho"], d["psi"], d["reachable"], s / s[-1])

    def _cell(self, model: TurntableModel, theta: float) -> tuple[int, int] | None:
        P = model.puzzle_T(theta)
        rho = float(np.hypot(P[0, 3], P[1, 3]))
        psi = np.arctan2(P[1, 0], P[0, 0]) - np.arctan2(P[1, 3], P[0, 3])
        psi = float(np.arctan2(np.sin(psi), np.cos(psi)))
        if rho < self.rho[0] - 0.005 or rho > self.rho[-1] + 0.005:
            return None  # outside the map: not reachable
        i = int(np.argmin(np.abs(self.rho - rho)))
        j = int(np.argmin(np.abs(np.angle(np.exp(1j * (self.psi - psi))))))
        return i, j

    def phases(self, model: TurntableModel) -> np.ndarray:
        """(checkpoints, 360): whether checkpoint k is reachable at each 1-deg angle."""
        out = np.zeros((self.reachable.shape[0], len(PHASES)), bool)
        for n, theta in enumerate(PHASES):
            cell = self._cell(model, float(theta))
            if cell is not None:
                out[:, n] = self.reachable[:, cell[0], cell[1]]
        return out

    def start_phases(self, model: TurntableModel,
                     traversal: float = TRAVERSAL_GUESS) -> np.ndarray:
        """(360,): start angles from which every checkpoint is reachable when reached."""
        per = self.phases(model)
        step = PHASES[1] - PHASES[0]
        ok = np.ones(len(PHASES), bool)
        for k, share in enumerate(self.share):
            shift = int(round(model.direction * model.omega * traversal * share / step))
            ok &= np.roll(per[k], -shift)
        return ok

    def check(self, model: TurntableModel, traversal: float = TRAVERSAL_GUESS
              ) -> tuple[bool, str]:
        """Whether the placement may be reachable at all, and a one-line report."""
        per = self.phases(model)
        shares = per.mean(axis=1)
        starts = self.start_phases(model, traversal)
        dead = [k for k, x in enumerate(shares) if x == 0.0]
        text = (f"checkpoints reachable in {', '.join(f'{100 * x:.0f}' for x in shares)} % of "
                f"the turntable's angles; start angles from which all are reachable in time: "
                f"{100 * starts.mean():.0f} %")
        if dead:
            return False, f"checkpoint(s) {dead} reachable at no angle: " + text
        if not starts.any():
            return False, "no start angle reaches every checkpoint in time: " + text
        return True, text
