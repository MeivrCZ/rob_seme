"""Safety margins from the run's uncertainties (k standard deviations, user 2026-11-26).

The planner's margins used to be fixed guesses (the hoop 3 mm from the puzzle, no
room at the joint limits). Now they follow from what the run measured:

* the turntable's angle at execution: the camera tracker's phase sd when it is
  frozen (PHASE_SD: one marker, the worse case; not known yet when planning), and
  the drift of the fitted speed over the traversal (its sd from the orbit fit's yaw
  residuals and the spread of the measurement times);
* the turntable's position: the orbit fit's position residual;
* the dimensions, measured with a 1 mm tape: each a uniform rounding error, sd
  1/sqrt(12) mm (user, 2026-11-26); DIMENSION_PARTS of them add up between the hoop
  and the puzzle (turntable, base, bar, robot position).

Margins, with K_SIGMA = 2 (user: about 95 % per check if the errors are normal):

* hoop against the puzzle: K * sqrt(sd_xy^2 + (reach * sd_angle)^2 + sd_dim^2),
  reach = how far the task reaches from the turntable axis (the worst point);
* joint limits: every configuration of the plan keeps K * sd_angle * JOINT_GAIN +
  LIMIT_RESERVE from its limits, so that the executor's correction (the target turned
  by up to K * sd_angle; up to JOINT_GAIN degrees of joint per degree, from the runs)
  never runs into a limit.

Example:
    ::

        margins = Margins.from_fit(fit, measurements, reach=scene.task_reach(0.0))
        scene = SoftScene(model, hoop_puzzle_margin=margins.hoop_puzzle)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_SIGMA = 2.0
DIMENSION_SD = 0.001 / np.sqrt(12)  # m, one tape-measured dimension (1 mm rounding, user)
DIMENSION_PARTS = 4  # turntable, base, bar, robot position
DIMENSION_SD_TOTAL = np.sqrt(DIMENSION_PARTS) * DIMENSION_SD  # m, in quadrature
PHASE_SD = np.deg2rad(0.8)  # rad, the camera tracker with one marker (0.4 deg with two)
# Floor under the fitted speed's sd: the stopwatch measurement of the period (E3; user
# 2026-11-26): T = 28 s read to 0.1 s (uniform, sd 0.1/sqrt(3) s) and the start/end
# mark to 4 deg (sd 4/sqrt(12) deg); omega = (2 pi + d_theta) / T, so
# sd = sqrt((sd_theta / T)^2 + (2 pi sd_T / T^2)^2) = 0.049 deg/s.
STOPWATCH_PERIOD, STOPWATCH_T_SD, STOPWATCH_ANGLE_SD = 28.0, 0.1 / np.sqrt(3), np.deg2rad(4.0) / np.sqrt(12)
MIN_OMEGA_SD = float(np.hypot(STOPWATCH_ANGLE_SD / STOPWATCH_PERIOD,
                              2 * np.pi * STOPWATCH_T_SD / STOPWATCH_PERIOD ** 2))  # rad/s
TRAVERSAL_GUESS = 12.0  # s, the longest traversals seen (the drift after the freeze)
JOINT_GAIN = 2.0  # deg of joint per deg of correction, at most (executor runs)
LIMIT_RESERVE = np.deg2rad(2.0)  # rad
MIN_HOOP_PUZZLE = 0.003  # m, never below the old fixed margin


@dataclass
class Margins:
    """The margins and where they come from."""

    hoop_puzzle: float  # m
    joint_limit: float  # rad
    sd_angle: float  # rad, at execution (phase + drift over the traversal)
    sd_xy: float  # m
    sd_omega: float  # rad/s
    reach: float  # m

    @staticmethod
    def from_fit(xy_rmse: float, yaw_rmse: float, times: np.ndarray, reach: float,
                 traversal: float = TRAVERSAL_GUESS) -> "Margins":
        """From the orbit fit's residuals and its measurement times [s].

        Args:
            xy_rmse: Position residual of the fit [m].
            yaw_rmse: Orientation residual of the fit [rad].
            times: Capture times of the fitted measurements [s].
            reach: Largest distance of the task from the turntable axis [m].
            traversal: Time from the freeze to the end of the traversal [s].
        """
        t = np.asarray(times, float)
        spread = float(np.sum((t - t.mean()) ** 2))
        sd_omega = max(MIN_OMEGA_SD, yaw_rmse / np.sqrt(spread) if spread > 0 else np.inf)
        sd_angle = float(np.hypot(PHASE_SD, sd_omega * traversal))
        sd_point = float(np.hypot(xy_rmse, reach * sd_angle))
        sd_total = float(np.hypot(sd_point, DIMENSION_SD_TOTAL))
        return Margins(hoop_puzzle=max(MIN_HOOP_PUZZLE, K_SIGMA * sd_total),
                       joint_limit=K_SIGMA * sd_angle * JOINT_GAIN + LIMIT_RESERVE,
                       sd_angle=sd_angle, sd_xy=float(xy_rmse), sd_omega=float(sd_omega),
                       reach=float(reach))

    def describe(self) -> str:
        return (f"hoop-puzzle {1000 * self.hoop_puzzle:.1f} mm, joint limits "
                f"{np.rad2deg(self.joint_limit):.1f} deg (k = {K_SIGMA:g}: angle sd "
                f"{np.rad2deg(self.sd_angle):.2f} deg incl. speed sd "
                f"{np.rad2deg(self.sd_omega):.3f} deg/s over {TRAVERSAL_GUESS:.0f} s, position sd "
                f"{1000 * self.sd_xy:.2f} mm, reach {1000 * self.reach:.0f} mm, dimensions sd "
                f"{1000 * DIMENSION_SD_TOTAL:.2f} mm)")
