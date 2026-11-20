"""P2.4: joint-limit, Jacobian, and collision safety metrics.

The metrics here use the project robot interface and the same finite-difference
Jacobian semantics as the robotics-lab HW04 material. Collision distances are
optional and use Prokop/collision.py with a placement and turntable angle.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROKOP_DIR = PROJECT_ROOT / "Prokop"
if str(PROKOP_DIR) not in sys.path:
    sys.path.insert(0, str(PROKOP_DIR))


class RobotKinematics(Protocol):
    q_min: ArrayLike
    q_max: ArrayLike

    def fk(self, q: ArrayLike) -> np.ndarray: ...

    def in_limits(self, q: ArrayLike) -> bool: ...


@dataclass(frozen=True)
class SafetyMetrics:
    """Safety values for one joint configuration."""

    smallest_singular_value: float
    manipulability: float
    joint_limit_margin: float
    clearance: float
    collision_name: str | None

    @property
    def margin(self) -> float:
        """Simple combined margin; collision-free, singularity, and joint limits."""
        singularity_margin = self.smallest_singular_value
        return float(min(singularity_margin, self.joint_limit_margin,
                         0.0 if self.collision_name is not None else self.clearance))


@dataclass(frozen=True)
class SegmentSafety:
    """Safety values sampled along a joint-space segment."""

    largest_joint_step: float
    samples: tuple[SafetyMetrics, ...]

    @property
    def worst_margin(self) -> float:
        return min((metric.margin for metric in self.samples), default=0.0)



def pose_error(T_current: np.ndarray, T_desired: np.ndarray) -> np.ndarray:
    """Position and rotation-vector error used by the finite-difference Jacobian."""
    current = np.asarray(T_current, dtype=float)
    desired = np.asarray(T_desired, dtype=float)
    if current.shape != (4, 4) or desired.shape != (4, 4):
        raise ValueError("poses must be 4x4 homogeneous transforms")
    cos_angle = np.clip((np.trace(desired[:3, :3] @ current[:3, :3].T) - 1.0)
                        / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    R = desired[:3, :3] @ current[:3, :3].T
    if angle < 1e-9:
        log_R = np.array([R[2, 1] - R[1, 2],
                          R[0, 2] - R[2, 0],
                          R[1, 0] - R[0, 1]]) / 2.0
    else:
        log_R = angle * np.array([R[2, 1] - R[1, 2],
                                  R[0, 2] - R[2, 0],
                                  R[1, 0] - R[0, 1]]) \
                / (2.0 * np.sin(angle))
    return np.concatenate((desired[:3, 3] - current[:3, 3], log_R))



def jacobian_finite_central_differences(robot: RobotKinematics, q: ArrayLike,
                               delta: float = 1e-6) -> np.ndarray:
    """Compute the 6xn flange Jacobian with finite central differences.

    The name distinguishes this method from forward differences. Each joint is
    evaluated at both q + delta e_i and q - delta e_i, giving
    (fk(q + delta e_i) - fk(q - delta e_i)) / (2 delta).

    Rows are position x/y/z followed by rotation-vector x/y/z in the robot
    frame. This is the lab-HW04 finite-difference method without planar-only
    assumptions.
    """
    configuration = np.asarray(q, dtype=float)
    if configuration.ndim != 1 or not np.all(np.isfinite(configuration)):
        raise ValueError("q must be a finite one-dimensional configuration")
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    Jacobian = np.zeros((6, configuration.size))
    T0 = np.asarray(robot.fk(configuration), dtype=float)
    for joint in range(configuration.size):
        plus = configuration.copy()
        minus = configuration.copy()
        plus[joint] += delta
        minus[joint] -= delta
        T_plus = np.asarray(robot.fk(plus), dtype=float)
        T_minus = np.asarray(robot.fk(minus), dtype=float)
        central = pose_error(T0, T_plus) - pose_error(T0, T_minus)
        Jacobian[:, joint] = central / (2.0 * delta)
    return Jacobian



def smallest_singular_value(J: ArrayLike) -> float:
    """Return sigma_min(J), zero at a singular configuration."""
    Jacobian = np.asarray(J, dtype=float)
    if Jacobian.ndim != 2:
        raise ValueError("Jacobian must be a 2D array")
    singular_values = np.linalg.svd(Jacobian, compute_uv=False)
    return 0.0 if singular_values.size == 0 else float(singular_values[-1])



def manipulability(J: ArrayLike) -> float:
    """Return Yoshikawa manipulability sqrt(det(J J^T))."""
    Jacobian = np.asarray(J, dtype=float)
    if Jacobian.ndim != 2:
        raise ValueError("Jacobian must be a 2D array")
    return float(np.sqrt(max(0.0, np.linalg.det(Jacobian @ Jacobian.T))))



def joint_limit_margin(robot: RobotKinematics, q: ArrayLike) -> float:
    """Return normalized clearance to the nearest lower or upper joint limit.

    A value of 0 means a limit is reached; 1 means the midpoint of all limits.
    This value is the minimum normalized distance in any joint coordinate.
    """
    configuration = np.asarray(q, dtype=float)
    q_min = np.asarray(robot.q_min, dtype=float)
    q_max = np.asarray(robot.q_max, dtype=float)
    if configuration.shape != q_min.shape or q_min.shape != q_max.shape:
        raise ValueError("q and joint-limit shapes do not match")
    if not np.all(np.isfinite(configuration)):
        raise ValueError("q must contain only finite values")

    span = q_max - q_min
    if np.any(span <= 0.0):
        raise ValueError("each joint must have a positive finite limit span")
    normalized = np.minimum((configuration - q_min) / span,
                            (q_max - configuration) / span)
    if np.any(normalized < -1e-12):
        return 0.0
    return float(np.clip(np.min(normalized), 0.0, 1.0))



def collision_metrics(robot: RobotKinematics, q: ArrayLike, checker=None,
                      placement=None, turntable_angle: float = 0.0,
                      ) -> tuple[float, str | None]:
    """Return the available environment/task collision clearance.

    If an ArmChecker is supplied, return its table/cell clearance. If a
    SceneCollider and placement are supplied as well, include task clearance and
    self-collision. Otherwise only the supplied checker is used.
    """
    configuration = np.asarray(q, dtype=float)
    clearance = np.inf
    collision_name = None

    if checker is not None:
        cell_clearance, name = checker.clearance(configuration)
        clearance = min(clearance, float(cell_clearance))
        if float(cell_clearance) <= 1e-9:
            collision_name = name

    if placement is not None:
        from collision import SceneCollider
        collider = checker if isinstance(checker, SceneCollider) else None
        if collider is None:
            raise ValueError("placement requires a SceneCollider as checker")
        collider.place(configuration, placement, turntable_angle)
        hits, task_clearance = collider.check(configuration, placement,
                                              turntable_angle,
                                              self_collision=True)
        clearance = min(clearance, float(task_clearance))
        if hits:
            collision_name = hits[0][1]

    if not np.isfinite(clearance):
        clearance = np.inf
    return float(clearance), collision_name



def configuration_metrics(robot: RobotKinematics, q: ArrayLike,
                          checker=None, placement=None,
                          turntable_angle: float = 0.0,
                          jacobian_delta: float = 1e-6) -> SafetyMetrics:
    """Compute all P2.4 safety metrics for one configuration."""
    configuration = np.asarray(q, dtype=float)
    if configuration.ndim != 1:
        raise ValueError("q must be a one-dimensional configuration")

    J = jacobian_finite_central_differences(robot, configuration, delta=jacobian_delta)
    smallest = smallest_singular_value(J)
    in_limits = bool(robot.in_limits(configuration))
    limit_margin = joint_limit_margin(robot, configuration) if in_limits else 0.0
    clearance, collision_name = collision_metrics(
        robot, configuration, checker=checker, placement=placement,
        turntable_angle=turntable_angle
    )
    return SafetyMetrics(
        smallest_singular_value=smallest,
        manipulability=manipulability(J),
        joint_limit_margin=limit_margin,
        clearance=clearance,
        collision_name=collision_name,
    )



def segment_metrics(robot: RobotKinematics, q0: ArrayLike, q1: ArrayLike,
                    checker=None, placement=None,
                    theta_at: Callable[[float], float] | None = None,
                    duration: float = 0.0, samples: int = 11,
                    jacobian_delta: float = 1e-6) -> SegmentSafety:
    """Sample configuration safety along the straight joint-space segment.

    Args:
        theta_at: Optional function of relative segment time [s] -> turntable
            angle [rad]. If omitted, turntable_angle is held at 0.
        duration: Segment duration [s], used with theta_at.
        samples: Number of uniform samples, including both endpoints.
    """
    start = np.asarray(q0, dtype=float)
    goal = np.asarray(q1, dtype=float)
    if start.shape != goal.shape or start.ndim != 1:
        raise ValueError("q0 and q1 must be matching one-dimensional configurations")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if duration < 0.0:
        raise ValueError("duration must be non-negative")

    sample_points = np.linspace(0.0, 1.0, samples)
    metrics = []
    for fraction in sample_points:
        q = start + fraction * (goal - start)
        angle = theta_at(float(fraction * duration)) if theta_at is not None else 0.0
        metrics.append(configuration_metrics(robot, q, checker=checker,
                                              placement=placement,
                                              turntable_angle=float(angle),
                                              jacobian_delta=jacobian_delta))
    return SegmentSafety(
        largest_joint_step=float(np.max(np.abs(goal - start))),
        samples=tuple(metrics),
    )
