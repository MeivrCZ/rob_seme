"""Task P1.1: the turntable, base, puzzle and path checkpoints as a one-joint robot.

The turntable is modelled like the spatial robot of lab 2 (HW02): a URDF whose
FK gives every frame. Its only joint is the turntable angle theta (continuous,
about z); everything else is fixed:

    world (robot base frame)
      -> turntable_axis        fixed: TurntableModel.axis_xy, on the table
      -> turntable_link        joint "theta" (continuous, z)
      -> top                   from turntable.urdf
      -> base_link             fixed: TurntableModel.base_offset, on the foam, base_yaw
           foam, guide_1, guide_2, puzzle_mount     from base.urdf
      -> puzzle_D_link         fixed: at puzzle_mount
      -> checkpoint_000 ...    fixed: PuzzlePath.frames

The model URDFs (turntable, base, puzzle D) are merged as they are, so the chain
carries their visual and collision geometry and helper frames (guide_1/2 are the
ArUco marker centres). FK follows SpatialManipulator.link_poses of our HW02
solution: update the configuration in yourdfpy, read the link transforms.

Example:
    ::

        chain = TurntableChain(model, path)
        T = chain.checkpoint_T(5, model.theta_at(t))
        poses = chain.link_poses(theta)          # every frame, by link name
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yourdfpy
from cell import TABLE_Z
from placement import BASE_FOAM, MODELS, TASK_PUZZLE

from solver.contracts import PuzzlePath, TurntableModel

JOINT = "theta"  # the turntable joint
WORLD = "world"  # the robot base frame
AXIS = "turntable_axis"  # on the table, at the turntable's axis
TURNTABLE = "turntable_link"
TOP = "top"
BASE = "base_link"
MOUNT = "puzzle_mount"
PUZZLE = f"{TASK_PUZZLE}_link"
MARKERS = ("guide_1", "guide_2")  # ArUco marker centres on the base


def checkpoint_name(i: int) -> str:
    """Link name of checkpoint @i."""
    return f"checkpoint_{i:03d}"


def _origin(T: np.ndarray) -> ET.Element:
    """URDF <origin> of a transform, in full precision.

    Args:
        T: 4x4 transform.

    Returns:
        The element, rpy as URDF's fixed-axis roll, pitch, yaw.
    """
    rpy = trimesh.transformations.euler_from_matrix(T, "sxyz")
    return ET.Element("origin", xyz=" ".join(f"{v:.17g}" for v in T[:3, 3]),
                      rpy=" ".join(f"{v:.17g}" for v in rpy))


def _joint(name: str, kind: str, parent: str, child: str, T: np.ndarray
           ) -> ET.Element:
    """A URDF <joint> element.

    Args:
        name: Joint name.
        kind: "fixed" or "continuous" (about z).
        parent: Parent link.
        child: Child link.
        T: Pose of the child in the parent frame at joint value 0.

    Returns:
        The element.
    """
    joint = ET.Element("joint", name=name, type=kind)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    joint.append(_origin(T))
    if kind == "continuous":
        ET.SubElement(joint, "axis", xyz="0 0 1")
    return joint


def _translation(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = x, y, z
    return T


def chain_urdf(model: TurntableModel, path: PuzzlePath | None = None) -> str:
    """Writes the chain as a URDF document.

    Args:
        model: The turntable model (fixed joints from its parameters).
        path: Checkpoints to add as fixed frames under the puzzle; None for none.

    Returns:
        The URDF text; mesh paths are relative to Prokop/models.
    """
    robot = ET.Element("robot", name="turntable_chain")
    for name in ("turntable", "base", TASK_PUZZLE):  # materials, links, joints as they are
        for element in ET.parse(MODELS / f"{name}.urdf").getroot():
            robot.append(element)
    robot.append(ET.Element("link", name=WORLD))
    robot.append(ET.Element("link", name=AXIS))
    robot.append(_joint("world_to_axis", "fixed", WORLD, AXIS,
                        _translation(*model.axis_xy, TABLE_Z)))
    robot.append(_joint(JOINT, "continuous", AXIS, TURNTABLE, np.eye(4)))
    c, s = np.cos(model.base_yaw), np.sin(model.base_yaw)
    T_base = _translation(*model.base_offset, BASE_FOAM)
    T_base[:2, :2] = [[c, -s], [s, c]]
    robot.append(_joint("top_to_base", "fixed", TOP, BASE, T_base))
    robot.append(_joint("mount_to_puzzle", "fixed", MOUNT, PUZZLE, np.eye(4)))
    if path is not None:
        for i, T in enumerate(path.frames):
            robot.append(ET.Element("link", name=checkpoint_name(i)))
            robot.append(_joint(f"puzzle_to_{checkpoint_name(i)}", "fixed",
                                PUZZLE, checkpoint_name(i), T))
    return ET.tostring(robot, encoding="unicode")


class TurntableChain:
    """FK of the turntable chain (lab 2 spatial robot semantics).

    Attributes:
        model: The turntable model the chain was built from.
        path: The checkpoints in the chain, or None.
        urdf: The yourdfpy model of the chain (left in the last FK configuration).
    """

    def __init__(self, model: TurntableModel, path: PuzzlePath | None = None,
                 load_meshes: bool = False) -> None:
        """Builds the chain.

        Args:
            model: The turntable model.
            path: Checkpoints to include; None for none.
            load_meshes: Load the visual / collision meshes too (for drawing or
                collision checks); FK does not need them.
        """
        self.model, self.path = model, path
        self.urdf = yourdfpy.URDF.load(
            io.StringIO(chain_urdf(model, path)), mesh_dir=str(MODELS),
            load_meshes=load_meshes, build_scene_graph=True,
            load_collision_meshes=load_meshes, build_collision_scene_graph=load_meshes)
        self._theta = None

    def _update_fk(self, theta: float) -> None:
        """Propagates the turntable angle into the URDF (skipped if unchanged)."""
        if theta != self._theta:
            self.urdf.update_cfg({JOINT: float(theta)})
            self._theta = theta

    def link_pose(self, link: str, theta: float) -> np.ndarray:
        """Pose of one frame of the chain.

        Args:
            link: Link name (e.g. PUZZLE, BASE, MARKERS[0], checkpoint_name(i)).
            theta: Turntable angle [rad].

        Returns:
            4x4 transform, robot frame.
        """
        self._update_fk(theta)
        return self.urdf.get_transform(link, WORLD)

    def link_poses(self, theta: float) -> dict[str, np.ndarray]:
        """Poses of all frames of the chain.

        Args:
            theta: Turntable angle [rad].

        Returns:
            4x4 transforms in the robot frame, by link name.
        """
        return {link: self.link_pose(link, theta) for link in self.urdf.link_map}

    def puzzle_T(self, theta: float) -> np.ndarray:
        """Pose of the puzzle frame at angle @theta (robot frame)."""
        return self.link_pose(PUZZLE, theta)

    def checkpoint_T(self, i: int, theta: float) -> np.ndarray:
        """Pose of checkpoint @i at angle @theta (robot frame).

        Raises:
            ValueError: If the chain was built without a path.
        """
        if self.path is None:
            raise ValueError("the chain has no checkpoints (built without a path)")
        return self.link_pose(checkpoint_name(i), theta)

    def checkpoint_poses(self, theta: float) -> np.ndarray:
        """Poses of all checkpoints at angle @theta, shape (N, 4, 4), robot frame."""
        n = 0 if self.path is None else len(self.path.frames)
        return np.array([self.checkpoint_T(i, theta) for i in range(n)])

