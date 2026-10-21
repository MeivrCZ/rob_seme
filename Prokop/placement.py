"""Task models and random placement: the turntable on the table, the base with the
puzzle on the turntable.

Placement rules (assignment + team decisions):

* The turntable with the base and puzzle must not touch the robot (with its
  adapter and the hoop at its nominal mount) in q_home, for any rotation of the
  turntable: MIN_CLEARANCE is kept to the volume swept by the spinning task, a
  solid of revolution.
* The base may overhang the turntable but must not tip over: the centre of mass
  of base + puzzle + magnets has to project into the support region, the
  intersection of the base footprint (125 x 125 mm) and the turntable top disc
  (both convex, so the intersection is convex), with STABILITY_MARGIN.
* The whole turntable stands inside the cage; a "viable" placement has its
  centre in the work area seen by the camera.

All frames are in the ctu_crs robot base frame [m]. The cell geometry (table,
cage, work area) is defined in cell.py and re-exported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from cell import CAGE_XY, TABLE_Z, WORK_AREA_XY, in_cage  # noqa: F401 (re-exported)
from ctu_crs import CRS93

MODELS = Path(__file__).resolve().parent / "models"  # URDFs and their meshes

# Table, cage and work area (the part the camera sees): see cell.py. Random
# placement puts the turntable inside the cage, the viable placement in the work
# area (random_placement(rng, WORK_AREA_XY)).
MIN_CLEARANCE = 0.01  # m, to the robot in q_home
STABILITY_MARGIN = 0.005  # m, centre of mass from the edge of the support region
BASE_HALF = 0.0625  # m, half side of the base plate footprint
BASE_FOAM = 0.025  # m, foam pad under the base plate (part of base.urdf)
FOAM_DENSITY = 30.0  # kg/m^3, ASSUMED (PU foam)
# Mass model for the centre of mass: printed parts are PETG (1.27 g/cm^3) with
# 3 perimeters + 10-20 % infill -> ASSUMED effective fill 0.5; magnets in base and
# puzzle foot: 3 + 3 neodymium discs 10 x 3 mm (7.5 g/cm^3) under the groove.
PRINT_DENSITY = 1270.0 * 0.5  # kg/m^3
MAGNET_MASS = 7500.0 * np.pi * 0.005**2 * 0.003  # kg, one magnet
MAGNETS_XY = np.array([[-0.015, 0.0], [0.0, -0.015], [0.015, 0.0]])  # base frame
TASK_PUZZLE = "puzzle_D"
# Nominal hoop pose in the adapter_front frame (see view_models.py).
HOOP_MOUNT = np.eye(4)


def load_urdf(path: Path) -> yourdfpy.URDF:
    """Loads a URDF with its visual and collision meshes.

    Mesh paths in the URDFs are relative ("meshes/x.stl") and resolved against the
    URDF's own directory.

    Args:
        path: The URDF file.

    Returns:
        The model, with a collision scene graph.
    """
    return yourdfpy.URDF.load(path, mesh_dir=str(path.parent),
                              build_collision_scene_graph=True,
                              load_collision_meshes=True)


def collision_meshes(urdf: yourdfpy.URDF, T_world: np.ndarray = np.eye(4)
                     ) -> list[trimesh.Trimesh]:
    """Collision meshes of a URDF in its current configuration.

    Args:
        urdf: The model (loaded with load_urdf).
        T_world: Pose of the model's base link in the world frame.

    Returns:
        Copies of the collision meshes in the world frame.
    """
    scene = urdf.collision_scene
    meshes = []
    for node in scene.graph.nodes_geometry:
        T, geom = scene.graph[node]
        meshes.append(scene.geometry[geom].copy().apply_transform(T_world @ T))
    return meshes


def yaw_matrix(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """Builds a transform from a translation and a rotation about z.

    Args:
        x: Translation x [m].
        y: Translation y [m].
        z: Translation z [m].
        yaw: Rotation about z [rad].

    Returns:
        The 4x4 homogeneous transform.
    """
    T = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    T[:3, 3] = x, y, z
    return T


@dataclass
class Placement:
    """Where the task stands.

    Attributes:
        turntable_xy: Turntable centre on the table, robot base frame [m].
        base_xy: Base centre relative to the turntable top centre [m].
        base_yaw: Base rotation about the turntable axis [rad].
    """

    turntable_xy: np.ndarray
    base_xy: np.ndarray
    base_yaw: float

    def turntable_T(self) -> np.ndarray:
        """Returns the pose of the turntable origin (bottom centre) on the table."""
        return yaw_matrix(*self.turntable_xy, TABLE_Z, 0.0)

    def base_T_top(self) -> np.ndarray:
        """Returns the pose of the base in the turntable top frame (on its foam)."""
        return yaw_matrix(*self.base_xy, BASE_FOAM, self.base_yaw)


class TaskScene:
    """Models of the task and of the robot in q_home, for placement checks.

    Attributes:
        turntable: Turntable model.
        base: ArUco base model (with its foam).
        puzzle: Puzzle model (TASK_PUZZLE).
        T_top: Turntable top frame in the turntable frame.
        T_mount: Puzzle mount frame in the base frame.
        turntable_radius: Radius of the turntable [m].
        com_xy: Centre of mass of base + puzzle + magnets, xy in the base frame [m].
        robot_home: Collision manager of the robot with the hoop in q_home.
    """

    def __init__(self) -> None:
        """Loads the models."""
        self.turntable = load_urdf(MODELS / "turntable.urdf")
        self.base = load_urdf(MODELS / "base.urdf")
        self.puzzle = load_urdf(MODELS / f"{TASK_PUZZLE}.urdf")
        self.T_top = self.turntable.get_transform("top", self.turntable.base_link)
        self.T_mount = self.base.get_transform("puzzle_mount", self.base.base_link)
        cyl = self.turntable.link_map["turntable_link"].collisions[0].geometry.cylinder
        self.turntable_radius = cyl.radius
        self.com_xy = self._com_xy()

        robot = load_urdf(MODELS / "crs93.urdf")
        robot.update_cfg(CRS93(tty_dev=None).q_home)
        hoop = load_urdf(MODELS / "hoop.urdf")
        T_hoop = robot.get_transform("adapter_front", robot.base_link) @ HOOP_MOUNT
        self.robot_home = trimesh.collision.CollisionManager()
        for i, m in enumerate(collision_meshes(robot) + collision_meshes(hoop, T_hoop)):
            self.robot_home.add_object(f"robot{i}", m)

    def _com_xy(self) -> np.ndarray:
        """Computes the centre of mass of base + puzzle + magnets.

        Returns:
            Its xy in the base frame [m].
        """
        parts = [(m, np.eye(4)) for m in collision_meshes(self.base)]
        parts += [(m, self.T_mount) for m in collision_meshes(self.puzzle)]
        masses, coms = [], []
        for m, T in parts:
            foam = m.bounds[1, 2] <= 1e-9  # the only part below the base plate
            masses.append((FOAM_DENSITY if foam else PRINT_DENSITY) * m.volume)
            coms.append(trimesh.transform_points([m.center_mass], T)[0, :2])
        masses += [2 * MAGNET_MASS] * len(MAGNETS_XY)  # base + puzzle magnet pairs
        coms += list(MAGNETS_XY)
        return np.average(coms, axis=0, weights=masses)

    def stability_margin(self, p: Placement) -> float:
        """Tip-over margin of a placement.

        Args:
            p: The placement.

        Returns:
            Distance [m] of the centre of mass from the edge of the support region
            (base footprint intersected with the turntable top); negative if the
            base tips over.
        """
        com_base = self.com_xy
        com_top = trimesh.transform_points([[*com_base, 0.0]], p.base_T_top())[0, :2]
        to_square = BASE_HALF - np.abs(com_base).max()
        to_disc = self.turntable_radius - np.linalg.norm(com_top)
        return min(to_square, to_disc)

    def task_meshes(self, p: Placement) -> list[trimesh.Trimesh]:
        """Collision meshes of the turntable, base and puzzle.

        Args:
            p: The placement.

        Returns:
            The meshes in the world frame.
        """
        T_tt = p.turntable_T()
        T_base = T_tt @ self.T_top @ p.base_T_top()
        return (collision_meshes(self.turntable, T_tt)
                + collision_meshes(self.base, T_base)
                + collision_meshes(self.puzzle, T_base @ self.T_mount))

    def swept_task_mesh(self, p: Placement, dz: float = 0.002) -> trimesh.Trimesh:
        """Volume swept by the turntable, base and puzzle during a full turn.

        A solid of revolution of the largest radius from the turntable axis per
        height slab (sampled on the surfaces, padded by the slab height).

        Args:
            p: The placement.
            dz: Slab height [m].

        Returns:
            The swept volume as a closed mesh in the world frame.
        """
        T_tt = p.turntable_T()
        local = [m.copy().apply_transform(np.linalg.inv(T_tt))
                 for m in self.task_meshes(p)]
        pts = np.vstack([np.vstack([m.vertices, m.sample(20000, seed=0)])
                         for m in local])
        r, z = np.linalg.norm(pts[:, :2], axis=1), pts[:, 2]
        edges = np.arange(z.min(), z.max() + dz, dz)
        slab = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
        r_max = np.zeros(len(edges) - 1)
        np.maximum.at(r_max, slab, r)
        # Pad, circumscribe the 64-gon of revolve() and round up to 0.1 mm (nearly
        # equal radii of neighbouring slabs would otherwise produce sliver faces).
        sections = 64
        r_max = np.ceil((r_max + dz) / np.cos(np.pi / sections) / 1e-4) * 1e-4
        profile = [[0.0, edges[0]]]
        for i, rm in enumerate(r_max):  # staircase profile (r, z), steps only
            if i == 0 or rm != r_max[i - 1]:
                profile += [[rm, edges[i]]]
            if i == len(r_max) - 1 or rm != r_max[i + 1]:
                profile += [[rm, edges[i + 1]]]
        profile.append([0.0, edges[-1]])
        swept = trimesh.creation.revolve(np.array(profile), sections=sections)
        if swept.volume < 0:
            swept.invert()
        return swept.apply_transform(T_tt)

    def clearance_to_robot(self, p: Placement) -> float:
        """Clearance of the spinning task to the robot in q_home.

        Args:
            p: The placement.

        Returns:
            Minimum distance [m] (0 if touching).
        """
        task = trimesh.collision.CollisionManager()
        task.add_object("swept", self.swept_task_mesh(p))
        return max(0.0, self.robot_home.min_distance_other(task))

    def random_placement(self, rng: np.random.Generator, area: np.ndarray = CAGE_XY,
                         max_tries: int = 1000) -> Placement:
        """Draws a uniformly random placement satisfying the rules of the module.

        Args:
            rng: Random generator.
            area: Region [[x_min, x_max], [y_min, y_max]] of the turntable centre:
                the cage by default, WORK_AREA_XY for a placement the camera
                sees. The whole turntable stays inside the cage walls in any case.
            max_tries: Number of draws before giving up.

        Returns:
            The placement.

        Raises:
            RuntimeError: If no draw satisfies the rules.
        """
        # Base centre within the disc where any part of the base can still touch it.
        reach = self.turntable_radius + BASE_HALF * np.sqrt(2)
        for _ in range(max_tries):
            r, phi = reach * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            p = Placement(rng.uniform(area[:, 0], area[:, 1]),
                          r * np.array([np.cos(phi), np.sin(phi)]),
                          rng.uniform(0, 2 * np.pi))
            if (in_cage(p.turntable_xy, self.turntable_radius)
                    and self.stability_margin(p) >= STABILITY_MARGIN
                    and self.clearance_to_robot(p) >= MIN_CLEARANCE):
                return p
        raise RuntimeError(f"no valid placement found in {max_tries} tries")
