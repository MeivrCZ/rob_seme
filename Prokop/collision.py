"""Collision checking with the exact meshes of the scene (the environment's
"hard" collisions: bodies touching as they would on the real station).

* ArmChecker: clearance of the CRS93 arm (with the adapter and the hoop) to the
  static work cell, the table and the cage, camera rail and camera of cell.py.
  Used by the tools to check poses before moving the real robot.
* SceneCollider: everything that can touch in the simulated scene: the arm with
  itself, with the cell and the table, and with the turntable, base and puzzle
  at their current placement and spin angle. CRS93Sim monitors it while the
  robot moves.

Example:
    ::

        checker = ArmChecker()
        d, what = checker.clearance(q)             # m, name of the nearest obstacle
        d, what = checker.path_clearance(q0, q1)   # along the joint-space line

        collider = SceneCollider(task)
        hits, d = collider.check(q, placement, turntable_angle)

These are the exact meshes, for the environment. A trajectory planner works with
its own simplified ("soft") collision model built from perceived poses.
"""

from __future__ import annotations

import numpy as np
import trimesh
from cell import TABLE_Z, cell_meshes
from placement import (HOOP_MOUNT, MODELS, Placement, TaskScene, collision_meshes,
                       load_urdf, yaw_matrix)

PATH_STEP = np.deg2rad(1.0)  # rad, max joint change between checked poses
# Robot parts never checked against each other: neighbours whose meshes overlap
# in every configuration, and parts mounted face to face (contact by design).
EXCLUDED_PAIRS = {frozenset(p) for p in (
    ("base_link", "link1"), ("link1", "link2"), ("link4", "link5"),
    ("link6", "adapter"), ("adapter", "hoop"))}
TABLE_PATCH = (0.12, 0.12, 0.004)  # m, table patch drawn at a contact


def arm_parts() -> tuple[object, list[tuple[str, str, trimesh.Trimesh, np.ndarray]]]:
    """Collision meshes of the robot (with the adapter) and the hoop.

    Returns:
        The robot URDF and its parts (link to move with, part name, mesh in the
        link frame, offset from the link frame). Part names are the link names,
        "adapter" and "hoop"; the hoop moves with "adapter_front" at HOOP_MOUNT.
    """
    urdf = load_urdf(MODELS / "crs93.urdf")
    hoop = load_urdf(MODELS / "hoop.urdf")
    parts = []
    for model, attach in ((urdf, None), (hoop, "adapter_front")):
        scene = model.collision_scene
        for node in scene.graph.nodes_geometry:
            link = scene.graph.transforms.parents[node]
            T, geom = scene.graph.get(frame_to=node, frame_from=link)
            mesh = scene.geometry[geom].copy().apply_transform(T)
            name = "hoop" if attach else link
            parts.append((attach or link, name, mesh,
                          HOOP_MOUNT if attach else np.eye(4)))
    return urdf, parts


class ArmChecker:
    """Distance of the arm with the hoop to the table and the cell.

    Attributes:
        urdf: The robot model, left in the last checked configuration.
    """

    def __init__(self) -> None:
        """Loads the robot, hoop and cell meshes."""
        self.urdf, self._parts = arm_parts()
        hoop = load_urdf(MODELS / "hoop.urdf")
        self._hoop_center = hoop.get_transform("hoop_center", hoop.base_link)
        self._arm = trimesh.collision.CollisionManager()
        for _, name, mesh, _ in self._parts:
            self._arm.add_object(name, mesh)
        self._cell = trimesh.collision.CollisionManager()
        for name, mesh in cell_meshes().items():
            self._cell.add_object(name, mesh)

    def _place(self, q: np.ndarray) -> None:
        """Moves the arm meshes to configuration @q [rad]."""
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        for link, name, _, offset in self._parts:
            T = self.urdf.get_transform(link, self.urdf.base_link) @ offset
            self._arm.set_transform(name, T)

    def clearance(self, q: np.ndarray) -> tuple[float, str]:
        """Clearance of the arm in one configuration.

        The robot's base link, which stands in a cut-out of the table, is not
        checked against the table.

        Args:
            q: Joint configuration [rad].

        Returns:
            The minimum distance [m] to the cell and the table (0 if touching) and
            the name of the nearest obstacle ("table" or a cell_meshes name).
        """
        self._place(q)
        d, (_, what) = self._arm.min_distance_other(self._cell, return_names=True)
        for link, name, mesh, offset in self._parts:
            if link == self.urdf.base_link:
                continue
            T = self.urdf.get_transform(link, self.urdf.base_link) @ offset
            z = trimesh.transform_points(mesh.vertices, T)[:, 2].min() - TABLE_Z
            if z < d:
                d, what = z, "table"
        return max(0.0, float(d)), what

    def hoop_center(self, q: np.ndarray) -> np.ndarray:
        """Position of the ring centre of the hoop.

        Args:
            q: Joint configuration [rad].

        Returns:
            The ring centre [m], robot base frame.
        """
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        T = (self.urdf.get_transform("adapter_front", self.urdf.base_link)
             @ HOOP_MOUNT @ self._hoop_center)
        return T[:3, 3]

    def path_clearance(self, q0: np.ndarray, q1: np.ndarray,
                       step: float = PATH_STEP) -> tuple[float, str]:
        """Clearance along the joint-space line from @q0 to @q1.

        That is the path of a single coordinated move.

        Args:
            q0: Start configuration [rad].
            q1: End configuration [rad].
            step: Largest joint change between checked configurations [rad].

        Returns:
            The minimum clearance [m] along the line and the nearest obstacle.
        """
        q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
        n = max(1, int(np.ceil(np.abs(q1 - q0).max() / step)))
        return min((self.clearance(q0 + s * (q1 - q0))
                    for s in np.linspace(0.0, 1.0, n + 1)), key=lambda r: r[0])


class SceneCollider:
    """Exact-mesh collision state of the whole simulated scene.

    Checked: the arm parts (links, adapter, hoop) with each other (except
    EXCLUDED_PAIRS), with the cell and the table (the robot's base link stands in
    a cut-out of the table and is not checked against it), and with the
    turntable, base and puzzle. The task objects are not checked against the
    table or each other (they stand on it, by construction).

    Attributes:
        bodies: Mesh of every body in its own frame, by name (for drawing).
    """

    def __init__(self, task: TaskScene) -> None:
        """Loads the meshes.

        Args:
            task: The task models (turntable, base, puzzle; their frames).
        """
        self._task = task
        self.urdf, self._parts = arm_parts()
        self._arm = trimesh.collision.CollisionManager()
        for _, name, mesh, _ in self._parts:
            self._arm.add_object(name, mesh)
        # Static obstacles. The table is the plane z = TABLE_Z, checked through
        # the lowest vertex of each arm part except the base link (exact for a
        # plane); its slab here is only for drawing.
        self._static = trimesh.collision.CollisionManager()
        self.bodies: dict[str, tuple[trimesh.Trimesh, np.ndarray]] = {}
        self._table_contact = np.zeros(2)  # xy of the last table contact
        for name, mesh in cell_meshes().items():
            self._static.add_object(name, mesh)
            self.bodies[name] = (mesh, np.eye(4))
        # Lowest points of a body are among its convex hull's vertices.
        self._hull_points = {name: mesh.convex_hull.vertices
                             for link, name, mesh, _ in self._parts
                             if link != self.urdf.base_link}
        # Task objects in their group frames (posed by placement and angle).
        self._task_objects = trimesh.collision.CollisionManager()
        self._task_meshes: dict[str, tuple[str, trimesh.Trimesh]] = {}
        for group, model in (("turntable", task.turntable), ("base", task.base),
                             ("puzzle", task.puzzle)):
            for i, mesh in enumerate(collision_meshes(model)):
                name = group if i == 0 else f"{group}_{i}"
                self._task_meshes[name] = (group, mesh)
                self._task_objects.add_object(name, mesh)
        self._poses: dict[str, np.ndarray] = {}

    def task_poses(self, placement: Placement,
                   turntable_angle: float) -> dict[str, np.ndarray]:
        """Poses of the task object groups.

        Args:
            placement: Turntable and base placement.
            turntable_angle: Rotation of the turntable top [rad].

        Returns:
            Pose (4 x 4) of "turntable", "base" and "puzzle".
        """
        T_tt = placement.turntable_T()
        T_base = (T_tt @ self._task.T_top @ yaw_matrix(0, 0, 0, turntable_angle)
                  @ placement.base_T_top())
        return {"turntable": T_tt, "base": T_base,
                "puzzle": T_base @ self._task.T_mount}

    def _place(self, q: np.ndarray, placement: Placement,
               turntable_angle: float) -> None:
        """Moves all bodies to the given state; remembers their poses."""
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        for link, name, _, offset in self._parts:
            T = self.urdf.get_transform(link, self.urdf.base_link) @ offset
            self._arm.set_transform(name, T)
            self._poses[name] = T
        groups = self.task_poses(placement, turntable_angle)
        for name, (group, _) in self._task_meshes.items():
            self._task_objects.set_transform(name, groups[group])
            self._poses[name] = groups[group]

    def body(self, name: str) -> tuple[trimesh.Trimesh, np.ndarray]:
        """A body's mesh (own frame) and its pose in the last checked state.

        Args:
            name: Body name as reported by check().

        Returns:
            Mesh and pose (4 x 4).
        """
        if name == "table":  # a patch of the table around the contact
            T = trimesh.transformations.translation_matrix(
                [*self._table_contact, TABLE_Z - TABLE_PATCH[2] / 2])
            return trimesh.creation.box(TABLE_PATCH), T
        if name in self.bodies:
            return self.bodies[name]
        for _, part, mesh, _ in self._parts:
            if part == name:
                return mesh, self._poses[name]
        return self._task_meshes[name][1], self._poses[name]

    def self_hits(self, q: np.ndarray) -> list[tuple[str, str]]:
        """Colliding pairs of arm parts (EXCLUDED_PAIRS left out).

        Args:
            q: Joint configuration [rad].

        Returns:
            The colliding pairs, empty if none.
        """
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        for link, name, _, offset in self._parts:
            self._arm.set_transform(
                name, self.urdf.get_transform(link, self.urdf.base_link) @ offset)
        _, pairs = self._arm.in_collision_internal(return_names=True)
        return sorted(tuple(sorted(p)) for p in pairs
                      if frozenset(p) not in EXCLUDED_PAIRS)

    def place(self, q: np.ndarray, placement: Placement,
              turntable_angle: float) -> None:
        """Puts the scene into a state for the queries below.

        Args:
            q: Joint configuration [rad].
            placement: Turntable and base placement.
            turntable_angle: Rotation of the turntable top [rad].
        """
        self._place(q, placement, turntable_angle)

    def placed_self_hits(self) -> list[tuple[str, str]]:
        """Colliding pairs of arm parts in the placed state (see self_hits)."""
        _, pairs = self._arm.in_collision_internal(return_names=True)
        return sorted(tuple(sorted(p)) for p in pairs
                      if frozenset(p) not in EXCLUDED_PAIRS)

    def placed_table(self) -> tuple[list[tuple[str, str]], float]:
        """Arm parts touching the table in the placed state.

        Returns:
            The colliding pairs (part, "table") and the clearance to the table [m].
        """
        hits, clearance = [], np.inf
        for name, points in self._hull_points.items():
            world = trimesh.transform_points(points, self._poses[name])
            z = world[:, 2].min()
            clearance = min(clearance, z - TABLE_Z)
            if z <= TABLE_Z:
                hits.append((name, "table"))
                self._table_contact = world[np.argmin(world[:, 2]), :2]
        return hits, float(max(clearance, 0.0))

    def placed_group(self, group: str) -> tuple[list[tuple[str, str]], float]:
        """Arm parts touching a group of obstacles in the placed state.

        Args:
            group: "cell" (cage, rail, camera) or "task" (turntable, base,
                puzzle).

        Returns:
            The colliding pairs (arm part, body) and the clearance [m].
        """
        other = self._static if group == "cell" else self._task_objects
        d = self._arm.min_distance_other(other)
        if d > 0:
            return [], float(d)
        _, pairs = self._arm.in_collision_other(other, return_names=True)
        return sorted(tuple(p) for p in pairs), 0.0

    def check(self, q: np.ndarray, placement: Placement, turntable_angle: float,
              self_collision: bool = True) -> tuple[list[tuple[str, str]], float]:
        """Collisions of the arm in one state of the scene.

        Args:
            q: Joint configuration [rad].
            placement: Turntable and base placement.
            turntable_angle: Rotation of the turntable top [rad].
            self_collision: Also check the arm parts against each other.

        Returns:
            The colliding pairs (arm part, other body), empty if none, and the
            clearance [m] of the arm to the cell, table and task objects (0 if
            touching; self-collision does not count in it).
        """
        self._place(q, placement, turntable_angle)
        hits = self.placed_self_hits() if self_collision else []
        table_hits, clearance = self.placed_table()
        hits += table_hits
        for group in ("cell", "task"):
            group_hits, d = self.placed_group(group)
            hits += group_hits
            clearance = min(clearance, d)
        if hits:
            return sorted(set(hits)), 0.0
        return [], clearance
