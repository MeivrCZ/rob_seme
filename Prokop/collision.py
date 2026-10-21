"""Clearance of the CRS93 arm (with the adapter and the hoop) to the static work
cell: the table and the cage, camera rail and camera of cell.py.

    checker = ArmChecker()
    d, what = checker.clearance(q)              # m, name of the nearest obstacle
    d, what = checker.path_clearance(q0, q1)    # along the joint-space line q0 -> q1

Not checked (yet): self-collision and the turntable with the puzzle.
"""

from __future__ import annotations

import numpy as np
import trimesh
from cell import TABLE_Z, cell_meshes
from placement import HOOP_MOUNT, MODELS, load_urdf

PATH_STEP = np.deg2rad(1.0)  # rad, max joint change between checked poses


class ArmChecker:
    def __init__(self) -> None:
        self.urdf = load_urdf(MODELS / "crs93.urdf")
        hoop = load_urdf(MODELS / "hoop.urdf")
        self._hoop_center = hoop.get_transform("hoop_center", hoop.base_link)
        # Collision meshes in their link frames: (link, name, mesh); the hoop is
        # attached to the adapter front at HOOP_MOUNT.
        self._parts: list[tuple[str, str, trimesh.Trimesh, np.ndarray]] = []
        for urdf, attach in ((self.urdf, None), (hoop, "adapter_front")):
            scene = urdf.collision_scene
            for node in scene.graph.nodes_geometry:
                link = scene.graph.transforms.parents[node]
                T, geom = scene.graph.get(frame_to=node, frame_from=link)
                mesh = scene.geometry[geom].copy().apply_transform(T)
                offset = HOOP_MOUNT if attach else np.eye(4)
                self._parts.append((attach or link, f"{link}/{node}", mesh, offset))
        self._arm = trimesh.collision.CollisionManager()
        for _, name, mesh, _ in self._parts:
            self._arm.add_object(name, mesh)
        self._cell = trimesh.collision.CollisionManager()
        for name, mesh in cell_meshes().items():
            self._cell.add_object(name, mesh)

    def _place(self, q: np.ndarray) -> None:
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        for link, name, _, offset in self._parts:
            T = self.urdf.get_transform(link, self.urdf.base_link) @ offset
            self._arm.set_transform(name, T)

    def clearance(self, q: np.ndarray) -> tuple[float, str]:
        """Minimum distance [m] of the arm in @q to the cell and the table (0 if
        touching) and the name of the nearest obstacle. The robot's base, which
        stands in a cut-out of the table, is not checked against the table."""
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
        """Position [m] of the ring centre of the hoop in configuration @q."""
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        T = (self.urdf.get_transform("adapter_front", self.urdf.base_link)
             @ HOOP_MOUNT @ self._hoop_center)
        return T[:3, 3]

    def path_clearance(self, q0: np.ndarray, q1: np.ndarray,
                       step: float = PATH_STEP) -> tuple[float, str]:
        """Minimum clearance along the joint-space line @q0 -> @q1 (the path of a
        single coordinated move), checked every @step rad of the largest joint
        change."""
        q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
        n = max(1, int(np.ceil(np.abs(q1 - q0).max() / step)))
        return min((self.clearance(q0 + s * (q1 - q0))
                    for s in np.linspace(0.0, 1.0, n + 1)), key=lambda r: r[0])
