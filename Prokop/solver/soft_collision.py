"""The solver's own (soft) collision model: boxes and the table plane.

The exact meshes (collision.py) are the simulator's ground truth; the
algorithm uses this model instead. All bodies are boxes fitted once to the
CAD (tools/fit_soft_boxes.py -> the soft URDFs models/soft/*_soft.urdf: the same
links and joints as the exact models, box collision elements only), every box
contains its piece of the body:

* the arm parts with the hoop (its ring as 36 sector boxes around the hole);
* the cell (cage walls, camera rail, post, camera) and the table (z <= TABLE_Z);
* the task, posed from the PERCEIVED TurntableModel: the turntable (in the
  intersection of two boxes), the base, and the puzzle's square bar.

Tests are dot products only. Box vs box: the largest gap over the 15
separating axes (3 + 3 face normals, 9 edge cross products), a LOWER BOUND on
their distance that is <= 0 exactly when they overlap. Box vs table: the
lowest corner (exact). A body in the intersection of boxes is at least as far
as the farthest of them.

Speed: the boxes form a hierarchy (enclosing boxes of whole parts, the
hoop's handle and ring quarters, the puzzle and its bend). A query tests the
enclosing boxes first and refines only the pairs closer than REFINE; the
result is a lower bound either way (an enclosing box contains its children).

Margins are subtracted from the distances: MARGIN for the arm against
everything, HOOP_PUZZLE_MARGIN (smaller: the ring has ~23 mm around the bar)
for the hoop against the puzzle, SELF_MARGIN between arm parts. Joint
neighbours are never checked against each other (their boxes overlap by
construction; the joint limits keep them apart), nor the pairs in
SELF_EXCLUDED (cannot touch within the joint limits), nor the robot's base
link against the table (it stands in a cut-out).

Example:
    ::

        scene = SoftScene(model)                # contracts.TurntableModel
        c = scene.clearance(q, theta)           # SoftClearance
        c.distance, c.pair, c.self_gap          # m (margins subtracted), names
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cell import TABLE_Z
from placement import HOOP_MOUNT
from solver.contracts import TurntableModel, _fixed_frames

MODELS = Path(__file__).resolve().parents[1] / "models" / "soft"
MARGIN = 0.010  # m, arm against the cell, table and task
HOOP_PUZZLE_MARGIN = 0.003  # m, hoop against the puzzle
SELF_MARGIN = 0.0  # m, between arm parts (the boxes are already larger)
REFINE = 0.02  # m, pairs closer than this (margin subtracted) are refined
# Non-neighbouring parts that cannot touch within the joint limits, not checked
# either: link 4 and link 6 stay >= 7.9 mm apart over the whole q5 x q6 range
# (exact meshes, 2026-11-25), but their boxes meet in most wrist poses.
SELF_EXCLUDED = {frozenset(("link4", "link6"))}
ARM_ORDER = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6",
             "adapter", "hoop")  # joint neighbours are adjacent here


def _origin(element: ET.Element | None) -> np.ndarray:
    """4 x 4 pose of a URDF <origin xyz rpy> (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    T = np.eye(4)
    if element is None:
        return T
    r, p, y = (float(v) for v in element.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = [float(v) for v in element.get("xyz", "0 0 0").split()]
    return T


def read_soft_urdf(name: str) -> tuple[dict[str, list[dict]], list[dict]]:
    """Boxes by link (name, pose T in the link frame, half extents) and the joints."""
    root = ET.parse(MODELS / f"{name}.urdf").getroot()
    boxes = {}
    for link in root.iter("link"):
        items = []
        for c in link.iter("collision"):
            size = np.array([float(v) for v in c.find("geometry/box").get("size").split()])
            items.append({"name": c.get("name"), "T": _origin(c.find("origin")),
                          "half": size / 2})
        boxes[link.get("name")] = items
    joints = [{"name": j.get("name"), "type": j.get("type"),
               "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
               "origin": _origin(j.find("origin")),
               "axis": np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis")
                                                     is not None else "1 0 0").split()])}
              for j in root.iter("joint")]
    return boxes, joints


@functools.cache
def _data() -> dict:
    """The soft model: the robot chain, the arm boxes and the obstacles."""
    robot, chain = read_soft_urdf("crs93_soft")
    hoop, _ = read_soft_urdf("hoop_soft")
    arm = [{"part": link, "link": link, "boxes": items} for link, items in robot.items() if items]
    arm.append({"part": "hoop", "link": "adapter_front",
                "boxes": [dict(b, T=HOOP_MOUNT @ b["T"]) for b in hoop["hoop_link"]]})
    return {"table_z": TABLE_Z, "chain": chain, "arm": arm,
            "turntable": read_soft_urdf("turntable_soft")[0]["turntable_link"],
            "base": read_soft_urdf("base_soft")[0]["base_link"],
            "puzzle": read_soft_urdf("puzzle_D_soft")[0]["puzzle_D_link"],
            "cell": read_soft_urdf("cell_soft")[0]["cell"]}


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation (4 x 4) about a unit axis."""
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    T = np.eye(4)
    T[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return T


def link_frames(q: np.ndarray) -> dict[str, np.ndarray]:
    """Frames of all robot links in the base frame (the URDF chain; q as ctu_crs)."""
    frames = {"base_link": np.eye(4)}
    k = 0
    for j in _data()["chain"]:
        T = frames[j["parent"]] @ np.array(j["origin"])
        if j["type"] == "revolute":
            T = T @ _rot(np.array(j["axis"]), float(q[k]))
            k += 1
        frames[j["child"]] = T
    return frames


@dataclass
class Boxes:
    """A set of boxes: centres (n, 3), axes (n, 3, 3) as columns, half extents (n, 3)."""

    c: np.ndarray
    R: np.ndarray
    h: np.ndarray

    @staticmethod
    def of(items: list[dict], T: np.ndarray | list[np.ndarray]) -> "Boxes":
        """Boxes of json items posed by T (one pose for all, or one each)."""
        Ts = [T] * len(items) if isinstance(T, np.ndarray) else T
        M = np.array([Ti @ np.array(b["T"]) for Ti, b in zip(Ts, items)])
        return Boxes(M[:, :3, 3], M[:, :3, :3], np.array([b["half"] for b in items]))

    def __len__(self) -> int:
        return len(self.c)


def box_gaps(A: Boxes, B: Boxes, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Separating-axis lower bounds of the distance of box pairs (A[ia], B[ib]) [m].

    <= 0 exactly when the boxes overlap; otherwise at most their distance.
    """
    Ra, Rb = A.R[ia], B.R[ib]  # (P, 3, 3), axes as columns
    ha, hb = A.h[ia], B.h[ib]
    d = B.c[ib] - A.c[ia]  # (P, 3)
    at, bt = Ra.transpose(0, 2, 1), Rb.transpose(0, 2, 1)  # axes as rows
    cross = np.cross(at[:, :, None, :], bt[:, None, :, :]).reshape(-1, 9, 3)
    L = np.concatenate([at, bt, cross], axis=1)  # (P, 15, 3)
    norm = np.sqrt((L * L).sum(axis=2))
    valid = norm > 1e-9
    L = L / np.where(valid, norm, 1.0)[:, :, None]
    ra = (np.abs(L @ Ra) * ha[:, None, :]).sum(axis=2)  # (P, 15)
    rb = (np.abs(L @ Rb) * hb[:, None, :]).sum(axis=2)
    gap = np.abs((L @ d[:, :, None])[:, :, 0]) - ra - rb
    return np.where(valid, gap, -np.inf).max(axis=1)


def lowest_z(A: Boxes) -> np.ndarray:
    """Lowest point of every box [m] (exact)."""
    return A.c[:, 2] - np.einsum("nk,nk->n", np.abs(A.R[:, 2, :]), A.h)


@dataclass
class SoftClearance:
    """Result of SoftScene.clearance.

    Attributes:
        distance: Least distance of the arm to the cell, table and task with
            the margins subtracted [m] (<= 0: too close or touching).
        pair: The pair it belongs to (arm box, obstacle box).
        self_gap: Least gap between non-neighbouring arm parts, margin
            subtracted [m] (<= 0: they may touch).
        self_pair: Its pair.
    """

    distance: float
    pair: tuple[str, str]
    self_gap: float
    self_pair: tuple[str, str]

    @property
    def free(self) -> bool:
        return self.distance > 0 and self.self_gap > 0


def enclosing_box(T: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A box (pose, half extents) containing boxes given by poses (k, 4, 4), halves (k, 3)."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = np.concatenate([Ti[:3, 3] + (signs * hi) @ Ti[:3, :3].T for Ti, hi in zip(T, half)])
    centre = corners.mean(axis=0)
    R = np.linalg.svd(corners - centre)[2].T  # principal axes as columns
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    local = (corners - centre) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, centre + R @ ((lo + hi) / 2)
    return out, (hi - lo) / 2 + 1e-6


class _Tree:
    """Boxes in a hierarchy: leaves first (the model's boxes), then enclosing nodes."""

    def __init__(self) -> None:
        self.local: list[np.ndarray] = []  # pose in its frame
        self.half: list[np.ndarray] = []
        self.frame: list[int] = []  # index of the frame it moves with
        self.children: list[list[int]] = []
        self.name: list[str] = []
        self.tag: list = []  # arm: part index; obstacles: group name

    def add(self, T, half, frame, name, tag, children=()) -> int:
        self.local.append(np.asarray(T, float))
        self.half.append(np.asarray(half, float))
        self.frame.append(frame)
        self.children.append(list(children))
        self.name.append(name)
        self.tag.append(tag)
        return len(self.name) - 1

    def group(self, nodes: list[int], name: str) -> int:
        """Adds a node enclosing the given nodes (all in one frame); one node: itself."""
        if len(nodes) == 1:
            return nodes[0]
        T, half = enclosing_box(np.array([self.local[k] for k in nodes]),
                                np.array([self.half[k] for k in nodes]))
        return self.add(T, half, self.frame[nodes[0]], name, self.tag[nodes[0]], nodes)

    def freeze(self) -> None:
        self.local, self.half = np.array(self.local), np.array(self.half)
        self.frame, self.name = np.array(self.frame), np.array(self.name)
        self.internal = np.array([len(c) > 0 for c in self.children])
        self.size = np.linalg.norm(self.half, axis=1)


class SoftScene:
    """The soft model of the scene for one (perceived) turntable model."""

    def __init__(self, model: TurntableModel, margin: float = MARGIN,
                 hoop_puzzle_margin: float = HOOP_PUZZLE_MARGIN,
                 self_margin: float = SELF_MARGIN) -> None:
        data = _data()
        self.model = model
        self.table_z = data["table_z"]
        self.margin, self.hoop_puzzle_margin = margin, hoop_puzzle_margin
        self.self_margin = self_margin
        # --- the arm: leaves, then per part its groups and its root ---
        arm = self.arm = _Tree()
        self._links = sorted({part["link"] for part in data["arm"]})
        leaves_of = {}
        for part in data["arm"]:
            idx = ARM_ORDER.index(part["part"])
            leaves_of[part["part"]] = [arm.add(b["T"], b["half"], self._links.index(part["link"]),
                                               b["name"], idx) for b in part["boxes"]]
        self._arm_leaves = np.arange(len(arm.name))
        roots = []
        for part in data["arm"]:
            leaves = leaves_of[part["part"]]
            if part["part"] == "hoop":
                names = [arm.name[k] for k in leaves]
                handle = [k for k, nm in zip(leaves, names) if nm.startswith("hoop_handle")]
                ring = [k for k, nm in zip(leaves, names) if nm.startswith("hoop_ring")]
                quarters = [arm.group(ring[q * len(ring) // 4:(q + 1) * len(ring) // 4],
                                      f"hoop_ring_quarter_{q}") for q in range(4)]
                roots.append(arm.group([arm.group(handle, "hoop_handle")] + quarters, "hoop"))
            else:
                roots.append(arm.group(leaves, part["part"]))
        arm.freeze()
        self._arm_roots = np.array(roots)
        part_of = np.array(arm.tag)
        self._arm_part = part_of
        self._arm_is_hoop = part_of == ARM_ORDER.index("hoop")
        self._table_boxes = self._arm_leaves[part_of[self._arm_leaves]
                                             != ARM_ORDER.index("base_link")]
        ri, rj = np.triu_indices(len(roots), 1)
        pr = part_of[self._arm_roots]
        keep = np.abs(pr[ri] - pr[rj]) > 1  # no joint neighbours
        keep &= np.array([frozenset((ARM_ORDER[a], ARM_ORDER[b])) not in SELF_EXCLUDED
                          for a, b in zip(pr[ri], pr[rj])])
        self._self_roots = (self._arm_roots[ri[keep]], self._arm_roots[rj[keep]])
        # --- the obstacles: cell, base, puzzle (with its bend as a group) ---
        obs = self.obs = _Tree()
        self._groups = ("cell", "base", "puzzle")
        roots = [obs.add(b["T"], b["half"], 0, b["name"], "cell") for b in data["cell"]]
        roots += [obs.add(b["T"], b["half"], 1, b["name"], "base") for b in data["base"]]
        puzzle = [obs.add(b["T"], b["half"], 2, b["name"], "puzzle") for b in data["puzzle"]]
        self._obs_leaves = np.arange(len(obs.name))
        bend = [k for k in puzzle if obs.name[k].startswith("puzzle_bend")]
        rest = [k for k in puzzle if k not in bend]
        roots.append(obs.group(rest + [obs.group(bend, "puzzle_bend")], "puzzle"))
        obs.freeze()
        self._obs_roots = np.array(roots)
        self._obs_is_puzzle = np.array(obs.tag) == "puzzle"
        self._turntable = data["turntable"]
        self._T_top_inv = np.linalg.inv(_fixed_frames()["T_top"])
        self._theta: float | None = None
        ia, ib = np.meshgrid(self._arm_roots, self._obs_roots, indexing="ij")
        self._env_roots = (ia.ravel(), ib.ravel())

    # --- posing ---

    def _pose_obstacles(self, theta: float) -> None:
        if self._theta == theta:
            return
        frames = np.array([np.eye(4), self.model.base_T(theta), self.model.puzzle_T(theta)])
        M = frames[self.obs.frame] @ self.obs.local
        self._obs_boxes = Boxes(M[:, :3, 3], M[:, :3, :3], self.obs.half)
        tt = self.model.top_T(theta) @ self._T_top_inv
        self._tt_boxes = Boxes.of(self._turntable, tt)
        self._theta = theta

    def arm_boxes(self, q: np.ndarray) -> Boxes:
        """All arm nodes (leaves first) in configuration q."""
        frames = link_frames(np.asarray(q, dtype=float))
        F = np.array([frames[link] for link in self._links])
        M = F[self.arm.frame] @ self.arm.local
        return Boxes(M[:, :3, 3], M[:, :3, :3], self.arm.half)

    def obstacle_boxes(self, theta: float) -> Boxes:
        """All obstacle nodes (leaves first) with the turntable at theta."""
        self._pose_obstacles(theta)
        return self._obs_boxes

    SWEPT_SIDES = 4  # boxes per swept cylinder, rotated by 90 / SWEPT_SIDES deg (a 16-gon)

    def _swept_boxes(self) -> tuple[Boxes, np.ndarray, list[str]]:
        """Boxes enclosing the volume the task sweeps in a full turn (cached).

        Every task box (turntable, base, puzzle) sweeps a cylinder about the
        turntable axis: radius = its farthest corner, height = its own. Each
        cylinder lies in the intersection of SWEPT_SIDES boxes rotated against
        each other. Returns the boxes (SWEPT_SIDES per cylinder, consecutive),
        which cylinders are the puzzle's, and the cylinders' names.
        """
        if getattr(self, "_swept", None) is not None:
            return self._swept
        self._pose_obstacles(0.0)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        task = np.flatnonzero(np.array(self.obs.tag) != "cell")
        task = task[task < len(self._obs_leaves)]  # the leaf boxes only
        sources = [(self._obs_boxes, k, str(self.obs.name[k]), self.obs.tag[k] == "puzzle")
                   for k in task]
        sources += [(self._tt_boxes, k, "turntable", False) for k in range(len(self._tt_boxes))]
        axis = np.asarray(self.model.axis_xy, float)
        c, R, h, is_puzzle, names = [], [], [], [], []
        for boxes, k, name, puzzle in sources:
            corners = boxes.c[k] + (signs * boxes.h[k]) @ boxes.R[k].T
            radius = float(np.linalg.norm(corners[:, :2] - axis, axis=1).max())
            z0, z1 = corners[:, 2].min(), corners[:, 2].max()
            for side in range(self.SWEPT_SIDES):
                a = side * (np.pi / 2) / self.SWEPT_SIDES
                c.append([axis[0], axis[1], (z0 + z1) / 2])
                R.append([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                h.append([radius, radius, (z1 - z0) / 2])
            is_puzzle.append(puzzle)
            names.append(f"{name} (swept)")
        self._swept = (Boxes(np.array(c), np.array(R), np.array(h)), np.array(is_puzzle), names)
        return self._swept

    def swept_clearance(self, q: np.ndarray) -> tuple[float, tuple[str, str]]:
        """Least distance of the arm to the volume the task sweeps in a full turn.

        For a pose the robot holds while the turntable turns (waiting for the
        puzzle to come round). Margins as in clearance(); a lower bound.

        Returns:
            Distance [m] (<= 0: too close) and the pair (arm box, swept body).
        """
        arm = self.arm_boxes(q)
        swept, is_puzzle, names = self._swept_boxes()
        leaves = self._arm_leaves
        k = self.SWEPT_SIDES
        n_cyl = len(names)
        ia = np.repeat(leaves, n_cyl * k)
        ib = np.tile(np.arange(n_cyl * k), len(leaves))
        gaps = box_gaps(arm, swept, ia, ib).reshape(len(leaves), n_cyl, k).max(axis=2)
        margin = np.where(self._arm_is_hoop[leaves][:, None] & is_puzzle[None, :],
                          self.hoop_puzzle_margin, self.margin)
        gaps = gaps - margin
        a, b = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
        return float(gaps[a, b]), (str(self.arm.name[leaves[a]]), names[b])

    def task_reach(self, theta: float) -> float:
        """Largest distance of the turntable, base and puzzle boxes from the turntable axis [m].

        With the turntable at omega, no task point moves faster than omega times this.
        """
        self._pose_obstacles(theta)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        reach = 0.0
        task = np.flatnonzero(np.array(self.obs.tag) != "cell")
        for b, idx in ((self._obs_boxes, task),
                       (self._tt_boxes, np.arange(len(self._tt_boxes)))):
            corners = b.c[idx, None] + np.einsum("nij,nkj->nki", b.R[idx],
                                                 signs[None] * b.h[idx, None])
            reach = max(reach, float(np.linalg.norm(corners[..., :2] - self.model.axis_xy,
                                                    axis=2).max()))
        return reach

    def lever_arms(self, q: np.ndarray) -> np.ndarray:
        """Per joint, the largest distance of any box it moves from its axis point [m], (6,).

        A joint change dq_k moves no point of the arm further than r_k |dq_k|, so
        sum_k r_k |dq_k| bounds the motion of every arm point (for small changes;
        callers keep a slack). Uses the distance to the axis POINT, which is at
        least the distance to the axis line.
        """
        q = np.asarray(q, dtype=float)
        frames = link_frames(q)
        leaves = self._arm_leaves
        M = np.array([frames[link] for link in self._links])[self.arm.frame[leaves]] \
            @ self.arm.local[leaves]
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        local = signs[None] * self.arm.half[leaves][:, None]  # (n, 8, 3)
        corners = M[:, None, :3, 3] + np.einsum("nij,nkj->nki", M[:, :3, :3], local)
        part = self._arm_part[leaves]
        r = np.zeros(6)
        for k in range(1, 7):  # joint k moves link k and everything after it
            moved = corners[part >= k].reshape(-1, 3)
            r[k - 1] = np.linalg.norm(moved - frames[f"link{k}"][:3, 3], axis=1).max()
        return r

    # --- hierarchical search ---

    @staticmethod
    def _descend(A: Boxes, ta: _Tree, B: Boxes, tb: _Tree, i: np.ndarray, j: np.ndarray,
                 margin) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gaps (margin subtracted) of node pairs, refining those closer than REFINE.

        Returns:
            The final gaps and their node pairs (leaves, or nodes left unrefined).
        """
        out_g, out_i, out_j = [], [], []
        while len(i):
            g = box_gaps(A, B, i, j) - margin(i, j)
            split = (g < REFINE) & (ta.internal[i] | tb.internal[j])
            out_g.append(g[~split]), out_i.append(i[~split]), out_j.append(j[~split])
            ni, nj = [], []
            for a, b in zip(i[split], j[split]):
                if ta.internal[a] and (not tb.internal[b] or ta.size[a] >= tb.size[b]):
                    ni += ta.children[a]
                    nj += [b] * len(ta.children[a])
                else:
                    ni += [a] * len(tb.children[b])
                    nj += tb.children[b]
            i, j = np.array(ni, dtype=int), np.array(nj, dtype=int)
        return np.concatenate(out_g), np.concatenate(out_i), np.concatenate(out_j)

    def _turntable_gaps(self, arm: Boxes) -> tuple[np.ndarray, np.ndarray]:
        """Arm nodes vs the turntable (inside both its boxes: at least the larger gap)."""
        out_g, out_i = [], []
        i = self._arm_roots
        k = len(self._tt_boxes)
        while len(i):
            ii = np.repeat(i, k)
            jj = np.tile(np.arange(k), len(i))
            g = box_gaps(arm, self._tt_boxes, ii, jj).reshape(-1, k).max(axis=1) - self.margin
            split = (g < REFINE) & self.arm.internal[i]
            out_g.append(g[~split]), out_i.append(i[~split])
            i = np.array([c for a in i[split] for c in self.arm.children[a]], dtype=int)
        return np.concatenate(out_g), np.concatenate(out_i)

    # --- queries ---

    def clearance(self, q: np.ndarray, theta: float) -> SoftClearance:
        """Distances of the arm in configuration q, turntable at theta (class doc)."""
        arm = self.arm_boxes(q)
        self._pose_obstacles(theta)
        best, pair = np.inf, ("", "")
        tb = self._table_boxes
        table = lowest_z(Boxes(arm.c[tb], arm.R[tb], arm.h[tb])) - self.table_z - self.margin
        k = int(np.argmin(table))
        best, pair = float(table[k]), (self.arm.name[tb[k]], "table")

        def env_margin(i, j):
            return np.where(self._arm_is_hoop[i] & self._obs_is_puzzle[j],
                            self.hoop_puzzle_margin, self.margin)

        g, i, j = self._descend(arm, self.arm, self._obs_boxes, self.obs, *self._env_roots,
                                env_margin)
        k = int(np.argmin(g))
        if g[k] < best:
            best, pair = float(g[k]), (self.arm.name[i[k]], self.obs.name[j[k]])
        g, i = self._turntable_gaps(arm)
        k = int(np.argmin(g))
        if g[k] < best:
            best, pair = float(g[k]), (self.arm.name[i[k]], "turntable")
        g, i, j = self._descend(arm, self.arm, arm, self.arm, *self._self_roots,
                                lambda i, j: self.self_margin)
        k = int(np.argmin(g))
        return SoftClearance(best, (str(pair[0]), str(pair[1])), float(g[k]),
                             (str(self.arm.name[i[k]]), str(self.arm.name[j[k]])))

    def group_distances(self, q: np.ndarray, theta: float) -> dict[str, float]:
        """Distances by obstacle group, all leaf pairs, NO margins: table, cell, task, self."""
        arm = self.arm_boxes(q)
        self._pose_obstacles(theta)
        tb = self._table_boxes
        table = float((lowest_z(Boxes(arm.c[tb], arm.R[tb], arm.h[tb])) - self.table_z).min())
        ia, ib = (x.ravel() for x in np.meshgrid(self._arm_leaves, self._obs_leaves, indexing="ij"))
        env = box_gaps(arm, self._obs_boxes, ia, ib)
        cell = np.array(self.obs.tag)[ib] == "cell"
        k = len(self._tt_boxes)
        tt = box_gaps(arm, self._tt_boxes, np.repeat(self._arm_leaves, k),
                      np.tile(np.arange(k), len(self._arm_leaves))).reshape(-1, k).max(axis=1)
        si, sj = np.triu_indices(len(self._arm_leaves), 1)
        part = self._arm_part
        keep = np.abs(part[si] - part[sj]) > 1
        keep &= np.array([frozenset((ARM_ORDER[a], ARM_ORDER[b])) not in SELF_EXCLUDED
                          for a, b in zip(part[si], part[sj])])
        selfg = box_gaps(arm, arm, si[keep], sj[keep])
        return {"table": table, "cell": float(env[cell].min()),
                "task": float(min(env[~cell].min(), tt.min())), "self": float(selfg.min())}
