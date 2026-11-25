"""Fits the solver's soft collision model (boxes) to the models; writes models/soft/*_soft.urdf.

The planner never uses the exact meshes (those are the simulator's ground
truth). Its own model, solver.soft_collision, consists of boxes and the table
plane, fitted here once:

* Arm parts: one oriented bounding box per part in its link frame; parts in
  SPLIT are cut into slabs along their longest axis first (a box per slab).
* Hoop: the handle (everything beyond the hole, y > ring centre + hole
  radius) as HANDLE_SLABS boxes, the ring as RING_SECTORS boxes, each covering an
  angular sector of the annulus in its own radial/tangential frame.
* Turntable: two boxes rotated 45 deg against each other; the body lies in
  their INTERSECTION (an octagonal prism around the disc).
* Base: one box. Puzzle: the square bar from solver.wire's centreline (one
  box per straight segment, BEND_PIECES boxes for the R50 bend) and its foot.
* Cell: the cage walls, the camera rail, its post and the camera (boxes).

Every box contains its piece of the mesh (a box is convex, so containing the
vertices of the piece suffices; cut pieces include the cut points). The
validation (experiments/validate_soft_collision.py) checks that.

Output: a soft URDF for each model, in models/soft/ (crs93_soft, hoop_soft, base_soft,
turntable_soft, puzzle_D_soft) with the same links and joints but no meshes:
every link's collision geometry is replaced by its boxes (<collision
name="..."><origin/><geometry><box/></geometry>). The exact models stay the
simulator's ground truth; these are read by solver.soft_collision only. The
cell (cell.py) gets cell_soft.urdf, one link "cell" in the robot base frame.
The turntable's two boxes are named intersect_*: the body lies in their
intersection (a URDF cannot say that).

    uv run python Prokop/tools/fit_soft_boxes.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

PROKOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROKOP))

from cell import cell_meshes  # noqa: E402
from collision import arm_parts  # noqa: E402
from placement import TaskScene, collision_meshes, load_urdf, MODELS  # noqa: E402
from solver.wire import ARC_CENTRE, ARC_RADIUS, BAR_HALF, OPEN_END_X, SEGMENTS  # noqa: E402

MODEL_DIR = PROKOP / "models" / "soft"  # a subfolder: the viewer's gallery and the model tests load models/*.urdf
SPLIT = {"link1": 2, "link2": 2, "link4": 3}  # parts cut into this many slabs
HANDLE_SLABS = 3  # the hoop handle, likewise
RING_SECTORS = 36
BEND_PIECES = 6
PAD = 1e-4  # m, added to every half extent (numerical safety)


def box(T: np.ndarray, half: np.ndarray, name: str) -> dict:
    """A box: pose of its centre (axes = columns) in its frame, half extents."""
    return {"name": name, "T": np.asarray(T).tolist(),
            "half": (np.asarray(half) + PAD).tolist()}


def aabb_in(points: np.ndarray, R: np.ndarray, name: str) -> dict:
    """Box with the axes R (columns) around points (frame coordinates)."""
    local = points @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, R @ ((lo + hi) / 2)
    return box(T, (hi - lo) / 2, name)


def obb(mesh: trimesh.Trimesh, name: str) -> dict:
    """Oriented bounding box of a mesh piece."""
    b = mesh.bounding_box_oriented
    return box(b.primitive.transform, np.asarray(b.primitive.extents) / 2, name)


def slabs(mesh: trimesh.Trimesh, n: int) -> list[trimesh.Trimesh]:
    """Cuts a mesh into n slabs along its oriented box's longest axis."""
    b = mesh.bounding_box_oriented
    axis_index = int(np.argmax(b.primitive.extents))
    u = b.primitive.transform[:3, axis_index]
    c = b.primitive.transform[:3, 3]
    L = b.primitive.extents[axis_index]
    cuts = [c + u * (-L / 2 + k * L / n) for k in range(1, n)]
    pieces, rest = [], mesh
    for p in cuts:
        pieces.append(trimesh.intersections.slice_mesh_plane(rest, -u, p))
        rest = trimesh.intersections.slice_mesh_plane(rest, u, p)
    return pieces + [rest]


def hoop_boxes(mesh: trimesh.Trimesh) -> list[dict]:
    """Handle box + ring sector boxes (hoop_link frame)."""
    hoop = load_urdf(MODELS / "hoop.urdf")
    centre = hoop.get_transform("hoop_center", "hoop_link")[:3, 3]
    v = mesh.vertices - centre
    hole = float(np.min(np.linalg.norm(v[:, :2], axis=1)))
    cut = centre + np.array([0.0, hole, 0.0])
    handle = trimesh.intersections.slice_mesh_plane(mesh, [0, 1, 0], cut)
    ring = trimesh.intersections.slice_mesh_plane(mesh, [0, -1, 0], cut)
    out = [obb(piece, f"hoop_handle_{i}") for i, piece in enumerate(slabs(handle, HANDLE_SLABS))]
    step = 2 * np.pi / RING_SECTORS
    for k in range(RING_SECTORS):
        a, b = -np.pi + k * step, -np.pi + (k + 1) * step
        piece = trimesh.intersections.slice_mesh_plane(ring, [-np.sin(a), np.cos(a), 0], centre)
        piece = trimesh.intersections.slice_mesh_plane(piece, [np.sin(b), -np.cos(b), 0], centre)
        if len(piece.vertices) == 0:
            continue
        m = (a + b) / 2
        R = np.array([[np.cos(m), -np.sin(m), 0], [np.sin(m), np.cos(m), 0], [0, 0, 1.0]])
        out.append(aabb_in(piece.vertices, R, f"hoop_ring_{k}"))
    return out


def arm() -> tuple[dict[str, list[dict]], list[dict]]:
    """Boxes of the robot's links (by link, link frames) and of the hoop (hoop_link frame)."""
    _, parts = arm_parts()
    robot: dict[str, list[dict]] = {}
    hoop: list[dict] = []
    for link, name, mesh, _ in parts:
        if name == "hoop":
            hoop = hoop_boxes(mesh)  # in hoop_link; mounted at adapter_front @ HOOP_MOUNT
        elif name in SPLIT:
            robot[link] = [obb(p, f"{name}_{i}") for i, p in enumerate(slabs(mesh, SPLIT[name]))]
        else:
            robot[link] = [obb(mesh, name)]
    return robot, hoop


def turntable(task: TaskScene) -> list[dict]:
    """Two boxes rotated 45 deg whose intersection contains the turntable."""
    v = np.vstack([m.vertices for m in collision_meshes(task.turntable)])
    out = []
    for k, a in enumerate((0.0, np.pi / 4)):
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        out.append(aabb_in(v, R, f"intersect_{k}"))
    return out


def base(task: TaskScene) -> list[dict]:
    """One box around the base (with its foam), base frame."""
    v = np.vstack([m.vertices for m in collision_meshes(task.base)])
    return [aabb_in(v, np.eye(3), "base")]


def puzzle(task: TaskScene) -> list[dict]:
    """The square bar along solver.wire's centreline, and the foot (puzzle frame)."""
    out = []
    names = {0: "top bar", 2: "vertical run", 3: "jog", 4: "stem"}
    y = np.array([0.0, 1.0, 0.0])
    for k, name in names.items():
        seg = SEGMENTS[k]
        a, b = seg.a.copy(), seg.b.copy()
        if k == 0:
            a[0] = OPEN_END_X  # the bar ends at the open end
        if k == 4:
            b[2] = 0.0  # the stem reaches down into the foot
        d = (b - a) / np.linalg.norm(b - a)
        R = np.column_stack([d, y, np.cross(d, y)])
        # Extend by the half width at the joints (the mitred corners stay inside).
        ext = 0.0 if k == 0 else BAR_HALF
        lo = 0.0 if k == 0 else -ext
        pts = np.array([a + lo * d, b + BAR_HALF * d])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, (pts[0] + pts[1]) / 2
        half = [np.linalg.norm(pts[1] - pts[0]) / 2, BAR_HALF, BAR_HALF]
        out.append(box(T, half, f"puzzle_{name.replace(' ', '_')}"))
    arc = SEGMENTS[1]
    da = (arc.angle_to - arc.angle_from) / BEND_PIECES
    r_in, r_out = ARC_RADIUS - BAR_HALF, ARC_RADIUS + BAR_HALF
    for k in range(BEND_PIECES):
        m = arc.angle_from + (k + 0.5) * da
        u = np.array([np.cos(m), 0.0, np.sin(m)])  # radial
        t = np.array([-np.sin(m), 0.0, np.cos(m)])  # tangential
        lo_r = r_in * np.cos(abs(da) / 2)
        R = np.column_stack([u, y, np.cross(u, y)])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, ARC_CENTRE + u * (lo_r + r_out) / 2
        half = [(r_out - lo_r) / 2, BAR_HALF, r_out * np.sin(abs(da) / 2)]
        # The third axis (u x y) is the tangent direction up to sign.
        assert abs(abs(np.cross(u, y) @ t) - 1) < 1e-9
        out.append(box(T, half, f"puzzle_bend_{k}"))
    mesh = trimesh.util.concatenate(collision_meshes(task.puzzle))
    foot = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, -1], [0, 0, 0.0])
    out.append(aabb_in(foot.vertices, np.eye(3), "puzzle_foot"))
    return out


def rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF roll, pitch, yaw of a rotation (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    if np.hypot(R[0, 0], R[1, 0]) < 1e-12:  # gimbal lock: put it all in the yaw
        return 0.0, float(pitch), float(np.arctan2(-R[0, 1], R[1, 1]))
    return (float(np.arctan2(R[2, 1], R[2, 2])), float(pitch),
            float(np.arctan2(R[1, 0], R[0, 0])))


def _num(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def collision_box(b: dict) -> ET.Element:
    """<collision name=...><origin/><geometry><box/></geometry></collision>."""
    T = np.array(b["T"])
    c = ET.Element("collision", name=b["name"])
    ET.SubElement(c, "origin", xyz=_num(T[:3, 3]), rpy=_num(rpy(T[:3, :3])))
    g = ET.SubElement(c, "geometry")
    ET.SubElement(g, "box", size=_num(2 * np.array(b["half"])))
    return c


def write_soft_urdf(source: Path | None, name: str, boxes: dict[str, list[dict]]) -> Path:
    """Copies a URDF's links and joints, replacing all geometry by the boxes.

    Args:
        source: The exact model (None: a new robot with the given links only).
        name: Output file stem.
        boxes: Boxes by link name, in that link's frame.
    """
    if source is None:
        root = ET.Element("robot", name=name)
        for link in boxes:
            ET.SubElement(root, "link", name=link)
    else:
        root = ET.parse(source).getroot()
        root.set("name", name)
    for link in root.iter("link"):
        for child in list(link):
            if child.tag in ("visual", "collision", "inertial"):
                link.remove(child)
        for b in boxes.get(link.get("name"), []):
            link.append(collision_box(b))
    missing = set(boxes) - {link.get("name") for link in root.iter("link")}
    assert not missing, f"no such links in {source}: {missing}"
    ET.indent(root)
    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / f"{name}.urdf"
    note = (f" {name}: soft collision boxes for the solver (solver.soft_collision),\n"
            f"     generated by tools/fit_soft_boxes.py"
            f"{' from ' + source.name if source else ''}; do not edit by hand. ")
    out.write_text('<?xml version="1.0"?>\n<!--' + note + "-->\n"
                   + ET.tostring(root, encoding="unicode") + "\n")
    return out


def main() -> None:
    task = TaskScene()
    robot, hoop = arm()
    written = [
        write_soft_urdf(MODELS / "crs93.urdf", "crs93_soft", robot),
        write_soft_urdf(MODELS / "hoop.urdf", "hoop_soft", {"hoop_link": hoop}),
        write_soft_urdf(MODELS / "base.urdf", "base_soft", {"base_link": base(task)}),
        write_soft_urdf(MODELS / "turntable.urdf", "turntable_soft",
                        {"turntable_link": turntable(task)}),
        write_soft_urdf(MODELS / "puzzle_D.urdf", "puzzle_D_soft",
                        {"puzzle_D_link": puzzle(task)}),
        write_soft_urdf(None, "cell_soft", {"cell": [
            aabb_in(m.vertices, _axes(m), name) for name, m in cell_meshes().items()]}),
    ]
    n = sum(len(b) for b in robot.values()) + len(hoop)
    print(f"wrote {', '.join(p.name for p in written)} ({n} arm boxes)")


def _axes(mesh: trimesh.Trimesh) -> np.ndarray:
    """Axes of a box mesh (its oriented bounding box is the box itself)."""
    return mesh.bounding_box_oriented.primitive.transform[:3, :3]


if __name__ == "__main__":
    main()
