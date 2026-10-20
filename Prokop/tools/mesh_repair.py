"""Rebuild a broken (holey, non-manifold) mesh into a watertight solid.

Used by robodk2urdf.py for the RoboDK link meshes, which miss triangles. Their
vertices are correct, so the solid is rebuilt from them: the surface is split into
convex patches, each patch is replaced by the convex hull of its vertices, and the
hulls are united. This keeps the original geometry exactly where it is convex and
closes the holes, as long as the part is a union of convex pieces (true for the CRS
links: cylinders, cones, plates, boxes).

Segmentation: greedy region growing over edge-adjacent triangles, largest first; a
triangle joins a patch only if all the patch triangles still lie on the hull of the
patch vertices with outward-agreeing normals, i.e. the patch stays convex. Flat
patches that do not lie on any hull (thin strips) are added as 1 mm thick plates.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh
from trimesh.proximity import closest_point

TOL = 0.05  # mm, distance of a triangle centre from its patch hull
FLAT_THICKNESS = 1.0  # mm, thickness given to flat strips that lie on no hull


def _hull(points: np.ndarray) -> trimesh.Trimesh | None:
    try:
        h = trimesh.convex.convex_hull(points)
    except Exception:  # (near) planar point set
        return None
    return h if h.volume > 1e-6 else None


def _on_hull(h: trimesh.Trimesh, centers: np.ndarray,
             normals: np.ndarray | None = None) -> np.ndarray:
    _, dist, tri = closest_point(h, centers)
    ok = dist < TOL
    if normals is not None:
        ok &= np.einsum("ij,ij->i", h.face_normals[tri], normals) > 0.9
    return ok


def _plate(m: trimesh.Trimesh, face: int) -> trimesh.Trimesh:
    """Triangle @face extruded FLAT_THICKNESS behind it (against its normal)."""
    v = m.vertices[m.faces[face]]
    return trimesh.convex.convex_hull(np.vstack([v, v - FLAT_THICKNESS * m.face_normals[face]]))


def convex_patches(m: trimesh.Trimesh) -> list[np.ndarray]:
    """Partition faces of @m into edge-connected patches whose hull they lie on."""
    neighbours = defaultdict(set)
    for a, b in m.face_adjacency:
        neighbours[a].add(b)
        neighbours[b].add(a)
    centers, normals = m.triangles_center, m.face_normals
    free = set(range(len(m.faces)))
    patches = []
    for seed in np.argsort(-m.area_faces):
        if seed not in free:
            continue
        patch = [seed]
        free.discard(seed)
        frontier, rejected = list(neighbours[seed]), set()
        while frontier:
            f = frontier.pop()
            if f not in free or f in rejected:
                continue
            cand = patch + [f]
            h = _hull(m.vertices[np.unique(m.faces[cand])])
            if h is None or _on_hull(h, centers[cand], normals[cand]).all():
                patch.append(f)
                free.discard(f)
                frontier += list(neighbours[f])
                rejected.clear()  # the patch changed, retry rejected ones
            else:
                rejected.add(f)
        patches.append(np.array(patch))
    return patches


def rebuild_watertight(m: trimesh.Trimesh, scale: float = 1000.0
                       ) -> tuple[trimesh.Trimesh, dict]:
    """Return a watertight rebuild of @m and a report of the deviation from it.
    @scale converts the mesh units to mm (the tolerances are in mm)."""
    mm = m.copy()
    mm.apply_scale(scale)
    hulls, flats = [], []
    for patch in convex_patches(mm):
        h = _hull(mm.vertices[np.unique(mm.faces[patch])])
        if h is not None:
            hulls.append(h)
        else:
            flats.append(patch)
    for patch in flats:
        c = mm.triangles_center[patch]
        if any(_on_hull(h, c).all() for h in hulls):
            continue
        # Thin strip lying on no hull: give its triangles a thickness behind them.
        hulls += [_plate(mm, f) for f in patch if mm.area_faces[f] > 1e-6]
    solid = trimesh.boolean.union(hulls, engine="manifold")

    # Original triangles still sticking out of the solid (e.g. a flat triangle
    # spanning two pieces whose union is slightly concave there) become plates too.
    pts, fid = trimesh.sample.sample_surface(mm, 20000, seed=0)
    sticking = np.unique(fid[trimesh.proximity.signed_distance(solid, pts) < -TOL])
    if len(sticking):
        solid = trimesh.boolean.union(
            [solid] + [_plate(mm, f) for f in sticking], engine="manifold")

    # Deviation: the original surface must not stick out of the solid (positive
    # signed distance = inside), and the solid must not swallow original surface.
    sd = trimesh.proximity.signed_distance(solid, pts)
    report = {
        "pieces": len(hulls),
        "outside_max_mm": float(np.maximum(0, -sd).max()),
        "swallowed_frac": float((sd > 0.5).mean()),
    }
    solid.apply_scale(1.0 / scale)
    return solid, report
