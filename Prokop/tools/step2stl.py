"""Converts the STEP models of the hardware repo into the STL meshes of the URDFs.

Usage:
    From the repo root (needs the optional `cad` dependency group)::

        uv run --group cad Prokop/tools/step2stl.py [output_dir]   # default Prokop/models/meshes

Meshes keep the STEP-file origin and units (mm); the URDFs scale them by 0.001.
The base is additionally split into base_plate.stl + base_markers.stl and the hoop
into hoop_white.stl + hoop_blue.stl for colouring.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.StlAPI import StlAPI_Writer
from OCP.STEPControl import STEPControl_Reader

PROKOP = Path(__file__).resolve().parent.parent
HW = PROKOP.parent / "ROB1-2025-hw-main"

# STEP file (relative to the hw repo) -> mesh name
SOURCES = {
    "puzzles/A.step": "puzzle_A",
    "puzzles/B.step": "puzzle_B",
    "puzzles/C.step": "puzzle_C",
    "puzzles/D.step": "puzzle_D",
    "puzzles/E.step": "puzzle_E",
    "robot_ee/CRS_adapter.step": "crs_adapter",
    "robot_ee/Hoop.step": "hoop",
    "bases/base 01-02.step": "base",
}


def convert(src: Path, dst: Path) -> float:
    """Tessellates a STEP model into a binary STL.

    Args:
        src: The STEP file.
        dst: The STL file to write.

    Returns:
        The volume of the CAD solid [mm^3], to compare with the mesh.
    """
    r = STEPControl_Reader()
    assert r.ReadFile(str(src)) == 1, src
    r.TransferRoots()
    shape = r.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.02, False, 0.2, True)
    w = StlAPI_Writer()
    w.ASCIIMode = False
    w.Write(shape, str(dst))
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def split_base_markers(base: Path) -> None:
    """Splits the base mesh into the white plate and the black ArUco markers.

    For colouring only, the URDF collision still uses the whole mesh. The black
    marker parts are a 0.6 mm relief on top of the plate surface (z = 5.0 .. 5.6
    mm); the white marker cells are holes in it down to the plate surface.

    Args:
        base: base.stl; base_plate.stl and base_markers.stl are written next to it.
    """
    m = trimesh.load(base)
    black = m.triangles_center[:, 2] > 5.0 + 1e-3
    for name, mask in [("base_plate", ~black), ("base_markers", black)]:
        part = m.submesh([np.nonzero(mask)[0]], append=True)
        part.export(base.with_name(f"{name}.stl"))


def split_hoop_colours(hoop: Path) -> None:
    """Splits the hoop mesh into its two filament colours.

    For colouring only. The hoop is printed flat (STEP z = -6.6 .. 0 mm) with
    filament changes at 0.8 mm and 5.8 mm (see print_files/hoop_dualcolor_*.gcode,
    the slicer marks the middle part #4835C7): the 0.8 mm layers on both faces
    are white, the core is blue (matches the photo of the setup). Faces lying in
    a filament-change plane (the floors of the 0.8 mm deep ring groove and
    triangle recess on both faces) are surfaces of the core, so blue. The two
    closed magnet cavities are dropped.

    Args:
        hoop: hoop.stl; hoop_white.stl and hoop_blue.stl are written next to it.
    """
    body = max(trimesh.load(hoop).split(only_watertight=False),
               key=lambda b: len(b.faces))
    slice_ = trimesh.intersections.slice_mesh_plane
    top = slice_(body, [0, 0, 1], [0, 0, -0.8])
    bottom = slice_(body, [0, 0, -1], [0, 0, -5.8])
    core = slice_(slice_(body, [0, 0, -1], [0, 0, -0.8]), [0, 0, 1], [0, 0, -5.8])
    white = trimesh.util.concatenate([top, bottom])
    z = white.vertices[white.faces][:, :, 2]  # (faces, 3) vertex heights
    in_plane = np.any([np.all(np.abs(z - h) < 1e-6, axis=1) for h in (-0.8, -5.8)],
                      axis=0)
    white.submesh([np.flatnonzero(~in_plane)], append=True).export(
        hoop.with_name("hoop_white.stl"))
    trimesh.util.concatenate([core, white.submesh([np.flatnonzero(in_plane)],
                                                  append=True)]).export(
        hoop.with_name("hoop_blue.stl"))


def main() -> None:
    """Converts all SOURCES and splits the base and hoop meshes for colouring."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", nargs="?", type=Path,
                        default=PROKOP / "models" / "meshes")
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    for rel, name in SOURCES.items():
        dst = out / f"{name}.stl"
        v = convert(HW / rel, dst)
        m = trimesh.load(dst)
        print(f"{name:12s} cad_vol={v:10.1f} mesh_vol={m.volume:10.1f} "
              f"wt={m.is_watertight} faces={len(m.faces)}")
        print("   bounds", m.bounds.round(2).tolist())
    split_base_markers(out / "base.stl")
    split_hoop_colours(out / "hoop.stl")


if __name__ == "__main__":
    main()
