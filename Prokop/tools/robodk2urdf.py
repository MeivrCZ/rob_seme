"""Converts the RoboDK robot file CRS-A465.robot into crs93.urdf and STL meshes.

Usage:
    From the repo root::

        uv run Prokop/tools/robodk2urdf.py

Kinematics are taken from ctu_crs (the driver of the real CRS93), so that the URDF
joint values are exactly the q used by robot.move_to_q() and the URDF "flange" frame
equals robot.fk(q). RoboDK provides only the link meshes (+ colours, joint limits for
a sanity check).

RoboDK .robot format (reverse engineered, no official spec):
  4 B   big-endian uncompressed size, then a zlib stream (Qt qCompress).
  Decompressed:
    header of float64; [1] = header length in doubles (4000), [22] = number of
    bodies (base + 6 links); row i of 20 doubles at 200 + 20 i holds the modified
    (Craig) DH parameters of joint i: alpha, a [mm], theta offset, d [mm];
    joint limits [deg] at 600.. (lower) and 620.. (upper).
    Then int64 n_bodies, and per body: int64 n_submeshes, per submesh
    int64 n_triangles, rgba float32[4], n_triangles * 3 vertices of
    (float32 xyz [mm], 2 B packed normal); after each body 16 B of padding.
  Each body's mesh is expressed in its own modified-DH frame.

Relation to ctu_crs (verified numerically by this script):
  same joint directions and zeros, ctu base frame = RoboDK base frame rotated by
  pi about z and shifted 24.8 mm up (ctu shoulder height 305.2 mm vs RoboDK 330 mm),
  ctu flange = RoboDK flange rotated by pi about z.
  Exception: the static base mesh is kept unrotated, so that its square box points
  to the back as on the real robot (RoboDK has it at the front).
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np
import trimesh
from ctu_crs import CRS93
from mesh_repair import rebuild_watertight

MODELS = Path(__file__).resolve().parent.parent / "models"
SRC = MODELS / "CRS-A465.robot"
URDF = MODELS / "crs93.urdf"
MESH_DIR = MODELS / "meshes" / "crs93"

# CRS adapter (meshes/crs_adapter.stl, STEP frame, mm). Its +z face (z = +2 mm) is
# screwed to the flange, its front face (z = -8 mm) carries the triangular hole whose
# vertex points to -y (between two screws). In q_home the flange x axis points
# down. Turned by ADAPTER_TURN = 0 the vertex would point down: adapter z = -flange
# z, adapter -y = flange x. The 4 screws allow turns by multiples of 90 deg about
# the flange z axis; on the CRS93 station it is turned by 180 deg: the vertex, and
# with it the hoop, points UP in q_home (seen on the robot 2026-10-21).
ADAPTER_TURN = np.pi
ADAPTER_R = (trimesh.transformations.rotation_matrix(ADAPTER_TURN, [0, 0, 1])[:3, :3]
             @ np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))
ADAPTER_T = np.array([0.0, 0.0, 0.002])
ADAPTER_FRONT_Z = -0.008  # front face in the adapter frame; the hoop seats here

# Modified DH structure expected in the RoboDK file (alpha, theta offset); lengths
# are checked separately against ctu_crs.
EXPECTED_ALPHA = np.array([0, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2])
EXPECTED_THETA = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, np.pi])


def read_robodk(path: Path) -> tuple[np.ndarray, np.ndarray, list[list[tuple]]]:
    """Reads a RoboDK .robot file (format in the module doc).

    Args:
        path: The .robot file.

    Returns:
        The modified DH table [6x4: alpha, a, theta, d; m, rad], the joint limits
        [6x2, deg] and the bodies: per body a list of (triangles [n, 3, 3] in m,
        rgba).
    """
    raw = path.read_bytes()
    data = zlib.decompress(raw[4:])
    assert len(data) == int.from_bytes(raw[:4], "big")

    header_len = int(struct.unpack_from("<d", data, 8)[0])
    header = np.frombuffer(data[: header_len * 8], "<f8")
    dhm = np.array([header[200 + 20 * i: 204 + 20 * i] for i in range(6)])
    dhm[:, [1, 3]] /= 1000.0
    limits = np.stack([header[600:606], header[620:626]], axis=1)

    p = header_len * 8
    (n_bodies,) = struct.unpack_from("<q", data, p)
    p += 8
    assert n_bodies == int(header[22]) == 7, n_bodies
    bodies = []
    for _ in range(n_bodies):
        (n_sub,) = struct.unpack_from("<q", data, p)
        p += 8
        subs = []
        for _ in range(n_sub):
            (n_tri,) = struct.unpack_from("<q", data, p)
            rgba = struct.unpack_from("<4f", data, p + 8)
            p += 24
            rec = np.frombuffer(data, np.uint8, n_tri * 3 * 14, p).reshape(-1, 14)
            p += n_tri * 3 * 14
            xyz = rec[:, :12].copy().view("<f4").reshape(n_tri, 3, 3) / 1000.0
            subs.append((xyz.astype(float), rgba))
        bodies.append(subs)
        p += 16
    assert p == len(data), (p, len(data))
    return dhm, limits, bodies


def ctu_dhm(robot: CRS93, rdk_dhm: np.ndarray) -> np.ndarray:
    """Replaces the link lengths of the RoboDK DH table by those of ctu_crs.

    Args:
        robot: The ctu_crs robot.
        rdk_dhm: Modified DH table from read_robodk.

    Returns:
        The DH table with the ctu_crs link lengths.
    """
    ll = robot.link_lengths  # base height, upper arm, forearm, wrist-flange
    dhm = rdk_dhm.copy()
    dhm[0, 3], dhm[2, 1], dhm[3, 3], dhm[5, 3] = ll[0], ll[1], ll[2], ll[3]
    return dhm


def fmt(v) -> str:
    """Formats numbers for a URDF attribute ("x y z")."""
    return " ".join(f"{x:.12g}" for x in v)


def write_urdf(dhm: np.ndarray, q_min: np.ndarray, q_max: np.ndarray,
               mesh_files: list[list[tuple[str, tuple]]], collision_files: list[str]
               ) -> str:
    """Builds the URDF text of the robot with its adapter.

    Args:
        dhm: Modified DH table [m, rad] (ctu_crs link lengths).
        q_min: Lower joint limits [rad].
        q_max: Upper joint limits [rad].
        mesh_files: Per body, the visual mesh files with their rgba colours.
        collision_files: Per body, the watertight collision mesh file.

    Returns:
        The URDF document.
    """
    base_drop = 0.330 - dhm[0, 3]  # RoboDK base frame lies this much below ctu's
    names = ["base_link"] + [f"link{i}" for i in range(1, 7)]
    out = [
        '<?xml version="1.0"?>',
        "<!-- CRS A465 (CTU station CRS93). GENERATED by robodk2urdf.py, do not edit.",
        "     Kinematics = ctu_crs (joint values == robot.get_q(), frame 'flange' ==",
        "     robot.fk(q), base_link == ctu base frame). Meshes from RoboDK library.",
        "     Units: m, rad. -->",
        '<robot name="crs93">',
    ]
    materials = {}
    for subs in mesh_files:
        for _, rgba in subs:
            materials.setdefault(rgba, f"crs_{len(materials)}")
    for rgba, mname in materials.items():
        out.append(f'  <material name="{mname}"><color rgba="{fmt(rgba)}"/></material>')

    for i, name in enumerate(names):
        # The base mesh is in the RoboDK base frame, which lies lower. It is NOT
        # rotated by pi like the RoboDK kinematics: on the real robot the square
        # box on the base cylinder points to the back (-x of ctu), opposite to the
        # arm in q_home, while RoboDK has it on the arm's side. Link meshes are in the
        # modified-DH frame, which lies d_i above the URDF link frame (see the joint
        # loop below).
        origin = (f'<origin xyz="0 0 {-base_drop:.12g}"/>'
                  if i == 0 else f'<origin xyz="0 0 {dhm[i - 1, 3]:.12g}"/>')
        out.append(f'  <link name="{name}">')
        for fname, rgba in mesh_files[i]:
            out += [
                "    <visual>",
                f"      {origin}" if origin else None,
                f'      <geometry><mesh filename="{fname}"/></geometry>',
                f'      <material name="{materials[rgba]}"/>',
                "    </visual>",
            ]
        out += [
            "    <collision>",
            f"      {origin}" if origin else None,
            f'      <geometry><mesh filename="{collision_files[i]}"/></geometry>',
            "    </collision>",
            "  </link>",
        ]
    out = [line for line in out if line is not None]

    # Modified DH: T_i = Rx(alpha) Tx(a) Rz(theta + q) Tz(d). Rz(theta + q) Tz(d)
    # = Rz(q) [Rz(theta) Tz(d)] can't be split into a URDF joint (origin, then
    # rotation about z) directly, so the d of joint i is moved into the origin of
    # joint i+1 (rotation about z commutes with translation along z).
    prev_d = 0.0
    for i in range(6):
        alpha, a, theta, d = dhm[i]
        # Joint 1 additionally carries the pi rotation between ctu and RoboDK base.
        yaw = theta + (np.pi if i == 0 else 0.0)
        R = (trimesh.transformations.translation_matrix([0, 0, prev_d])
             @ trimesh.transformations.rotation_matrix(alpha, [1, 0, 0])
             @ trimesh.transformations.translation_matrix([a, 0, 0])
             @ trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]))
        rpy = trimesh.transformations.euler_from_matrix(R, "sxyz")
        out += [
            f'  <joint name="joint{i + 1}" type="revolute">',
            f'    <parent link="{names[i]}"/>',
            f'    <child link="{names[i + 1]}"/>',
            f'    <origin xyz="{fmt(R[:3, 3])}" rpy="{fmt(rpy)}"/>',
            '    <axis xyz="0 0 1"/>',
            f'    <limit lower="{q_min[i]:.12g}" upper="{q_max[i]:.12g}" '
            'velocity="1.0" effort="100"/>',
            "  </joint>",
        ]
        prev_d = d
    out += [
        '  <link name="flange"/>',
        '  <joint name="link6_to_flange" type="fixed">',
        '    <parent link="link6"/>',
        '    <child link="flange"/>',
        f'    <origin xyz="0 0 {prev_d:.12g}" rpy="0 0 {np.pi:.12g}"/>',
        "  </joint>",
    ]
    rpy = trimesh.transformations.euler_from_matrix(ADAPTER_R, "sxyz")
    adapter_mesh = '<mesh filename="meshes/crs_adapter.stl" scale="0.001 0.001 0.001"/>'
    out += [
        "  <!-- CRS adapter; the hoop is NOT part of the robot: its magnetic mount is",
        "       ambiguous and not rigid, it is to be attached from measurements. -->",
        '  <material name="crs_adapter_color"><color rgba="0.12 0.22 0.6 1"/></material>',
        '  <link name="adapter">',
        f"    <visual><geometry>{adapter_mesh}</geometry>",
        '      <material name="crs_adapter_color"/></visual>',
        f"    <collision><geometry>{adapter_mesh}</geometry></collision>",
        "  </link>",
        '  <joint name="flange_to_adapter" type="fixed">',
        '    <parent link="flange"/>',
        '    <child link="adapter"/>',
        f'    <origin xyz="{fmt(ADAPTER_T)}" rpy="{fmt(rpy)}"/>',
        "  </joint>",
        "  <!-- centre of the adapter front face, axes as in the adapter/hoop STEP files",
        "       (triangle vertex and nominal hoop ring direction = -y) -->",
        '  <link name="adapter_front"/>',
        '  <joint name="adapter_to_adapter_front" type="fixed">',
        '    <parent link="adapter"/>',
        '    <child link="adapter_front"/>',
        f'    <origin xyz="0 0 {ADAPTER_FRONT_Z:.12g}" rpy="0 0 0"/>',
        "  </joint>",
        "</robot>",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    """Writes the meshes and the URDF, then verifies them against ctu_crs."""
    rdk_dhm, limits_deg, bodies = read_robodk(SRC)
    assert np.allclose(rdk_dhm[:, 0], EXPECTED_ALPHA), rdk_dhm
    assert np.allclose(rdk_dhm[:, 2], EXPECTED_THETA), rdk_dhm

    robot = CRS93(tty_dev=None)
    dhm = ctu_dhm(robot, rdk_dhm)
    print("link lengths RoboDK [mm]:", rdk_dhm[[0, 2, 3, 5], [3, 1, 3, 3]] * 1000)
    print("link lengths ctu    [mm]:", robot.link_lengths * 1000)
    print("limits RoboDK [deg]:\n", limits_deg.T.round(1))
    print("limits ctu    [deg]:\n", np.rad2deg([robot.q_min, robot.q_max]).round(1))

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    rel = MESH_DIR.relative_to(MODELS)
    mesh_files, collision_files = [], []
    for i, subs in enumerate(bodies):
        files, parts = [], []
        for j, (tris, rgba) in enumerate(subs):
            m = trimesh.Trimesh(**trimesh.triangles.to_kwargs(tris))
            m.merge_vertices()
            if not m.is_watertight:
                # RoboDK meshes miss triangles (holes) -> rebuild a closed solid.
                m, rep = rebuild_watertight(m)
                print(f"link{i}: rebuilt from {rep['pieces']} convex pieces, original "
                      f"surface outside it <= {rep['outside_max_mm']:.2f} mm, "
                      f"swallowed {100 * rep['swallowed_frac']:.1f} %")
                assert m.is_watertight
                assert rep["outside_max_mm"] < 1.0 and rep["swallowed_frac"] < 0.01
            name = f"link{i}" + (f"_{j}" if len(subs) > 1 else "") + ".stl"
            m.export(MESH_DIR / name)
            files.append((f"{rel}/{name}", tuple(round(c, 3) for c in rgba)))
            parts.append(m)
        if len(subs) > 1:
            m = trimesh.util.concatenate(parts)
            m.export(MESH_DIR / f"link{i}.stl")
        collision_files.append(f"{rel}/link{i}.stl")
        mesh_files.append(files)
        print(f"link{i}: {sum(len(p.faces) for p in parts)} triangles, "
              f"watertight={all(p.is_watertight for p in parts)}")

    URDF.write_text(write_urdf(dhm, robot.q_min, robot.q_max, mesh_files,
                               collision_files))
    print("wrote", URDF)
    verify(robot)


def verify(robot: CRS93) -> None:
    """Checks the written URDF against ctu_crs.

    The URDF flange must equal ctu_crs fk for random configurations, and in
    q_home the adapter must be turned as ADAPTER_TURN says.

    Args:
        robot: The ctu_crs robot.

    Raises:
        AssertionError: If a check fails.
    """
    import yourdfpy

    urdf = yourdfpy.URDF.load(URDF, mesh_dir=str(MODELS))
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(200):
        q = rng.uniform(robot.q_min, robot.q_max)
        urdf.update_cfg(q)
        T = urdf.get_transform("flange", "base_link")
        err = max(err, np.abs(T - robot.fk(q)).max())
    print(f"max |URDF flange - ctu fk| over 200 random q: {err:.2e}")
    assert err < 1e-9

    # In q_home the adapter's triangle vertex (-y) must point up (ADAPTER_TURN = pi),
    # its front (-z) away from the flange.
    urdf.update_cfg(robot.q_home)
    T = urdf.get_transform("adapter_front", "base_link")
    F = robot.fk(robot.q_home)
    print("q_home: triangle vertex dir", (-T[:3, 1]).round(6),
          "front dir", (-T[:3, 2]).round(6))
    assert np.allclose(-T[:3, 1], [0, 0, -np.cos(ADAPTER_TURN)])
    assert np.allclose(-T[:3, 2], F[:3, 2])
    assert np.allclose(T[:3, 3], F[:3, 3] + 0.010 * F[:3, 2])


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter
                            ).parse_args()
    main()
