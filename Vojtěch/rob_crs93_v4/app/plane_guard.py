"""Conservative tabletop clearance preflight for CRS93 station.

The Prokop URDF is only usable after its computed flange FK agrees with the
hardware driver's FK. Collision vertices cover entire robot arm surfaces,
not merely joint origins. It is NOT a certified collision-avoidance system:
other obstacles, people, table inclination, loose hoop mounts, controller
tracking and asynchronous stop latency are not modeled.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


CLEARANCE_M = .010
# Largest cross-section from the measured 69.6-mm ring OD. Add 2 mm
# uncertainty for a conservative envelope from flange to ring centre.
HOOP_ENVELOPE_RADIUS_M = .037


def _rpy_rotation(rpy):
    a, b, c = rpy
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return np.array([[cc*cb, cc*sb*sa-sc*ca, cc*sb*ca+sc*sa],
                     [sc*cb, sc*sb*sa+cc*ca, sc*sb*ca-cc*sa],
                     [-sb, cb*sa, cb*ca]], dtype=float)


def _joint_rot(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, k = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    return np.array([[c+x*x*k, x*y*k-z*s, x*z*k+y*s],
                     [y*x*k+z*s, c+y*y*k, y*z*k-x*s],
                     [z*x*k-y*s, z*y*k+x*s, c+z*z*k]])


def _origin(element):
    t = np.eye(4)
    if element is None:
        return t
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    if xyz.size != 3 or rpy.size != 3:
        raise ValueError('Invalid URDF origin')
    t[:3, 3] = xyz
    t[:3, :3] = _rpy_rotation(rpy)
    return t


def _stl_vertices(path: Path):
    """Read binary STL without additional dependencies; full vertex set."""
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f'Empty mesh: {path}')
    n = int.from_bytes(raw[80:84], 'little')
    if 84 + 50*n != len(raw) or n == 0:
        raise ValueError(f'Expected binary STL: {path}')
    faces = np.ndarray(shape=(n,), dtype=np.dtype([
        ('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')]), buffer=raw, offset=84)
    pts = faces['vertices'].reshape(-1, 3).astype(float, copy=True)
    if not np.isfinite(pts).all():
        raise ValueError(f'Invalid mesh: {path}')
    return pts


class TableClearanceGuard:
    def __init__(self, root: Path, table_z_m: float, *, clearance_m=CLEARANCE_M):
        self.table_z_m = float(table_z_m)
        self.clearance_m = float(clearance_m)
        if not np.isfinite(self.table_z_m) or not np.isfinite(self.clearance_m) or self.clearance_m < .01:
            raise ValueError('Table Z must be finite; minimum clearance 0.010 m')
        base = Path(root)
        urdf = base / "models" / 'crs93.urdf'
        if not urdf.is_file():
            raise FileNotFoundError(f'Cannot check arm geometry; missing {urdf}')
        doc = ET.parse(urdf).getroot()
        self.links = {}
        for elem in doc.findall('link'):
            name = elem.get('name')
            collision = elem.find('collision')
            if collision is None:
                self.links[name] = None
                continue
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                self.links[name] = None
                continue
            mesh_path = (urdf.parent / mesh.get('filename')).resolve()
            # Prevent unexpected paths outside the checked model directory.
            if not mesh_path.is_relative_to(urdf.parent.resolve()):
                raise ValueError('Unsafe mesh path')
            points = _stl_vertices(mesh_path)
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            if scale.size != 3:
                raise ValueError(f'Invalid mesh scale: {mesh_path}')
            self.links[name] = (points * scale, _origin(collision.find('origin')))
        self.joints = []
        for j in doc.findall('joint'):
            parent, child = j.find('parent'), j.find('child')
            if parent is None or child is None:
                continue
            a = j.find('axis')
            axis = np.fromstring(a.get('xyz', '0 0 1') if a is not None else '0 0 1', sep=' ')
            self.joints.append((j.get('name'), j.get('type'), parent.get('link'), child.get('link'),
                                _origin(j.find('origin')), axis))
        if [j[0] for j in self.joints[:6]] != [f'joint{k}' for k in range(1, 7)]:
            raise ValueError('Unexpected robot kinematic chain in URDF')

    def transforms(self, q):
        q = np.asarray(q, float).reshape(-1)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Need six finite joint angles')
        poses = {'base_link': np.eye(4)}
        for name, kind, parent, child, origin, axis in self.joints:
            if parent not in poses:
                # Only the connected robot chain matters.
                continue
            T = poses[parent] @ origin
            if kind == 'revolute':
                i = int(name.removeprefix('joint')) - 1
                J = np.eye(4)
                J[:3, :3] = _joint_rot(axis, float(q[i]))
                T = T @ J
            poses[child] = T
        if 'flange' not in poses:
            raise ValueError('URDF is missing flange transform')
        return poses

    def check(self, q, robot_fk, hoop_offset):
        """Check arm mesh vertices + hoop capsule at a joint-space sample."""
        transforms = self.transforms(q)
        T_flange = np.asarray(robot_fk(q), float)
        if T_flange.shape != (4, 4) or not np.isfinite(T_flange).all():
            raise RuntimeError('Invalid CRS93 FK')
        urdf_flange = transforms['flange']
        if (np.linalg.norm(T_flange[:3, 3] - urdf_flange[:3, 3]) > .003 or
                np.linalg.norm(T_flange[:3, :3] - urdf_flange[:3, :3]) > .025):
            raise RuntimeError('Prokop URDF does not agree with CRS93 FK; clearance cannot be verified')
        floor = self.table_z_m + self.clearance_m
        # Exclude fixed base_link itself, which may be below tabletop.
        for name in ('link1', 'link2', 'link3', 'link4', 'link5', 'link6'):
            if name not in transforms or self.links.get(name) is None:
                raise RuntimeError(f'Incomplete robot geometry: {name}')
            points, local = self.links[name]
            global_t = transforms[name] @ local
            low = np.min(points @ global_t[2, :3] + global_t[2, 3])
            if low < floor:
                raise ValueError(f'Collision risk: {name} {low:.3f} m below 1 cm clearance over table Z={self.table_z_m:.3f} m')
            # The physical joint centers are included explicitly.
            if name != 'link1' and transforms[name][2, 3] < floor:
                raise ValueError(f'Collision risk: {name} joint center below minimum clearance')
        offset = np.asarray(hoop_offset, float).reshape(-1)
        if offset.shape != (3,) or not np.isfinite(offset).all() or not .05 < np.linalg.norm(offset) < .25:
            raise ValueError('Measured hoop centre offset TCP required before motion')
        center = (T_flange @ np.r_[offset, 1.])[:3]
        # Capsule bounds the ring + full handle in *any* rotation about its
        # mounting vector, provided measured flange-to-hoop centre is correct.
        if min(T_flange[2, 3], center[2]) - HOOP_ENVELOPE_RADIUS_M < floor:
            raise ValueError('Hoop envelope gets closer than 1 cm to the table')
        return center
