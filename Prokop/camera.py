"""Simulated station camera: an image of the scene rendered from the camera pose.

The perception module gets the same input as on the real station, a
1920 x 1200 BGR uint8 image, but rendered from the simulation's scene: the robot
with the adapter and hoop, the turntable with the base (ArUco markers) and the
puzzle, on the table. Flat colours with simple lighting: good for checking the
logic of a pipeline, not for tuning detection.

Example:
    ::

        renderer = SceneRenderer(TaskScene())
        image = renderer.render(q, placement, turntable_angle)

CRS93Sim.grab_image uses it with the measured timing of the real camera. The
camera parameters (Camera) default to the camera model of cell.py; going to the
real robot means replacing them by the calibrated ones on the perception side.

Rendering is off-screen OpenGL (moderngl; EGL on Linux, so it works headless and
without a GPU, e.g. on WSL). An OpenGL context belongs to the thread that made
it, so a SceneRenderer keeps one per calling thread (made by prepare(), ~0.6 s).
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field

import numpy as np
from cell import CAMERA_K, CAMERA_RESOLUTION, CAMERA_T, TABLE_Z
from placement import HOOP_MOUNT, MODELS, Placement, TaskScene, load_urdf, yaw_matrix

TABLE_COLOR = (0.84, 0.84, 0.84)  # renders as ~215 grey, like the real table image
TABLE_HALF = 1.5  # m, half side of the table quad
AMBIENT, DIFFUSE = 0.55, 0.45  # lighting: ambient + Lambert from LIGHT_DIR
LIGHT_DIR = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0])
NEAR, FAR = 0.01, 5.0  # m, clipping planes
SAMPLES = 4  # multisampling (smooth edges)

_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform mat3 rot;
in vec3 position;
in vec3 normal;
out vec3 n_world;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    n_world = rot * normal;
}
"""
_FRAGMENT_SHADER = """
#version 330
uniform vec3 color;
uniform vec3 light;
uniform float ambient;
uniform float diffuse;
in vec3 n_world;
out vec4 frag;
void main() {
    // Flat triangles seen from either side: light the side facing the light.
    float lambert = abs(dot(normalize(n_world), light));
    frag = vec4(color * (ambient + diffuse * lambert), 1.0);
}
"""


@dataclass
class Camera:
    """Pinhole camera, OpenCV convention.

    Attributes:
        T: Pose of the camera frame in the robot base frame (4 x 4): z the
            optical axis, x image right, y image down.
        K: Intrinsic matrix (3 x 3) [px].
        resolution: Image size (width, height) [px].
    """

    T: np.ndarray = field(default_factory=lambda: CAMERA_T.copy())
    K: np.ndarray = field(default_factory=lambda: CAMERA_K.copy())
    resolution: tuple[int, int] = CAMERA_RESOLUTION

    def project(self, p: np.ndarray) -> np.ndarray:
        """Projects points into the image.

        Args:
            p: Point(s) [m], robot base frame, shape (3,) or (n, 3).

        Returns:
            Pixel coordinates (u, v), shape (2,) or (n, 2).
        """
        p = np.atleast_2d(p)
        pc = (np.linalg.inv(self.T) @ np.column_stack([p, np.ones(len(p))]).T)[:3]
        uv = (self.K @ pc)[:2] / pc[2]
        return uv.T[0] if len(p) == 1 else uv.T

    def gl_projection(self) -> np.ndarray:
        """OpenGL projection matrix from the camera frame to clip space.

        Image row v = 0 maps to the first row read back from the framebuffer,
        so the image needs no flip.

        Returns:
            4 x 4 matrix.
        """
        (fx, _, cx), (_, fy, cy), _ = self.K
        w, h = self.resolution
        return np.array([
            [2 * fx / w, 0, 2 * (cx + 0.5) / w - 1, 0],
            [0, 2 * fy / h, 2 * (cy + 0.5) / h - 1, 0],
            [0, 0, (FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)],
            [0, 0, 1, 0]])


def _flat(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle vertices and face normals (flat shading).

    Args:
        mesh: trimesh.Trimesh.

    Returns:
        Vertices (3 n, 3) and normals (3 n, 3), float32.
    """
    v = mesh.vertices[mesh.faces].reshape(-1, 3)
    n = np.repeat(mesh.face_normals, 3, axis=0)
    return v.astype("f4"), n.astype("f4")


def _visuals(urdf) -> list[tuple[str, object, np.ndarray]]:
    """Visual meshes of a URDF with their colours.

    Args:
        urdf: yourdfpy.URDF (its current configuration is irrelevant).

    Returns:
        (scene node, mesh in the node frame, RGB colour 0..1) per geometry.
    """
    scene = urdf.scene
    out = []
    for node in scene.graph.nodes_geometry:
        _, geom = scene.graph[node]
        mesh = scene.geometry[geom]
        rgb = np.asarray(mesh.visual.face_colors[0][:3], float) / 255.0
        out.append((node, mesh, rgb))
    return out


class SceneRenderer:
    """Renders camera images of the task scene.

    Attributes:
        camera: The camera model used for rendering.
    """

    def __init__(self, task: TaskScene, camera: Camera | None = None) -> None:
        """Loads the visual meshes; the OpenGL context is made on the first render.

        Args:
            task: The task models (turntable, base, puzzle; their frames).
            camera: Camera model; default: cell.py's.
        """
        self.camera = camera or Camera()
        self._task = task
        self._robot = load_urdf(MODELS / "crs93.urdf")
        hoop = load_urdf(MODELS / "hoop.urdf")
        # (group, key, mesh, colour). Robot: key = scene node, posed by q. Other
        # groups: key = the mesh's static pose in the group frame.
        self._items = [("robot", n, m, c) for n, m, c in _visuals(self._robot)]
        for group, urdf in (("hoop", hoop), ("turntable", task.turntable),
                            ("base", task.base), ("puzzle", task.puzzle)):
            self._items += [(group, urdf.scene.graph[n][0], m, c)
                            for n, m, c in _visuals(urdf)]
        self._gl: dict[int, dict] = {}  # OpenGL state per thread

    def prepare(self) -> None:
        """Makes the OpenGL context of the calling thread, if it has none yet.

        render() does it too; call it ahead to keep the first render fast.
        """
        if threading.get_ident() not in self._gl:
            self._gl[threading.get_ident()] = self._init_gl()

    def _init_gl(self) -> dict:
        """Creates a context with the shaders, meshes and framebuffers.

        Returns:
            The OpenGL objects of the calling thread.
        """
        import moderngl
        if sys.platform.startswith("linux"):
            try:
                ctx = moderngl.create_standalone_context(backend="egl")
            except Exception:
                ctx = moderngl.create_standalone_context()
        else:
            ctx = moderngl.create_standalone_context()
        prog = ctx.program(vertex_shader=_VERTEX_SHADER,
                           fragment_shader=_FRAGMENT_SHADER)
        vaos = []
        table = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]],
                         "f4") * TABLE_HALF
        geometry = [(np.column_stack([table, np.full(6, TABLE_Z, "f4")]),
                     np.tile(np.array([0, 0, 1], "f4"), (6, 1)))]
        geometry += [_flat(mesh) for _, _, mesh, _ in self._items]
        for v, n in geometry:
            vbo = ctx.buffer(np.hstack([v, n]).astype("f4").tobytes())
            vaos.append(ctx.vertex_array(prog, [(vbo, "3f 3f", "position", "normal")]))
        w, h = self.camera.resolution
        msaa = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((w, h), samples=SAMPLES)],
            depth_attachment=ctx.depth_renderbuffer((w, h), samples=SAMPLES))
        fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h))])
        return {"ctx": ctx, "prog": prog, "vaos": vaos, "msaa": msaa, "fbo": fbo,
                "resolution": (w, h)}

    def poses(self, q: np.ndarray, placement: Placement,
              turntable_angle: float) -> dict[str, np.ndarray]:
        """World poses of the groups of the scene.

        Args:
            q: Robot configuration [rad].
            placement: Turntable and base placement.
            turntable_angle: Rotation of the turntable top [rad], counter-clockwise
                from above.

        Returns:
            Pose (4 x 4) of each group's frame: "turntable", "base" (in its groove
            frame), "puzzle", "hoop" (hoop link).
        """
        self._robot.update_cfg(np.asarray(q, dtype=float))
        T_tt = placement.turntable_T()
        T_base = (T_tt @ self._task.T_top @ yaw_matrix(0, 0, 0, turntable_angle)
                  @ placement.base_T_top())
        return {
            "turntable": T_tt,
            "base": T_base,
            "puzzle": T_base @ self._task.T_mount,
            "hoop": self._robot.get_transform("adapter_front", self._robot.base_link)
            @ HOOP_MOUNT,
        }

    def render(self, q: np.ndarray, placement: Placement,
               turntable_angle: float) -> np.ndarray:
        """Renders the camera image of the scene.

        Args:
            q: Robot configuration [rad].
            placement: Turntable and base placement.
            turntable_angle: Rotation of the turntable top [rad].

        Returns:
            Image (height, width, 3), BGR uint8, like CRS93.grab_image.
        """
        self.prepare()
        gl = self._gl[threading.get_ident()]
        if gl["resolution"] != tuple(self.camera.resolution):
            raise ValueError("the camera resolution changed after prepare()")
        groups = self.poses(q, placement, turntable_angle)
        view_proj = self.camera.gl_projection() @ np.linalg.inv(self.camera.T)
        ctx, prog = gl["ctx"], gl["prog"]
        gl["msaa"].use()
        ctx.enable(ctx.DEPTH_TEST)
        ctx.clear(*TABLE_COLOR, 1.0)
        prog["light"].value = tuple(LIGHT_DIR)
        prog["ambient"].value = AMBIENT
        prog["diffuse"].value = DIFFUSE
        draws = [(np.eye(4), TABLE_COLOR)]
        for group, key, _, rgb in self._items:
            if group == "robot":
                T, _ = self._robot.scene.graph[key]
            else:
                T = groups[group] @ key
            draws.append((T, rgb))
        for vao, (T, rgb) in zip(gl["vaos"], draws):
            prog["mvp"].write((view_proj @ T).T.astype("f4").tobytes())
            prog["rot"].write(np.ascontiguousarray(T[:3, :3].T).astype("f4").tobytes())
            prog["color"].value = tuple(float(c) for c in rgb)
            vao.render()
        ctx.copy_framebuffer(gl["fbo"], gl["msaa"])
        w, h = self.camera.resolution
        rgb = np.frombuffer(gl["fbo"].read(components=3), np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(rgb[:, :, ::-1])
