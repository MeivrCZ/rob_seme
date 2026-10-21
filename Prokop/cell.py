"""The static work cell around the CRS93: table, work area, cage, camera rail and
camera, as constants and collision meshes.

Names: the RoboDK base mesh is the robot's black steel "foot" (square, 203 mm,
50.8 mm high, bolted to the table) with the "round base" (a truncated cone) on
top of it.

Robot base frame (ctu_crs, m): x forward (the arm in q_home), y to the left, z up,
origin on the robot's vertical axis. "Right" below is -y, as seen from the flange
in q_home looking forward.

Measured (tape, 2026-10-21): table height, work area, cage side, rail height and
section, camera position along the rail. PLACEHOLDER marks what is not measured.
"""

from __future__ import annotations

import numpy as np
import trimesh

# The (wooden) table is 25 mm below the top of the foot (where the screws go in).
# The foot top is 50.8 mm above the base mesh bottom, which is 24.8 mm below the
# ctu base frame; so the table is ~1 mm above the ctu base frame and the model's
# foot reaches 26 mm below the table (it sits in a cut-out in the table).
FOOT_TOP_Z = -0.0248 + 0.0508
TABLE_Z = FOOT_TOP_Z - 0.025
# Front of the robot's round base (x max of the cone bottom, base mesh).
ROBOT_FRONT_X = 0.099
# Work area on the table [[x_min, x_max], [y_min, y_max]]: the part the camera sees,
# 30 cm deep, 48.5 cm wide, centred on the robot's axis, starting 23.8 cm in front
# of the round base.
WORK_AREA_XY = np.array([[ROBOT_FRONT_X + 0.238, ROBOT_FRONT_X + 0.238 + 0.30],
                         [-0.2425, 0.2425]])

# Cage: regular octagon, inner side 75 cm, centred on the robot, a flat panel in
# front (wall normals every 45 deg starting at +x); frame of 40 x 40 mm profiles,
# so the walls are 40 mm thick.
CAGE_SIDE = 0.75
CAGE_APOTHEM = CAGE_SIDE / (2 * np.tan(np.pi / 8))  # axis to inner wall, 0.905 m
CAGE_WALL = 0.04
CAGE_HEIGHT = 1.08  # m, top edge above the table
# Bounding box of the cage interior, [[x_min, x_max], [y_min, y_max]].
CAGE_XY = np.array([[-CAGE_APOTHEM, CAGE_APOTHEM], [-CAGE_APOTHEM, CAGE_APOTHEM]])

# Camera: Basler ace 2 a2A1920-51gcBAS (Sony IMX392, colour, GigE), pointing
# straight down, on the robot's axis (y = 0), 50 cm from the outer face of the
# front cage wall, clamped to the right side of the rail. Not centred over the
# work area: ~4 cm closer to the robot (confirmed by eye).
CAMERA_RESOLUTION = (1920, 1200)  # px (width, height)
CAMERA_PIXEL = 3.45e-6  # m, IMX392 pixel pitch
# Body 29 x 29 mm footprint, 42 mm deep (ace 2 GigE housing), + the lens, a 28 mm
# tall cylinder, as one box (x, y, z).
CAMERA_SIZE = np.array([0.029, 0.029, 0.042 + 0.028])
CAMERA_X = CAGE_APOTHEM + CAGE_WALL - 0.50
# Camera rail: 40 x 40 mm aluminium profile along x, top 131 cm above the table,
# its right side touching the camera; held by a vertical 40 x 40 mm post standing
# on the front cage wall.
RAIL_SECTION = 0.04
RAIL_TOP_Z = TABLE_Z + 1.31
RAIL_Y = CAMERA_SIZE[1] / 2 + RAIL_SECTION / 2
POST_X = CAGE_APOTHEM + CAGE_WALL / 2
RAIL_X = (-POST_X, POST_X)  # over the whole cage
CAMERA_TOP_Z = RAIL_TOP_Z  # camera body flush with the rail top; PLACEHOLDER
# Camera frame (OpenCV: z = optical axis, x = image right, y = image down) at the
# front of the lens (PLACEHOLDER for the optical centre). The image's top edge
# faces the robot: image down = +x (away from the robot), image right = +y.
CAMERA_T = np.eye(4)
CAMERA_T[:3, :3] = np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
CAMERA_T[:3, 3] = CAMERA_X, 0.0, CAMERA_TOP_Z - CAMERA_SIZE[2]
# Focal length, ESTIMATED (not printed on the lens; a homography of the table seen
# straight down cannot separate it from the camera height): the image just fits
# the sticker rectangle of the work area, seen from the lens; mean of the fits of
# the image width and height, ~17 mm.
_lens_height = CAMERA_T[2, 3] - TABLE_Z
CAMERA_FOCAL = CAMERA_PIXEL * _lens_height * float(np.mean(
    np.array(CAMERA_RESOLUTION) / np.ptp(WORK_AREA_XY, axis=1)[::-1]))
# Intrinsics (pinhole, no distortion; Vojtěch's calibration will replace them).
CAMERA_K = np.array([[CAMERA_FOCAL / CAMERA_PIXEL, 0, (CAMERA_RESOLUTION[0] - 1) / 2],
                     [0, CAMERA_FOCAL / CAMERA_PIXEL, (CAMERA_RESOLUTION[1] - 1) / 2],
                     [0, 0, 1]])


def in_cage(xy: np.ndarray, margin: float = 0.0) -> bool:
    """Whether the point @xy is inside the cage, at least @margin from its walls."""
    angles = np.arange(8) * np.pi / 4
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    return bool(np.all(normals @ np.asarray(xy) <= CAGE_APOTHEM - margin))


def _box(size, center, yaw: float = 0.0) -> trimesh.Trimesh:
    T = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    T[:3, 3] = center
    return trimesh.creation.box(extents=size, transform=T)


def cell_meshes() -> dict[str, trimesh.Trimesh]:
    """Collision meshes of the cage walls, the camera rail with its post and the
    camera, in the robot base frame (the table itself is the plane z = TABLE_Z)."""
    meshes = {}
    for k in range(8):
        a = k * np.pi / 4
        r = CAGE_APOTHEM + CAGE_WALL / 2
        meshes[f"cage_wall_{k}"] = _box(
            (CAGE_WALL, CAGE_SIDE, CAGE_HEIGHT),
            (r * np.cos(a), r * np.sin(a), TABLE_Z + CAGE_HEIGHT / 2), a)
    meshes["camera_rail"] = _box(
        (RAIL_X[1] - RAIL_X[0], RAIL_SECTION, RAIL_SECTION),
        (sum(RAIL_X) / 2, RAIL_Y, RAIL_TOP_Z - RAIL_SECTION / 2))
    post_height = RAIL_TOP_Z - RAIL_SECTION - (TABLE_Z + CAGE_HEIGHT)
    meshes["camera_rail_post"] = _box(
        (RAIL_SECTION, RAIL_SECTION, post_height),
        (POST_X, RAIL_Y, TABLE_Z + CAGE_HEIGHT + post_height / 2))
    meshes["camera"] = _box(
        CAMERA_SIZE, (CAMERA_X, 0.0, CAMERA_TOP_Z - CAMERA_SIZE[2] / 2))
    return meshes
