# Measurements

Every number the simulation uses that is not in a CAD file or in `ctu_crs`:
its value, how it was obtained, and where it lives in the code. The experiments
behind them are described in [EXPERIMENTS.md](EXPERIMENTS.md) (E1 to E9).

**Frame:** the `ctu_crs` robot base frame, in metres. x points forward (where
the arm points in `q_home`), y to the left, z up. The origin is on the robot's
vertical axis, 305.2 mm below the shoulder axis. "Right" means −y, as seen from
the flange in `q_home`, looking forward.

**Kinds:** *measured* = tape, stopwatch or robot log; *derived* = computed from
measured values; *estimated* = indirect, with the method given; *assumed* = not
measured; *decided* = a team choice.

## Robot and table

| Quantity | Value | Kind | Source | Code |
|---|---|---|---|---|
| Robot foot (square steel block under the round base) | 203 mm square, 50.8 mm high | from the model | bottom of the RoboDK base mesh | `models/crs93.urdf` |
| Table surface below the top of the foot | 25 mm | measured (tape) | E1 | `cell.TABLE_Z` |
| Table height in the robot frame | z = +1.0 mm | derived | −24.8 mm (mesh bottom) + 50.8 mm (foot) − 25 mm; the foot sits 26 mm deep in a cut-out of the table | `cell.TABLE_Z` |
| Front of the round base | x = 0.099 m | from the model | x max of the cone bottom | `cell.ROBOT_FRONT_X` |
| Hard home pose (encoders zeroed) | (163.4, 0.3, −56.0, 176.7, 14.9, −172.0)° | measured (`get_q`) | E4; equals `ctu_crs` IRC 0 within 0.1° | `CRS93Sim.hard_home` |
| Soft home | `q_home` = (0, 0, −45, 0, −45, 0)° | measured | E4 | `ctu_crs` |
| Adapter turn on the flange | 180° (hoop points **up** in `q_home`) | measured (seen) | E4 | `tools/robodk2urdf.ADAPTER_TURN`, `models/crs93.urdf` |
| Hoop on the adapter | triangle in triangle, as in CAD | decided | a small shift on the magnets is accepted | `placement.HOOP_MOUNT` |

## Work area, cage, camera rail

| Quantity | Value | Kind | Source | Code |
|---|---|---|---|---|
| Work area (sticker rectangle, about the camera view) | 30 cm deep × 48.5 cm wide, centred on the robot's axis, starting 23.8 cm in front of the round base: x 0.337–0.637 m, y ±0.2425 m | measured (tape) | E2 | `cell.WORK_AREA_XY` |
| Cage | regular octagon, inner side 75 cm (axis to inner wall 0.905 m), a flat panel in front | measured (tape) | E2 | `cell.CAGE_SIDE`, `CAGE_APOTHEM` |
| Cage wall thickness | 40 mm (frame of 40 × 40 mm profiles) | measured | E2 | `cell.CAGE_WALL` |
| Cage height | 108 cm above the table | measured (tape) | E2 | `cell.CAGE_HEIGHT` |
| Camera rail | 40 × 40 mm profile along x over the whole cage, top 131 cm above the table, on a vertical post standing on the front cage wall | measured (tape) | E2 | `cell.RAIL_*`, `POST_X` |

## Camera

| Quantity | Value | Kind | Source | Code |
|---|---|---|---|---|
| Model | Basler ace 2 a2A1920-51gcBAS, Sony IMX392, colour, GigE | given | station | `cell.py` |
| Resolution | 1920 × 1200 × 3 | measured | E7 | `cell.CAMERA_RESOLUTION` |
| Pixel pitch | 3.45 µm | datasheet (IMX392) | one retailer lists 3.5 µm | `cell.CAMERA_PIXEL` |
| Body + lens | 29 × 29 mm footprint, 42 mm body (depth not confirmed) + 28 mm lens cylinder | given / measured (lens) | E2 | `cell.CAMERA_SIZE` |
| Mounting | on the robot's axis (y = 0), on the right side of the rail, pointing straight down; top flush with the rail top (assumed) | measured / assumed | E2, E8 (the centre sticker strip runs through the image centre) | `cell.CAMERA_T` |
| Image orientation | top edge towards the robot: image right = +y, image down = +x | measured | E2, E8 | `cell.CAMERA_T` |
| Camera x | **0.475 m** (≈ 47 cm from the outer face of the front cage wall) | measured (camera image) | E8; the tape said 50 cm (x 0.445 m), the hoop in the image showed 3.0 cm more | `cell.CAMERA_X` |
| Lens height above the table | 1.24 m | derived | rail top − body − lens | `cell.CAMERA_T` |
| Focal length | 17.0 mm (fx = fy = 4933 px), pinhole, no distortion | estimated | E9: the image just fits the sticker rectangle; checked by the ring size in E8 (1 %) | `cell.CAMERA_FOCAL`, `CAMERA_K` |
| `grab_image` duration | 186 ms (first call 0.8 s) | measured | E7: `ctu_crs` starts and stops the camera per call, so at most ~5 images/s | `simulation.GRAB_TIME` |
| Image delay (age when `grab_image` returns) | 108 ms → 0.11 s | measured | E8 | `simulation.CAMERA_DELAY` |

## Turntable, base, puzzle

| Quantity | Value | Kind | Source | Code |
|---|---|---|---|---|
| Turntable | cylinder Ø 137 mm, 34 mm high | measured (tape) | E2 | `models/turntable.urdf` |
| Period at the slowest speed | **28.0 s/rev**, the same both ways: CW 28.00 ± 0.05 s (5 turns), CCW 28.06 ± 0.02 s (4 turns) | measured (stopwatch) | E3 | `view_models.SPIN_PERIOD` |
| Base on its foam | 30 mm total: 5 mm printed plate (+0.6 mm marker relief) on 25 mm foam with the base footprint (125 × 125 mm) | measured (tape) | E2 | `placement.BASE_FOAM`, `models/base.urdf` |
| Densities for the tip-over check | PETG 1270 kg/m³ × 0.5 fill; foam 30 kg/m³; magnets 10 × 3 mm NdFeB | assumed | print settings of the hardware repo | `placement.PRINT_DENSITY`, `FOAM_DENSITY`, `MAGNET_MASS` |

## Motion and serial line of the control unit

All are measured with `tools/log_motion.py` (E5, E6) and are the defaults of
`CRS93Sim`; pass `0` for an ideal robot.

| Quantity | Value | Kind | Code |
|---|---|---|---|
| Joint speeds | match the `ctu_crs` settings (IRC·256/ms) within 0.1°/s: j1 17.6°/s at relative speed 0.25, 29.8°/s at 0.5; j4 58.8°/s, j6 147.1°/s at 0.5 | measured | `CRS93Sim._joint_limits_rad` |
| Accelerations | 170–390°/s² (j1), 276°/s² (j4), 3500°/s² (j6); the sim's assumed units (IRC·256/ms²) give 140–180, 283, 3400 | measured (to about ×2: a ramp lasts 1–2 samples) | `CRS93Sim._joint_limits_rad` |
| Motion profile | coordinated: all joints start and stop together, trapezoidal speed | measured | `simulation.Segment` |
| Start of motion after `move_to_q` | 93–147 ms, mean 126 ms → 0.12 s | measured (profile fit) | `simulation.COMMAND_DELAY` |
| `move_to_q` blocks | ~25 ms per call (serial write) | measured | `simulation.SEND_DELAY` |
| `get_q` round trip | 51 ms median (41–154 ms) | measured | `simulation.QUERY_DELAY` |
| `in_motion` round trip | ~10 ms | measured (loop period − `get_q`) | `simulation.STATUS_DELAY` |
| Sampling with `get_q` + `in_motion` | ~18 Hz | measured | — |
| End of a move | overshoot ~0.3°, settled in ~0.25 s | measured | not simulated |
| Queued moves | blended only if the next move continues in exactly the same joint-space direction; corners and reversals stop; the vias of a bending path nearly stop (down to a few °/s) | measured (E6) | `CRS93Sim._enqueue`, `BLEND_TOL` |
| Queue limit | never reached in the tests ("coordmv queue full" never printed) | — | not simulated |

## Superseded values

Kept for the record, so that old notes and logs can be read:

| Quantity | Old value | Replaced by | Why |
|---|---|---|---|
| Table height | 31 mm below the model's base bottom (z = −55.8 mm) | 25 mm below the top of the foot (z = +1.0 mm) | the RoboDK base mesh already contains the foot; the first reading assumed it did not (E1) |
| Work area | 48 × 30 cm, 22.2 cm in front of the robot | 48.5 × 30 cm, 23.8 cm in front of the round base | re-measured at the CRS93 station (E2) |
| Camera x | 0.445 m (taped 50 cm from the cage) | 0.475 m | the camera image (E8) |
| Turntable | Ø 160 mm × 45 mm, 16 s/rev (guesses) | Ø 137 mm × 34 mm, 28.0 s/rev | E2, E3 |
| Hard home in the sim | q = 0 per joint group | the encoder-zero pose | E4 |
| Joint-6 view offset of 180° (old viewer) | — | adapter turned 180° in the URDF | the offset broke `fk(q)` = drawn flange (E4) |
