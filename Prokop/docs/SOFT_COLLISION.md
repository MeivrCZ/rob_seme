# Soft collision model for planning

The simulator's exact meshes (`collision.py`, "hard" collisions) are the
environment's ground truth: they decide whether the simulated robot crashed.
Planning code must not use them. It uses this **soft model** instead: every
body as a few boxes, built from the *perceived* poses (the `TurntableModel`),
checked with dot products only.

Module: `solver/soft_collision.py`. Boxes: `models/soft/*_soft.urdf`. Fitting:
`tools/fit_soft_boxes.py`. Validation: `experiments/validate_soft_collision.py`.
Tests: `tests/test_soft_collision.py`.

## Usage

```python
from solver.soft_collision import SoftScene

scene = SoftScene(model)          # contracts.TurntableModel (perceived placement)
c = scene.clearance(q, theta)     # joint configuration [rad], turntable angle [rad]
c.free                            # True: no box closer than its margin
c.distance, c.pair                # least distance to the cell/table/task [m], margin subtracted
c.self_gap, c.self_pair           # least gap between arm parts [m]
```

A query takes about 1.4 ms (2.4 ms with the hoop around the bar), against
8.6 ms for the exact meshes. `SoftScene` keeps the obstacle poses of the last
`theta`, so build one per placement and reuse it.

## What is modelled

| Body | Boxes | Frame |
|---|---|---|
| Robot links | one per link; `link1`, `link2` in 2 slabs, `link4` in 3 | the link (`crs93_soft.urdf`) |
| Adapter | 1 | `adapter` |
| Hoop | handle in 3 slabs, the ring as 36 sector boxes around the hole | `hoop_link` (`hoop_soft.urdf`), mounted at `adapter_front` @ `HOOP_MOUNT` |
| Puzzle D | the 10 x 10 mm bar: top bar, vertical run, jog, stem (1 each), the R50 bend (6), the foot | `puzzle_D_link` (`puzzle_D_soft.urdf`) |
| Base | 1 (plate, guides and foam) | `base_link` (`base_soft.urdf`) |
| Turntable | 2, rotated 45°: the body lies in their **intersection** (named `intersect_*`) | `turntable_link` (`turntable_soft.urdf`) |
| Cell | cage walls, camera rail, post, camera | robot base frame (`cell_soft.urdf`) |
| Table | the plane z = `cell.TABLE_Z` | — |

The soft URDFs have the same links and joints as the exact models but only
`<collision>` elements with `<box>` geometry, so the exact models stay
untouched (the simulator reads every collision element of those). The robot's
link frames come from the joint chain in `crs93_soft.urdf` (the same as the
URDF to 1e-15). The task boxes are posed from the `TurntableModel`
(`base_T`, `puzzle_T`, `top_T`), i.e. from what perception measured.

The puzzle boxes are built from the wire's centreline in `solver/wire.py`
(measured on the mesh), not fitted to the mesh; the validation checks that
they contain it.

## How the test works

- **Box against box:** the separating-axis test with the 15 axes (3 face
  normals of each box, 9 cross products of their edges). The largest gap
  along these axes is a **lower bound** of the distance between the boxes,
  and it is <= 0 exactly when they overlap. Far apart it underestimates
  (by up to about √3 for boxes separated diagonally); near contact it is
  tight.
- **Box against the table:** the lowest corner (exact).
- **Turntable:** inside both of its boxes, so at least as far as the farther
  one.
- **Hierarchy:** the boxes of a part (and the hoop's handle and ring
  quarters, the puzzle and its bend) have enclosing boxes. A query tests
  those first and refines only pairs closer than `REFINE` (2 cm). An
  enclosing box contains its children, so the result is still a lower bound.
- **Not checked:** joint neighbours against each other (their boxes overlap
  by construction; the joint limits keep them apart), the robot's base link
  against the table (it stands in a cut-out), the task objects against the
  table and each other.

## Margins

Subtracted from the distances (`SoftScene(model, margin, hoop_puzzle_margin,
self_margin)`):

| Pair | Margin | Default |
|---|---|---|
| arm and hoop against the cell, the table, the turntable and the base | `MARGIN` | 10 mm |
| hoop against the puzzle | `HOOP_PUZZLE_MARGIN` | 3 mm (the ring has about 23 mm around the bar) |
| arm parts against each other | `SELF_MARGIN` | 0 (the boxes are larger than the parts already) |

Perception error should be added to these once it is known.

## Validation (2026-11-25)

`uv run python Prokop/experiments/validate_soft_collision.py --states 500`:

1. **Forward kinematics** of the soft chain vs the URDF: 4e-16.
2. **Coverage:** every vertex of every mesh lies inside its boxes (all arm
   parts, the hoop's 3310, the puzzle's 604, base, turntable, cell).
3. **500 random states** (placement in the work area, turntable angle, q
   within the limits): the soft distance **never** exceeded the exact one
   (table, cell, task, and the hierarchical query), and every exact contact
   was a soft contact. Soft self-contacts: 6, exact: 3. How much smaller the
   soft distances are (median / 90 %): table 0.1 / 0.5 mm, cell 7 / 27 mm,
   task 65 / 122 mm (mostly far-away pairs).
4. **Rings around the bar** (IK candidates at the checkpoints, exactly
   collision free): the soft model accepts 94–95 % of them without margins,
   67–71 % with the default margins.
5. **Speed:** 1.4 ms per query (random q), 2.4 ms with the ring around the bar.

## Changing the boxes

Edit `tools/fit_soft_boxes.py` (e.g. `SPLIT`, `RING_SECTORS`, `BEND_PIECES`),
then regenerate and validate:

```bash
uv run python Prokop/tools/fit_soft_boxes.py
uv run python Prokop/experiments/validate_soft_collision.py
uv run pytest Prokop/tests/test_soft_collision.py
```

Notes from fitting: the hoop's hole has a radius of 30.0 mm in the CAD
(`models/meshes/hoop.stl`; outer radius 34.8 mm, 6.6 mm thick), not the
31.2 mm of the nominal 62.4 mm diameter. One box per long link was too coarse
(link 4 and link 6 boxes overlapped in most wrist poses); the slabs fixed that.
