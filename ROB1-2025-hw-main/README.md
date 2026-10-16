# ROB1 2025 semestral work hardware files

## Structure of the repository

- `bases`: Base plates for holding the different puzzle pieces. It has ArUco 4x4 markers in a diagonal pattern.
- `drawings`: Technical (amateur) drawings of all the parts. **All dimensions are in Milimeters (mm).**
- `print_files`: G-codes for printing all the parts on a Prusa MK3 with a 0.4 nozzle and 3mf files (usual slicer format).
- `robot_ee`: 3D models of adapters for all 3 robot models used (CRS, Bosch and Mitsubishi) and an universal "Hoop" that connects via two magnets onto them.
- `puzzles`: 3D models of different geometric challenges (A,B,C,D,E), again they held to the base plate via two to three magnets.

You'll find every 3D model file in STEP form, which can be opened and edited in any CAD software.

## Hardware information

- Printed out of PETG, 3 perimeters, 10-20% infill. (Cubic or Gyroid infill recommended)
- Robot adapters are two-piece and are designed for easy assembly and disassembly, magnets can be easily harvested, no glue is used.
- Hoop piece is the only part with printed-in magnets.
- Magnets for Hoop: Diameter = 10 mm, height = 5 mm, neodymium.
- Magnets for base & puzzles: Diameter = 10 mm, height = 3 mm, neodymium.