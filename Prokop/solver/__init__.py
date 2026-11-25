"""The puzzle-D solver (docs/SOLVER_PLAN.md); shared types in solver.contracts.

The solver builds on Natka's planning code (Natka/: puzzle path, hoop targets,
IK, safety metrics, selection, timed plan), so her directory is put on the
import path here (appended: Prokop's own modules win a name clash).
"""

import sys
from pathlib import Path

NATKA = Path(__file__).resolve().parents[2] / "Natka"
if str(NATKA) not in sys.path:
    sys.path.append(str(NATKA))
