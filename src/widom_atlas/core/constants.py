"""Physical and tolerance constants — single source of truth for the package."""

from __future__ import annotations

from typing import Final

KB_EV_PER_K: Final[float] = 8.617333262e-5
EV_TO_KJMOL: Final[float] = 96.48533212331001  # 1 eV = 96.485332... kJ/mol

DEFAULT_SYMPREC: Final[float] = 1e-2
DEFAULT_ANGLE_TOLERANCE_DEG: Final[float] = 5.0
DEFAULT_BASIN_MATCH_TOL_A: Final[float] = 0.35
DEFAULT_ENERGY_MATCH_TOL_KJMOL: Final[float] = 2.0
DEFAULT_DENSITY_GRID_SHAPE: Final[tuple[int, int, int]] = (48, 48, 48)

ALLOWED_GASES_V1: Final[frozenset[str]] = frozenset({"CO2", "N2", "CH4"})
