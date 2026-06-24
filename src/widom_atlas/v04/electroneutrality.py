"""T010: electroneutrality check over CIF atomic partial charges."""
from __future__ import annotations

import re
from pathlib import Path


def _parse_cif_charges(text: str) -> list[float]:
    """Extract values from any `_atom_site_charge` column in a P1 CIF.

    The IZA zeolite CIFs in our fixtures do NOT carry a charge column;
    DDEC-style CIFs (e.g., FIQCEN_clean_min_charges.cif, RUBTAK01_SL_DDEC.cif)
    do. If no charge column is present, this returns an empty list and the
    caller should treat the framework as neutral-by-construction.
    """
    lines = text.splitlines()
    in_loop = False
    columns: list[str] = []
    rows_start: int | None = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "loop_":
            in_loop = True
            columns = []
            rows_start = None
            continue
        if in_loop:
            if s.startswith("_"):
                columns.append(s)
                continue
            # First non-keyword data row
            if rows_start is None:
                rows_start = i
            if "_atom_site_charge" in columns:
                break
    else:
        return []
    if rows_start is None or "_atom_site_charge" not in columns:
        return []
    charge_idx = columns.index("_atom_site_charge")
    charges: list[float] = []
    for ln in lines[rows_start:]:
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("_") or s.startswith("loop_"):
            # End of data block
            if not s or s.startswith("loop_"):
                break
            continue
        fields = re.split(r"\s+", s)
        if len(fields) <= charge_idx:
            continue
        try:
            charges.append(float(fields[charge_idx]))
        except ValueError:
            continue
    return charges


def check_electroneutrality(
    cif_path: Path, tolerance_e: float = 1e-2
) -> tuple[bool, float, int]:
    """Sum charges from a CIF; return (passes, sum, n_atoms).

    `passes` is True if |sum| < tolerance OR the CIF has no charge column.
    """
    text = cif_path.read_text()
    charges = _parse_cif_charges(text)
    if not charges:
        return (True, 0.0, 0)
    total = sum(charges)
    return (abs(total) < tolerance_e, total, len(charges))
