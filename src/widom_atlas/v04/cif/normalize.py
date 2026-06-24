"""CIF label normalization for RASPA3 input generation.

RASPA3 matches `_atom_site_label` against `PseudoAtoms.name` exactly, so a
CIF with labels `O1, O2, …, Si1, Si2, …` must either be re-labelled OR
the force_field must enumerate every label. We re-label.

Rules:
- For Si and O in pure-silica zeolites (IZA CIFs): collapse all O* → O,
  all Si* → Si.
- For Na-Rho fixtures: collapse Si* → Si, O* → O, Al* → Al, Na* → Na.
  Carbon/oxygen of CO2 (CO1, CO2, OC1, OC2, etc.) keep their distinctive
  labels.
- For DDEC-charged MOF CIFs (RUBTAK01_SL_DDEC.cif, FIQCEN_clean_min_charges.cif):
  preserve original labels (they carry the per-charge identity).
"""
from __future__ import annotations

import re
from pathlib import Path


def normalize_zeolite_labels(cif_text: str) -> tuple[str, dict[str, int]]:
    """Collapse Si* → Si, O* → O, Al* → Al, Na* → Na in atom-site rows.

    Rows that look like adsorbed CO2 atoms (labels OC1..OC4, CO1, CO2_atom)
    are STRIPPED from the framework — RASPA3 must insert its own probes.
    """
    lines = cif_text.splitlines()
    out: list[str] = []
    in_atom_loop = False
    columns: list[str] = []
    counts: dict[str, int] = {}
    drop_co2_labels = {"OC1", "OC2", "OC3", "OC4", "CO1", "CO2_atom", "CO2"}

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_loop = False
            columns = []
            out.append(line)
            continue
        if stripped.startswith("_atom_site_"):
            columns.append(stripped)
            if "_atom_site_label" in columns and "_atom_site_type_symbol" in columns:
                in_atom_loop = True
            out.append(line)
            continue
        if in_atom_loop and stripped and not stripped.startswith("_") and not stripped.startswith("#"):
            fields = re.split(r"\s+", stripped)
            if len(fields) < 2:
                out.append(line)
                continue
            label = fields[0]
            element = fields[1]
            # Drop refinement-stored CO2 atoms — RASPA3 inserts its own probes
            if label in drop_co2_labels:
                continue
            new_label = element if element in ("Si", "O", "Al", "Na", "Cs", "Li", "K") else label
            counts[new_label] = counts.get(new_label, 0) + 1
            leading_ws = line[: len(line) - len(line.lstrip())]
            rest = re.split(r"\s+", line.strip())
            rest[0] = new_label
            out.append(leading_ws + "  ".join(rest))
            continue
        if in_atom_loop and (not stripped or stripped.startswith("loop_")):
            in_atom_loop = False
        out.append(line)
    return "\n".join(out) + ("\n" if cif_text.endswith("\n") else ""), counts


_LABEL_ELEMENT = re.compile(r"^([A-Z][a-z]?)")


def _ensure_type_symbol_column(text: str) -> str:
    """Inject `_atom_site_type_symbol` after `_atom_site_label` if missing.

    RASPA3 v3.0.29 silently drops every atom row whose element it cannot
    resolve from the CIF (Number Of Atoms = 0 in the output). Adding the
    element symbol (derived from the label prefix) fixes parsing for the
    DDEC CIFs that omit this column.
    """
    lines = text.splitlines()
    columns: list[str] = []
    column_start: int | None = None
    rows_start: int | None = None
    in_atom_loop = False
    type_symbol_col: int | None = None
    label_col: int | None = None
    for idx, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            columns = []
            column_start = idx + 1
            rows_start = None
            in_atom_loop = False
            continue
        if s.startswith("_atom_site_"):
            columns.append(s)
            if "_atom_site_label" in columns:
                in_atom_loop = True
            continue
        if in_atom_loop and column_start is not None and rows_start is None and s and not s.startswith("#") and not s.startswith("_"):
            rows_start = idx
            if "_atom_site_type_symbol" in columns:
                type_symbol_col = columns.index("_atom_site_type_symbol")
            label_col = columns.index("_atom_site_label")
            break
    if rows_start is None or label_col is None:
        return text
    if type_symbol_col is not None:
        return text  # already present
    # Insert column header right after _atom_site_label and add a type column to each data row.
    new_lines: list[str] = list(lines[: rows_start])
    # Find where to insert the new column header: after _atom_site_label
    for i, c in enumerate(new_lines):
        if "_atom_site_label" in c:
            insert_idx = i + 1
            indent = c[: len(c) - len(c.lstrip())]
            new_lines.insert(insert_idx, f"{indent}_atom_site_type_symbol")
            break
    # Now process rows from `rows_start` (in the original lines)
    for line in lines[rows_start:]:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("_") or s.startswith("loop_"):
            new_lines.append(line)
            continue
        fields = re.split(r"\s+", s)
        if len(fields) <= label_col:
            new_lines.append(line)
            continue
        label = fields[label_col]
        m = _LABEL_ELEMENT.match(label)
        element = m.group(1) if m else label
        new_fields = [*fields[: label_col + 1], element, *fields[label_col + 1 :]]
        leading_ws = line[: len(line) - len(line.lstrip())]
        new_lines.append(leading_ws + " ".join(new_fields))
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def normalize_cif_to_workdir(
    src: Path, dst: Path, mode: str
) -> dict[str, int]:
    """Normalize a CIF for RASPA3 consumption.

    mode='zeolite' → collapse Si*/O*/Al*/Na* labels + inject type column if needed
    mode='preserve' → inject _atom_site_type_symbol column if missing (DDEC CIFs)
    """
    text = src.read_text()
    if mode == "preserve":
        dst.write_text(_ensure_type_symbol_column(text))
        return {}
    if mode == "zeolite":
        new_text, counts = normalize_zeolite_labels(text)
        new_text = _ensure_type_symbol_column(new_text)
        dst.write_text(new_text)
        return counts
    raise ValueError(f"unknown mode: {mode}")
