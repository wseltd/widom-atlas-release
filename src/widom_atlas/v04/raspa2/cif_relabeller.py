"""VOGTIV (Mg-MOF-74) CIF → Lin/Mercado sublattice relabelling.

Writes a new CIF in the working directory with atom labels rewritten to
match the sublattice-specific Mof_Mg / Mof_Oa / Mof_Ob / Mof_Oc / Mof_Ca
/ Mof_Cb / Mof_Cc / Mof_Cd / Mof_H tags used in Lin/Mercado's
raspa_pseudo_atoms.def. The element symbol in the `_atom_site_type_symbol`
column is also kept up to date for RASPA2 CIF parsing.

The classification is purely geometric: each atom's neighbourhood
(O-bond count for C atoms, Mg-bond count for O atoms, etc.) decides its
sublattice label. This was added 2026-05-17 after a verified bug in the
old index-based pattern relabeller: in the operator-supplied VOGTIV CIF
(CoRE-MOF July 2014), the per-ligand atom ordering was [Cd, Cc, Cb, Ca]
rather than [Ca, Cb, Cc, Cd], so the old code was putting Mof_Ca's
Buckingham parameters on the carboxylate C and Mof_Cd's on the
phenoxide-attached C — a swap that distorted the Mg-MOF-74 / CO2 K_H
and Q_st by attaching the wrong cross-pair parameters to the wrong
sites.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..cif.mg_mof74_relabel import classify_vogtiv_geometric


def _lattice_matrix_from_params(
    a: float, b: float, c: float,
    alpha_deg: float, beta_deg: float, gamma_deg: float,
) -> np.ndarray:
    """Standard CIF lattice vectors with a along x, b in xy-plane."""
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz_sq, 0.0))
    return np.array([
        [a, 0.0, 0.0],
        [b * cos_g, b * sin_g, 0.0],
        [cx, cy, cz],
    ])


def _parse_vogtiv_cif(cif_path: Path):
    """Minimal CIF parser tailored to the CoRE-MOF VOGTIV_clean_h.cif schema.

    Returns (lattice_matrix, atoms) where each atom is a dict with keys
    {original_line_index, label, element, frac, cart}.
    """
    text = cif_path.read_text()
    lines = text.splitlines()

    a = b = c = None
    alpha = beta = gamma = None
    for line in lines:
        m = re.match(r"_cell_length_a\s+([0-9.eE+-]+)", line.strip())
        if m:
            a = float(m.group(1))
        m = re.match(r"_cell_length_b\s+([0-9.eE+-]+)", line.strip())
        if m:
            b = float(m.group(1))
        m = re.match(r"_cell_length_c\s+([0-9.eE+-]+)", line.strip())
        if m:
            c = float(m.group(1))
        m = re.match(r"_cell_angle_alpha\s+([0-9.eE+-]+)", line.strip())
        if m:
            alpha = float(m.group(1))
        m = re.match(r"_cell_angle_beta\s+([0-9.eE+-]+)", line.strip())
        if m:
            beta = float(m.group(1))
        m = re.match(r"_cell_angle_gamma\s+([0-9.eE+-]+)", line.strip())
        if m:
            gamma = float(m.group(1))
    if any(v is None for v in (a, b, c, alpha, beta, gamma)):
        raise ValueError(f"missing cell parameters in {cif_path}")
    lattice = _lattice_matrix_from_params(a, b, c, alpha, beta, gamma)

    columns: list[str] = []
    in_atom_loop = False
    label_col = None
    type_col = None
    fx_col = fy_col = fz_col = None
    atoms: list[dict] = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            columns = []
            in_atom_loop = False
            continue
        if s.startswith("_atom_site_"):
            columns.append(s)
            continue
        if not in_atom_loop and columns and all(
            k in columns for k in (
                "_atom_site_label",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            )
        ):
            in_atom_loop = True
            label_col = columns.index("_atom_site_label")
            fx_col = columns.index("_atom_site_fract_x")
            fy_col = columns.index("_atom_site_fract_y")
            fz_col = columns.index("_atom_site_fract_z")
            if "_atom_site_type_symbol" in columns:
                type_col = columns.index("_atom_site_type_symbol")
        if in_atom_loop and s and not s.startswith("#") and not s.startswith("_"):
            fields = re.split(r"\s+", s)
            if label_col is None or len(fields) <= max(label_col, fx_col, fy_col, fz_col):
                continue
            label = fields[label_col]
            element = fields[type_col] if type_col is not None and len(fields) > type_col else None
            if element is None:
                m = re.match(r"^([A-Z][a-z]?)", label)
                element = m.group(1) if m else label
            try:
                fx = float(fields[fx_col])
                fy = float(fields[fy_col])
                fz = float(fields[fz_col])
            except ValueError:
                continue
            cart = np.array([fx, fy, fz]) @ lattice
            atoms.append({
                "original_line_index": idx,
                "label": label,
                "element": element,
                "frac": (fx, fy, fz),
                "cart": (float(cart[0]), float(cart[1]), float(cart[2])),
            })

    return lattice, atoms


def relabel_vogtiv_cif(src_cif: Path, dst_cif: Path) -> dict[str, int]:
    """Rewrite a VOGTIV-style CIF with geometry-based sublattice labels.

    Returns a dict mapping each new sublattice label to its count for
    quick sanity assertions. Raises ValueError if the classifier can't
    label every C or O atom.
    """
    lattice, atoms = _parse_vogtiv_cif(src_cif)
    elements = [a["element"] for a in atoms]
    cartesian = np.array([a["cart"] for a in atoms])
    fractional = np.array([a["frac"] for a in atoms])

    new_labels = classify_vogtiv_geometric(
        elements=elements,
        cartesian=cartesian,
        fractional=fractional,
        lattice_matrix=lattice,
    )

    counts: dict[str, int] = {}
    label_by_line: dict[int, tuple[str, str]] = {}
    for atom, new_label in zip(atoms, new_labels, strict=True):
        counts[new_label] = counts.get(new_label, 0) + 1
        label_by_line[atom["original_line_index"]] = (new_label, atom["element"])

    src_text = src_cif.read_text()
    lines = src_text.splitlines()
    out: list[str] = []

    columns: list[str] = []
    in_atom_loop = False
    label_col = None
    type_col = None
    for idx, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            columns = []
            in_atom_loop = False
            out.append(line)
            continue
        if s.startswith("_atom_site_"):
            columns.append(s)
            if "_atom_site_label" in columns:
                in_atom_loop = True
                label_col = columns.index("_atom_site_label")
                if "_atom_site_type_symbol" in columns:
                    type_col = columns.index("_atom_site_type_symbol")
            out.append(line)
            continue
        if in_atom_loop and idx in label_by_line:
            new_label, element = label_by_line[idx]
            fields = re.split(r"\s+", s)
            if label_col is None or len(fields) <= label_col:
                out.append(line)
                continue
            leading_ws = line[: len(line) - len(line.lstrip())]
            fields[label_col] = new_label
            if type_col is not None and len(fields) > type_col:
                fields[type_col] = element
            out.append(leading_ws + "  ".join(fields))
            continue
        out.append(line)

    dst_cif.write_text("\n".join(out) + ("\n" if src_text.endswith("\n") else ""))
    return counts
