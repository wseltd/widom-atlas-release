"""Loader for the 1a Mg-MOF-74 + CO2 Lin/Mercado Buckingham system.

Reads:
  * VOGTIV CIF (relabelled geometrically per Mg-MOF-74 sublattice convention).
  * Lin/Mercado `raspa_pseudo_atoms.def` for per-atom charges + masses.
  * Lin/Mercado `raspa_force_field.def` for per-pair Buckingham + LJ entries
    (after the `(C, A, B)` → `(A, B, C)` column-order fix and the
    `buckingham` → `BUCKINGHAM2` hard-core upgrade documented in
    `raspa2/input_writer.py::_rewrite_lin_mercado_force_field_acb_to_abc`).

Returns a `NativeSystem` ready for `run_native_widom(... enable_ewald=True)`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..raspa2.cif_relabeller import _parse_vogtiv_cif, relabel_vogtiv_cif
from .potentials import BuckinghamAExpC6, LennardJones12_6, PairTable
from .system import NativeSystem, ProbeMolecule


def _parse_lin_mercado_pseudo_atoms(
    pseudo_path: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (charges_e_by_type, mass_amu_by_type) from the Lin/Mercado file."""
    charges: dict[str, float] = {}
    masses: dict[str, float] = {}
    for line in pseudo_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        fields = re.split(r"\s+", s)
        if len(fields) < 6:
            continue
        if fields[0].isdigit():
            continue
        type_name = fields[0]
        try:
            mass = float(fields[4])
            charge = float(fields[5])
        except ValueError:
            continue
        masses[type_name] = mass
        charges[type_name] = charge
    return charges, masses


def _parse_lin_mercado_force_field(ff_path: Path) -> dict[tuple[str, str], dict]:
    """Parse the per-pair entries in raspa_force_field.def.

    The operator-supplied file's per-pair Buckingham rows are in `(C, A, B)`
    order (see `raspa2/input_writer.py::_rewrite_lin_mercado_force_field_acb_to_abc`
    for the verification cross-check against Lin 2014 SI Table S7). We
    rebuild the (A, B, C) representation here.

    Returns a dict mapping (type_a, type_b) → params dict with form name
    ("BUCKINGHAM" or "LJ") and parameters.
    """
    pairs: dict[tuple[str, str], dict] = {}
    text = ff_path.read_text().splitlines()
    in_pairs = False
    for line in text:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("# number of defined interactions"):
            in_pairs = True
            continue
        fields = re.split(r"\s+", s)
        if len(fields) < 5:
            continue
        kind = fields[2].lower() if len(fields) > 2 else ""
        if kind == "buckingham":
            try:
                t1, t2 = fields[0], fields[1]
                published_C = float(fields[3])
                published_A = float(fields[4])
                published_B = float(fields[5])
                pairs[(t1, t2)] = {
                    "kind": "BUCKINGHAM",
                    "A_K": published_A,
                    "B_inv_angstrom": published_B,
                    "C_K_angstrom6": published_C,
                }
            except (ValueError, IndexError):
                continue
        elif kind in ("lennard-jones", "lennard_jones", "lj"):
            try:
                t1, t2 = fields[0], fields[1]
                eps = float(fields[3])
                sig = float(fields[4])
                pairs[(t1, t2)] = {
                    "kind": "LJ",
                    "epsilon_K": eps,
                    "sigma_angstrom": sig,
                }
            except (ValueError, IndexError):
                continue
    return pairs


def load_1a_native_lin_mercado(
    repo_root: Path,
    lin_mercado_pkg: Path | None = None,
    cif_path: Path | None = None,
    cutoff_angstrom: float = 12.8,
    hardcore_angstrom: float = 1.0,
) -> NativeSystem:
    """Build a NativeSystem for 1a Mg-MOF-74 + CO2 Lin/Mercado."""
    if lin_mercado_pkg is None:
        lin_mercado_pkg = repo_root / "docs/research/dataset-research-for-v0.4/9"
    if cif_path is None:
        cif_path = (
            repo_root
            / "docs/research/dataset-research-for-v0.4/15/core-mof-sep2014/core-mof-july2014/VOGTIV_clean_h.cif"
        )

    tmp_relabelled = repo_root / "evidence" / "_native_vogtiv_relabelled.cif"
    tmp_relabelled.parent.mkdir(parents=True, exist_ok=True)
    relabel_vogtiv_cif(cif_path, tmp_relabelled)

    lattice, atoms = _parse_vogtiv_cif(tmp_relabelled)
    text = tmp_relabelled.read_text().splitlines()
    label_by_line: dict[int, str] = {}
    columns: list[str] = []
    in_loop = False
    label_col = None
    for idx, line in enumerate(text):
        s = line.strip()
        if s == "loop_":
            columns = []
            in_loop = False
            continue
        if s.startswith("_atom_site_"):
            columns.append(s)
            continue
        if not in_loop and columns and all(
            k in columns for k in (
                "_atom_site_label", "_atom_site_fract_x",
                "_atom_site_fract_y", "_atom_site_fract_z",
            )
        ):
            in_loop = True
            label_col = columns.index("_atom_site_label")
        if in_loop and s and not s.startswith("#") and not s.startswith("_"):
            fields = re.split(r"\s+", s)
            if label_col is not None and len(fields) > label_col:
                label_by_line[idx] = fields[label_col]

    framework_types: list[str] = []
    framework_cart: list[np.ndarray] = []
    for atom in atoms:
        framework_types.append(label_by_line[atom["original_line_index"]])
        framework_cart.append(np.asarray(atom["cart"]))
    framework_cart_arr = np.vstack(framework_cart)

    charges, masses = _parse_lin_mercado_pseudo_atoms(
        lin_mercado_pkg / "raspa_pseudo_atoms.def"
    )
    framework_charges = np.array(
        [charges.get(t, 0.0) for t in framework_types]
    )
    type_to_mass = {t: float(masses.get(t, 0.0)) for t in set(framework_types)}
    # Add CO2 probe types
    type_to_mass.setdefault("C_co2", float(masses.get("C_co2", 12.0107)))
    type_to_mass.setdefault("O_co2", float(masses.get("O_co2", 15.9994)))

    pair_entries = _parse_lin_mercado_force_field(
        lin_mercado_pkg / "raspa_force_field.def"
    )

    pair_table = PairTable()
    for (t1, t2), entry in pair_entries.items():
        if entry["kind"] == "BUCKINGHAM":
            pair_table.set(
                t1, t2,
                BuckinghamAExpC6(
                    A_K=entry["A_K"],
                    B_inv_angstrom=entry["B_inv_angstrom"],
                    C_K_angstrom6=entry["C_K_angstrom6"],
                    hardcore_angstrom=hardcore_angstrom,
                    cutoff_A=cutoff_angstrom,
                ),
            )
        elif entry["kind"] == "LJ":
            pair_table.set(
                t1, t2,
                LennardJones12_6(
                    epsilon_K=entry["epsilon_K"],
                    sigma_angstrom=entry["sigma_angstrom"],
                    cutoff_A=cutoff_angstrom,
                ),
            )

    probe = ProbeMolecule(
        name="CO2",
        types=["O_co2", "C_co2", "O_co2"],
        body_positions=np.array([
            [0.0, 0.0, 1.149],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.149],
        ]),
        charges_e=np.array([
            charges.get("O_co2", -0.3256),
            charges.get("C_co2", 0.6512),
            charges.get("O_co2", -0.3256),
        ]),
    )

    target = 2.0 * cutoff_angstrom
    n_a = max(1, math.ceil(target / np.linalg.norm(lattice[0])))
    n_b = max(1, math.ceil(target / np.linalg.norm(lattice[1])))
    n_c = max(1, math.ceil(target / np.linalg.norm(lattice[2])))

    return NativeSystem(
        framework_types=framework_types,
        framework_cart_angstrom=framework_cart_arr,
        framework_charges_e=framework_charges,
        cell_matrix_angstrom=lattice,
        pair_table=pair_table,
        probe=probe,
        type_to_mass_amu=type_to_mass,
        supercell_replicas=(n_a, n_b, n_c),
        energy_cutoff_angstrom=cutoff_angstrom,
    )
