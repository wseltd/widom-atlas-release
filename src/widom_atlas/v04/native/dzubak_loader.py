"""Loader for the 1b Mg-MOF-74 + CO2 Dzubak 2012 system.

Parameters transcribed verbatim from Dzubak 2012 Nature Chemistry SI
(DOI 10.1038/nchem.1432) — file
`docs/research/dataset-research-for-v0.4/9/dzubak_2012_si.pdf`:

  Table SI 4 — pairwise CO2-framework parameters (modified Buckingham:
              V(r) = A·exp(-B·r) - C/r^5 - D/r^6).
  Table SI 8 — framework atomic partial charges (LoProp on Mg-MOF-74).
  Table SI 12 — guest LJ self-parameters (header columns mislabeled:
                actual order is (Epsilon[K], Sigma[Å]) despite the
                printed `(Sigma, Epsilon)` heading; verified by
                physical-magnitude check vs TraPPE-CO2).
  Table SI 13 — guest charges (TraPPE-CO2: q_C = +0.700, q_O = -0.350).

CO2 model: TraPPE rigid linear, C-O bond length 1.16 Å, q_C = +0.700,
q_O = -0.350 (different from Lin/Mercado's EPM2 q_C=+0.6512, q_O=-0.3256).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..raspa2.cif_relabeller import _parse_vogtiv_cif, relabel_vogtiv_cif
from .potentials import BuckinghamAExpC6, DzubakAExpC5D6, LennardJones12_6, PairTable
from .system import NativeSystem, ProbeMolecule


# Dzubak 2012 SI Table SI 4 — Mg-MOF-74 + CO2 Buckingham/Dzubak pairs.
# Tuple = (A_K, B_inv_angstrom, C_K_angstrom5, D_K_angstrom6)
# A reported as ×10⁷ K; C as ×10⁵ K·Å⁻⁵; D as ×10⁵ K·Å⁻⁶. Scaled here.
DZUBAK_2012_TABLE_SI_4: dict[tuple[str, str], tuple[float, float, float, float]] = {
    # framework_type, molecule_type: A_K, B, C5, D6
    ("Mof_Mg", "O_co2"): (4.067e7, 4.152, 0.0,        4.062e5),
    ("Mof_Oa", "O_co2"): (1.401e7, 3.330, 0.636e5,    0.0     ),
    ("Mof_Ob", "O_co2"): (1.673e7, 3.520, 0.0,        0.891e5),
    ("Mof_Oc", "O_co2"): (1.468e7, 3.399, 1.160e5,    0.0     ),
    ("Mof_Ca", "O_co2"): (2.280e7, 4.065, 1.445e5,    0.0     ),
    ("Mof_Cb", "O_co2"): (1.408e7, 3.348, 0.0,        0.907e5),
    ("Mof_Cc", "O_co2"): (2.139e7, 3.786, 1.194e5,    0.0     ),
    ("Mof_Cd", "O_co2"): (0.562e7, 3.006, 0.0,        1.756e5),
    ("Mof_H",  "O_co2"): (2.153e7, 4.180, 0.0,        0.824e5),
    ("Mof_Mg", "C_co2"): (7.395e7, 4.770, 0.0,        0.0     ),
    ("Mof_Oa", "C_co2"): (23.047e7, 4.990, 0.0,        0.0     ),
    ("Mof_Ob", "C_co2"): (23.047e7, 4.990, 0.0,        0.0     ),
    ("Mof_Oc", "C_co2"): (23.047e7, 4.990, 0.0,        0.0     ),
    ("Mof_Ca", "C_co2"): (6.900e7, 4.190, 0.0,        0.0     ),
    ("Mof_Cb", "C_co2"): (6.900e7, 4.190, 0.0,        0.0     ),
    ("Mof_Cc", "C_co2"): (6.900e7, 4.190, 0.0,        0.0     ),
    ("Mof_Cd", "C_co2"): (4.584e7, 4.050, 0.0,        0.0     ),
    ("Mof_H",  "C_co2"): (6.261e7, 5.000, 0.0,        0.0     ),
}

# Dzubak 2012 SI Table SI 8 — Mg-MOF-74 framework charges (e).
DZUBAK_2012_TABLE_SI_8: dict[str, float] = {
    "Mof_Mg": +1.5637,
    "Mof_Oa": -0.7654,
    "Mof_Ob": -0.7088,
    "Mof_Oc": -0.8328,
    "Mof_Ca": +0.4820,
    "Mof_Cb": -0.1354,
    "Mof_Cc": +0.1890,
    "Mof_Cd": -0.1814,
    "Mof_H":  +0.3891,
}

# Dzubak 2012 SI Table SI 12 — guest LJ self (column labels SWAPPED in PDF;
# actual order is Epsilon[K], Sigma[Å]). Verified by physical magnitude.
DZUBAK_2012_CO2_SELF_LJ: dict[str, tuple[float, float]] = {
    # (epsilon_K, sigma_angstrom)
    "C_co2": (27.0, 2.800),
    "O_co2": (79.0, 3.050),
}

# Dzubak 2012 SI Table SI 13 — guest charges (TraPPE-CO2).
DZUBAK_2012_CO2_CHARGES: dict[str, float] = {
    "C_co2": +0.700,
    "O_co2": -0.350,
}

# Atomic masses (g/mol) — matches RASPA2 stock pseudo_atoms.def values.
DZUBAK_2012_FRAMEWORK_MASSES: dict[str, float] = {
    "Mof_Mg": 24.305,
    "Mof_Oa": 15.9994, "Mof_Ob": 15.9994, "Mof_Oc": 15.9994,
    "Mof_Ca": 12.0107, "Mof_Cb": 12.0107, "Mof_Cc": 12.0107, "Mof_Cd": 12.0107,
    "Mof_H":  1.00794,
    "C_co2":  12.0107, "O_co2":  15.9994,
}


def load_1b_native_dzubak(
    repo_root: Path,
    cif_path: Path | None = None,
    cutoff_angstrom: float = 12.8,
    hardcore_angstrom: float = 1.0,
) -> NativeSystem:
    """Build a NativeSystem for 1b Mg-MOF-74 + CO2 Dzubak 2012."""
    if cif_path is None:
        cif_path = (
            repo_root
            / "docs/research/dataset-research-for-v0.4/15/core-mof-sep2014/core-mof-july2014/VOGTIV_clean_h.cif"
        )

    tmp_relabelled = repo_root / "evidence" / "_native_vogtiv_relabelled_dzubak.cif"
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

    framework_charges = np.array(
        [DZUBAK_2012_TABLE_SI_8.get(t, 0.0) for t in framework_types]
    )
    type_to_mass = {t: DZUBAK_2012_FRAMEWORK_MASSES.get(t, 0.0) for t in set(framework_types)}
    type_to_mass.setdefault("C_co2", DZUBAK_2012_FRAMEWORK_MASSES["C_co2"])
    type_to_mass.setdefault("O_co2", DZUBAK_2012_FRAMEWORK_MASSES["O_co2"])

    pair_table = PairTable()
    # Cross pairs (framework × CO2) via Dzubak Table SI 4. Use the C5+D6 form
    # consistently — entries with C5=0 reduce to a Buckingham; entries with
    # D6=0 reduce to A·exp - C/r^5.
    for (ftype, ptype), (A, B, C5, D6) in DZUBAK_2012_TABLE_SI_4.items():
        pair_table.set(
            ftype, ptype,
            DzubakAExpC5D6(
                A_K=A,
                B_inv_angstrom=B,
                C_K_angstrom5=C5,
                D_K_angstrom6=D6,
                hardcore_angstrom=hardcore_angstrom,
                cutoff_A=cutoff_angstrom,
            ),
        )
    # CO2 self-LJ (used only for completeness — Widom on a single CO2 doesn't
    # invoke self-LJ).
    for ptype, (eps, sig) in DZUBAK_2012_CO2_SELF_LJ.items():
        pair_table.set(
            ptype, ptype,
            LennardJones12_6(
                epsilon_K=eps, sigma_angstrom=sig,
                cutoff_A=cutoff_angstrom,
            ),
        )

    # CO2 probe with TraPPE bond length 1.16 Å and TraPPE charges.
    probe = ProbeMolecule(
        name="CO2",
        types=["O_co2", "C_co2", "O_co2"],
        body_positions=np.array([
            [0.0, 0.0, 1.16],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.16],
        ]),
        charges_e=np.array([
            DZUBAK_2012_CO2_CHARGES["O_co2"],
            DZUBAK_2012_CO2_CHARGES["C_co2"],
            DZUBAK_2012_CO2_CHARGES["O_co2"],
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
