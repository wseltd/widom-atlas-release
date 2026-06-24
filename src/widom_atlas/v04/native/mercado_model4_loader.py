"""Loader for 1d Mg-MOF-74 + CO2 Mercado et al. 2016 ("Model 4") FF.

Source: Mercado, Vlaisavljevich, Lin et al., J. Phys. Chem. C 120, 12590
(2016), DOI 10.1021/acs.jpcc.6b03393. DFT-derived M-MOF-74 + CO2 force
field. Operator-supplied verbatim 2026-05-19 pass-5; cross-checked vs
the EPFL repository tables.

  Table S1 — framework atomic charges (LoProp).
  Table S2 — guest charges (EPM2 CO2: q_C=+0.6512, q_O=-0.3256).
  Table S7 — Model 4 cross-pair table (global scaling factor Sg = 1.7
             applied to derived O_CO2-M cross potentials; the values
             below are the FINAL post-scaling parameters, ready to use).

Mixed potential set per operator's packet:

  * O_CO2-framework (Mg/Oa/Ob/Oc/Ca/Cb/Cc/Cd):
      Buckingham A*exp(-B*r) - C/r^6  (A in K, B in 1/A, C in K*A^6)
  * O_CO2-H:  LJ 12-6 (eps in K, sigma in A)
  * C_CO2-framework + C_CO2-H:
      LJ 12-6 (eps in K, sigma in A)

CO2 model: EPM2 rigid linear, C-O bond length 1.149 A, q_C=+0.6512,
q_O=-0.3256 (same EPM2 charges Lin/Mercado 2014 uses; differs from
TraPPE which 1b Dzubak and 1c Becker use).

Atom-type mapping: identical to 1c Becker (Mercado uses Oa/Ob/Oc/Ca/Cb/
Cc/Cd labels that match the VOGTIV relabeller directly — no remapping
needed).

  Mg / Oa / Ob / Oc / Ca / Cb / Cc / Cd / H  ->
  Mof_Mg / Mof_Oa / Mof_Ob / Mof_Oc / Mof_Ca / Mof_Cb / Mof_Cc / Mof_Cd / Mof_H

The framework partial charges are identical to Becker Table S3 (operator
confirmed: Mercado SI Table S1 = Becker SI Table S3 framework charge
column). The MERCADO_MODEL4_FRAMEWORK_CHARGES dict below uses the same
numerical values; the duplication is intentional for source provenance.

NOTE on the Buckingham convention: per Mercado/Vlaisavljevich/Lin's
github.com/rociomer/DFT-derived-force-field README, COTA writes
coefficients in (A, C, B) column order while RASPA expects (A, B, C).
The operator-supplied table uses (Aij, Bij, Cij) order — this loader
treats them as A, B, C in the standard form V(r) = A*exp(-B*r) - C/r^6
without re-ordering.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..raspa2.cif_relabeller import _parse_vogtiv_cif, relabel_vogtiv_cif
from .potentials import BuckinghamAExpC6, LennardJones12_6, PairTable
from .system import NativeSystem, ProbeMolecule


MERCADO_MODEL4_FRAMEWORK_CHARGES_E: dict[str, float] = {
    "Mof_Mg": +1.560,
    "Mof_Oa": -0.899,
    "Mof_Ob": -0.752,
    "Mof_Oc": -0.903,
    "Mof_Ca": +0.900,
    "Mof_Cb": -0.314,
    "Mof_Cc": +0.456,
    "Mof_Cd": -0.234,
    "Mof_H":  +0.186,
}

# Table S7 Model 4: O_CO2 x framework Buckingham parameters
# Tuple = (A_K, B_inv_angstrom, C_K_angstrom6)
MERCADO_MODEL4_O_CO2_FRAMEWORK_BUCKINGHAM: dict[str, tuple[float, float, float]] = {
    "Mof_Mg": (2.47320e7, 3.965, 4.08795e5),
    "Mof_Oa": (3.37882e7, 3.805, 1.43132e5),
    "Mof_Ob": (2.67786e7, 3.780, 1.43132e5),
    "Mof_Oc": (2.63432e7, 3.705, 1.43132e5),
    "Mof_Ca": (2.76500e7, 3.840, 2.26311e5),
    "Mof_Cb": (2.14836e7, 3.515, 2.26311e5),
    "Mof_Cc": (2.99216e7, 3.840, 2.26311e5),
    "Mof_Cd": (1.31469e7, 3.315, 2.26311e5),
}

# Table S7 Model 4: O_CO2 x H Lennard-Jones
MERCADO_MODEL4_O_CO2_H_LJ: tuple[float, float] = (56.900, 2.343)

# Table S7 Model 4: C_CO2 x framework Lennard-Jones
MERCADO_MODEL4_C_CO2_FRAMEWORK_LJ: dict[str, tuple[float, float]] = {
    "Mof_Mg": (190.6212, 2.816),
    "Mof_Oa": (69.958, 2.794),
    "Mof_Ob": (69.958, 2.794),
    "Mof_Oc": (69.958, 2.794),
    "Mof_Ca": (87.738, 2.904),
    "Mof_Cb": (87.738, 2.904),
    "Mof_Cc": (87.738, 2.904),
    "Mof_Cd": (87.738, 2.904),
    "Mof_H":  (68.317, 2.453),
}

# CO2 model: EPM2 (Harris-Yung 1995), bond C-O = 1.149 A
MERCADO_MODEL4_CO2_CHARGES_E: dict[str, float] = {
    "C_co2": +0.6512,
    "O_co2": -0.3256,
}

# CO2 self-LJ (EPM2). Used for completeness; Widom on a single CO2
# does not invoke gas-gas self-LJ in the framework-guest energy.
MERCADO_MODEL4_CO2_SELF_LJ: dict[str, tuple[float, float]] = {
    "C_co2": (28.129, 2.757),   # EPM2/Harris-Yung 1995 C
    "O_co2": (80.507, 3.033),   # EPM2/Harris-Yung 1995 O
}

MERCADO_MODEL4_FRAMEWORK_MASSES: dict[str, float] = {
    "Mof_Mg": 24.305,
    "Mof_Oa": 15.9994, "Mof_Ob": 15.9994, "Mof_Oc": 15.9994,
    "Mof_Ca": 12.0107, "Mof_Cb": 12.0107, "Mof_Cc": 12.0107, "Mof_Cd": 12.0107,
    "Mof_H":  1.00794,
    "C_co2":  12.0107, "O_co2":  15.9994,
}


def load_1d_native_mercado_model4(
    repo_root: Path,
    cif_path: Path | None = None,
    cutoff_angstrom: float = 12.8,
    hardcore_angstrom: float = 1.0,
) -> NativeSystem:
    """Build a NativeSystem for 1d Mg-MOF-74 + CO2 Mercado 2016 Model 4."""
    if cif_path is None:
        cif_path = (
            repo_root
            / "docs/research/dataset-research-for-v0.4/15/core-mof-sep2014/core-mof-july2014/VOGTIV_clean_h.cif"
        )

    tmp_relabelled = repo_root / "evidence" / "_native_vogtiv_relabelled_mercado_model4.cif"
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
        [MERCADO_MODEL4_FRAMEWORK_CHARGES_E.get(t, 0.0) for t in framework_types]
    )
    type_to_mass = {t: MERCADO_MODEL4_FRAMEWORK_MASSES.get(t, 0.0) for t in set(framework_types)}
    type_to_mass.setdefault("C_co2", MERCADO_MODEL4_FRAMEWORK_MASSES["C_co2"])
    type_to_mass.setdefault("O_co2", MERCADO_MODEL4_FRAMEWORK_MASSES["O_co2"])

    pair_table = PairTable()

    # O_CO2 x framework metal/O/C (Buckingham A*exp(-Br) - C/r^6)
    for ftype, (A, B, C) in MERCADO_MODEL4_O_CO2_FRAMEWORK_BUCKINGHAM.items():
        pair_table.set(
            ftype, "O_co2",
            BuckinghamAExpC6(
                A_K=A,
                B_inv_angstrom=B,
                C_K_angstrom6=C,
                hardcore_angstrom=hardcore_angstrom,
                cutoff_A=cutoff_angstrom,
            ),
        )

    # O_CO2 x H (LJ)
    eps, sig = MERCADO_MODEL4_O_CO2_H_LJ
    pair_table.set(
        "Mof_H", "O_co2",
        LennardJones12_6(epsilon_K=eps, sigma_angstrom=sig, cutoff_A=cutoff_angstrom),
    )

    # C_CO2 x framework (LJ)
    for ftype, (eps, sig) in MERCADO_MODEL4_C_CO2_FRAMEWORK_LJ.items():
        pair_table.set(
            ftype, "C_co2",
            LennardJones12_6(epsilon_K=eps, sigma_angstrom=sig, cutoff_A=cutoff_angstrom),
        )

    # CO2 self-LJ for completeness
    for ptype, (eps, sig) in MERCADO_MODEL4_CO2_SELF_LJ.items():
        pair_table.set(
            ptype, ptype,
            LennardJones12_6(epsilon_K=eps, sigma_angstrom=sig, cutoff_A=cutoff_angstrom),
        )

    # EPM2 CO2 probe: rigid linear, C-O bond = 1.149 A
    probe = ProbeMolecule(
        name="CO2",
        types=["O_co2", "C_co2", "O_co2"],
        body_positions=np.array([
            [0.0, 0.0, 1.149],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.149],
        ]),
        charges_e=np.array([
            MERCADO_MODEL4_CO2_CHARGES_E["O_co2"],
            MERCADO_MODEL4_CO2_CHARGES_E["C_co2"],
            MERCADO_MODEL4_CO2_CHARGES_E["O_co2"],
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
