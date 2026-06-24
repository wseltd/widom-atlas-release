"""Loader for 2b HKUST-1 + CO2 Ongari 2017 modified Cu-O(CO2) force field.

Source: Ongari, Tiana, Stoneburner, Gagliardi, Smit, J. Phys. Chem. C 121,
15135-15144 (2017), DOI 10.1021/acs.jpcc.7b02302. Modified UFF force
field with a refitted Cu-O(CO2) cross-pair derived from ROS-MP2 cluster
calculations on Cu2(formate)4 + CO2. Operator-supplied verbatim values
from the SI (jp7b02302_si_001.pdf section 4) on 2026-05-19 pass-6.

Functional form for the modified Cu-O(CO2) only:
    V(r) = A*exp(-B*r) - C6/r^6 - C8/r^8     (RASPA "generic", eq. 3.94)

    A  = 1.0e8  K
    B  = 4.19   1/Angstrom
    C6 = 3.196e4 K*Angstrom^6
    C8 = 5.0e6  K*Angstrom^8
    Hard core: V = inf for r < 1.8 A (SI rule, 1.0e15 K in source).

ALL OTHER framework x CO2 cross-pairs use Lorentz-Berthelot mixing
between UFF base LJ (framework) and TraPPE CO2 (gas). UFF base values
from Ongari SI section 4:
    H   eps/kB = 22.14 K,  sigma = 2.57 A
    C   eps/kB = 52.84 K,  sigma = 3.43 A
    O   eps/kB = 30.20 K,  sigma = 3.12 A
    Cu  eps/kB = 2.52  K,  sigma = 3.11 A    (used for Cu-C_CO2 only;
                                              Cu-O_CO2 uses the modified
                                              generic potential above)

REPEAT charges (Ongari SI Table S12):
    Cu                 = +0.914 e
    O (carboxylic)     = -0.534 e
    H (benzene)        = +0.159 e
    C (carboxylic)     = +0.586 e
    C (benzene, CH)    = +0.062 e
    C (benzene, CC)    = -0.196 e

CO2 model: TraPPE (q_C = +0.700, q_O = -0.350, bond C-O = 1.16 A,
C eps/kB = 27.0 K sigma = 2.80 A, O eps/kB = 79.0 K sigma = 3.05 A).

Atom-type mapping from FIQCEN CIF (HKUST-1 CoRE-MOF charged variant) to
Ongari atom types:
    Cu1 -> Cu               (paddle-wheel Cu)
    O1  -> O (carboxylic)   (carboxylate O coordinated to Cu)
    C2  -> C (carboxylic)   (carboxylate C, OOC-)
    C1  -> C (benzene, CH)  (aromatic C bonded to H)
    C3  -> C (benzene, CC)  (aromatic C bonded to another C, no H)
    H1  -> H (benzene)      (aromatic ring H)

The atom_label_map field in YAML 2a documents this mapping verbatim
(Cu->Cu1, C_aromatic_CH->C1, C_carboxylate->C2, C_ipso->C3,
 O_carboxylate->O1, H->H1). 2b inherits the same CIF + same mapping.

Note: the Nazarian DDEC charges baked into the CIF file (Cu1 = +0.9360
etc.) are NOT used here. Ongari REPEAT charges OVERRIDE them via the
loader. Electroneutrality verified per HKUST-1 Cu3(BTC)2 asymmetric unit:
sum_q = 0.000 e (3*(+0.914) + 12*(-0.534) + 6*(+0.586) + 6*(+0.062) +
6*(-0.196) + 6*(+0.159) = 0.000).

Per operator pass-6 simulation protocol:
- Truncated potential cutoff = 13 A (NOT shifted, NOT tail-corrected)
- Ewald for charges in periodic cells
- (GCMC 50k equil + 50k production per Ongari SI is for absolute uptake;
  for Widom we use 3 seeds x 2 T x 100k insertions = 600k aggregate,
  matching the 1c/1d Mg-MOF-74 protocol.)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .loaders import _parse_cif_simple
from .potentials import LennardJones12_6, OngariAExpC6C8, PairTable
from .system import NativeSystem, ProbeMolecule


ONGARI_2017_REPEAT_CHARGES_E: dict[str, float] = {
    "Cu1": +0.914,
    "O1":  -0.534,
    "C2":  +0.586,
    "C1":  +0.062,
    "C3":  -0.196,
    "H1":  +0.159,
}

UFF_FRAMEWORK_LJ: dict[str, tuple[float, float]] = {
    "H1":  (22.14, 2.57),
    "C1":  (52.84, 3.43),
    "C2":  (52.84, 3.43),
    "C3":  (52.84, 3.43),
    "O1":  (30.20, 3.12),
    "Cu1": (2.52,  3.11),
}

TRAPPE_CO2_SELF_LJ: dict[str, tuple[float, float]] = {
    "C_co2": (27.00, 2.80),
    "O_co2": (79.00, 3.05),
}

TRAPPE_CO2_CHARGES_E: dict[str, float] = {
    "C_co2": +0.700,
    "O_co2": -0.350,
}

ONGARI_CU_O_CO2_GENERIC = {
    "A_K":             1.0e8,
    "B_inv_angstrom":  4.19,
    "C6_K_angstrom6":  3.196e4,
    "C8_K_angstrom8":  5.0e6,
    "hardcore_angstrom": 1.8,
}

FRAMEWORK_MASSES: dict[str, float] = {
    "Cu1": 63.546,
    "O1":  15.9994,
    "C1":  12.0107,
    "C2":  12.0107,
    "C3":  12.0107,
    "H1":  1.00794,
    "C_co2": 12.0107,
    "O_co2": 15.9994,
}


def _lb_epsilon(a: float, b: float) -> float:
    return math.sqrt(a * b)


def _lb_sigma(a: float, b: float) -> float:
    return 0.5 * (a + b)


def load_2b_native_ongari_2017(
    repo_root: Path,
    cif_path: Path | None = None,
    cutoff_angstrom: float = 13.0,
) -> NativeSystem:
    """Build a NativeSystem for 2b HKUST-1 + CO2 Ongari 2017 modified-Cu FF."""
    if cif_path is None:
        cif_path = (
            repo_root
            / "docs/research/dataset-research-for-v0.4/15/CoRE-MOF-1.0-DFT-minimized"
            / "CoRE-MOF-1.0-DFT-Minimized/minimized_structures_with_DDEC_charges"
            / "FIQCEN_clean_min_charges.cif"
        )

    lattice, types, cart = _parse_cif_simple(cif_path)
    framework_types: list[str] = []
    for t in types:
        if t == "Cu":
            framework_types.append("Cu1")
        elif t == "H":
            framework_types.append("H1")
        elif t == "O":
            framework_types.append("O1")
        else:
            framework_types.append(t)

    label_by_line = _read_cif_label_column(cif_path, len(types))
    if label_by_line is not None and len(label_by_line) == len(types):
        framework_types = label_by_line

    framework_charges = np.array(
        [ONGARI_2017_REPEAT_CHARGES_E.get(t, 0.0) for t in framework_types]
    )
    type_to_mass: dict[str, float] = {
        t: FRAMEWORK_MASSES.get(t, 0.0) for t in set(framework_types)
    }
    type_to_mass.setdefault("C_co2", FRAMEWORK_MASSES["C_co2"])
    type_to_mass.setdefault("O_co2", FRAMEWORK_MASSES["O_co2"])

    pair_table = PairTable()

    pair_table.set(
        "Cu1", "O_co2",
        OngariAExpC6C8(
            A_K=ONGARI_CU_O_CO2_GENERIC["A_K"],
            B_inv_angstrom=ONGARI_CU_O_CO2_GENERIC["B_inv_angstrom"],
            C6_K_angstrom6=ONGARI_CU_O_CO2_GENERIC["C6_K_angstrom6"],
            C8_K_angstrom8=ONGARI_CU_O_CO2_GENERIC["C8_K_angstrom8"],
            hardcore_angstrom=ONGARI_CU_O_CO2_GENERIC["hardcore_angstrom"],
            cutoff_A=cutoff_angstrom,
        ),
    )

    for ftype, (f_eps, f_sig) in UFF_FRAMEWORK_LJ.items():
        for ptype, (p_eps, p_sig) in TRAPPE_CO2_SELF_LJ.items():
            if ftype == "Cu1" and ptype == "O_co2":
                continue
            pair_table.set(
                ftype, ptype,
                LennardJones12_6(
                    epsilon_K=_lb_epsilon(f_eps, p_eps),
                    sigma_angstrom=_lb_sigma(f_sig, p_sig),
                    cutoff_A=cutoff_angstrom,
                ),
            )

    for ptype, (eps, sig) in TRAPPE_CO2_SELF_LJ.items():
        pair_table.set(
            ptype, ptype,
            LennardJones12_6(epsilon_K=eps, sigma_angstrom=sig, cutoff_A=cutoff_angstrom),
        )

    probe = ProbeMolecule(
        name="CO2",
        types=["O_co2", "C_co2", "O_co2"],
        body_positions=np.array([
            [0.0, 0.0, 1.16],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.16],
        ]),
        charges_e=np.array([
            TRAPPE_CO2_CHARGES_E["O_co2"],
            TRAPPE_CO2_CHARGES_E["C_co2"],
            TRAPPE_CO2_CHARGES_E["O_co2"],
        ]),
    )

    target = 2.0 * cutoff_angstrom
    n_a = max(1, math.ceil(target / np.linalg.norm(lattice[0])))
    n_b = max(1, math.ceil(target / np.linalg.norm(lattice[1])))
    n_c = max(1, math.ceil(target / np.linalg.norm(lattice[2])))

    return NativeSystem(
        framework_types=framework_types,
        framework_cart_angstrom=cart,
        framework_charges_e=framework_charges,
        cell_matrix_angstrom=lattice,
        pair_table=pair_table,
        probe=probe,
        type_to_mass_amu=type_to_mass,
        supercell_replicas=(n_a, n_b, n_c),
        energy_cutoff_angstrom=cutoff_angstrom,
    )


def _read_cif_label_column(cif_path: Path, n_atoms_expected: int) -> list[str] | None:
    """Read _atom_site_label column from CIF; returns list in CIF row order."""
    text = cif_path.read_text().splitlines()
    columns: list[str] = []
    in_loop = False
    label_col = None
    labels: list[str] = []
    for line in text:
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
                labels.append(fields[label_col])
    if len(labels) == n_atoms_expected:
        return labels
    return None
