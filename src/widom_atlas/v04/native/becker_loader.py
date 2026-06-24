"""Loader for 1c Mg-MOF-74 + CO2 Becker 2018 reduced-LJ approximation.

Source: Becker, Heinen, Dubbeldam, Lin, Vlugt, J. Phys. Chem. C 121, 4659 (2017), DOI 10.1021/acs.jpcc.6b12052 (vendored: docs/research/dataset-research-for-v0.4/1c_becker_2017/). Mg charges = SI Table S6.
(Related but NOT the charge source: the QM-derived 2018 paper is JPCC 122, 24488,
DOI 10.1021/acs.jpcc.8b08639.)

  Table S6 (2017) — Mg-MOF-74 framework atom-type LJ + fixed charges (digit-confirmed)
  Table S1 (2017) — unscaled UFF + TraPPE reference parameters

WARNING — POLARIZATION NOT IMPLEMENTED IN THIS BACKEND.

Becker 2018 is a polarizable force field: the CO2 molecule has induced
dipoles (CO2 atom polarizabilities alpha_C = 0.878 Angstrom^3,
alpha_O = 0.465 Angstrom^3) that respond to the framework field, screening
the Mg(+1.560 e) ... O_CO2(-0.35 e) coulomb at the open metal site.

The native widom-atlas evaluator only supports rigid charges + LJ 12-6 +
Buckingham + Dzubak A*exp(-B*r) - C5/r^5 - D6/r^6. No induced dipoles.

Per operator directive (verbatim, 2026-05-19 pass 4): "If the current backend
cannot implement induced dipoles, run a clearly labelled non-polarizable
reduced-LJ/charge version first and mark it as an approximation. Do not
present it as the full Becker polarizable force field unless polarisation
is actually implemented."

This loader therefore produces a NON-POLARIZABLE REDUCED APPROXIMATION:
  * Becker LJ 12-6 used as-is for framework atom types.
  * Becker fixed charges used as-is (no induced-dipole screening of Mg field).
  * CO2 polarizabilities recorded in metadata but NOT used in the energy.
  * Cross-pairs (framework x CO2) via Lorentz-Berthelot mixing.

Expected impact of the approximation: the unscreened Mg(+1.56) coulomb is
likely to over-attract O_CO2(-0.35) at the OMS, producing a Q_st higher
than the Becker-polarizable target. The K_H/Q_st verdict from this loader
must therefore be interpreted as the reduced-approximation outcome, NOT
the Becker 2018 full force-field outcome. Promotion to a polarizable
result would require an induced-dipole / Drude oscillator backend, which
is out of scope for v0.4.

Atom-type mapping (Becker labels -> VOGTIV-relabeller labels in repo):

  Mg    -> Mof_Mg
  O1    -> Mof_Oa   (carboxylate O, Mg-coordinated)
  O2    -> Mof_Ob   (phenolate / bridging mu-2 O, smaller charge magnitude
                     reflects more covalent character)
  O3    -> Mof_Oc   (carboxylate O, distal)
  C1    -> Mof_Ca   (carboxylate C, large positive charge)
  C2    -> Mof_Cb   (aromatic C type 1)
  C3    -> Mof_Cc   (aromatic C type 2)
  C4    -> Mof_Cd   (aromatic C type 3)
  H     -> Mof_H    (ring H)

The mapping rationale is documented in this docstring; the Becker paper's
explicit atom-type structural assignment was not in the repo at loader-
writing time. The mapping is consistent with: (a) charge-magnitude
correspondence (Becker O2 has smallest magnitude -0.752 e, matching the
more-covalent bridging-O assignment of Mof_Ob); (b) C1 largest positive
charge (+0.900 e) matches the carboxylate C / Mof_Ca assignment; (c) the
VOGTIV geometric relabeller's canonical ordering. Electroneutrality
verified: sum over Mg2(dobdc) unit (2 Mg + 2 of each O type + 2 of each
C type + 2 H per asymmetric unit) = 0.000 e.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..raspa2.cif_relabeller import _parse_vogtiv_cif, relabel_vogtiv_cif
from .potentials import LennardJones12_6, PairTable
from .system import NativeSystem, ProbeMolecule


BECKER_2018_FRAMEWORK_LJ: dict[str, tuple[float, float]] = {
    "Mof_Mg": (5.00,  3.00),
    "Mof_Oa": (30.19, 3.12),
    "Mof_Ob": (30.19, 3.12),
    "Mof_Oc": (30.19, 3.12),
    "Mof_Ca": (52.84, 3.42),
    "Mof_Cb": (52.84, 3.42),
    "Mof_Cc": (52.84, 3.42),
    "Mof_Cd": (52.84, 3.42),
    "Mof_H":  (22.14, 2.57),
}

BECKER_2018_FRAMEWORK_CHARGES_E: dict[str, float] = {
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

BECKER_2018_CO2_SELF_LJ: dict[str, tuple[float, float]] = {
    "C_co2": (27.0, 2.80),
    "O_co2": (79.0, 3.05),
}

BECKER_2018_CO2_CHARGES_E: dict[str, float] = {
    "C_co2": +0.700,
    "O_co2": -0.350,
}

BECKER_2018_CO2_POLARIZABILITY_ANG3: dict[str, float] = {
    "C_co2": 0.878,
    "O_co2": 0.465,
}

BECKER_2018_FRAMEWORK_MASSES: dict[str, float] = {
    "Mof_Mg": 24.305,
    "Mof_Oa": 15.9994, "Mof_Ob": 15.9994, "Mof_Oc": 15.9994,
    "Mof_Ca": 12.0107, "Mof_Cb": 12.0107, "Mof_Cc": 12.0107, "Mof_Cd": 12.0107,
    "Mof_H":  1.00794,
    "C_co2":  12.0107, "O_co2":  15.9994,
}


def _lb_epsilon(eps_a_K: float, eps_b_K: float) -> float:
    return math.sqrt(eps_a_K * eps_b_K)


def _lb_sigma(sig_a_ang: float, sig_b_ang: float) -> float:
    return 0.5 * (sig_a_ang + sig_b_ang)


def load_1c_native_becker_reduced(
    repo_root: Path,
    cif_path: Path | None = None,
    cutoff_angstrom: float = 12.8,
    hardcore_angstrom: float = 1.0,
) -> NativeSystem:
    """Build a NativeSystem for 1c Mg-MOF-74 + CO2 Becker 2018 reduced-LJ
    (non-polarizable approximation).

    Lorentz-Berthelot mixing for framework x CO2 cross-pairs.
    """
    if cif_path is None:
        cif_path = (
            repo_root
            / "docs/research/dataset-research-for-v0.4/15/core-mof-sep2014/core-mof-july2014/VOGTIV_clean_h.cif"
        )

    tmp_relabelled = repo_root / "evidence" / "_native_vogtiv_relabelled_becker.cif"
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
        [BECKER_2018_FRAMEWORK_CHARGES_E.get(t, 0.0) for t in framework_types]
    )
    type_to_mass = {t: BECKER_2018_FRAMEWORK_MASSES.get(t, 0.0) for t in set(framework_types)}
    type_to_mass.setdefault("C_co2", BECKER_2018_FRAMEWORK_MASSES["C_co2"])
    type_to_mass.setdefault("O_co2", BECKER_2018_FRAMEWORK_MASSES["O_co2"])

    pair_table = PairTable()

    framework_self_lj: dict[str, tuple[float, float]] = {}
    for t in set(framework_types):
        if t in BECKER_2018_FRAMEWORK_LJ:
            framework_self_lj[t] = BECKER_2018_FRAMEWORK_LJ[t]

    for ftype, (f_eps, f_sig) in framework_self_lj.items():
        for ptype, (p_eps, p_sig) in BECKER_2018_CO2_SELF_LJ.items():
            eps_cross = _lb_epsilon(f_eps, p_eps)
            sig_cross = _lb_sigma(f_sig, p_sig)
            pair_table.set(
                ftype, ptype,
                LennardJones12_6(
                    epsilon_K=eps_cross,
                    sigma_angstrom=sig_cross,
                    cutoff_A=cutoff_angstrom,
                ),
            )

    for ptype, (eps, sig) in BECKER_2018_CO2_SELF_LJ.items():
        pair_table.set(
            ptype, ptype,
            LennardJones12_6(
                epsilon_K=eps, sigma_angstrom=sig,
                cutoff_A=cutoff_angstrom,
            ),
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
            BECKER_2018_CO2_CHARGES_E["O_co2"],
            BECKER_2018_CO2_CHARGES_E["C_co2"],
            BECKER_2018_CO2_CHARGES_E["O_co2"],
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
