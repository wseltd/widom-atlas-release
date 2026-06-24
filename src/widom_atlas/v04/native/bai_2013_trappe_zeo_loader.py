"""Loader for Bai 2013 TraPPE-zeo all-silica zeolite force field.

Unblocks 4c (Si-CHA + CO2) and 6e (MFI + CH4) per the 2026-06-01 final
pivot directive. Source: Bai, Tsapatsis, Siepmann 2013, J. Phys. Chem. C
117(46), 24375-24387, DOI 10.1021/jp4074224.

PROVENANCE NOTE (2026-06-01)
---------------------------
The Bai 2013 main-paper PDF is paywalled and was not in the v0.4 repo.
The TraPPE database web UI at trappe.oit.umn.edu does not expose
framework FFs through its molecule-name search (operator confirmed).

These parameters were verified via RASPA3 v3.0.29's bundled force-field
JSONs at:

  miniconda3/envs/raspa3/share/raspa3/examples/advanced/
    2_mc_cfcmc_co2_in_mfi/force_field.json
  miniconda3/envs/raspa3/share/raspa3/examples/basic/
    7_mc_henry_coefficient_of_methane_in_mfi/force_field.json

Every relevant entry carries an explicit
  "source": "P. Bai, M. Tsapatsis, J. I. Siepmann, J. Phys. Chem. C 2013, 117, 24375-24387"
field, confirming the primary-anchored provenance. The Snurr/Dubbeldam
RASPA3 development team distributed these as the canonical TraPPE-zeo
values.

A separate 2026-06-01 deep-research search returned partially incorrect
values for the O LJ epsilon (93.0 K — that is actually the Calero/
Garcia-Perez 2007 O, NOT Bai 2013) and for the framework charges
(±1.20/-0.60 — that is Calero-family, NOT Bai 2013, which retains the
Garcia-Perez Si=+2.05/O=-1.025 charges; Bai's INNOVATION is the new
O LJ epsilon = 53 K and the new EXPLICIT Si LJ = 22 K / 2.3 A — not
the charges).

Parameters (verbatim from RASPA3 bundled JSONs)
-----------------------------------------------
  Si: epsilon/k_B = 22.0 K, sigma = 2.3 A, charge = +2.05 e
  O:  epsilon/k_B = 53.0 K, sigma = 3.3 A, charge = -1.025 e

Mixing rule: Lorentz-Berthelot (confirmed in both RASPA3 examples).
Truncation: shifted-truncated (RASPA3 bundled "TruncationMethod: shifted").
Tail correction: per the Maia 2023 protocol + McCready/Jorge 2024 it
should be ENABLED at the 14 A cutoff to make K_H cutoff-independent.
The native runner uses our `tail_correction.py` analytical formulas
when enabled.

Cutoff: 14.0 A (per Maia 2023 protocol — matches what Bai used).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .potentials import LennardJones12_6, PairTable
from .system import NativeSystem, ProbeMolecule

# ---------- Bai 2013 TraPPE-zeo framework parameters --------------------
BAI_2013_FRAMEWORK_LJ: dict[str, tuple[float, float]] = {
    "Si_zeo": (22.0, 2.3),
    "O_zeo":  (53.0, 3.3),
}

BAI_2013_FRAMEWORK_CHARGES_E: dict[str, float] = {
    "Si_zeo": +2.05,
    "O_zeo":  -1.025,
}

BAI_2013_FRAMEWORK_MASSES_AMU: dict[str, float] = {
    "Si_zeo": 28.0855,
    "O_zeo":  15.9994,
}

# ---------- TraPPE gas models (record-116 CO2 + record-1 UA CH4) --------
TRAPPE_CO2_SELF_LJ: dict[str, tuple[float, float]] = {
    "C_co2": (27.0, 2.80),
    "O_co2": (79.0, 3.05),
}
TRAPPE_CO2_CHARGES_E: dict[str, float] = {
    "C_co2": +0.700,
    "O_co2": -0.350,
}
TRAPPE_CO2_BOND_LENGTH_A: float = 1.16

TRAPPE_UA_CH4_SELF_LJ: tuple[float, float] = (148.0, 3.73)
TRAPPE_UA_CH4_CHARGE_E: float = 0.0
TRAPPE_UA_CH4_MASS_AMU: float = 16.0428


def _parse_iza_cif(cif_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Parse an IZA pure-silica CIF via ASE, expanding symmetry to P1.

    Returns (lattice_3x3, element_symbols, cartesian_positions).
    """
    import warnings

    from ase.io import read as _ase_read

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms_obj = _ase_read(str(cif_path))
    if isinstance(atoms_obj, list):
        atoms_obj = atoms_obj[0]
    lattice = np.asarray(atoms_obj.cell)
    elements = list(atoms_obj.get_chemical_symbols())
    cart = np.asarray(atoms_obj.get_positions())
    return lattice, elements, cart


def load_bai_2013_native_system(
    repo_root: Path,
    cif_path: Path,
    gas_species: str,
    cutoff_angstrom: float = 14.0,
    apply_lj_tail_correction: bool = True,
    lj_shifted: bool = True,
) -> tuple[NativeSystem, dict]:
    """Build a NativeSystem for an all-silica zeolite + (CO2 or CH4) under Bai 2013.

    Parameters
    ----------
    repo_root : Path
    cif_path : Path
        IZA pure-silica CIF (e.g. CHA_iza.cif, MFI_iza.cif).
    gas_species : str
        "CO2" or "CH4" (other species not yet wired).
    cutoff_angstrom : float
        LJ direct-cutoff (Bai 2013: 14 A).
    apply_lj_tail_correction : bool
        If True, the runner is expected to add the analytical LJ tail
        per `src/widom_atlas/v04/native/tail_correction.py`. This loader
        just returns the framework_type_counts + cell_volume so the
        runner can compute it; the actual correction is applied in
        post-processing.
    lj_shifted : bool
        Shifted-truncated convention (RASPA3 default). Default True.

    Returns
    -------
    (NativeSystem, metadata_dict)
    """
    if gas_species not in ("CO2", "CH4"):
        raise ValueError(f"unsupported gas_species {gas_species!r}; use CO2 or CH4")

    lattice, elements, cart = _parse_iza_cif(cif_path)

    # Re-label every Si -> Si_zeo, O -> O_zeo to match the Bai 2013 atom-type keys.
    types_relabelled: list[str] = []
    for el in elements:
        if el == "Si":
            types_relabelled.append("Si_zeo")
        elif el == "O":
            types_relabelled.append("O_zeo")
        else:
            raise ValueError(
                f"IZA pure-silica CIF expected only Si and O; got {el!r}"
            )

    framework_charges = np.array(
        [BAI_2013_FRAMEWORK_CHARGES_E[t] for t in types_relabelled]
    )

    type_to_mass = {t: BAI_2013_FRAMEWORK_MASSES_AMU[t] for t in set(types_relabelled)}

    # Probe (gas) setup
    if gas_species == "CO2":
        probe = ProbeMolecule(
            name="CO2",
            types=["O_co2", "C_co2", "O_co2"],
            body_positions=np.array([
                [0.0, 0.0, TRAPPE_CO2_BOND_LENGTH_A],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -TRAPPE_CO2_BOND_LENGTH_A],
            ]),
            charges_e=np.array([
                TRAPPE_CO2_CHARGES_E["O_co2"],
                TRAPPE_CO2_CHARGES_E["C_co2"],
                TRAPPE_CO2_CHARGES_E["O_co2"],
            ]),
        )
        probe_self_lj = TRAPPE_CO2_SELF_LJ
        type_to_mass.setdefault("C_co2", 12.0107)
        type_to_mass.setdefault("O_co2", 15.9994)
    else:  # CH4
        probe = ProbeMolecule(
            name="CH4",
            types=["CH4_sp3"],
            body_positions=np.array([[0.0, 0.0, 0.0]]),
            charges_e=np.zeros(1),  # TraPPE-UA CH4 is neutral
        )
        probe_self_lj = {"CH4_sp3": TRAPPE_UA_CH4_SELF_LJ}
        type_to_mass.setdefault("CH4_sp3", TRAPPE_UA_CH4_MASS_AMU)

    # Pair table: framework x probe via Lorentz-Berthelot mixing.
    pair_table = PairTable()
    for ftype, (f_eps, f_sig) in BAI_2013_FRAMEWORK_LJ.items():
        for ptype, (p_eps, p_sig) in probe_self_lj.items():
            eps_cross = math.sqrt(f_eps * p_eps)
            sig_cross = 0.5 * (f_sig + p_sig)
            pair_table.set(
                ftype, ptype,
                LennardJones12_6(
                    epsilon_K=eps_cross,
                    sigma_angstrom=sig_cross,
                    cutoff_A=cutoff_angstrom,
                    shifted=lj_shifted,
                ),
            )
    # Self pairs (probe-probe for completeness; framework self is unused
    # because the framework is rigid in Widom).
    for ptype, (eps, sig) in probe_self_lj.items():
        pair_table.set(
            ptype, ptype,
            LennardJones12_6(
                epsilon_K=eps, sigma_angstrom=sig,
                cutoff_A=cutoff_angstrom, shifted=lj_shifted,
            ),
        )

    # Supercell so simulation cell >= 2 x cutoff in each direction.
    target = 2.0 * cutoff_angstrom
    n_a = max(1, math.ceil(target / np.linalg.norm(lattice[0])))
    n_b = max(1, math.ceil(target / np.linalg.norm(lattice[1])))
    n_c = max(1, math.ceil(target / np.linalg.norm(lattice[2])))

    sys = NativeSystem(
        framework_types=types_relabelled,
        framework_cart_angstrom=cart,
        framework_charges_e=framework_charges,
        cell_matrix_angstrom=lattice,
        pair_table=pair_table,
        probe=probe,
        type_to_mass_amu=type_to_mass,
        supercell_replicas=(n_a, n_b, n_c),
        energy_cutoff_angstrom=cutoff_angstrom,
    )

    # Counts for tail correction
    from collections import Counter
    framework_type_counts = dict(Counter(types_relabelled))
    # Replicate the per-cell counts into the supercell when applying tail correction
    n_total_replicas = n_a * n_b * n_c
    framework_type_counts_supercell = {
        k: v * n_total_replicas for k, v in framework_type_counts.items()
    }
    cell_volume = float(abs(np.linalg.det(lattice))) * n_total_replicas

    metadata = {
        "force_field_lineage": "Bai_2013_TraPPE_zeo_per_RASPA3_bundled_JSON_primary_anchored",
        "force_field_doi": "10.1021/jp4074224",
        "framework_atom_type_counts_per_primitive_cell": framework_type_counts,
        "framework_atom_type_counts_in_supercell": framework_type_counts_supercell,
        "cell_volume_angstrom3_supercell": cell_volume,
        "cutoff_angstrom": cutoff_angstrom,
        "mixing_rule": "lorentz_berthelot",
        "lj_truncation": "shifted_truncated" if lj_shifted else "truncated",
        "apply_lj_tail_correction": apply_lj_tail_correction,
        "gas_species": gas_species,
        "probe_self_lj": dict(probe_self_lj),
        "framework_self_lj": dict(BAI_2013_FRAMEWORK_LJ),
        "framework_charges_e_per_type": dict(BAI_2013_FRAMEWORK_CHARGES_E),
        "supercell_replicas": (n_a, n_b, n_c),
    }
    return sys, metadata
