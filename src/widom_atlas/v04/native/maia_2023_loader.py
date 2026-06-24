"""Loader for 3b UiO-66 + CO2 Maia 2023 force-field variants.

Source: Maia, R.A.; Ribeiro, A.M.; Mota, J.P.B. Prediction of Carbon
Dioxide and Methane Adsorption on UiO-66 MOF via Molecular Simulation.
Crystals 2023, 13, 1523. DOI 10.3390/cryst13101523. CC-BY-4.0 (MDPI open).

  Table 1 — TraPPE gas models (CO2 + CH4)
  Table 2 — UiO-66 framework UA/UAq parameters (8 atom types)
  Table 3 — UiO-66 framework EHq parameters (9 atom types, explicit H)

Three variants:

  UA   : United-atom framework, NO framework charges, NO electrostatics.
         Maia (text + Figure 5) reports this as the BEST match to Cavka
         et al. experimental CO2 isotherms. Aromatic H atoms are dropped
         (lumped into C25 CH united-atom). Hydroxyl H is kept (H25 has
         sigma=0 and epsilon=0; only its mass and position contribute,
         and even mass is incidental for Widom).

  UAq  : Same atom typing as UA, with Maia Table 2 charges applied
         (q derived from Yang et al. 2011 JPC C 115, 13768, DOI
         10.1021/jp202633t). Electrostatics enabled (native Ewald).

  EHq  : Explicit-H framework. Maia Table 3 has 9 atom types: same
         Zr/C1/C13/O1/O25/O29 as UAq, plus reparameterised C25 (smaller
         sigma/epsilon because H is explicit) and H1 aromatic H
         (sigma=2.36 A, epsilon=25.45 K, q=+0.127 e), and H25 hydroxyl H
         (sigma=epsilon=0, q=+0.495 e).

ATOM-TYPE CLASSIFICATION

UiO-66 = Zr6O4(OH)4(BDC)6 per Zr6 cluster. The primitive cell of
RUBTAK01_SL_DDEC.cif (used by 3a) has 6 Zr + 32 O + 48 C + 28 H = 114
atoms (1 Zr6 cluster + 6 BDC ligands per primitive cell).

The loader classifies CIF atoms into Maia's atom types by DDEC charge
bucket — using the CIF's existing DDEC charge column as a *topology
proxy*. This is justified because each chemically distinct site in
UiO-66 has a distinct DDEC charge band:

  - Zr        : metal node (1 type)
  - O (mu3-O) : q <= -0.9 (4/cluster, the bridging oxide)             -> O25
  - O (mu3-OH): -0.9 < q <= -0.65 (4/cluster, the hydroxyl-bridging)  -> O29
  - O (carbox): q > -0.65 (24/cluster, carboxylate O bound to Zr)     -> O1
  - C (carbox): q > 0.5 (12/cluster, carboxylate C)                   -> C1
  - C (ipso)  : -0.05 < q < 0.05 (12/cluster, aromatic C-C, no H)     -> C13
  - C (CH)    : q < -0.05 (24/cluster, aromatic CH carbons)           -> C25
  - H (OH)    : q > 0.3 (4/cluster, hydroxyl H)                       -> H25
  - H (CH)    : q < 0.3 (24/cluster, aromatic CH H)                   -> dropped (UA/UAq) or H1 (EHq)

Bucket boundaries: chosen to lie in gap regions between groups in the
RUBTAK01_SL_DDEC charge histogram (well-separated by ~0.1 e). Classifier
verifies the expected per-cluster counts before returning.

Charges in UAq/EHq are the *Maia table* values, NOT the CIF DDEC values
(Maia's source is Yang 2011 JPC C; DDEC is a different method). The CIF
DDEC charges are used only to TYPE the atoms, then OVERWRITTEN by Maia's
Table 2/3 charges.

CO2 MODEL

Maia Table 1: TraPPE-CO2 fully-rigid 3-site C-O = 1.16 A, O-C-O = 180 deg.
  C : sigma=2.80 A, epsilon=27.0 K, q=+0.70 e
  O : sigma=3.05 A, epsilon=79.0 K, q=-0.35 e

CROSS-PAIRS

Lorentz-Berthelot mixing (Maia text). No explicit cross-pair table.

CUTOFF + LONG-RANGE

Maia: 14 A truncated LJ + analytical tail correction; Ewald 5 k-vectors
with alpha = 5.6 / L_min. Native widom-atlas backend honours the LJ
cutoff (no tail correction yet — flagged in verdict). Ewald via the
existing native Ewald module.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .potentials import LennardJones12_6, PairTable
from .system import NativeSystem, ProbeMolecule

# ---------- Maia 2023 Table 2 (UA / UAq) ----------------------------------
# epsilon in K, sigma in Angstrom
MAIA_2023_UA_LJ: dict[str, tuple[float, float]] = {
    "Maia_Zr":  (34.72, 2.78),
    "Maia_O1":  (55.00, 2.80),
    "Maia_O25": (93.00, 3.02),
    "Maia_O29": (55.00, 2.80),
    "Maia_C1":  (41.00, 3.90),
    "Maia_C13": (21.00, 3.88),
    "Maia_C25": (48.00, 3.74),
    "Maia_H25": (0.00,  0.00),
}

MAIA_2023_UAq_CHARGES_E: dict[str, float] = {
    "Maia_Zr":  +2.008,
    "Maia_O1":  -0.582,
    "Maia_O25": -1.179,
    "Maia_O29": -0.741,
    "Maia_C1":  +0.625,
    "Maia_C13": -0.002,
    "Maia_C25": +0.006,
    "Maia_H25": +0.495,
}

# ---------- Maia 2023 Table 3 (EHq) ---------------------------------------
# Same Zr/C1/C13/O1/O25/O29 as Table 2; reparameterised C25 (smaller, H is
# explicit); new H1 aromatic H.
MAIA_2023_EHq_LJ: dict[str, tuple[float, float]] = {
    "Maia_Zr":  (34.72, 2.78),
    "Maia_O1":  (55.00, 2.80),
    "Maia_O25": (93.00, 3.02),
    "Maia_O29": (55.00, 2.80),
    "Maia_C1":  (41.00, 3.90),
    "Maia_C13": (21.00, 3.88),
    "Maia_C25": (30.70, 3.60),
    "Maia_H1":  (25.45, 2.36),
    "Maia_H25": (0.00,  0.00),
}

MAIA_2023_EHq_CHARGES_E: dict[str, float] = {
    "Maia_Zr":  +2.008,
    "Maia_O1":  -0.582,
    "Maia_O25": -1.179,
    "Maia_O29": -0.741,
    "Maia_C1":  +0.625,
    "Maia_C13": -0.002,
    "Maia_C25": -0.121,
    "Maia_H1":  +0.127,
    "Maia_H25": +0.495,
}

# ---------- TraPPE CO2 (Maia Table 1) -------------------------------------
TRAPPE_CO2_SELF_LJ: dict[str, tuple[float, float]] = {
    "C_co2": (27.0, 2.80),
    "O_co2": (79.0, 3.05),
}
TRAPPE_CO2_CHARGES_E: dict[str, float] = {
    "C_co2": +0.700,
    "O_co2": -0.350,
}
TRAPPE_CO2_BOND_LENGTH_A: float = 1.16  # Maia Table 1

# ---------- Element masses (amu) ------------------------------------------
ELEMENT_MASS_AMU: dict[str, float] = {
    "Zr":  91.224,
    "O":   15.9994,
    "C":   12.0107,
    "H":   1.00794,
}

# Per-Maia-atom masses keyed by Maia label
MAIA_TYPE_MASS_AMU: dict[str, float] = {
    "Maia_Zr":  ELEMENT_MASS_AMU["Zr"],
    "Maia_O1":  ELEMENT_MASS_AMU["O"],
    "Maia_O25": ELEMENT_MASS_AMU["O"] + ELEMENT_MASS_AMU["H"],  # O + H (mu3-O paired)
    "Maia_O29": ELEMENT_MASS_AMU["O"],
    "Maia_C1":  ELEMENT_MASS_AMU["C"],
    "Maia_C13": ELEMENT_MASS_AMU["C"],
    # In UA: C25 lumps aromatic C with its H (united-atom CH); for EH: C only.
    "Maia_C25": ELEMENT_MASS_AMU["C"] + ELEMENT_MASS_AMU["H"],
    "Maia_H1":  ELEMENT_MASS_AMU["H"],
    "Maia_H25": ELEMENT_MASS_AMU["H"],
    "C_co2":    ELEMENT_MASS_AMU["C"],
    "O_co2":    ELEMENT_MASS_AMU["O"],
}


# ---------- Charge-bucket classifier --------------------------------------
def _classify_atom_by_ddec_charge(element: str, q_ddec: float) -> str:
    """Return Maia atom-type label from element + CIF DDEC charge.

    Bucket boundaries chosen to lie in well-separated gap regions of the
    RUBTAK01_SL_DDEC charge histogram (see module docstring). Caller is
    expected to verify per-cluster counts after classification.
    """
    if element == "Zr":
        return "Maia_Zr"
    if element == "O":
        # Charge-bucket boundaries derived from the RUBTAK01_SL_DDEC
        # histogram: O charges cluster at -1.196 (mu3-O, 4 atoms), -1.057
        # (mu3-OH, 4 atoms), -0.593 (carboxylate, 24 atoms). Boundaries are
        # placed in gap regions (~0.07 e gap between -1.196 and -1.057;
        # ~0.46 e gap between -1.057 and -0.593).
        if q_ddec <= -1.10:
            return "Maia_O25"  # mu3-O bridging oxide
        if q_ddec <= -0.80:
            return "Maia_O29"  # mu3-OH bridging hydroxyl
        return "Maia_O1"  # carboxylate O
    if element == "C":
        if q_ddec > 0.5:
            return "Maia_C1"  # carboxylate C
        if -0.05 < q_ddec < 0.05:
            return "Maia_C13"  # aromatic ipso (no H)
        return "Maia_C25"  # aromatic CH
    if element == "H":
        if q_ddec > 0.3:
            return "Maia_H25"  # hydroxyl H
        return "Maia_H_aromatic"  # aromatic H; UA: drop, EHq: H1
    raise ValueError(f"unknown element for Maia classifier: {element!r}")


def _verify_classification_counts(
    counts: dict[str, int],
) -> dict[str, str]:
    """Verify per-cluster counts match the expected UiO-66 Zr6O4(OH)4(BDC)6
    stoichiometry. Returns a {label: status} dict for diagnostics.

    Expected per Zr6-cluster (asymmetric unit; multiply by cluster count):
      Zr  : 6
      O25 : 4   (mu3-O)
      O29 : 4   (mu3-OH)
      O1  : 24  (carboxylate, 2 per BDC ligand x 6 ligands)
      C1  : 12  (carboxylate C, 2 per BDC x 6)
      C13 : 12  (aromatic ipso, 2 per BDC x 6)
      C25 : 24  (aromatic CH, 4 per BDC x 6)
      H25 : 4   (hydroxyl H)
      H_aro: 24 (aromatic CH H; dropped in UA)
    """
    expected = {
        "Maia_Zr":  6,
        "Maia_O25": 4,
        "Maia_O29": 4,
        "Maia_O1":  24,
        "Maia_C1":  12,
        "Maia_C13": 12,
        "Maia_C25": 24,
        "Maia_H25": 4,
        "Maia_H_aromatic": 24,
    }
    status: dict[str, str] = {}
    n_zr = counts.get("Maia_Zr", 0)
    if n_zr == 0 or n_zr % 6 != 0:
        status["Maia_Zr"] = (
            f"FAIL: expected n_Zr % 6 == 0, got {n_zr}"
        )
        return status
    n_clusters = n_zr // 6
    for label, per_cluster in expected.items():
        observed = counts.get(label, 0)
        target = per_cluster * n_clusters
        if observed != target:
            status[label] = (
                f"FAIL: expected {target} (per_cluster={per_cluster} x "
                f"{n_clusters} clusters), got {observed}"
            )
        else:
            status[label] = "OK"
    return status


# ---------- CIF parser (DDEC-aware) ---------------------------------------
def _parse_cif_with_charges(cif_path: Path) -> tuple[
    np.ndarray, list[str], np.ndarray, np.ndarray
]:
    """Parse a CIF that has columns label/element/x/y/z/charge.

    Returns (lattice_3x3, elements, cart_xyz, ddec_charges).

    Uses ASE for cell + symmetry expansion to ensure Cartesian positions
    are correct; then re-reads the file line-by-line to extract per-atom
    DDEC charges in the SAME order as ASE returns the atoms (assumes
    explicit P1 with no symmetry expansion needed — true for RUBTAK01_SL_DDEC).
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

    # Read raw CIF for charge column
    text = cif_path.read_text().splitlines()
    columns: list[str] = []
    in_loop = False
    label_col = None
    type_col = None
    charge_col = None
    rows: list[list[str]] = []
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
            if "_atom_site_charge" in columns:
                charge_col = columns.index("_atom_site_charge")
            if "_atom_site_label" in columns:
                label_col = columns.index("_atom_site_label")
            if "_atom_site_type_symbol" in columns:
                type_col = columns.index("_atom_site_type_symbol")
        if in_loop and s and not s.startswith("#") and not s.startswith("_"):
            fields = re.split(r"\s+", s)
            rows.append(fields)
    if not rows or charge_col is None:
        raise ValueError(
            f"CIF {cif_path} missing _atom_site_charge column; cannot apply "
            f"Maia 2023 charge-bucket classifier"
        )
    # If type_col exists, use it; else derive from label by leading letters
    raw_elements: list[str] = []
    raw_charges: list[float] = []
    for row in rows:
        if type_col is not None and len(row) > type_col:
            raw_elements.append(row[type_col])
        elif label_col is not None and len(row) > label_col:
            m = re.match(r"^([A-Z][a-z]?)", row[label_col])
            if m is None:
                raise ValueError(f"cannot parse element from label {row[label_col]!r}")
            raw_elements.append(m.group(1))
        else:
            raise ValueError("cannot determine element column in CIF")
        if len(row) > charge_col:
            try:
                raw_charges.append(float(row[charge_col]))
            except ValueError as exc:
                raise ValueError(
                    f"cannot parse charge {row[charge_col]!r} in CIF row {row}"
                ) from exc
        else:
            raise ValueError(f"row missing charge column: {row}")
    n_asym = len(rows)
    n_expanded = len(elements)
    if n_asym == n_expanded:
        return lattice, elements, cart, np.array(raw_charges)
    # Replicate if symmetry-expanded
    if n_expanded % n_asym == 0:
        repeat = n_expanded // n_asym
        return (
            lattice,
            elements,
            cart,
            np.repeat(np.array(raw_charges), repeat),
        )
    raise ValueError(
        f"CIF {cif_path} symmetry expansion mismatch: "
        f"n_asym={n_asym}, n_expanded={n_expanded}; cannot replicate charges"
    )


# ---------- Loader entry point --------------------------------------------
def load_3b_native_maia_2023(
    repo_root: Path,
    cif_path: Path | None = None,
    variant: str = "UA",
    cutoff_angstrom: float = 14.0,
) -> NativeSystem:
    """Build a NativeSystem for 3b UiO-66 + CO2 in a Maia 2023 variant.

    Parameters
    ----------
    repo_root : Path
        Repo root for path resolution.
    cif_path : Path | None
        Override CIF path. Default: fixtures/v04/RUBTAK01_SL_DDEC.cif
        (Cavka 2008 structure refined + DDEC6 charges — used by 3a too).
    variant : str
        One of:
        - "UA"  : united-atom framework, no charges (Maia best match).
        - "UAq" : united-atom + Yang/Maia charges.
        - "EHq" : explicit H + Yang/Maia charges.
    cutoff_angstrom : float
        LJ direct cutoff. Maia uses 14 A.

    Returns
    -------
    NativeSystem
    """
    variant = variant.upper()
    if variant not in ("UA", "UAQ", "EHQ"):
        raise ValueError(f"variant must be UA / UAq / EHq, got {variant!r}")
    if cif_path is None:
        cif_path = (
            repo_root / "fixtures" / "v04" / "RUBTAK01_SL_DDEC.cif"
        )

    lattice, elements, cart, ddec_charges = _parse_cif_with_charges(cif_path)

    # Step 1: classify all atoms
    initial_types: list[str] = [
        _classify_atom_by_ddec_charge(el, q)
        for el, q in zip(elements, ddec_charges, strict=True)
    ]
    counts: dict[str, int] = {}
    for t in initial_types:
        counts[t] = counts.get(t, 0) + 1

    classification_status = _verify_classification_counts(counts)
    failures = {k: v for k, v in classification_status.items() if not v.startswith("OK")}
    if failures:
        raise ValueError(
            f"Maia atom-type classification failed (per-cluster counts off): {failures}"
        )

    # Step 2: drop aromatic H for UA variants, re-label H_aromatic for EHq
    keep_mask: list[bool] = []
    types_final: list[str] = []
    cart_final: list[np.ndarray] = []
    drop_aromatic_H = variant in ("UA", "UAQ")
    for _el, t, xyz in zip(elements, initial_types, cart, strict=True):
        if t == "Maia_H_aromatic":
            if drop_aromatic_H:
                keep_mask.append(False)
                continue
            t = "Maia_H1"  # EHq
        keep_mask.append(True)
        types_final.append(t)
        cart_final.append(xyz)
    cart_final_arr = np.asarray(cart_final)

    # Step 3: build charge array per variant
    if variant == "UA":
        charges_table = {t: 0.0 for t in MAIA_2023_UA_LJ}
        charges_table["Maia_H_aromatic"] = 0.0
        lj_table = MAIA_2023_UA_LJ
    elif variant == "UAQ":
        charges_table = dict(MAIA_2023_UAq_CHARGES_E)
        lj_table = MAIA_2023_UA_LJ
    else:  # EHQ
        charges_table = dict(MAIA_2023_EHq_CHARGES_E)
        lj_table = MAIA_2023_EHq_LJ

    framework_charges = np.array(
        [charges_table.get(t, 0.0) for t in types_final]
    )

    # Per-type mass map
    type_to_mass: dict[str, float] = {}
    for t in set(types_final):
        if t == "Maia_C25" and variant == "EHQ":
            type_to_mass[t] = ELEMENT_MASS_AMU["C"]  # H is explicit, not lumped
        elif t == "Maia_C25":
            type_to_mass[t] = ELEMENT_MASS_AMU["C"] + ELEMENT_MASS_AMU["H"]  # UA: CH lumped
        else:
            type_to_mass[t] = MAIA_TYPE_MASS_AMU.get(t, ELEMENT_MASS_AMU.get(t.split("_")[1][:2], 12.0))
    type_to_mass.setdefault("C_co2", ELEMENT_MASS_AMU["C"])
    type_to_mass.setdefault("O_co2", ELEMENT_MASS_AMU["O"])

    # Pair table: framework × CO2 via Lorentz-Berthelot mixing
    pair_table = PairTable()
    for ftype, (f_eps, f_sig) in lj_table.items():
        if f_eps == 0.0 and f_sig == 0.0:
            continue  # H25 sigma=epsilon=0 contributes nothing
        for ptype, (p_eps, p_sig) in TRAPPE_CO2_SELF_LJ.items():
            eps_cross = math.sqrt(f_eps * p_eps)
            sig_cross = 0.5 * (f_sig + p_sig)
            pair_table.set(
                ftype, ptype,
                LennardJones12_6(
                    epsilon_K=eps_cross,
                    sigma_angstrom=sig_cross,
                    cutoff_A=cutoff_angstrom,
                ),
            )
    # Self-pairs (probe-probe; only relevant if probe atoms see each other)
    for ptype, (eps, sig) in TRAPPE_CO2_SELF_LJ.items():
        pair_table.set(
            ptype, ptype,
            LennardJones12_6(
                epsilon_K=eps, sigma_angstrom=sig,
                cutoff_A=cutoff_angstrom,
            ),
        )

    # CO2 partial charges: for UA (no framework charges), the framework-probe
    # Coulomb cross-term is identically zero regardless of probe charge, and
    # intra-probe Coulomb is a constant that does not affect Widom DeltaU.
    # So we zero the CO2 charges in UA to avoid the Ewald machinery overhead
    # while remaining physically equivalent. UAq / EHq use Maia's TraPPE-CO2
    # partial charges.
    if variant == "UA":
        probe_charges_arr = np.zeros(3)
    else:
        probe_charges_arr = np.array([
            TRAPPE_CO2_CHARGES_E["O_co2"],
            TRAPPE_CO2_CHARGES_E["C_co2"],
            TRAPPE_CO2_CHARGES_E["O_co2"],
        ])
    probe = ProbeMolecule(
        name="CO2",
        types=["O_co2", "C_co2", "O_co2"],
        body_positions=np.array([
            [0.0, 0.0, TRAPPE_CO2_BOND_LENGTH_A],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -TRAPPE_CO2_BOND_LENGTH_A],
        ]),
        charges_e=probe_charges_arr,
    )

    # Supercell to ensure simulation cell >= 2 x cutoff in each direction
    target = 2.0 * cutoff_angstrom
    n_a = max(1, math.ceil(target / np.linalg.norm(lattice[0])))
    n_b = max(1, math.ceil(target / np.linalg.norm(lattice[1])))
    n_c = max(1, math.ceil(target / np.linalg.norm(lattice[2])))

    return NativeSystem(
        framework_types=types_final,
        framework_cart_angstrom=cart_final_arr,
        framework_charges_e=framework_charges,
        cell_matrix_angstrom=lattice,
        pair_table=pair_table,
        probe=probe,
        type_to_mass_amu=type_to_mass,
        supercell_replicas=(n_a, n_b, n_c),
        energy_cutoff_angstrom=cutoff_angstrom,
    )
