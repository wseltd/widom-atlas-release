"""Auto-loaders that build :class:`NativeSystem` from v04 YAML branch entries.

Currently supports:

  * LJ-only branches with optional `cross_LJ_explicit_optional` (6c MFI+Ar,
    6a MFI+CH4, 6b MFI+Kr, 6d MFI+CH4 numerical).
  * Branches with `framework_charges` declared in YAML.
  * Single-atom rigid probes (Ar, Kr, CH4-sp3, etc.) plus rigid 3-atom CO2.

Per-branch FF lineage (Lin/Mercado Buckingham, Dzubak C5+D6) is handled
elsewhere: those branches need the `.def` parsers in `raspa2/input_writer`
or a dedicated Dzubak parser, not this LJ-focused loader. The native
evaluator's Buckingham/Dzubak modes are exercised through hand-built
NativeSystem objects in tests until those parsers are wired through.

Inputs:
  * `repo_root` (path to widom-atlas/)
  * `branch_raw` (a YAML branch entry dict)

Outputs: a :class:`NativeSystem` ready for `run_native_widom`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..raspa3.binary_interactions import binary_interactions_for_branch
from ..raspa3.input_writer import _ELEMENT_LJ as _DEFAULT_FRAMEWORK_LJ
from ..raspa3.input_writer import _is_zeolite_branch, _uff_lj_K
from .potentials import LennardJones12_6, PairTable
from .system import NativeSystem, ProbeMolecule


ELEMENT_MASS_AMU: dict[str, float] = {
    "H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811,
    "C": 12.0107, "N": 14.00674, "O": 15.9994, "F": 18.9984032, "Ne": 20.1797,
    "Na": 22.98977, "Mg": 24.305, "Al": 26.981538, "Si": 28.0855, "P": 30.973761,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Mn": 54.938049, "Fe": 55.845, "Co": 58.9332, "Ni": 58.6934, "Cu": 63.546,
    "Zn": 65.409, "Br": 79.904, "Kr": 83.798, "Mo": 95.94, "Pd": 106.42,
    "Cd": 112.411, "Sn": 118.71, "I": 126.90447, "Xe": 131.293,
    "Zr": 91.224, "Y": 88.906, "Nb": 92.906, "Ru": 101.07, "Rh": 102.906,
    "Ag": 107.868, "Sb": 121.76, "Te": 127.60, "Cs": 132.9054,
    "Ba": 137.327, "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Pt": 195.084,
    "Au": 196.967, "Pb": 207.2,
}

# Per-probe-atom Lennard-Jones self-parameters. Sourced from the same
# Harris-Yung 1995 / Garcia-Perez 2007 / TraPPE conventions RASPA3 uses
# in its `force_field.json` SelfInteractions block. Without these, the
# LB mixing for cross-pairs (framework × probe atom) would generate zero
# LJ between framework Si/O and probe C/O atoms, and the bare Coulomb at
# close distances becomes catastrophically attractive.
PROBE_ATOM_SELF_LJ: dict[str, tuple[float, float]] = {
    # Harris-Yung 1995 EPM2 CO2 (tabulated by Garcia-Perez 2007 Table 1):
    "C_co2": (28.129, 2.76),
    "O_co2": (80.507, 3.033),
    # Single-site noble gases / united-atom CH4 from the YAML's gas.self_LJ
    # entries — overridden in load_native_system_for_branch.
}


def _lattice_matrix_from_params(
    a: float, b: float, c: float,
    alpha_deg: float, beta_deg: float, gamma_deg: float,
) -> np.ndarray:
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * cos_g) / sin_g
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([
        [a, 0.0, 0.0],
        [b * cos_g, b * sin_g, 0.0],
        [cx, cy, cz],
    ])


def _supercell_for_cutoff(cell_matrix: np.ndarray, cutoff_angstrom: float) -> tuple[int, int, int]:
    """Pick n_a × n_b × n_c so each box vector is > 2·cutoff."""
    target = 2.0 * cutoff_angstrom
    n_a = max(1, math.ceil(target / np.linalg.norm(cell_matrix[0])))
    n_b = max(1, math.ceil(target / np.linalg.norm(cell_matrix[1])))
    n_c = max(1, math.ceil(target / np.linalg.norm(cell_matrix[2])))
    return (n_a, n_b, n_c)


def _parse_cif_simple(cif_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read a CIF and return (lattice_matrix, atom_types, cartesian_positions),
    expanding the asymmetric unit via the declared space-group symmetry.

    Uses ASE's CIF reader, which correctly applies symmetry operations and
    deduplicates equivalent positions. Required for any CIF that lists only
    the asymmetric unit (e.g. the IZA MFI Pnma file has 38 unique sites; the
    P1 expansion has 288).
    """
    import warnings
    from ase.io import read as _ase_read
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms_obj = _ase_read(str(cif_path))
    if isinstance(atoms_obj, list):
        atoms_obj = atoms_obj[0]
    lattice = np.asarray(atoms_obj.cell)
    types = list(atoms_obj.get_chemical_symbols())
    cart = np.asarray(atoms_obj.get_positions())
    return lattice, types, cart


def _parse_cif_charges(cif_path: Path, n_expected_atoms: int) -> np.ndarray | None:
    """Read the `_atom_site_charge` column from a CIF, if present, and
    duplicate across symmetry-expanded sites.

    Returns an array of length n_expected_atoms with the per-site charge,
    or None if the CIF has no `_atom_site_charge` column.

    Because ASE's CIF reader (used in `_parse_cif_simple`) expands the
    asymmetric unit by space-group symmetry, the per-site charge must be
    duplicated to match. We do this by reading the asymmetric-unit charges
    in original CIF order and repeating each value `n_expected_atoms //
    n_asym_atoms` times. Atoms on special positions reduce the symmetry
    multiplicity, so the duplication is only exact when every site has
    full multiplicity. For CoRE-MOF / FIQCEN-style CIFs (which list every
    P1 site explicitly), this is correct. For Pnma / Fm-3m CIFs with
    symmetry-reduced sites, we fall back to None (asymmetric expansion
    is not safe to charge-replicate).
    """
    text = cif_path.read_text().splitlines()
    columns: list[str] = []
    in_loop = False
    label_col = None
    charge_col = None
    charges: list[float] = []
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
        if in_loop and s and not s.startswith("#") and not s.startswith("_"):
            fields = re.split(r"\s+", s)
            if charge_col is not None and len(fields) > charge_col:
                try:
                    charges.append(float(fields[charge_col]))
                except (ValueError, IndexError):
                    pass
    if not charges:
        return None
    n_asym = len(charges)
    if n_asym == n_expected_atoms:
        return np.array(charges)
    if n_expected_atoms % n_asym == 0:
        repeat = n_expected_atoms // n_asym
        return np.repeat(np.array(charges), repeat)
    return None


def _probe_for_species(species: str, self_lj: dict | None) -> ProbeMolecule:
    """Return a rigid probe for a known species."""
    if species in ("Ar", "Kr", "Xe", "He", "Ne"):
        return ProbeMolecule(
            name=species, types=[species],
            body_positions=np.array([[0.0, 0.0, 0.0]]),
        )
    if species in ("CH4", "methane"):
        return ProbeMolecule(
            name="CH4", types=["CH4_sp3"],
            body_positions=np.array([[0.0, 0.0, 0.0]]),
        )
    if species == "CO2":
        # EPM2-style rigid linear CO2: O-C-O along z, C-O = 1.149 Å
        return ProbeMolecule(
            name="CO2",
            types=["O_co2", "C_co2", "O_co2"],
            body_positions=np.array([
                [0.0, 0.0, 1.149],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.149],
            ]),
            charges_e=np.array([-0.3256, 0.6512, -0.3256]),
        )
    raise ValueError(f"unsupported probe species: {species!r}")


def load_native_system_for_branch(
    branch_raw: dict,
    repo_root: Path,
) -> NativeSystem:
    """Build a NativeSystem for an LJ-only branch (used by V1, V2).

    For Buckingham/Dzubak branches, use the dedicated `from_raspa2_package`
    loader (not implemented in this module).
    """
    fw = branch_raw["framework"]
    cif_path = (repo_root / fw["source_cif_path"]).resolve()
    if not cif_path.exists():
        raise FileNotFoundError(f"framework CIF not found: {cif_path}")
    lattice, types_raw, cart = _parse_cif_simple(cif_path)

    label_map = fw.get("atom_label_map", {})
    types = [label_map.get(t, t) for t in types_raw]

    # Build type_to_mass keyed by the relabelled sub-lattice label (in `types`).
    # We resolve atomic mass by element symbol — extracted either from the
    # YAML's `atom_label_map` key (when it IS an element symbol like `Mg`,
    # `Cu`) or from the CIF's own _atom_site_type_symbol (which is already
    # `types_raw[i]`). Skip YAML keys that are descriptors (e.g.
    # `C_aromatic_CH` in the 2a HKUST-1 YAML) — they're documentation of the
    # role, not the element.
    type_to_mass: dict[str, float] = {}
    for cif_type_symbol, sub_label in zip(types_raw, types, strict=True):
        if sub_label in type_to_mass:
            continue
        m = ELEMENT_MASS_AMU.get(cif_type_symbol)
        if m is None:
            # Fall back to the leading element pattern in the sub_label
            base = re.match(r"^([A-Z][a-z]?)", cif_type_symbol)
            el = base.group(1) if base else cif_type_symbol
            m = ELEMENT_MASS_AMU.get(el)
        if m is None:
            raise KeyError(
                f"no atomic mass for CIF type {cif_type_symbol!r} "
                f"(sublattice label {sub_label!r}); add it to ELEMENT_MASS_AMU"
            )
        type_to_mass[sub_label] = m

    ep = branch_raw.get("electrostatics_per_branch") or {}
    cutoff = float(ep.get("direct_cutoff_angstrom", 12.0))
    lj_shifted = (ep.get("lj_treatment") == "shifted_truncated")

    probe_species = branch_raw["gas"]["species"]
    probe_self_lj = (branch_raw["gas"].get("self_LJ") or {})
    probe = _probe_for_species(probe_species, probe_self_lj)
    for ptype, mass in zip(probe.types, [
        ELEMENT_MASS_AMU.get(re.match(r"^([A-Z][a-z]?)", t).group(1), 12.0) for t in probe.types
    ]):
        type_to_mass.setdefault(ptype, mass)

    pair_table = PairTable()
    ff = branch_raw["force_field"]
    framework_q_map: dict[str, float] = ff.get("framework_charges", {}) or {}
    framework_q_resolved: dict[str, float] = {}
    for cif_el, sub_label in (label_map.items() if label_map else []):
        if cif_el in framework_q_map:
            framework_q_resolved[sub_label] = float(framework_q_map[cif_el])
        elif sub_label in framework_q_map:
            framework_q_resolved[sub_label] = float(framework_q_map[sub_label])
    for sub_label in set(types):
        if sub_label not in framework_q_resolved:
            framework_q_resolved[sub_label] = float(framework_q_map.get(sub_label, 0.0))

    framework_charges = np.array([framework_q_resolved.get(t, 0.0) for t in types])

    # If the YAML didn't supply framework_charges (e.g. 2a HKUST-1 reads
    # Nazarian DDEC from the CIF column directly), fall back to the CIF's
    # `_atom_site_charge` column when present.
    if float(np.abs(framework_charges).sum()) == 0.0:
        cif_charges = _parse_cif_charges(cif_path, n_expected_atoms=len(types))
        if cif_charges is not None and float(np.abs(cif_charges).sum()) > 0.0:
            framework_charges = cif_charges

    framework_types_set = set(types)
    probe_types_set = set(probe.types)
    # Accept the YAML's gas species name ("CH4", "Ar", "Kr", ...) as an alias for
    # the internal RASPA probe-atom type ("CH4_sp3", "Ar", "Kr", "C_co2", "O_co2"),
    # so cross-LJ pair-keys like `O_zeo_CH4` resolve cleanly.
    probe_species_aliases = {probe_species: probe.types[0]}
    if probe_species == "CO2":
        probe_species_aliases.update({
            "C_CO2": "C_co2", "C_co2": "C_co2",
            "O_CO2": "O_co2", "O_co2": "O_co2",
        })
    # Inverse alias map for framework atoms: when the YAML's atom_label_map
    # renames a CIF element symbol (`O` → `O_zeo`, `Si` → `Si`), the RASPA3
    # binary_interactions_for_branch table uses the ORIGINAL element symbol.
    # Map element-symbol keys back to their renamed labels so the cross-LJ
    # splitter can find them in framework_types_set.
    framework_label_aliases: dict[str, str] = {}
    if label_map:
        for cif_el_key, renamed in label_map.items():
            framework_label_aliases[cif_el_key] = renamed
            # Also try the base element symbol of the descriptor key
            m_el = re.match(r"^([A-Z][a-z]?)", cif_el_key)
            if m_el:
                framework_label_aliases[m_el.group(1)] = renamed

    def _resolve_alias(name: str) -> str:
        if name in probe_species_aliases:
            return probe_species_aliases[name]
        if name in framework_label_aliases:
            return framework_label_aliases[name]
        return name

    # Framework self-LJ (Garcia-Perez / TraPPE-zeo defaults for Si, O, Al, etc.)
    # Required so cross-pairs between framework atom types and probe atom types
    # that aren't listed in `cross_LJ_explicit_optional` can be generated by
    # Lorentz-Berthelot mixing.
    framework_self_lj: dict[str, tuple[float, float]] = {}
    # Branch-aware framework self-LJ (v0.4.2 fix, parallel to the RASPA3 generator):
    # zeolite branches keep the TraPPE-zeo / cation table (`_ELEMENT_LJ`), correctly
    # tagged; MOF frameworks use genuine UFF per element so the provenance is truthful
    # (the old shared table mislabelled guest/zeolite values as "UFF"). The native
    # charged-MOF cell path remains a disclosed limitation; this only fixes the FF
    # provenance, not the cell handling.
    is_zeolite_fw = _is_zeolite_branch(branch_raw)
    for cif_type_symbol, sub_label in zip(types_raw, types, strict=True):
        if sub_label in framework_self_lj:
            continue
        m_el = re.match(r"^([A-Z][a-z]?)", cif_type_symbol)
        el = m_el.group(1) if m_el else cif_type_symbol
        if is_zeolite_fw:
            if cif_type_symbol in _DEFAULT_FRAMEWORK_LJ:
                framework_self_lj[sub_label] = _DEFAULT_FRAMEWORK_LJ[cif_type_symbol]
            elif el in _DEFAULT_FRAMEWORK_LJ:
                framework_self_lj[sub_label] = _DEFAULT_FRAMEWORK_LJ[el]
        else:
            framework_self_lj[sub_label] = _uff_lj_K(el)
    # Set framework self pairs in the table for completeness (not used by Widom
    # on a frozen framework, but written for round-trip).
    for ftype, (eps, sig) in framework_self_lj.items():
        pair_table.set(
            ftype, ftype,
            LennardJones12_6(
                epsilon_K=eps, sigma_angstrom=sig,
                cutoff_A=cutoff, shifted=lj_shifted,
            ),
        )

    # Collect cross-LJ entries strictly from the RASPA3 path's
    # `binary_interactions_for_branch` — this is what RASPA3 actually emits
    # (e.g. for 6c it emits Talu-Myers Ar-O; for 6a it emits nothing and
    # falls back to LB mixing). Loose `cross_LJ_*` YAML keys are NOT honoured
    # by the native loader, mirroring RASPA3's behaviour so V1-V4 validate
    # against the same physics RASPA3 ran.
    branch_id = branch_raw.get("branch_id", "")
    framework_elements_for_bi = set()
    for cif_el in (label_map.keys() if label_map else set(types_raw)):
        framework_elements_for_bi.add(cif_el)
    bi_entries = binary_interactions_for_branch(
        branch_id=branch_id,
        present_framework_elements=framework_elements_for_bi,
        gas_species=probe_species,
    )
    cross_lj: dict[str, dict] = {}
    for bi in bi_entries:
        if bi.get("type") != "lennard-jones":
            continue
        names = bi.get("names") or []
        if len(names) != 2:
            continue
        a, b = names
        pair_key = f"{a}_{b}"
        cross_lj[pair_key] = {
            "epsilon_K": bi["parameters"][0],
            "sigma_angstrom": bi["parameters"][1],
        }
    for pair_key, params in cross_lj.items():
        type_a = type_b = None
        # Try every `_` split point; pick the one that matches a framework
        # type with a probe type (in either order). Probe-species aliases
        # ("CH4" → "CH4_sp3", "C_CO2" → "C_co2") are accepted.
        for split_pos in range(1, len(pair_key)):
            if pair_key[split_pos] != "_":
                continue
            left_raw = pair_key[:split_pos]
            right_raw = pair_key[split_pos + 1:]
            left = _resolve_alias(left_raw)
            right = _resolve_alias(right_raw)
            if (
                (left in framework_types_set and right in probe_types_set)
                or (left in probe_types_set and right in framework_types_set)
            ):
                type_a, type_b = left, right
                break
        if type_a is None:
            raise ValueError(
                f"cross-LJ pair-key {pair_key!r} could not be split into "
                f"framework × probe types (framework={framework_types_set}, "
                f"probe={probe_types_set})"
            )
        eps = float(params["epsilon_K"])
        sig = float(params["sigma_angstrom"])
        pair_table.set(
            type_a, type_b,
            LennardJones12_6(
                epsilon_K=eps, sigma_angstrom=sig,
                cutoff_A=cutoff, shifted=lj_shifted,
            ),
        )

    probe_eps_sigma = probe_self_lj.get("epsilon_K"), probe_self_lj.get("sigma_angstrom")
    probe_self_resolved: dict[str, tuple[float, float]] = {}
    if probe_eps_sigma[0] is not None and probe_eps_sigma[1] is not None:
        # Single-site probe — apply YAML gas.self_LJ to every probe atom (Ar, Kr,
        # CH4_sp3, etc.).
        for ptype in probe.types:
            probe_self_resolved[ptype] = (
                float(probe_eps_sigma[0]), float(probe_eps_sigma[1]),
            )
    # Multi-atom rigid probes (CO2) — look up per-atom-type LJ from the
    # PROBE_ATOM_SELF_LJ table.
    for ptype in probe.types:
        if ptype not in probe_self_resolved and ptype in PROBE_ATOM_SELF_LJ:
            probe_self_resolved[ptype] = PROBE_ATOM_SELF_LJ[ptype]
    for ptype, (eps, sig) in probe_self_resolved.items():
        pair_table.set(
            ptype, ptype,
            LennardJones12_6(
                epsilon_K=eps, sigma_angstrom=sig,
                cutoff_A=cutoff, shifted=lj_shifted,
            ),
        )

    # Lorentz-Berthelot mixing for any (framework, probe) cross-pair not
    # already set by `cross_LJ_explicit_optional`. ε = √(ε_a ε_b), σ = (σ_a + σ_b)/2.
    for ftype, (eps_f, sig_f) in framework_self_lj.items():
        for ptype, (eps_p, sig_p) in probe_self_resolved.items():
            if pair_table.get(ftype, ptype) is not None:
                continue
            eps_cross = math.sqrt(max(eps_f * eps_p, 0.0))
            sig_cross = 0.5 * (sig_f + sig_p)
            pair_table.set(
                ftype, ptype,
                LennardJones12_6(
                    epsilon_K=eps_cross, sigma_angstrom=sig_cross,
                    cutoff_A=cutoff, shifted=lj_shifted,
                ),
            )

    n_a, n_b, n_c = _supercell_for_cutoff(lattice, cutoff)

    return NativeSystem(
        framework_types=types,
        framework_cart_angstrom=cart,
        framework_charges_e=framework_charges,
        cell_matrix_angstrom=lattice,
        pair_table=pair_table,
        probe=probe,
        type_to_mass_amu=type_to_mass,
        supercell_replicas=(n_a, n_b, n_c),
        energy_cutoff_angstrom=cutoff,
    )
