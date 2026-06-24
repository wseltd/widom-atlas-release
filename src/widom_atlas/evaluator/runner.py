"""Widom-evaluator orchestrator.

Inputs:
- ASE Atoms (host framework, with cell)
- a ``UserParameterFile`` (validated)
- a ``Component`` (multi-site rigid gas)
- temperature K
- n_insertions (and optional grid size)

Outputs (all on a results dataclass):
- per-insertion (E_LJ, E_coul, E_total) arrays in K
- K_H estimator (mol/kg/Pa) and Q_ads estimator (kJ/mol)
- provenance dict (FF source SHA-256, component name, RNG seed,
  cutoff, alpha, n_insertions, grid mode, framework SHA-256)

Design choices:
- ``min_cell_face_distance >= 2 * r_cut`` is enforced by multiplying the
  primitive cell along each underspec'd axis. This is the standard supercell
  policy in molecular simulation; we record the multiplier in provenance.
- The host atom-type table is built by mapping ASE ``chemical_symbols`` →
  UserParameterFile.framework_atom_types[].label. If a host atom has no
  match the runner returns ``RunnerStatus.UNRESOLVED_FRAMEWORK_ATOMS``.
- Energies in K are converted to J/mol via Avogadro / Boltzmann factors
  for the final K_H / Q_ads scalars.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from widom_atlas.backends.user_parameterised import UserParameterFile

from .component import Component, orient
from .energy import cell_face_distances, insertion_energy_K
from .ff_loader import FFTables, lift_to_tables, lj_pair_tables
from .grid import deterministic_uniform_grid, stochastic_uniform_random

N_AVOGADRO = 6.022_140_76e23
K_BOLTZMANN_J_PER_K = 1.380_649e-23
K_BOLTZMANN_KJ_MOL_PER_K = K_BOLTZMANN_J_PER_K * N_AVOGADRO * 1e-3  # 0.008314 kJ/mol/K
GAS_CONSTANT_J_MOL_K = K_BOLTZMANN_J_PER_K * N_AVOGADRO  # 8.314 J/mol/K


RunnerStatusLiteral = Literal[
    "ok",
    "unresolved_framework_atoms",
    "unresolved_gas_sites",
    "cell_too_small",
    "non_neutral_after_neutralisation",
]


@dataclass(frozen=True)
class WidomResult:
    """Per-run scalar + per-insertion result bundle."""

    status: RunnerStatusLiteral
    n_insertions_attempted: int
    n_insertions_used: int
    framework_name: str
    component_name: str
    temperature_K: float
    e_lj_K: np.ndarray            # (n_used,) Lennard-Jones contribution
    e_coul_K: np.ndarray          # (n_used,) Coulomb contribution
    e_total_K: np.ndarray         # (n_used,) e_lj + e_coul
    boltzmann_weights: np.ndarray  # (n_used,) exp(-beta U)
    insertion_centres_A: np.ndarray  # (n_used, 3) Cartesian position of each insertion centre
    KH_mol_per_kg_per_Pa: float | None
    Qads_kJ_per_mol: float | None
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_atom_id_arrays(
    atoms: Atoms, ff: FFTables
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Map host symbols to integer ids in the FF framework table.

    Returns (host_ids, host_charges, host_positions, missing_labels).
    Missing labels indicate a host atom with no match.
    """
    symbols = atoms.get_chemical_symbols()
    fw_labels = list(ff.framework.labels)
    label_to_idx: dict[str, int] = {label: i for i, label in enumerate(fw_labels)}
    host_ids = np.full(len(symbols), -1, dtype=int)
    missing: list[str] = []
    for i, sym in enumerate(symbols):
        if sym in label_to_idx:
            host_ids[i] = label_to_idx[sym]
        else:
            missing.append(sym)
    if missing:
        return host_ids, np.zeros(len(symbols)), atoms.get_positions(), sorted(set(missing))
    host_charges = ff.framework.charge_e[host_ids]
    return host_ids, host_charges, atoms.get_positions(), []


def _resolve_gas_site_ids(component: Component, ff: FFTables) -> tuple[np.ndarray, list[str]]:
    """Map component site labels to integer ids in the FF gas table."""
    gas_labels = list(ff.gas.labels)
    label_to_idx: dict[str, int] = {label: i for i, label in enumerate(gas_labels)}
    ids = np.full(component.n_sites, -1, dtype=int)
    missing: list[str] = []
    for i, lab in enumerate(component.site_labels):
        if lab in label_to_idx:
            ids[i] = label_to_idx[lab]
        else:
            missing.append(lab)
    return ids, missing


def _grow_supercell(atoms: Atoms, cell: np.ndarray, target_min_face_A: float) -> tuple[Atoms, tuple[int, int, int]]:
    face = cell_face_distances(cell)
    multiplier = tuple(int(np.ceil(target_min_face_A / d)) if d < target_min_face_A else 1 for d in face)
    multiplier_typed: tuple[int, int, int] = (multiplier[0], multiplier[1], multiplier[2])
    if multiplier_typed == (1, 1, 1):
        return atoms, multiplier_typed
    return atoms.repeat(multiplier_typed), multiplier_typed


def run_widom_evaluator(
    *,
    atoms: Atoms,
    framework_name: str,
    user_parameter_file: UserParameterFile,
    component: Component,
    temperature_K: float,
    n_insertions: int,
    seed: int,
    r_cut_A: float = 12.0,
    grid_mode: Literal["deterministic_uniform", "stochastic_uniform"] = "stochastic_uniform",
    framework_source_path: Path | None = None,
    ff_source_sha: str | None = None,
) -> WidomResult:
    """Run a Widom evaluation and return scalar + per-insertion results."""
    ff = lift_to_tables(user_parameter_file)
    rng = np.random.default_rng(seed)

    cell = np.asarray(atoms.cell.array, dtype=float)
    atoms_super, multiplier = _grow_supercell(atoms, cell, target_min_face_A=2.0 * r_cut_A)
    cell_super = np.asarray(atoms_super.cell.array, dtype=float)
    cell_inv = np.linalg.inv(cell_super)

    host_ids, host_charges, host_pos, missing_fw = _build_atom_id_arrays(atoms_super, ff)
    if missing_fw:
        return WidomResult(
            status="unresolved_framework_atoms",
            n_insertions_attempted=n_insertions,
            n_insertions_used=0,
            framework_name=framework_name,
            component_name=component.name,
            temperature_K=temperature_K,
            e_lj_K=np.array([]),
            e_coul_K=np.array([]),
            e_total_K=np.array([]),
            boltzmann_weights=np.array([]),
            insertion_centres_A=np.zeros((0, 3)),
            KH_mol_per_kg_per_Pa=None,
            Qads_kJ_per_mol=None,
            provenance={
                "framework_name": framework_name,
                "missing_framework_labels": missing_fw,
                "supercell_multiplier": list(multiplier),
            },
            warnings=[
                f"Framework atom labels {missing_fw} not in UserParameterFile.framework_atom_types"
            ],
        )

    site_ids, missing_gas = _resolve_gas_site_ids(component, ff)
    if missing_gas:
        return WidomResult(
            status="unresolved_gas_sites",
            n_insertions_attempted=n_insertions,
            n_insertions_used=0,
            framework_name=framework_name,
            component_name=component.name,
            temperature_K=temperature_K,
            e_lj_K=np.array([]),
            e_coul_K=np.array([]),
            e_total_K=np.array([]),
            boltzmann_weights=np.array([]),
            insertion_centres_A=np.zeros((0, 3)),
            KH_mol_per_kg_per_Pa=None,
            Qads_kJ_per_mol=None,
            provenance={
                "framework_name": framework_name,
                "missing_gas_site_labels": missing_gas,
            },
            warnings=[
                f"Gas site labels {missing_gas} not in UserParameterFile.gas_sites"
            ],
        )

    sigma_pair, epsilon_pair = lj_pair_tables(ff.framework, ff.gas, ff.mixing_rule)
    coulomb = ff.coulomb_method

    # Pre-allocate per-insertion arrays
    e_lj = np.zeros(n_insertions)
    e_coul = np.zeros(n_insertions)
    if grid_mode == "deterministic_uniform":
        n_per_axis = round(n_insertions ** (1.0 / 3.0))
        n_per_axis = max(2, n_per_axis)
        centres = deterministic_uniform_grid(cell_super, n_per_axis)
        n_used = centres.shape[0]
        if n_used != n_insertions:
            n_insertions = n_used
            e_lj = np.zeros(n_used)
            e_coul = np.zeros(n_used)
    elif grid_mode == "stochastic_uniform":
        centres = stochastic_uniform_random(cell_super, n_insertions, rng)
    else:
        raise ValueError(f"unknown grid_mode {grid_mode!r}")

    for i in range(centres.shape[0]):
        site_offsets_i = orient(component, rng)
        e_lj_i, e_coul_i = insertion_energy_K(
            framework_positions=host_pos,
            framework_atom_ids=host_ids,
            cell=cell_super,
            cell_inv=cell_inv,
            insertion_centre=centres[i],
            site_offsets=site_offsets_i,
            site_atom_ids=site_ids,
            sigma_pair=sigma_pair,
            epsilon_pair=epsilon_pair,
            fw_charges=host_charges,
            gas_charges=component.site_charges,
            r_cut_A=r_cut_A,
            coulomb=coulomb,
        )
        e_lj[i] = e_lj_i
        e_coul[i] = e_coul_i

    e_total = e_lj + e_coul
    e_clip = np.clip(e_total / temperature_K, -200.0, 200.0)  # avoid overflow on hard clashes
    bz = np.exp(-e_clip)
    n_used = bz.shape[0]

    # K_H (mol/kg/Pa) ≈ <exp(-βU)> / (rho_host * R * T)
    # rho_host (kg/m^3) = total_mass_amu * 1.66054e-27 / volume_m3
    volume_m3 = float(abs(np.linalg.det(cell_super))) * 1e-30
    masses = atoms_super.get_masses()
    total_mass_kg = float(masses.sum()) * 1.66053906660e-27
    rho_host = total_mass_kg / volume_m3 if volume_m3 > 0 else 0.0
    mean_bz = float(bz.mean()) if n_used > 0 else 0.0
    if rho_host > 0 and mean_bz > 0:
        KH = mean_bz / (rho_host * GAS_CONSTANT_J_MOL_K * temperature_K)
    else:
        KH = None

    # <U exp(-βU)> / <exp(-βU)>  →  Q_ads ≈ -<E> - kT
    if mean_bz > 0:
        avg_E_K = float(np.sum(e_total * bz) / np.sum(bz))
        avg_E_kJ_mol = avg_E_K * K_BOLTZMANN_KJ_MOL_PER_K
        Qads = -avg_E_kJ_mol - K_BOLTZMANN_KJ_MOL_PER_K * temperature_K
    else:
        Qads = None

    return WidomResult(
        status="ok",
        n_insertions_attempted=n_insertions,
        n_insertions_used=n_used,
        framework_name=framework_name,
        component_name=component.name,
        temperature_K=temperature_K,
        e_lj_K=e_lj,
        e_coul_K=e_coul,
        e_total_K=e_total,
        boltzmann_weights=bz,
        insertion_centres_A=centres,
        KH_mol_per_kg_per_Pa=KH,
        Qads_kJ_per_mol=Qads,
        provenance={
            "framework_name": framework_name,
            "framework_source_path": str(framework_source_path) if framework_source_path else "",
            "framework_sha256": (
                _sha256_of_file(framework_source_path)
                if framework_source_path is not None and Path(framework_source_path).exists()
                else ""
            ),
            "ff_source_sha256": ff_source_sha or "",
            "n_insertions": n_used,
            "seed": seed,
            "r_cut_A": r_cut_A,
            "grid_mode": grid_mode,
            "supercell_multiplier": list(multiplier),
            "coulomb_method": coulomb,
            "mixing_rule": ff.mixing_rule,
            "host_total_mass_kg": total_mass_kg,
            "host_volume_A3": float(abs(np.linalg.det(cell_super))),
            "host_density_kg_per_m3": rho_host,
            "redistribution_status": ff.redistribution_status,
        },
        warnings=list(component.warnings),
    )


def load_atoms(structure_path: Path) -> Atoms:
    """Load an ASE Atoms from a CIF / XYZ etc."""
    raw = ase_read(str(structure_path))
    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"no atoms parsed from {structure_path}")
        return raw[0]
    return raw


__all__ = [
    "GAS_CONSTANT_J_MOL_K",
    "K_BOLTZMANN_KJ_MOL_PER_K",
    "WidomResult",
    "load_atoms",
    "run_widom_evaluator",
]
