"""Native Widom runner.

Drives the Widom insertion loop for a :class:`NativeSystem` and a probe
molecule. Per insertion:

  1. Pick a random Cartesian point uniformly in the simulation cell.
  2. Pick a random orientation in SO(3).
  3. Place the probe atoms at that pose.
  4. Sum the framework-probe pair energy (direct space, minimum-image,
     pair-table-driven), with a hard short-range cutoff inherited from the
     pair table.
  5. Stream the total energy (in K) into the :class:`WidomAccumulator`.

Electrostatics: this runner does **direct-space** electrostatics under
the same cutoff as the van der Waals pair table; Ewald is a follow-on.
For verdict-affecting branches that need accurate Ewald (i.e. any
charged framework), the runner refuses to run until `enable_ewald=True`
is wired through OR the system carries no partial charges.

The runner returns a `NativeWidomResult` with K_H, Q_st, seed manifest,
and a per-batch energy histogram for diagnostics.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .ewald import (
    EwaldParameters,
    build_framework_ewald_cache,
    widom_ewald_delta_U,
)
from .system import NativeSystem, insert_probe_at, random_rotation_matrix
from .widom import WidomAccumulator, framework_mass_kg

COULOMB_K_ANGSTROM_PER_E2 = 167101.0
"""Coulomb prefactor in K·Å·e^-2: k_e / k_B, with k_e = 1/(4πε_0).

Numerical value: (e^2 / (4πε_0)) / k_B = 1.4399645 eV·Å / (8.617e-5 eV/K)
                                       ≈ 1.6710e5 K·Å·e^-2.
"""


@dataclass
class StrongestInsertion:
    """The lowest-energy Widom insertion recorded during a native run.

    Used by the site-truth verdict to extract atom-pair distances (e.g.
    Mg-O(CO₂), Cu-O(CO₂), Na-OC1/OC3) from the strongest-binding probe pose.
    """
    U_K: float
    probe_types: list[str]
    probe_cartesian_angstrom: np.ndarray  # (n_probe_atoms, 3)
    framework_types: list[str]
    framework_cartesian_angstrom: np.ndarray  # (n_framework_atoms, 3)
    supercell_matrix_angstrom: np.ndarray  # (3, 3) — needed for PBC distance calc
    temperature_K: float
    seed: int

    def min_image_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Minimum-image distance under the supercell lattice."""
        inv = np.linalg.inv(self.supercell_matrix_angstrom)
        d = q - p
        frac_d = d @ inv
        frac_d -= np.round(frac_d)
        return float(np.linalg.norm(frac_d @ self.supercell_matrix_angstrom))

    def closest_framework_atom_of_type(
        self, probe_atom_idx: int, framework_type: str,
    ) -> tuple[int, float]:
        """Return (framework_index, distance) for the closest framework atom of
        the requested type to the given probe atom under PBC."""
        best_idx = -1
        best_d = float("inf")
        p = self.probe_cartesian_angstrom[probe_atom_idx]
        for fi, (ftype, fcart) in enumerate(
            zip(self.framework_types, self.framework_cartesian_angstrom)
        ):
            if ftype != framework_type:
                continue
            d = self.min_image_distance(p, fcart)
            if d < best_d:
                best_d = d
                best_idx = fi
        return best_idx, best_d


@dataclass
class NativeWidomResult:
    K_H_mol_per_kg_per_Pa: float
    Q_st_kJ_per_mol: float
    mean_boltzmann_factor: float
    mean_U_K: float
    n_insertions: int
    n_overlaps: int
    seed: int
    temperature_K: float
    duration_s: float
    framework_mass_kg: float
    backend_tag: str = "native_widom_v04"
    notes: list[str] = field(default_factory=list)
    strongest_insertion: StrongestInsertion | None = None


def run_native_widom(
    system: NativeSystem,
    temperature_K: float,
    n_insertions: int,
    seed: int,
    enable_ewald: bool = False,
    ewald_parameters: EwaldParameters | None = None,
    batch_size: int = 1024,
) -> NativeWidomResult:
    """Run Widom insertions and return aggregated K_H + Q_st.

    If `enable_ewald=True`, electrostatic energies are added via the
    Ewald cross-term to every insertion. If `enable_ewald=False` and the
    framework or probe carries any non-zero partial charges, raises a
    RuntimeError (we refuse to silently truncate electrostatics).
    """
    has_framework_charges = (
        system.framework_charges_e is not None
        and float(np.abs(system.framework_charges_e).sum()) > 0.0
    )
    has_probe_charges = (
        system.probe.charges_e is not None
        and float(np.abs(system.probe.charges_e).sum()) > 0.0
    )
    if (has_framework_charges or has_probe_charges) and not enable_ewald:
        raise RuntimeError(
            "native Widom: framework or probe carries partial charges but "
            "enable_ewald=False — refusing to run with truncated electrostatics. "
            "Set enable_ewald=True (and supply ewald_parameters)."
        )

    rng = np.random.default_rng(seed)
    types_super, carts_super = system.supercell_positions()
    supercell = system.supercell_cell()
    inv_supercell = np.linalg.inv(supercell)
    masses_per_super_atom = [system.type_to_mass_amu[t] for t in types_super]
    M_kg = sum(masses_per_super_atom) * 1.66053906660e-27
    cutoff_sq = system.energy_cutoff_angstrom ** 2

    # Ewald cache (built once per system) for Widom electrostatic cross terms.
    ewald_cache = None
    if enable_ewald and (has_framework_charges or has_probe_charges):
        if ewald_parameters is None:
            ewald_parameters = EwaldParameters(
                alpha_inv_angstrom=0.3,
                real_cutoff_angstrom=system.energy_cutoff_angstrom,
                k_max_inv_angstrom=1.4,
            )
        # Replicate framework charges to supercell shape (per-atom in types_super order).
        framework_q_super = np.concatenate(
            [system.framework_charges_e] * (len(types_super) // system.n_framework_atoms),
            axis=0,
        ) if system.framework_charges_e is not None else np.zeros(len(types_super))
        ewald_cache = build_framework_ewald_cache(
            framework_charges_e=framework_q_super,
            framework_positions_angstrom=carts_super,
            cell_matrix_angstrom=supercell,
            parameters=ewald_parameters,
        )

    probe_charges = (
        system.probe.charges_e
        if system.probe.charges_e is not None
        else np.zeros(system.probe.n_atoms())
    )

    acc = WidomAccumulator()
    n_overlaps = 0
    t0 = time.time()

    # Track the strongest (lowest-U) insertion across all batches for site-truth.
    strongest_U_K = float("inf")
    strongest_probe_carts: np.ndarray | None = None

    remaining = n_insertions
    while remaining > 0:
        n_batch = min(batch_size, remaining)
        # Random fractional centres in [0, 1)^3
        frac_centres = rng.random((n_batch, 3))
        centres = frac_centres @ supercell  # (n_batch, 3) cartesian
        energies_K = np.empty(n_batch, dtype=float)

        for i in range(n_batch):
            rotation = random_rotation_matrix(rng)
            probe_carts = insert_probe_at(system.probe, centres[i], rotation)
            U_total = 0.0
            min_pair_r = float("inf")
            for j, ptype in enumerate(system.probe.types):
                # Vector from each framework atom to this probe atom under PBC.
                d = probe_carts[j][np.newaxis, :] - carts_super
                # Minimum image (fractional wrap to (-0.5, 0.5])
                frac_d = d @ inv_supercell
                frac_d -= np.round(frac_d)
                d = frac_d @ supercell
                r2 = np.einsum("ij,ij->i", d, d)
                within = r2 < cutoff_sq
                if not np.any(within):
                    continue
                r = np.sqrt(r2[within])
                local_min = float(r.min()) if r.size else float("inf")
                if local_min < min_pair_r:
                    min_pair_r = local_min
                framework_types_within = [types_super[k] for k in np.where(within)[0]]
                # Group by framework type so the pair-table dispatch is one call per type
                # instead of one per atom — saves big constant factors for medium-sized cells.
                u_arr = np.zeros_like(r)
                for ftype in set(framework_types_within):
                    mask = np.fromiter(
                        (ft == ftype for ft in framework_types_within),
                        count=len(framework_types_within),
                        dtype=bool,
                    )
                    if not np.any(mask):
                        continue
                    u_arr[mask] = system.pair_table.pair_energy(ftype, ptype, r[mask])
                U_total += float(np.sum(u_arr))
                if not math.isfinite(U_total):
                    U_total = float("inf")
                    break
            # Electrostatic hard-core: reject insertions where ANY framework
            # atom is closer than 1.5 Å to ANY probe atom. The LJ hard wall
            # (σ/4 ≈ 0.8 Å) catches truly-overlapping insertions, but the
            # Ewald real-space term grows as ~q_a·q_b/r → -∞ at small r if
            # signs are opposite, well before the LJ wall kicks in. Reject
            # before the Boltzmann factor can overflow.
            ELECTROSTATIC_HARDCORE_A = 1.5
            if min_pair_r < ELECTROSTATIC_HARDCORE_A:
                U_total = float("inf")
            # Electrostatics via Ewald cross terms (only if charges + cache present)
            if ewald_cache is not None and math.isfinite(U_total):
                U_total += widom_ewald_delta_U(ewald_cache, probe_charges, probe_carts)
            energies_K[i] = U_total
            if not math.isfinite(U_total):
                n_overlaps += 1
            elif U_total < strongest_U_K:
                # Capture the lowest-U accepted insertion for site-truth extraction
                strongest_U_K = U_total
                strongest_probe_carts = probe_carts.copy()

        acc.update(energies_K, beta_inv_K=temperature_K)
        remaining -= n_batch

    duration = time.time() - t0
    # Supercell volume in m³ (V_supercell from cell vectors, converted from Å³)
    V_super_A3 = float(abs(np.dot(
        supercell[0], np.cross(supercell[1], supercell[2])
    )))
    V_super_m3 = V_super_A3 * 1e-30
    K_H = acc.K_H_mol_per_kg_per_Pa(
        T_K=temperature_K,
        M_framework_kg=M_kg,
        V_supercell_m3=V_super_m3,
    )
    Q_st = acc.Q_st_kJ_per_mol(T_K=temperature_K)
    strongest = None
    if strongest_probe_carts is not None and math.isfinite(strongest_U_K):
        strongest = StrongestInsertion(
            U_K=strongest_U_K,
            probe_types=list(system.probe.types),
            probe_cartesian_angstrom=strongest_probe_carts,
            framework_types=list(types_super),
            framework_cartesian_angstrom=carts_super,
            supercell_matrix_angstrom=supercell,
            temperature_K=temperature_K,
            seed=seed,
        )
    return NativeWidomResult(
        K_H_mol_per_kg_per_Pa=K_H,
        Q_st_kJ_per_mol=Q_st,
        mean_boltzmann_factor=acc.mean_boltzmann_factor(),
        mean_U_K=acc.mean_boltzmann_U_K(),
        n_insertions=n_insertions,
        n_overlaps=n_overlaps,
        seed=seed,
        temperature_K=temperature_K,
        duration_s=duration,
        framework_mass_kg=M_kg,
        strongest_insertion=strongest,
    )
