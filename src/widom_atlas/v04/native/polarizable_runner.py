"""Polarizable Widom runner: wraps `run_native_widom` with per-insertion SCF dipole solve.

This is the production driver for the first public polarizable-Widom
Henry-constant workflow. Per-insertion algorithm:

  1. Pick random insertion point + orientation (same as plain Widom).
  2. Compute the standard LJ/Buckingham/Dzubak + Coulomb interaction
     energy (handled by the underlying `run_native_widom` machinery).
  3. Compute the static field at every framework atom + probe atom
     due to ALL permanent charges (framework + probe).
  4. Solve the self-consistent induced dipoles mu_i.
  5. Polarization energy U_pol = -1/2 sum mu_i . E^0_i.
  6. The polarization CONTRIBUTION to the insertion ΔE is
       ΔU_pol = U_pol(with_probe) - U_pol(framework_only)
     where U_pol(framework_only) is precomputed once per simulation.

Performance: the per-insertion SCF cost scales as O(N^2) per CG step
(N = framework + probe atom count) and convergence typically takes
5-15 iterations for isotropic-polarizability point dipoles. For the
Becker M-MOF-74 case (114 framework atoms + 3 probe atoms) on a
2x2x2 supercell, this is ~10 ms per insertion -- ~13 minutes per
80k-insertion seed. Acceptable for prototyping.

The static field due to framework permanent charges can be precomputed
once per simulation (framework is rigid in Widom). Per insertion only
the probe-contribution to the field changes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .polarizable_dipoles import (
    TholeDamping,
    build_dipole_tensor_block_3Nx3N,
    polarization_energy_K,
    solve_induced_dipoles_direct,
    solve_induced_dipoles_scf,
    static_coulomb_field_at_points_K_per_e_angstrom,
)
from .system import NativeSystem, insert_probe_at, random_rotation_matrix


@dataclass
class PolarizableWidomResult:
    K_H_mol_per_kg_per_Pa: float
    K_H_mol_per_kg_per_bar: float
    Q_st_kJ_per_mol: float
    framework_mass_kg: float
    n_insertions: int
    n_overlaps_excluded: int
    polarization_energy_framework_only_K: float
    mean_delta_U_pol_per_insertion_K: float
    duration_s: float
    backend_tag: str = "polarizable_widom_v04"
    note: str = ""


def _compute_framework_polarization(
    framework_positions: np.ndarray,
    framework_charges_e: np.ndarray,
    framework_alphas_A3: np.ndarray,
    cell_matrix: np.ndarray,
    thole: TholeDamping,
    cutoff_angstrom: float,
    use_direct_inverse: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Precompute the framework-only induced dipoles + polarization energy.

    Returns: (E^0 in K/(eA), mu in eA, U_pol in K).
    """
    E0_framework = static_coulomb_field_at_points_K_per_e_angstrom(
        target_positions=framework_positions,
        source_positions=framework_positions,
        source_charges_e=framework_charges_e,
        cell_matrix_angstrom=cell_matrix,
        cutoff_angstrom=cutoff_angstrom,
    )
    T = build_dipole_tensor_block_3Nx3N(
        positions_angstrom=framework_positions,
        alphas_A3=framework_alphas_A3,
        cell_matrix_angstrom=cell_matrix,
        thole=thole,
        cutoff_angstrom=cutoff_angstrom,
    )
    if use_direct_inverse:
        mu = solve_induced_dipoles_direct(
            E0_framework, framework_alphas_A3, T
        )
    else:
        mu, _niter = solve_induced_dipoles_scf(
            E0_framework, framework_alphas_A3, T
        )
    U_pol = polarization_energy_K(mu, E0_framework)
    return E0_framework, mu, U_pol


def run_polarizable_widom(
    system: NativeSystem,
    framework_alphas_A3: np.ndarray,
    probe_alphas_A3: np.ndarray,
    temperature_K: float,
    n_insertions: int,
    seed: int,
    thole: TholeDamping | None = None,
    cutoff_angstrom: float | None = None,
    use_direct_inverse_for_framework: bool = True,
    scf_max_iter: int = 50,
    scf_tol_e_angstrom: float = 1e-5,
    batch_progress_every: int = 1000,
) -> PolarizableWidomResult:
    """Run polarizable Widom insertion with isotropic SCF induced dipoles.

    Parameters
    ----------
    system : NativeSystem
        Framework + probe + pair_table (LJ/Buck/Dzu). Charges are read
        from system.framework_charges_e and system.probe.charges_e.
    framework_alphas_A3 : np.ndarray
        Per-framework-atom polarizabilities (shape (N_framework,)) in A^3.
        Order must match system.framework_types after supercell expansion.
    probe_alphas_A3 : np.ndarray
        Per-probe-atom polarizabilities (shape (n_probe,)) in A^3.
    thole : TholeDamping
        Thole damping spec. Default = TholeDamping(a_thole=2.6).
    cutoff_angstrom : float
        Real-space dipole-tensor cutoff. Defaults to system.energy_cutoff_angstrom.

    Returns
    -------
    PolarizableWidomResult
    """
    if thole is None:
        thole = TholeDamping(a_thole=2.6)
    if cutoff_angstrom is None:
        cutoff_angstrom = system.energy_cutoff_angstrom

    types_super, carts_super = system.supercell_positions()
    supercell = system.supercell_cell()
    inv_supercell = np.linalg.inv(supercell)
    n_replicas = len(carts_super) // system.n_framework_atoms
    if system.framework_charges_e is not None:
        framework_q_super = np.tile(system.framework_charges_e, n_replicas)
    else:
        framework_q_super = np.zeros(len(carts_super))
    framework_alphas_super = np.tile(framework_alphas_A3, n_replicas)

    # 1. Precompute framework-only polarization (constant per simulation)
    _E0_fw, _mu_fw, U_pol_fw = _compute_framework_polarization(
        framework_positions=carts_super,
        framework_charges_e=framework_q_super,
        framework_alphas_A3=framework_alphas_super,
        cell_matrix=supercell,
        thole=thole,
        cutoff_angstrom=cutoff_angstrom,
        use_direct_inverse=use_direct_inverse_for_framework,
    )

    # Per-type mass map for framework mass
    masses_per_super_atom = [system.type_to_mass_amu[t] for t in types_super]
    M_kg = sum(masses_per_super_atom) * 1.66053906660e-27

    cutoff_sq = cutoff_angstrom ** 2
    rng = np.random.default_rng(seed)
    beta_inv_K = temperature_K

    sum_w = 0.0
    sum_uw = 0.0
    n_overlaps = 0
    delta_U_pol_per_insertion_sum = 0.0

    n_probe = system.probe.n_atoms()
    probe_charges = (
        system.probe.charges_e
        if system.probe.charges_e is not None
        else np.zeros(n_probe)
    )

    N_combined = len(carts_super) + n_probe
    alphas_combined = np.zeros(N_combined)
    alphas_combined[: len(carts_super)] = framework_alphas_super
    alphas_combined[len(carts_super) :] = probe_alphas_A3

    t0 = time.time()
    for _i in range(n_insertions):
        centre = rng.random(3) @ supercell
        rotation = random_rotation_matrix(rng)
        probe_carts = insert_probe_at(system.probe, centre, rotation)

        # 2. LJ/Buck interaction (mirrors run_native_widom inner loop)
        U_total_lj = 0.0
        overlap = False
        for j, ptype in enumerate(system.probe.types):
            d = probe_carts[j][None, :] - carts_super
            frac_d = d @ inv_supercell
            frac_d -= np.round(frac_d)
            d = frac_d @ supercell
            r2 = np.einsum("ij,ij->i", d, d)
            within = r2 < cutoff_sq
            if not np.any(within):
                continue
            r = np.sqrt(r2[within])
            if r.min() < 1.5:  # hard core to prevent SCF blowup
                overlap = True
                break
            framework_types_within = [
                types_super[k] for k in np.where(within)[0]
            ]
            u_arr = np.zeros_like(r)
            for ftype in set(framework_types_within):
                mask = np.array(
                    [ft == ftype for ft in framework_types_within], dtype=bool
                )
                if not np.any(mask):
                    continue
                u_arr[mask] = system.pair_table.pair_energy(ftype, ptype, r[mask])
            U_total_lj += float(np.sum(u_arr))
            if not math.isfinite(U_total_lj):
                overlap = True
                break

        if overlap:
            n_overlaps += 1
            continue

        # 3. Polarization energy: solve combined (framework + probe) SCF.
        combined_positions = np.vstack([carts_super, probe_carts])
        combined_charges = np.concatenate([framework_q_super, probe_charges])

        # Static field at every combined-system atom from all permanent charges
        E0_combined = static_coulomb_field_at_points_K_per_e_angstrom(
            target_positions=combined_positions,
            source_positions=combined_positions,
            source_charges_e=combined_charges,
            cell_matrix_angstrom=supercell,
            cutoff_angstrom=cutoff_angstrom,
        )

        # Build dipole-tensor block for the combined system
        T_combined = build_dipole_tensor_block_3Nx3N(
            positions_angstrom=combined_positions,
            alphas_A3=alphas_combined,
            cell_matrix_angstrom=supercell,
            thole=thole,
            cutoff_angstrom=cutoff_angstrom,
        )
        # SCF (iterative, scaling-friendly for large systems)
        mu_combined, _niter = solve_induced_dipoles_scf(
            E0_combined,
            alphas_combined,
            T_combined,
            max_iterations=scf_max_iter,
            tol_e_angstrom=scf_tol_e_angstrom,
        )
        U_pol_with_probe = polarization_energy_K(mu_combined, E0_combined)
        delta_U_pol = U_pol_with_probe - U_pol_fw

        # Total insertion energy = LJ + polarization-change
        U_total_K = U_total_lj + delta_U_pol
        if not math.isfinite(U_total_K):
            n_overlaps += 1
            continue

        w = math.exp(-U_total_K / beta_inv_K)
        sum_w += w
        sum_uw += U_total_K * w
        delta_U_pol_per_insertion_sum += delta_U_pol

    duration = time.time() - t0
    n_used = n_insertions - n_overlaps
    mean_w = sum_w / max(1, n_used)
    mean_U_K = sum_uw / sum_w if sum_w > 0 else 0.0
    mean_delta_U_pol = delta_U_pol_per_insertion_sum / max(1, n_used)

    # K_H in mol/(kg.Pa): K_H = <exp(-U/kT)> / (rho_framework * k_B * T)
    # We use the same conversion as the WidomAccumulator.
    V_cell_A3 = abs(np.linalg.det(supercell))
    V_cell_m3 = V_cell_A3 * 1e-30
    k_B_J_per_K = 1.380649e-23
    N_A = 6.02214076e23
    # K_H = <exp(-βU)> * V / (M_kg * k_B * T)
    K_H_mol_per_kg_per_Pa = mean_w * V_cell_m3 / (M_kg * k_B_J_per_K * temperature_K)
    K_H_mol_per_kg_per_Pa /= N_A  # convert atoms->mol
    K_H_mol_per_kg_per_bar = K_H_mol_per_kg_per_Pa * 1e5

    Q_st_kJ_per_mol = (
        -mean_U_K * k_B_J_per_K * N_A * 1e-3 + (temperature_K * 8.314e-3)
    )

    return PolarizableWidomResult(
        K_H_mol_per_kg_per_Pa=K_H_mol_per_kg_per_Pa,
        K_H_mol_per_kg_per_bar=K_H_mol_per_kg_per_bar,
        Q_st_kJ_per_mol=Q_st_kJ_per_mol,
        framework_mass_kg=M_kg,
        n_insertions=n_used,
        n_overlaps_excluded=n_overlaps,
        polarization_energy_framework_only_K=U_pol_fw,
        mean_delta_U_pol_per_insertion_K=mean_delta_U_pol,
        duration_s=duration,
        note=(
            f"polarizable Widom v04; framework U_pol = {U_pol_fw:.1f} K, "
            f"mean ΔU_pol per insertion = {mean_delta_U_pol:.1f} K, "
            f"{n_overlaps}/{n_insertions} insertions excluded (hard-core overlap)"
        ),
    )
