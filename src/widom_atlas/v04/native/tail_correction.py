"""Analytical long-range tail corrections for the native Widom evaluator.

For a probe atom of type p inserted into a framework with N_f atoms of
type f in volume V at number density n_f = N_f / V, the mean-field tail
correction (assuming the pair correlation g(r) = 1 beyond the cutoff
r_c) for the probe-framework pair is

    U_tail_pf = 4 pi n_f integral_{r_c}^{inf} r^2 V_pf(r) dr

For multi-atom probes the contribution is summed over probe atoms;
across framework atom types it is summed independently per type.

Closed forms (in K · Å):
  LJ 12-6   V(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6]
       U_tail = (16 pi / 3) n_f epsilon sigma^3 [(1/3)(sigma/r_c)^9 - (sigma/r_c)^3]
  Buckingham V(r) = A exp(-B r) - C/r^6
       (exponential is negligible beyond ~10 A for B >~ 3 / A; we skip it)
       U_tail = -(4 pi / 3) n_f C / r_c^3
  Dzubak    V(r) = A exp(-B r) - C/r^5 - D/r^6
       U_tail = -4 pi n_f [C/(2 r_c^2) + D/(3 r_c^3)]
  Ongari    V(r) = A exp(-B r) - C6/r^6 - C8/r^8 (RASPA generic special case)
       U_tail = -4 pi n_f [C6/(3 r_c^3) + C8/(5 r_c^5)]

Units throughout: distances in Å, energies in K (k_B T at T = 1 K = 1 K).
n_f in atoms / Å^3.

The exponential term in Buckingham / Dzubak / Ongari decays many orders of
magnitude faster than the dispersion tails at r_c = 12-14 Å for typical
B in [3.0, 4.5] Å^-1; we drop it analytically (the integrand integral
is bounded by exp(-B r_c) / B^3 which is <~ 1e-19 for B=4 r_c=14).

Frenkel and Smit 2002 (Understanding Molecular Simulation, 2nd ed., 37-38)
is the canonical derivation for LJ; the other dispersion tails follow the
same prescription.
"""
from __future__ import annotations

import math

_FOUR_PI = 4.0 * math.pi


def lj_12_6_tail_correction_K(
    epsilon_K: float,
    sigma_angstrom: float,
    cutoff_angstrom: float,
    n_framework_per_angstrom3: float,
) -> float:
    """Closed-form analytical LJ 12-6 tail for one probe-framework pair type."""
    s_over_rc = sigma_angstrom / cutoff_angstrom
    s_over_rc_3 = s_over_rc ** 3
    s_over_rc_9 = s_over_rc_3 ** 3
    return (
        (16.0 * math.pi / 3.0)
        * n_framework_per_angstrom3
        * epsilon_K
        * (sigma_angstrom ** 3)
        * ((1.0 / 3.0) * s_over_rc_9 - s_over_rc_3)
    )


def buckingham_a_exp_c6_tail_correction_K(
    C_K_angstrom6: float,
    cutoff_angstrom: float,
    n_framework_per_angstrom3: float,
) -> float:
    """Buckingham A exp(-Br) - C/r^6 tail. Exponential dropped (<1e-19 at r_c=12 A)."""
    return -(_FOUR_PI / 3.0) * n_framework_per_angstrom3 * C_K_angstrom6 / (
        cutoff_angstrom ** 3
    )


def dzubak_a_exp_c5_d6_tail_correction_K(
    C_K_angstrom5: float,
    D_K_angstrom6: float,
    cutoff_angstrom: float,
    n_framework_per_angstrom3: float,
) -> float:
    """Dzubak A exp(-Br) - C/r^5 - D/r^6 tail. Exponential dropped."""
    r5_term = C_K_angstrom5 / (2.0 * cutoff_angstrom ** 2)
    r6_term = D_K_angstrom6 / (3.0 * cutoff_angstrom ** 3)
    return -_FOUR_PI * n_framework_per_angstrom3 * (r5_term + r6_term)


def ongari_a_exp_c6_c8_tail_correction_K(
    C6_K_angstrom6: float,
    C8_K_angstrom8: float,
    cutoff_angstrom: float,
    n_framework_per_angstrom3: float,
) -> float:
    """Ongari A exp(-Br) - C6/r^6 - C8/r^8 tail. Exponential dropped."""
    r6_term = C6_K_angstrom6 / (3.0 * cutoff_angstrom ** 3)
    r8_term = C8_K_angstrom8 / (5.0 * cutoff_angstrom ** 5)
    return -_FOUR_PI * n_framework_per_angstrom3 * (r6_term + r8_term)


def total_lj_tail_for_probe(
    probe_atom_types: list[str],
    framework_type_counts: dict[str, int],
    framework_self_lj: dict[str, tuple[float, float]],
    probe_self_lj: dict[str, tuple[float, float]],
    cell_volume_angstrom3: float,
    cutoff_angstrom: float,
) -> float:
    """Sum the LJ tail correction over every probe-framework atom-type pair.

    Cross-pair epsilon and sigma are derived via Lorentz-Berthelot mixing
    from the probe-self and framework-self LJ parameters.

    Returns total tail energy in K to add as a constant offset to each
    Widom-insertion energy.
    """
    total_K = 0.0
    for ptype in probe_atom_types:
        if ptype not in probe_self_lj:
            continue
        p_eps, p_sig = probe_self_lj[ptype]
        for ftype, n_atoms in framework_type_counts.items():
            if n_atoms == 0 or ftype not in framework_self_lj:
                continue
            f_eps, f_sig = framework_self_lj[ftype]
            if f_eps == 0.0 or p_eps == 0.0:
                continue
            eps_cross = math.sqrt(f_eps * p_eps)
            sig_cross = 0.5 * (f_sig + p_sig)
            n_density = n_atoms / cell_volume_angstrom3
            total_K += lj_12_6_tail_correction_K(
                epsilon_K=eps_cross,
                sigma_angstrom=sig_cross,
                cutoff_angstrom=cutoff_angstrom,
                n_framework_per_angstrom3=n_density,
            )
    return total_K


def total_buckingham_tail_for_probe(
    cross_C_table: dict[tuple[str, str], float],
    framework_type_counts: dict[str, int],
    cell_volume_angstrom3: float,
    cutoff_angstrom: float,
) -> float:
    """Sum Buckingham tail across explicit cross-pair (framework_type, probe_type) C entries."""
    total_K = 0.0
    for (ftype, _ptype), C_K_angstrom6 in cross_C_table.items():
        n_atoms = framework_type_counts.get(ftype, 0)
        if n_atoms == 0:
            continue
        n_density = n_atoms / cell_volume_angstrom3
        total_K += buckingham_a_exp_c6_tail_correction_K(
            C_K_angstrom6=C_K_angstrom6,
            cutoff_angstrom=cutoff_angstrom,
            n_framework_per_angstrom3=n_density,
        )
    return total_K


def total_dzubak_tail_for_probe(
    cross_C5_table: dict[tuple[str, str], float],
    cross_D6_table: dict[tuple[str, str], float],
    framework_type_counts: dict[str, int],
    cell_volume_angstrom3: float,
    cutoff_angstrom: float,
) -> float:
    """Sum Dzubak tail across explicit cross-pair entries."""
    total_K = 0.0
    all_keys = set(cross_C5_table) | set(cross_D6_table)
    for key in all_keys:
        ftype, _ptype = key
        n_atoms = framework_type_counts.get(ftype, 0)
        if n_atoms == 0:
            continue
        n_density = n_atoms / cell_volume_angstrom3
        C5 = cross_C5_table.get(key, 0.0)
        D6 = cross_D6_table.get(key, 0.0)
        total_K += dzubak_a_exp_c5_d6_tail_correction_K(
            C_K_angstrom5=C5,
            D_K_angstrom6=D6,
            cutoff_angstrom=cutoff_angstrom,
            n_framework_per_angstrom3=n_density,
        )
    return total_K


def total_ongari_tail_for_probe(
    cross_C6_table: dict[tuple[str, str], float],
    cross_C8_table: dict[tuple[str, str], float],
    framework_type_counts: dict[str, int],
    cell_volume_angstrom3: float,
    cutoff_angstrom: float,
) -> float:
    """Sum Ongari (RASPA generic) tail across explicit cross-pair entries."""
    total_K = 0.0
    all_keys = set(cross_C6_table) | set(cross_C8_table)
    for key in all_keys:
        ftype, _ptype = key
        n_atoms = framework_type_counts.get(ftype, 0)
        if n_atoms == 0:
            continue
        n_density = n_atoms / cell_volume_angstrom3
        C6 = cross_C6_table.get(key, 0.0)
        C8 = cross_C8_table.get(key, 0.0)
        total_K += ongari_a_exp_c6_c8_tail_correction_K(
            C6_K_angstrom6=C6,
            C8_K_angstrom8=C8,
            cutoff_angstrom=cutoff_angstrom,
            n_framework_per_angstrom3=n_density,
        )
    return total_K
