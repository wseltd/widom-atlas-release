"""Regression tests for analytical tail corrections."""
from __future__ import annotations

from widom_atlas.v04.native.tail_correction import (
    buckingham_a_exp_c6_tail_correction_K,
    dzubak_a_exp_c5_d6_tail_correction_K,
    lj_12_6_tail_correction_K,
    ongari_a_exp_c6_c8_tail_correction_K,
)


def test_lj_tail_at_cutoff_equals_sigma_is_zero_for_specific_ratio():
    """Sanity: closed-form formula structure.

    U_tail = (16 pi / 3) n_f eps sig^3 [(1/3)(σ/rc)^9 - (σ/rc)^3]
    Zero when (1/3)(σ/rc)^9 = (σ/rc)^3 => σ/rc = 3^(1/6) ≈ 1.20094.
    """
    sigma_over_rc = 3 ** (1.0 / 6.0)  # exact
    sigma = sigma_over_rc
    rc = 1.0
    u = lj_12_6_tail_correction_K(
        epsilon_K=10.0, sigma_angstrom=sigma,
        cutoff_angstrom=rc, n_framework_per_angstrom3=0.1,
    )
    assert abs(u) < 1e-9, f"Expected tail ≈ 0 at exact σ/rc = 3^(1/6), got {u}"


def test_lj_tail_is_attractive_for_normal_conditions():
    """For r_c > sigma * 3^(1/6) ≈ 1.2 sigma, tail should be negative (attractive)."""
    u = lj_12_6_tail_correction_K(
        epsilon_K=100.0, sigma_angstrom=3.0,
        cutoff_angstrom=12.8, n_framework_per_angstrom3=0.05,
    )
    assert u < 0


def test_buckingham_tail_scales_as_inverse_cube_of_cutoff():
    """U_tail ∝ 1/r_c^3 for Buckingham."""
    u_12 = buckingham_a_exp_c6_tail_correction_K(C_K_angstrom6=1e5, cutoff_angstrom=12.0, n_framework_per_angstrom3=0.05)
    u_24 = buckingham_a_exp_c6_tail_correction_K(C_K_angstrom6=1e5, cutoff_angstrom=24.0, n_framework_per_angstrom3=0.05)
    # Halving r_c should give 8x larger |tail|
    assert abs(u_12 / u_24 - 8.0) < 1e-9


def test_dzubak_tail_has_both_r5_and_r6_contributions():
    """U_tail = -4pi n_f [C/(2 r_c^2) + D/(3 r_c^3)]: the two terms must combine additively."""
    u_C_only = dzubak_a_exp_c5_d6_tail_correction_K(C_K_angstrom5=1e4, D_K_angstrom6=0.0, cutoff_angstrom=12.0, n_framework_per_angstrom3=0.05)
    u_D_only = dzubak_a_exp_c5_d6_tail_correction_K(C_K_angstrom5=0.0, D_K_angstrom6=1e5, cutoff_angstrom=12.0, n_framework_per_angstrom3=0.05)
    u_both = dzubak_a_exp_c5_d6_tail_correction_K(C_K_angstrom5=1e4, D_K_angstrom6=1e5, cutoff_angstrom=12.0, n_framework_per_angstrom3=0.05)
    assert abs((u_C_only + u_D_only) - u_both) < 1e-9


def test_dzubak_r5_term_scales_as_inverse_square_cutoff():
    u_12 = dzubak_a_exp_c5_d6_tail_correction_K(C_K_angstrom5=1e4, D_K_angstrom6=0.0, cutoff_angstrom=12.0, n_framework_per_angstrom3=0.05)
    u_24 = dzubak_a_exp_c5_d6_tail_correction_K(C_K_angstrom5=1e4, D_K_angstrom6=0.0, cutoff_angstrom=24.0, n_framework_per_angstrom3=0.05)
    assert abs(u_12 / u_24 - 4.0) < 1e-9  # halving r_c → 4x


def test_ongari_tail_includes_r6_and_r8_terms():
    u_C6 = ongari_a_exp_c6_c8_tail_correction_K(C6_K_angstrom6=3.196e4, C8_K_angstrom8=0.0, cutoff_angstrom=13.0, n_framework_per_angstrom3=0.01)
    u_C8 = ongari_a_exp_c6_c8_tail_correction_K(C6_K_angstrom6=0.0, C8_K_angstrom8=5e6, cutoff_angstrom=13.0, n_framework_per_angstrom3=0.01)
    u_both = ongari_a_exp_c6_c8_tail_correction_K(C6_K_angstrom6=3.196e4, C8_K_angstrom8=5e6, cutoff_angstrom=13.0, n_framework_per_angstrom3=0.01)
    assert abs((u_C6 + u_C8) - u_both) < 1e-9


def test_lj_tail_proportional_to_density():
    """U_tail scales linearly with n_framework."""
    u_low = lj_12_6_tail_correction_K(epsilon_K=50, sigma_angstrom=3.0, cutoff_angstrom=14.0, n_framework_per_angstrom3=0.01)
    u_high = lj_12_6_tail_correction_K(epsilon_K=50, sigma_angstrom=3.0, cutoff_angstrom=14.0, n_framework_per_angstrom3=0.10)
    assert abs(u_high / u_low - 10.0) < 1e-9


def test_typical_lj_tail_value_within_expected_magnitude():
    """For a zeolite-like system (rho ~ 0.05 atoms/A^3, eps=50K, sigma=3.0A, rc=14A),
    the LJ tail should be a few K — small but not negligible relative to typical
    Widom insertion energies of a few hundred K."""
    u = lj_12_6_tail_correction_K(epsilon_K=50, sigma_angstrom=3.0, cutoff_angstrom=14.0, n_framework_per_angstrom3=0.05)
    assert -50.0 < u < -1.0
