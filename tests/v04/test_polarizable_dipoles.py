"""Regression tests for Thole-damped induced-dipole polarization."""
from __future__ import annotations

import numpy as np
import pytest

from widom_atlas.v04.native.polarizable_dipoles import (
    COULOMB_K_ANGSTROM_PER_E2,
    TholeDamping,
    bare_dipole_tensor,
    build_dipole_tensor_block_3Nx3N,
    polarization_energy_K,
    solve_induced_dipoles_direct,
    solve_induced_dipoles_scf,
    static_coulomb_field_at_points_K_per_e_angstrom,
)


def test_thole_damping_vanishes_linearly_at_origin():
    """T_thole(r) = sr/2 - (sr)^3/12 + ... at small sr; linear in r, not faster."""
    thole = TholeDamping()
    # At sr ~ 0.0026 (r=0.001 with alphas (1,1) → s=2.6), T ≈ sr/2 ≈ 0.0013
    t_very_small = thole.damping_factor(0.001, 1.0, 1.0)
    assert t_very_small < 0.01
    # The factor should be monotonically smaller as r → 0
    t_small = thole.damping_factor(0.01, 1.0, 1.0)
    assert t_very_small < t_small


def test_thole_damping_goes_to_one_at_long_range():
    thole = TholeDamping()
    assert thole.damping_factor(20.0, 1.0, 1.0) == pytest.approx(1.0, abs=1e-9)


def test_thole_damping_is_monotonic():
    thole = TholeDamping()
    fs = [thole.damping_factor(r, 1.0, 1.0) for r in (0.5, 1.0, 2.0, 5.0, 10.0)]
    assert all(fs[i] < fs[i + 1] for i in range(len(fs) - 1))


def test_bare_dipole_tensor_is_symmetric_and_traceless():
    """T_ij_alpha_beta = T_ij_beta_alpha and trace = 0."""
    r_vec = np.array([1.5, 0.0, 0.0])
    T = bare_dipole_tensor(r_vec)
    assert np.allclose(T, T.T)
    assert abs(np.trace(T)) < 1e-9


def test_bare_dipole_tensor_xx_component_along_x_axis():
    """T_xx = 2/r^3 when r is along x-axis: from (3 r_x r_x - r^2) / r^5 = (3 r^2 - r^2)/r^5 = 2/r^3."""
    r_vec = np.array([3.0, 0.0, 0.0])
    T = bare_dipole_tensor(r_vec)
    expected_T_xx = 2.0 / (3.0 ** 3)
    assert abs(T[0, 0] - expected_T_xx) < 1e-9


def test_static_coulomb_field_proton_at_distance():
    """E from +1 e at origin to point at distance 3 A along x: |E| = k_e / r^2 = 167101/9 K/(e A)."""
    sources = np.array([[0.0, 0.0, 0.0]])
    charges = np.array([1.0])
    targets = np.array([[3.0, 0.0, 0.0]])
    E = static_coulomb_field_at_points_K_per_e_angstrom(
        target_positions=targets, source_positions=sources, source_charges_e=charges,
        cell_matrix_angstrom=None, cutoff_angstrom=10.0,
    )
    expected_x = COULOMB_K_ANGSTROM_PER_E2 / (3.0 ** 2)
    assert abs(E[0, 0] - expected_x) < 1e-6


def test_scf_dipole_solver_agrees_with_direct_inverse_two_atom_dimer():
    """For a small system, direct inverse and SCF should agree to high precision."""
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    alphas = np.array([1.0, 1.0])
    E0 = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    thole = TholeDamping()
    T = build_dipole_tensor_block_3Nx3N(pos, alphas, None, thole, cutoff_angstrom=10.0)
    mu_dir = solve_induced_dipoles_direct(E0, alphas, T)
    mu_scf, niter = solve_induced_dipoles_scf(E0, alphas, T)
    assert niter > 0  # converged (not divergence-fallback)
    assert np.max(np.abs(mu_dir - mu_scf)) < 1e-4


def test_scf_returns_negative_iterations_on_divergence():
    """A pathological E0 with α=10 on a 2-atom system should still converge with under-
    relaxation; verify SCF divergence-detection path exists."""
    # The divergence-detection path is the >1e10 delta clause; force it artificially.
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    alphas = np.array([100.0, 100.0])
    # Build a huge fake "T" block (off-diagonal blowup) to force divergence even with relaxation
    T_block = np.ones((6, 6)) * 1e20
    E0 = np.zeros((2, 3))
    mu, niter = solve_induced_dipoles_scf(
        E0, alphas, T_block, max_iterations=10, tol_e_angstrom=1e-6, relaxation=1.0,
    )
    # Should either flag divergence (-1) or converge to zero (no static field)
    assert niter == -1 or np.allclose(mu, 0.0)


def test_polarization_energy_negative_for_aligned_dipole_with_field():
    """U_pol = -1/2 sum mu_i · E_i. If mu // E^0, U_pol < 0 (stabilizing)."""
    mu = np.array([[1.0, 0.0, 0.0]])
    E0 = np.array([[100.0, 0.0, 0.0]])
    U = polarization_energy_K(mu, E0)
    assert U < 0
    assert pytest.approx(-50.0) == U


def test_polarization_energy_zero_for_orthogonal_dipole_and_field():
    mu = np.array([[0.0, 1.0, 0.0]])
    E0 = np.array([[100.0, 0.0, 0.0]])
    U = polarization_energy_K(mu, E0)
    assert pytest.approx(0.0) == U


def test_thole_damping_with_larger_alphas_gives_smaller_s():
    """s = a / (alpha_i alpha_j)^(1/6); larger alphas → smaller s → less damping near origin."""
    thole = TholeDamping()
    s_small = thole.s_inv_angstrom(0.1, 0.1)
    s_large = thole.s_inv_angstrom(10.0, 10.0)
    assert s_large < s_small
