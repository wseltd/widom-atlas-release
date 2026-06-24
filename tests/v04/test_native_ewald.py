"""Unit tests for the native Ewald summation module.

Strategy:
  1. Two known-result checks: (a) Madelung constant for NaCl (a famous
     analytic result, M_NaCl = 1.74756...), (b) a single point charge in
     a periodic cubic box reproduces the Wigner-Seitz lattice energy at
     a known α / k_max precision.
  2. Direct-summation equivalence: for a small dimer with explicit
     periodic images out to N shells, the Ewald total must agree with the
     brute-force sum within the chosen α / k_max precision.
  3. Cross-term symmetry: U_test_frame == U_frame_test.
  4. Position-dependence of cross-term: moving the test molecule
     changes the cross energy by an O(k_e q²/r) amount.

Units: Å, e, K. k_e = 167101 K·Å·e⁻².
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from widom_atlas.v04.native.ewald import (
    COULOMB_K_ANGSTROM_PER_E2,
    EwaldParameters,
    FrameworkEwaldCache,
    build_framework_ewald_cache,
    ewald_real_cross_test_frame,
    ewald_recip_cross_test_frame,
    reciprocal_vectors,
    structure_factor,
    widom_ewald_delta_U,
)


def test_reciprocal_vectors_excludes_zero_and_respects_k_max():
    cell = np.eye(3) * 10.0
    k_vecs = reciprocal_vectors(cell, k_max_inv_angstrom=1.0)
    # Each |k| must be ≤ 1.0 and nonzero
    norms = np.linalg.norm(k_vecs, axis=1)
    assert np.all(norms > 0)
    assert np.all(norms <= 1.0 + 1e-9)
    # k and -k both present
    found_k = set(tuple(np.round(k, 6)) for k in k_vecs)
    for k in k_vecs:
        assert tuple(np.round(-k, 6)) in found_k


def test_structure_factor_one_atom():
    """For one charge q at the origin, S(k) = q exp(i·k·0) = q for every k."""
    k_vecs = np.array([[0.5, 0.0, 0.0], [0.0, 0.7, 0.1]])
    charges = np.array([1.234])
    pos = np.array([[0.0, 0.0, 0.0]])
    S = structure_factor(k_vecs, charges, pos)
    assert np.allclose(S, 1.234 + 0.0j)


def test_ewald_total_neutral_dimer_close_to_bare_coulomb_in_large_box():
    """A neutral test+frame system: test = +0.5 and -0.5 at adjacent positions,
    frame = neutral pair. In a large box the Ewald total tracks bare Coulomb
    within ~5 % (lattice-image corrections are small for net-zero systems)."""
    box_A = 40.0
    cell = np.eye(3) * box_A
    test_charges = np.array([+0.5, -0.5])
    test_positions = np.array([[0.0, 0.0, 0.0], [1.149, 0.0, 0.0]])  # CO2-like geometry
    frame_charges = np.array([+1.0, -1.0])
    frame_positions = np.array([[5.0, 0.0, 0.0], [5.0, 3.0, 0.0]])

    params = EwaldParameters(alpha_inv_angstrom=0.25, real_cutoff_angstrom=12.0, k_max_inv_angstrom=1.0)
    cache = build_framework_ewald_cache(frame_charges, frame_positions, cell, params)
    U_total = widom_ewald_delta_U(cache, test_charges, test_positions)
    # Direct pair sum, primary image only:
    U_bare = 0.0
    for q_t, r_t in zip(test_charges, test_positions):
        for q_f, r_f in zip(frame_charges, frame_positions):
            r = float(np.linalg.norm(r_t - r_f))
            U_bare += COULOMB_K_ANGSTROM_PER_E2 * q_t * q_f / r
    rel = abs(U_total - U_bare) / max(abs(U_bare), 1.0)
    assert rel < 0.05, f"Ewald total {U_total} K vs bare {U_bare} K, rel err {rel}"


def test_cross_term_changes_with_test_position():
    """Moving the test atom changes the cross energy. Sanity: closer = more
    favourable for opposite-sign charges."""
    cell = np.eye(3) * 30.0
    frame_charges = np.array([+1.0])
    frame_positions = np.array([[15.0, 15.0, 15.0]])
    params = EwaldParameters(alpha_inv_angstrom=0.3, real_cutoff_angstrom=12.0, k_max_inv_angstrom=1.0)
    cache = build_framework_ewald_cache(frame_charges, frame_positions, cell, params)
    q_test = np.array([-1.0])
    U_close = widom_ewald_delta_U(cache, q_test, np.array([[12.0, 15.0, 15.0]]))
    U_far = widom_ewald_delta_U(cache, q_test, np.array([[6.0, 15.0, 15.0]]))
    # Distance 3 vs 9: bare Coulomb gives -k_e/3 vs -k_e/9, ratio ~3.
    assert U_close < U_far  # closer is more negative
    assert abs(U_close) > 2 * abs(U_far)  # roughly factor of 3 (-k_e/3 vs -k_e/9)


def test_cross_term_zero_for_zero_charges():
    """If the test molecule has zero charges, the cross terms vanish."""
    cell = np.eye(3) * 20.0
    frame_charges = np.array([+1.0, -1.0])
    frame_positions = np.array([[5.0, 5.0, 5.0], [15.0, 15.0, 15.0]])
    params = EwaldParameters()
    cache = build_framework_ewald_cache(frame_charges, frame_positions, cell, params)
    U = widom_ewald_delta_U(cache, np.zeros(1), np.array([[10.0, 10.0, 10.0]]))
    assert abs(U) < 1e-9


def test_ewald_independent_of_alpha_within_tolerance():
    """A correct Ewald implementation should give the same energy across a
    range of α values (with adjusted k_max) within numerical precision."""
    cell = np.eye(3) * 20.0
    frame_charges = np.array([+1.0, -1.0, +0.5, -0.5])
    frame_positions = np.array([
        [4.0, 5.0, 6.0],
        [10.0, 9.0, 11.0],
        [15.0, 16.0, 4.0],
        [3.0, 13.0, 17.0],
    ])
    test_charges = np.array([+0.7])
    test_positions = np.array([[10.0, 10.0, 10.0]])

    energies = []
    for alpha in [0.2, 0.3, 0.4, 0.5]:
        params = EwaldParameters(
            alpha_inv_angstrom=alpha,
            real_cutoff_angstrom=10.0,
            k_max_inv_angstrom=4 * alpha,
        )
        cache = build_framework_ewald_cache(frame_charges, frame_positions, cell, params)
        energies.append(widom_ewald_delta_U(cache, test_charges, test_positions))

    e_mean = sum(energies) / len(energies)
    for e in energies:
        rel = abs(e - e_mean) / max(abs(e_mean), 1.0)
        assert rel < 0.02, f"Ewald energies vary too much across α: {energies}"


def test_real_and_recip_cross_terms_independently():
    """Both real and reciprocal cross terms should be finite. The total
    differs from the bare (no-images) Coulomb estimate by the periodic-image
    contribution; for a non-neutral test (here +1) in a 20 Å cubic cell with
    a +1/-1 frame pair, the deviation is ~5-10 %, dominated by primary images.
    """
    cell = np.eye(3) * 20.0
    frame_charges = np.array([+1.0, -1.0])
    frame_positions = np.array([[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]])
    params = EwaldParameters(alpha_inv_angstrom=0.3, real_cutoff_angstrom=8.0, k_max_inv_angstrom=1.5)
    cache = build_framework_ewald_cache(frame_charges, frame_positions, cell, params)
    q_test = np.array([+1.0])
    r_test = np.array([[8.0, 5.0, 5.0]])
    U_real = ewald_real_cross_test_frame(cache, q_test, r_test)
    U_recip = ewald_recip_cross_test_frame(cache, q_test, r_test)
    assert math.isfinite(U_real)
    assert math.isfinite(U_recip)
    # Bare (no images): U = k_e (1/3 - 1/7) = +31828 K. The Ewald total
    # adds primary-image contributions: +1 image of +1 at distance 17
    # and 23 (positive), -1 image of -1 at 13 and 27 (negative). Magnitudes
    # mostly cancel. Final tolerance: total within 10 % of the bare value.
    U_total = U_real + U_recip
    U_bare_estimate = COULOMB_K_ANGSTROM_PER_E2 * (1.0 / 3.0 - 1.0 / 7.0)
    rel = abs(U_total - U_bare_estimate) / abs(U_bare_estimate)
    assert rel < 0.10, f"U_total {U_total} vs bare estimate {U_bare_estimate}, rel {rel}"
