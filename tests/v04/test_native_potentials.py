"""Unit tests for the native Widom evaluator's typed potentials + log-sum-exp."""
from __future__ import annotations

import math

import numpy as np
import pytest

from widom_atlas.v04.native.potentials import (
    BuckinghamAExpC6,
    DzubakAExpC5D6,
    LennardJones12_6,
    PairTable,
)
from widom_atlas.v04.native.widom import (
    K_B_J_PER_K,
    N_AVOGADRO,
    WidomAccumulator,
    framework_mass_kg,
)


def test_lj_well_depth_and_zero():
    """LJ minimum is at r = 2^(1/6)σ with V = -ε."""
    p = LennardJones12_6(epsilon_K=100.0, sigma_angstrom=3.0)
    r_min = 2.0 ** (1.0 / 6.0) * 3.0
    v = p.energy(np.array([r_min, 3.0]))
    assert math.isclose(v[0], -100.0, abs_tol=1e-9)
    # V = 0 at r = σ
    assert math.isclose(v[1], 0.0, abs_tol=1e-9)


def test_lj_truncation_beyond_cutoff_returns_zero():
    p = LennardJones12_6(epsilon_K=100.0, sigma_angstrom=3.0, cutoff_A=12.8)
    v = p.energy(np.array([13.0, 12.7]))
    assert v[0] == 0.0
    assert v[1] != 0.0


def test_lj_hard_wall_below_quarter_sigma():
    """Insertions r < σ/4 must give +∞ so the Boltzmann factor is zero."""
    p = LennardJones12_6(epsilon_K=100.0, sigma_angstrom=3.0)
    v = p.energy(np.array([0.05]))
    assert math.isinf(v[0]) and v[0] > 0


def test_buckingham_known_value_lin_mercado_mg_Oco2():
    """Cross-check the Mg-O(CO2) Buckingham well from Lin 2014 SI Table S7.

    Parameters: A=2.4732e7 K, B=3.965 Å^-1, C=4.08795e5 K·Å^6.
    At r=3 Å: V = 2.4732e7 × exp(-11.895) - 4.08795e5/729
            ≈ 168 - 561 = -393 K = -3.27 kJ/mol per pair.
    """
    p = BuckinghamAExpC6(
        A_K=2.4732e7, B_inv_angstrom=3.965, C_K_angstrom6=4.08795e5,
        hardcore_angstrom=1.0,
    )
    v = p.energy(np.array([3.0]))
    assert -450.0 < v[0] < -340.0, f"V(3.0)={v[0]} outside expected -393 K band"
    # And at r=4 Å:  3 - 100 = -97 K
    v4 = p.energy(np.array([4.0]))
    assert -120.0 < v4[0] < -80.0


def test_buckingham_hardcore_returns_inf():
    p = BuckinghamAExpC6(
        A_K=1e6, B_inv_angstrom=3.0, C_K_angstrom6=1e4, hardcore_angstrom=1.0,
    )
    v = p.energy(np.array([0.5, 1.5]))
    assert math.isinf(v[0]) and v[0] > 0
    assert math.isfinite(v[1])


def test_dzubak_form_includes_both_C_and_D_terms():
    """V_Dzubak(r) = A exp(-Br) - C/r^5 - D/r^6.

    With A=B=0 → only the dispersion terms remain. At r=2 Å with C=8, D=64:
    V = 0 - 8/32 - 64/64 = -0.25 - 1.0 = -1.25 K.
    """
    p = DzubakAExpC5D6(
        A_K=0.0, B_inv_angstrom=0.0, C_K_angstrom5=8.0, D_K_angstrom6=64.0,
        hardcore_angstrom=0.5,
    )
    v = p.energy(np.array([2.0]))
    assert math.isclose(v[0], -1.25, rel_tol=1e-9)


def test_dzubak_recovers_pure_lj_dispersion_when_C_is_zero():
    """Setting Dzubak's r^-5 coefficient to zero gives the same r-dependence
    as Buckingham (modulo the exponential)."""
    pd = DzubakAExpC5D6(
        A_K=1e6, B_inv_angstrom=3.0, C_K_angstrom5=0.0, D_K_angstrom6=4e5,
    )
    pb = BuckinghamAExpC6(
        A_K=1e6, B_inv_angstrom=3.0, C_K_angstrom6=4e5,
    )
    r = np.linspace(2.0, 6.0, 9)
    vd = pd.energy(r)
    vb = pb.energy(r)
    np.testing.assert_allclose(vd, vb, rtol=1e-12)


def test_pair_table_symmetric():
    t = PairTable()
    p = LennardJones12_6(50.0, 3.0)
    t.set("A", "B", p)
    assert t.get("A", "B") is p
    assert t.get("B", "A") is p


def test_widom_accumulator_handles_extreme_attractive_energies():
    """A few deeply-attractive insertions must not overflow the Z accumulator."""
    acc = WidomAccumulator()
    # T = 298 K, deeply attractive insertions
    T = 298.0
    energies_K = np.array([-12000.0, -10000.0, -5000.0, +500.0])
    acc.update(energies_K, beta_inv_K=T)
    # Total should be dominated by the -12000 K insertion: exp(12000/298) = exp(40.3) ≈ 3.2e17.
    # Boltzmann mean for 4 samples = 3.2e17 / 4 ≈ 8e16. (Order of magnitude check.)
    z = acc.mean_boltzmann_factor()
    assert z > 1e15, f"Z={z} unexpectedly small for -12000 K insertion"
    # <U>_B is dominated by the most negative U (largest weight).
    u_mean = acc.mean_boltzmann_U_K()
    assert -12500.0 < u_mean < -11500.0, f"<U>_B={u_mean} not dominated by -12000 K"


def test_widom_accumulator_handles_overlap_only():
    """If every insertion is an overlap (U=+inf), Z=0 and Q_st=0."""
    acc = WidomAccumulator()
    acc.update(np.array([np.inf, np.inf, np.inf]), beta_inv_K=298.0)
    assert acc.mean_boltzmann_factor() == 0.0


def test_widom_accumulator_streaming_matches_batched():
    """Repeated streaming updates with chunks of energies must give the same
    Z + <U>_B as a single bulk update."""
    rng = np.random.default_rng(42)
    energies = rng.normal(loc=200.0, scale=300.0, size=2000)
    energies = np.clip(energies, -10000.0, 10000.0)
    T = 298.0

    bulk = WidomAccumulator()
    bulk.update(energies, beta_inv_K=T)

    stream = WidomAccumulator()
    for chunk in np.array_split(energies, 17):
        stream.update(chunk, beta_inv_K=T)

    assert math.isclose(bulk.mean_boltzmann_factor(), stream.mean_boltzmann_factor(), rel_tol=1e-10)
    assert math.isclose(bulk.mean_boltzmann_U_K(), stream.mean_boltzmann_U_K(), rel_tol=1e-10)


def test_K_H_matches_analytic_formula():
    """For a deterministic small set of U values, K_H = <e^{-βU}>·V/(M·R·T).

    With three insertions at U = [0, -1000K, +inf] in a system with framework
    mass M = 1 g/mol and V_supercell = 1 nm³:
      <e^{-βU}> = (1 + e^{1000/T} + 0)/3
    K_H = <e^{-βU}> × V / (M × R × T)
    """
    acc = WidomAccumulator()
    acc.update(np.array([0.0, -1000.0, np.inf]), beta_inv_K=298.0)
    T = 298.0
    M_kg = 1.0 * 1.66053906660e-27  # 1 amu
    V_m3 = 1e-27  # 1 nm³
    K_H = acc.K_H_mol_per_kg_per_Pa(
        T_K=T, M_framework_kg=M_kg, V_supercell_m3=V_m3,
    )
    z = acc.mean_boltzmann_factor()
    z_expected = (1.0 + math.exp(1000.0 / T) + 0.0) / 3.0
    assert math.isclose(z, z_expected, rel_tol=1e-9)
    R = K_B_J_PER_K * N_AVOGADRO
    expected = z * V_m3 / (M_kg * R * T)
    assert math.isclose(K_H, expected, rel_tol=1e-12)


def test_framework_mass_kg_sum_matches_amu_table():
    types = ["Mg", "C", "O", "C", "H"]
    masses = {"Mg": 24.305, "C": 12.0, "O": 15.9994, "H": 1.0}
    expected_amu = 24.305 + 12.0 + 15.9994 + 12.0 + 1.0
    m_kg = framework_mass_kg(types, masses)
    assert math.isclose(m_kg, expected_amu * 1.66053906660e-27, rel_tol=1e-12)
