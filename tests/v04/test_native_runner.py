"""Smoke test for the native Widom runner on a trivial periodic LJ system.

This is *not* the V1-V4 validation against RASPA3 — those require real
fixtures (MFI+Ar, MFI+CH4, etc.). It only verifies the runner's API +
internal consistency:

  * K_H returned is positive and finite.
  * Q_st is a sensible number.
  * Multi-seed determinism: same seed → same answer.
  * Different seeds → different answer.
  * Insertions count matches input.
  * Refuses to run with truncated electrostatics on a charged system.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from widom_atlas.v04.native.potentials import LennardJones12_6, PairTable
from widom_atlas.v04.native.runner import run_native_widom
from widom_atlas.v04.native.system import NativeSystem, ProbeMolecule


def _cubic_test_system(box_A: float = 15.0) -> NativeSystem:
    """3x3x3 = 27 Si-like LJ atoms on a cubic lattice, neutral, with an Ar probe."""
    coords = []
    n = 3
    spacing = box_A / n
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                coords.append([(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing])
    framework = np.array(coords)
    cell = np.eye(3) * box_A
    pt = PairTable()
    pt.set(
        "Si_test", "Ar_test",
        LennardJones12_6(epsilon_K=120.0, sigma_angstrom=3.5, cutoff_A=12.0),
    )
    probe = ProbeMolecule(
        name="Ar",
        types=["Ar_test"],
        body_positions=np.array([[0.0, 0.0, 0.0]]),
    )
    return NativeSystem(
        framework_types=["Si_test"] * len(framework),
        framework_cart_angstrom=framework,
        framework_charges_e=None,
        cell_matrix_angstrom=cell,
        pair_table=pt,
        probe=probe,
        type_to_mass_amu={"Si_test": 28.0855, "Ar_test": 39.948},
        energy_cutoff_angstrom=12.0,
    )


def test_runner_returns_positive_finite_K_H():
    sys_ = _cubic_test_system()
    r = run_native_widom(sys_, temperature_K=298.0, n_insertions=2000, seed=42)
    assert r.n_insertions == 2000
    assert math.isfinite(r.K_H_mol_per_kg_per_Pa)
    assert r.K_H_mol_per_kg_per_Pa > 0.0
    assert math.isfinite(r.Q_st_kJ_per_mol)
    assert r.duration_s > 0.0


def test_runner_seed_deterministic():
    sys_ = _cubic_test_system()
    r1 = run_native_widom(sys_, temperature_K=298.0, n_insertions=500, seed=11)
    r2 = run_native_widom(sys_, temperature_K=298.0, n_insertions=500, seed=11)
    assert math.isclose(r1.K_H_mol_per_kg_per_Pa, r2.K_H_mol_per_kg_per_Pa, rel_tol=1e-12)
    assert math.isclose(r1.Q_st_kJ_per_mol, r2.Q_st_kJ_per_mol, rel_tol=1e-12)


def test_runner_different_seeds_diverge():
    sys_ = _cubic_test_system()
    r1 = run_native_widom(sys_, temperature_K=298.0, n_insertions=500, seed=11)
    r2 = run_native_widom(sys_, temperature_K=298.0, n_insertions=500, seed=12)
    assert r1.K_H_mol_per_kg_per_Pa != r2.K_H_mol_per_kg_per_Pa


def test_runner_refuses_charged_system_without_ewald():
    """A framework with non-zero partial charges + enable_ewald=False must fail."""
    sys_ = _cubic_test_system()
    sys_.framework_charges_e = np.full(sys_.n_framework_atoms, 0.1)
    with pytest.raises(RuntimeError, match="enable_ewald=False"):
        run_native_widom(sys_, temperature_K=298.0, n_insertions=100, seed=1)


def test_runner_refuses_charged_probe_without_ewald():
    sys_ = _cubic_test_system()
    sys_.probe = ProbeMolecule(
        name="ChargedAr",
        types=["Ar_test"],
        body_positions=np.array([[0.0, 0.0, 0.0]]),
        charges_e=np.array([1.0]),
    )
    with pytest.raises(RuntimeError, match="enable_ewald=False"):
        run_native_widom(sys_, temperature_K=298.0, n_insertions=100, seed=1)
