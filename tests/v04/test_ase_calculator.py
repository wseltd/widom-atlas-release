"""Regression tests for the ASE Calculator wrapper around the native evaluator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def test_ase_calculator_wraps_native_system_lj_only():
    """ASE Calculator should evaluate the native pair-table for a probe pose without crashing."""
    pytest.importorskip("ase")
    from ase.atoms import Atoms

    from widom_atlas.v04.native.ase_calculator import (
        make_native_ase_calculator,
    )
    from widom_atlas.v04.native.maia_2023_loader import load_3b_native_maia_2023

    sys = load_3b_native_maia_2023(REPO, variant="UA")
    calc = make_native_ase_calculator(sys, treat_all_atoms_as_test_particle=False)

    # Build an Atoms object: framework atoms + 3 probe CO2 atoms at the cell center
    framework_types, framework_carts = sys.supercell_positions()
    # Use chemical symbols for framework; ASE wants element names
    framework_symbols = ["Zr" if t.startswith("Maia_Zr") else "C" if t.startswith("Maia_C") else "O" if t.startswith("Maia_O") else "H" for t in framework_types]
    cell_a = sys.supercell_cell()
    centre = 0.5 * (cell_a[0] + cell_a[1] + cell_a[2])
    probe_pos = np.array([
        centre + np.array([0.0, 0.0, 1.16]),
        centre,
        centre + np.array([0.0, 0.0, -1.16]),
    ])
    all_carts = np.vstack([framework_carts, probe_pos])
    all_symbols = [*framework_symbols, "O", "C", "O"]
    atoms = Atoms(symbols=all_symbols, positions=all_carts, cell=cell_a, pbc=True)
    atoms.calc = calc

    energy_eV = atoms.get_potential_energy()
    assert np.isfinite(energy_eV)
    # Test pose is at cell center which clashes with UiO-66 framework atoms — the
    # LJ repulsive wall dominates. We just verify the calculator returns a finite,
    # positive (repulsive) value as expected for a clash.
    assert energy_eV > 0.0


def test_ase_calculator_reports_K_to_eV_conversion_constant():
    from widom_atlas.v04.native.ase_calculator import K_to_eV
    # k_B in eV/K = 8.617333262e-5
    assert abs(K_to_eV - 8.617333262e-5) < 1e-12
