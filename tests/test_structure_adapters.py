"""Tests for widom_atlas.io.structure_adapters (T016)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from widom_atlas.io.structure_adapters import (
    ase_to_pymatgen,
    get_cell_matrix,
    pymatgen_to_ase,
)


def _ase_water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 0.95, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )


def test_ase_to_pymatgen_round_trip_positions() -> None:
    a = _ase_water()
    s = ase_to_pymatgen(a)
    a2 = pymatgen_to_ase(s)
    np.testing.assert_allclose(a.positions, a2.positions, atol=1e-10)


def test_ase_to_pymatgen_round_trip_symbols() -> None:
    a = _ase_water()
    s = ase_to_pymatgen(a)
    a2 = pymatgen_to_ase(s)
    assert a.get_chemical_symbols() == a2.get_chemical_symbols()


def test_ase_to_pymatgen_round_trip_cell() -> None:
    a = _ase_water()
    s = ase_to_pymatgen(a)
    np.testing.assert_allclose(get_cell_matrix(a), get_cell_matrix(s), atol=1e-10)


def test_adapters_reject_non_periodic_atoms() -> None:
    a = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=False)
    with pytest.raises(ValueError):
        ase_to_pymatgen(a)


def test_adapters_reject_singular_cell() -> None:
    cell = np.eye(3) * 10.0
    cell[2, 2] = 0.0
    a = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    with pytest.raises(ValueError):
        ase_to_pymatgen(a)


def test_get_cell_matrix_agrees_for_ase_and_pymatgen() -> None:
    a = _ase_water()
    s = ase_to_pymatgen(a)
    np.testing.assert_allclose(get_cell_matrix(a), get_cell_matrix(s), atol=1e-12)


def test_round_trip_preserves_fractional_coords_to_1e10() -> None:
    a = _ase_water()
    f1 = a.get_scaled_positions()
    s = ase_to_pymatgen(a)
    a2 = pymatgen_to_ase(s)
    f2 = a2.get_scaled_positions()
    np.testing.assert_allclose(f1, f2, atol=1e-10)


def test_adapters_reject_2d_periodic_structures() -> None:
    a = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=[True, True, False],
    )
    with pytest.raises(NotImplementedError):
        ase_to_pymatgen(a)
