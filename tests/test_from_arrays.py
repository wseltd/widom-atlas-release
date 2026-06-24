"""Tests for widom_atlas.io.from_arrays (T014)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from widom_atlas.io.from_arrays import from_arrays


def _atoms(cell_diag: float = 10.0) -> Atoms:
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * cell_diag, pbc=True)


def _frac_cart(n: int = 50, seed: int = 0, cell_diag: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frac = rng.random((n, 3))
    cart = frac * cell_diag
    return frac, cart


def test_from_arrays_wraps_fractional_coords() -> None:
    atoms = _atoms()
    rng = np.random.default_rng(0)
    frac = rng.random((10, 3)) + 1.7  # outside [0,1)
    e = rng.normal(-0.1, 0.05, 10)
    ai = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CO2")
    arr = np.asarray(ai.positions_frac)
    assert ((arr >= 0.0) & (arr < 1.0)).all()


def test_from_arrays_computes_cart_from_frac() -> None:
    atoms = _atoms(cell_diag=10.0)
    frac, _ = _frac_cart()
    e = np.zeros(50)
    ai = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CO2")
    cart = np.asarray(ai.positions_cart_A)
    np.testing.assert_allclose(cart, frac * 10.0, atol=1e-12)


def test_from_arrays_computes_frac_from_cart() -> None:
    atoms = _atoms(cell_diag=10.0)
    _, cart = _frac_cart()
    e = np.zeros(50)
    ai = from_arrays(structure=atoms, positions_cart=cart, energies_eV=e, temperature_K=298.15, gas="CO2")
    frac = np.asarray(ai.positions_frac)
    np.testing.assert_allclose(frac, cart / 10.0, atol=1e-12)


def test_from_arrays_rejects_inconsistent_cart_and_frac() -> None:
    atoms = _atoms(cell_diag=10.0)
    frac, cart = _frac_cart()
    cart_wrong = cart + 1.0  # 1 A offset, > 1e-6
    e = np.zeros(50)
    with pytest.raises(ValueError, match="inconsistent"):
        from_arrays(
            structure=atoms,
            positions_cart=cart_wrong,
            positions_frac=frac,
            energies_eV=e,
            temperature_K=298.15,
            gas="CO2",
        )


def test_from_arrays_rejects_lists_requires_ndarray() -> None:
    atoms = _atoms()
    with pytest.raises(TypeError):
        from_arrays(
            structure=atoms,
            positions_frac=[[0.1, 0.1, 0.1]],
            energies_eV=np.array([0.0]),
            temperature_K=298.15,
            gas="CO2",
        )
    with pytest.raises(TypeError):
        from_arrays(
            structure=atoms,
            positions_frac=np.zeros((1, 3)),
            energies_eV=[0.0],  # type: ignore[arg-type]
            temperature_K=298.15,
            gas="CO2",
        )


def test_from_arrays_rejects_mismatched_shapes() -> None:
    atoms = _atoms()
    with pytest.raises(ValueError):
        from_arrays(
            structure=atoms,
            positions_frac=np.zeros((10, 3)),
            energies_eV=np.zeros(11),
            temperature_K=298.15,
            gas="CO2",
        )


def test_from_arrays_input_hash_is_deterministic() -> None:
    atoms = _atoms()
    frac, _ = _frac_cart()
    e = np.linspace(-0.3, -0.1, 50)
    a1 = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CO2")
    a2 = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CO2")
    assert a1.input_hash == a2.input_hash


def test_from_arrays_input_hash_changes_with_energies() -> None:
    atoms = _atoms()
    frac, _ = _frac_cart()
    e1 = np.linspace(-0.3, -0.1, 50)
    e2 = e1 + 0.01
    a1 = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e1, temperature_K=298.15, gas="CO2")
    a2 = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e2, temperature_K=298.15, gas="CO2")
    assert a1.input_hash != a2.input_hash


def test_from_arrays_accepts_pymatgen_structure() -> None:
    from pymatgen.core import Lattice, Structure

    lat = Lattice.cubic(10.0)
    s = Structure(lat, ["H"], [[0.0, 0.0, 0.0]])
    rng = np.random.default_rng(2)
    frac = rng.random((20, 3))
    e = rng.normal(-0.2, 0.05, 20)
    ai = from_arrays(structure=s, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="N2")
    assert ai.gas == "N2"
    assert ai.samples.energies_eV.shape == (20,)


def test_from_arrays_default_accessible_all_true() -> None:
    atoms = _atoms()
    frac, _ = _frac_cart(n=15)
    e = np.zeros(15)
    ai = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CH4")
    assert all(ai.accessible)
    assert len(ai.accessible) == 15


def test_from_arrays_rejects_unsupported_gas() -> None:
    atoms = _atoms()
    frac, _ = _frac_cart(n=5)
    e = np.zeros(5)
    with pytest.raises(ValueError):
        from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="H2O")
    with pytest.raises(ValueError):
        from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="Ar")
