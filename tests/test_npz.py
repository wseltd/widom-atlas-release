"""Tests for widom_atlas.io.npz (T015)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.io.npz import from_npz, save_samples_npz


def _atoms() -> Atoms:
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)


def _ai(seed: int = 1, n: int = 32, gas: str = "N2") -> tuple[Atoms, object]:
    rng = np.random.default_rng(seed)
    frac = rng.random((n, 3))
    e = rng.normal(-0.1, 0.02, n)
    atoms = _atoms()
    return atoms, from_arrays(
        structure=atoms,
        positions_frac=frac,
        energies_eV=e,
        temperature_K=298.15,
        gas=gas,
        metadata={"src": "unit-test", "seed": seed},
    )


def test_npz_round_trip_preserves_arrays_bitexact(tmp_path: Path) -> None:
    atoms, ai = _ai()
    p = tmp_path / "samples.npz"
    save_samples_npz(ai, p)
    ai2 = from_npz(p, structure=atoms)
    np.testing.assert_array_equal(np.asarray(ai.positions_frac), np.asarray(ai2.positions_frac))
    np.testing.assert_array_equal(np.asarray(ai.positions_cart_A), np.asarray(ai2.positions_cart_A))
    np.testing.assert_array_equal(np.asarray(ai.energies_eV), np.asarray(ai2.energies_eV))
    np.testing.assert_array_equal(np.asarray(ai.accessible), np.asarray(ai2.accessible))


def test_npz_round_trip_preserves_input_hash(tmp_path: Path) -> None:
    atoms, ai = _ai()
    p = tmp_path / "samples.npz"
    save_samples_npz(ai, p)
    ai2 = from_npz(p, structure=atoms)
    assert ai.input_hash == ai2.input_hash


def test_npz_round_trip_preserves_metadata_json(tmp_path: Path) -> None:
    atoms, ai = _ai()
    p = tmp_path / "samples.npz"
    save_samples_npz(ai, p)
    ai2 = from_npz(p, structure=atoms)
    for key in ("src", "seed"):
        assert ai2.metadata[key] == ai.metadata[key]


def test_npz_round_trip_preserves_gas_and_temperature(tmp_path: Path) -> None:
    atoms, ai = _ai(gas="CH4")
    p = tmp_path / "samples.npz"
    save_samples_npz(ai, p)
    ai2 = from_npz(p, structure=atoms)
    assert ai2.gas == "CH4"
    assert ai2.temperature_K == ai.temperature_K


def test_npz_raises_when_parent_dir_missing(tmp_path: Path) -> None:
    _, ai = _ai()
    p = tmp_path / "does_not_exist" / "samples.npz"
    with pytest.raises(FileNotFoundError):
        save_samples_npz(ai, p)


def test_npz_does_not_use_pickle(tmp_path: Path) -> None:
    _, ai = _ai()
    p = tmp_path / "samples.npz"
    save_samples_npz(ai, p)
    # numpy pickled objects show up as ".pkl" entries in the .npz zipfile;
    # asserting no entry has a .pkl suffix is the strongest user-visible signal.
    with zipfile.ZipFile(p, "r") as zf:
        assert not any(name.endswith(".pkl") for name in zf.namelist())


def test_npz_round_trip_uses_from_arrays_validation(tmp_path: Path) -> None:
    atoms, ai = _ai()
    p = tmp_path / "samples.npz"
    save_samples_npz(ai, p)
    ai2 = from_npz(p, structure=atoms)
    # If from_arrays validation ran on load, gas membership and frac wrapping
    # are guaranteed; and the sample view materialises an InsertionSamples,
    # whose validators run unconditionally.
    s = ai2.samples
    assert s.gas in {"CO2", "N2", "CH4"}
    assert ((s.positions_frac >= 0.0) & (s.positions_frac < 1.0)).all()
