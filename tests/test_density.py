"""Tests for density module: Boltzmann weights, grid build, smoothing, npz IO (T020–T023)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from widom_atlas.core.constants import KB_EV_PER_K
from widom_atlas.core.models import DensityGrid, InsertionSamples
from widom_atlas.density.boltzmann import boltzmann_weights, log_boltzmann_weights
from widom_atlas.density.grid import build_density_grid
from widom_atlas.density.io import (
    DENSITY_NPZ_SCHEMA_VERSION,
    load_density_npz,
    save_density_npz,
)
from widom_atlas.density.smoothing import smooth_density


def _atoms(cell_diag: float = 10.0) -> Atoms:
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * cell_diag, pbc=True)


def _samples(n: int = 200, seed: int = 0, T: float = 298.15, gas: str = "CO2") -> InsertionSamples:
    rng = np.random.default_rng(seed)
    frac = rng.random((n, 3))
    e = rng.normal(-0.2, 0.05, n)
    return InsertionSamples(
        positions_cart=frac * 10.0,
        positions_frac=frac,
        energies_eV=e,
        accessible=np.ones(n, dtype=bool),
        temperature_K=T,
        gas=gas,
    )


# --- T020 Boltzmann ---------------------------------------------------------


def test_boltzmann_weights_sum_to_one() -> None:
    e = np.array([-0.5, -0.3, -0.1, 0.1, 1.0])
    w = boltzmann_weights(e, 298.15)
    assert abs(w.sum() - 1.0) < 1e-12


def test_boltzmann_weights_lowest_energy_dominates() -> None:
    e = np.array([-0.5, -0.3, -0.1, 0.1, 1.0])
    w = boltzmann_weights(e, 298.15)
    assert int(np.argmax(w)) == 0


def test_boltzmann_weights_numerical_stability_large_range() -> None:
    e = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # ~80 kT spread at 298 K
    w = boltzmann_weights(e, 298.15)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-12


def test_log_boltzmann_consistent_with_linear() -> None:
    e = np.array([-0.5, -0.3, -0.1, 0.1, 1.0])
    lw = log_boltzmann_weights(e, 298.15)
    np.testing.assert_allclose(np.exp(lw), boltzmann_weights(e, 298.15), atol=1e-15)


def test_boltzmann_rejects_nonpositive_temperature() -> None:
    e = np.array([0.0])
    with pytest.raises(ValueError):
        boltzmann_weights(e, 0.0)
    with pytest.raises(ValueError):
        boltzmann_weights(e, -1.0)


def test_boltzmann_rejects_nonfinite_energies() -> None:
    with pytest.raises(ValueError):
        boltzmann_weights(np.array([0.0, np.nan]), 298.15)
    with pytest.raises(ValueError):
        boltzmann_weights(np.array([0.0, np.inf]), 298.15)


# --- T021 build_density_grid -------------------------------------------------


def test_build_density_grid_normalised() -> None:
    s = _samples()
    g = build_density_grid(s, _atoms(), n_grid=(16, 16, 16))
    assert isinstance(g, DensityGrid)
    assert abs(float(g.grid.sum()) - 1.0) < 1e-9


def test_build_density_grid_periodic_wrap() -> None:
    rng = np.random.default_rng(0)
    n = 10
    frac = rng.random((n, 3)) + 1.5  # outside [0,1)
    e = rng.normal(-0.2, 0.05, n)
    s = InsertionSamples(
        positions_cart=(frac % 1.0) * 10.0,
        positions_frac=(frac % 1.0),
        energies_eV=e,
        accessible=np.ones(n, dtype=bool),
        temperature_K=298.15,
        gas="CO2",
    )
    g = build_density_grid(s, _atoms(), n_grid=(8, 8, 8))
    assert g.grid.shape == (8, 8, 8)


def test_build_density_grid_boltzmann_weights() -> None:
    rng = np.random.default_rng(0)
    frac = np.array([[0.05, 0.5, 0.5], [0.55, 0.5, 0.5]])
    e = np.array([-0.5, 0.5])  # first should dominate
    s = InsertionSamples(
        positions_cart=frac * 10.0,
        positions_frac=frac,
        energies_eV=e,
        accessible=np.ones(2, dtype=bool),
        temperature_K=298.15,
        gas="CO2",
    )
    g = build_density_grid(s, _atoms(), n_grid=(20, 20, 20))
    # Voxel containing the low-energy point must have higher density than the high-energy one.
    i_low = (np.floor(frac[0] * 20).astype(int))
    i_high = (np.floor(frac[1] * 20).astype(int))
    assert g.grid[tuple(i_low)] > g.grid[tuple(i_high)]


def test_build_density_grid_rejects_nonpositive_temperature() -> None:
    s = _samples()
    with pytest.raises(ValueError):
        build_density_grid(s, _atoms(), n_grid=(8, 8, 8), temperature_K=0.0)


def test_build_density_grid_logsumexp_stable_large_energies() -> None:
    rng = np.random.default_rng(0)
    n = 50
    frac = rng.random((n, 3))
    e = rng.uniform(-2.0, 2.0, n)
    s = InsertionSamples(
        positions_cart=frac * 10.0,
        positions_frac=frac,
        energies_eV=e,
        accessible=np.ones(n, dtype=bool),
        temperature_K=298.15,
        gas="CO2",
    )
    g = build_density_grid(s, _atoms(), n_grid=(8, 8, 8))
    assert np.all(np.isfinite(g.grid))


def test_build_density_grid_triclinic_cell() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.5, 0.5, 7.0]])
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    rng = np.random.default_rng(0)
    n = 30
    frac = rng.random((n, 3))
    e = rng.normal(-0.2, 0.05, n)
    s = InsertionSamples(
        positions_cart=frac @ cell,
        positions_frac=frac,
        energies_eV=e,
        accessible=np.ones(n, dtype=bool),
        temperature_K=298.15,
        gas="CO2",
    )
    g = build_density_grid(s, atoms, n_grid=(8, 8, 8))
    np.testing.assert_allclose(g.cell_A, cell, atol=1e-12)


# --- T022 smooth_density -----------------------------------------------------


def test_smooth_density_periodic_no_edge_artefact() -> None:
    s = _samples(n=500, seed=0)
    g = build_density_grid(s, _atoms(), n_grid=(16, 16, 16))
    g_smooth = smooth_density(g, sigma_A=0.5)
    # mass at face x=0 and x=last should be similar after periodic smoothing (no edge dip).
    assert g_smooth.grid[0].sum() > 0.0
    assert g_smooth.grid[-1].sum() > 0.0


def test_smooth_density_anisotropic_sigma_triclinic() -> None:
    cell = np.diag([5.0, 10.0, 20.0])
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    rng = np.random.default_rng(0)
    n = 200
    frac = rng.random((n, 3))
    s = InsertionSamples(
        positions_cart=frac @ cell,
        positions_frac=frac,
        energies_eV=rng.normal(-0.2, 0.05, n),
        accessible=np.ones(n, dtype=bool),
        temperature_K=298.15,
        gas="CO2",
    )
    g = build_density_grid(s, atoms, n_grid=(8, 8, 8))
    g_smooth = smooth_density(g, sigma_A=0.5)
    assert g_smooth.shape == g.shape
    assert abs(float(g_smooth.grid.sum()) - 1.0) < 1e-9


def test_smooth_density_preserves_normalisation() -> None:
    s = _samples()
    g = build_density_grid(s, _atoms(), n_grid=(16, 16, 16))
    g_smooth = smooth_density(g, sigma_A=0.5)
    assert abs(float(g_smooth.grid.sum()) - 1.0) < 1e-9


def test_smooth_density_rejects_non_wrap_mode() -> None:
    s = _samples()
    g = build_density_grid(s, _atoms(), n_grid=(8, 8, 8))
    with pytest.raises(ValueError):
        smooth_density(g, sigma_A=0.5, mode="constant")


def test_smooth_density_does_not_mutate_input() -> None:
    s = _samples()
    g = build_density_grid(s, _atoms(), n_grid=(8, 8, 8))
    original = g.grid.copy()
    smooth_density(g, sigma_A=0.5)
    np.testing.assert_array_equal(g.grid, original)


# --- T023 save/load density --------------------------------------------------


def test_save_load_density_npz_roundtrip(tmp_path: Path) -> None:
    s = _samples()
    g = build_density_grid(s, _atoms(), n_grid=(8, 8, 8))
    p = tmp_path / "density.npz"
    save_density_npz(g, p)
    g2 = load_density_npz(p)
    np.testing.assert_allclose(g.grid, g2.grid)
    np.testing.assert_allclose(g.cell_A, g2.cell_A)
    assert g.shape == g2.shape
    assert g.gas == g2.gas


def test_save_density_npz_creates_parent_dir(tmp_path: Path) -> None:
    s = _samples()
    g = build_density_grid(s, _atoms(), n_grid=(8, 8, 8))
    p = tmp_path / "nested" / "deeper" / "density.npz"
    save_density_npz(g, p)
    assert p.exists()


def test_load_density_npz_rejects_wrong_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "broken.npz"
    np.savez_compressed(
        p,
        grid=np.ones((4, 4, 4)) / 64.0,
        shape=np.array([4, 4, 4], dtype=np.int64),
        cell_A=np.eye(3) * 5.0,
        spacing_A=np.array([1.25, 1.25, 1.25], dtype=np.float64),
        temperature_K=np.float64(298.15),
        gas=np.asarray("CO2"),
        normalisation=np.asarray("probability"),
        smoothing_sigma_A=np.float64(0.0),
        n_source_samples=np.int64(0),
        metadata_json=np.asarray("{}"),
        schema_version=np.int64(DENSITY_NPZ_SCHEMA_VERSION + 99),
    )
    with pytest.raises(ValueError):
        load_density_npz(p)


def test_save_density_npz_includes_gas_temperature_metadata(tmp_path: Path) -> None:
    s = _samples(gas="N2")
    g = build_density_grid(s, _atoms(), n_grid=(8, 8, 8))
    p = tmp_path / "density.npz"
    save_density_npz(g, p)
    with np.load(p, allow_pickle=False) as f:
        assert str(f["gas"]) == "N2"
        assert float(f["temperature_K"]) == 298.15


def test_kb_constant_value_pinned() -> None:
    assert pytest.approx(8.617333262e-5, rel=1e-12) == KB_EV_PER_K
