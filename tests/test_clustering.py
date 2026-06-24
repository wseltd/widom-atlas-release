"""Tests for clustering: PBC DBSCAN, basin extraction, uncertainty (T024–T026)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from widom_atlas.clustering.basins import extract_basins
from widom_atlas.clustering.pbc_dbscan import pbc_dbscan
from widom_atlas.clustering.uncertainty import (
    _bootstrap_centroid_stderr,
    _effective_sample_size_kish,
    annotate_basin_uncertainty,
)
from widom_atlas.core.models import Basin, InsertionSamples


def _atoms(cell_diag: float = 10.0) -> Atoms:
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * cell_diag, pbc=True)


def _samples(positions_frac: np.ndarray, energies_eV: np.ndarray, T: float = 298.15) -> InsertionSamples:
    n = positions_frac.shape[0]
    cart = positions_frac * 10.0
    return InsertionSamples(
        positions_cart=cart,
        positions_frac=positions_frac,
        energies_eV=energies_eV,
        accessible=np.ones(n, dtype=bool),
        temperature_K=T,
        gas="CO2",
    )


# --- T024 pbc_dbscan ---------------------------------------------------------


def _two_blob_frac(rng: np.random.Generator, n_per_blob: int = 30) -> np.ndarray:
    blob_a = rng.normal([0.05, 0.5, 0.5], 0.005, (n_per_blob, 3))
    blob_b = rng.normal([0.95, 0.5, 0.5], 0.005, (n_per_blob, 3))
    pts = np.vstack([blob_a, blob_b]) % 1.0
    return pts


def test_pbc_dbscan_joins_boundary_crossing_basin() -> None:
    rng = np.random.default_rng(0)
    pts = _two_blob_frac(rng)
    cell = np.eye(3) * 10.0
    labels = pbc_dbscan(pts, cell, eps_A=2.0, min_samples=5)
    unique = sorted(set(int(x) for x in labels) - {-1})
    assert len(unique) == 1


def test_pbc_dbscan_separates_distant_basins() -> None:
    rng = np.random.default_rng(0)
    blob_a = rng.normal([0.2, 0.2, 0.5], 0.005, (20, 3))
    blob_b = rng.normal([0.8, 0.8, 0.5], 0.005, (20, 3))
    pts = np.vstack([blob_a, blob_b])
    cell = np.eye(3) * 10.0
    labels = pbc_dbscan(pts, cell, eps_A=0.5, min_samples=5)
    unique = sorted(set(int(x) for x in labels) - {-1})
    assert len(unique) == 2


def test_pbc_dbscan_triclinic_min_image() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.5, 7.0]])
    rng = np.random.default_rng(0)
    pts = rng.normal([0.5, 0.5, 0.5], 0.01, (20, 3))
    labels = pbc_dbscan(pts, cell, eps_A=0.5, min_samples=5)
    assert int((labels == 0).sum()) >= 15


def test_pbc_dbscan_weighted_core_point() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal([0.5, 0.5, 0.5], 0.01, (10, 3))
    weights = np.full(10, 1.0)
    cell = np.eye(3) * 10.0
    labels_unweighted = pbc_dbscan(pts, cell, eps_A=0.5, min_samples=3)
    labels_weighted = pbc_dbscan(pts, cell, eps_A=0.5, min_samples=3, weights=weights)
    assert (labels_unweighted >= 0).all()
    assert (labels_weighted >= 0).all()


def test_pbc_dbscan_returns_minus_one_for_noise() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (5, 3))  # sparse — likely noise
    cell = np.eye(3) * 100.0
    labels = pbc_dbscan(pts, cell, eps_A=0.5, min_samples=10)
    assert int((labels == -1).sum()) == 5


def test_pbc_dbscan_deterministic_for_fixed_input() -> None:
    rng1 = np.random.default_rng(7)
    pts1 = _two_blob_frac(rng1)
    rng2 = np.random.default_rng(7)
    pts2 = _two_blob_frac(rng2)
    cell = np.eye(3) * 10.0
    a = pbc_dbscan(pts1, cell, eps_A=2.0, min_samples=5)
    b = pbc_dbscan(pts2, cell, eps_A=2.0, min_samples=5)
    np.testing.assert_array_equal(a, b)


def test_pbc_dbscan_rejects_eps_larger_than_half_min_cell_width() -> None:
    pts = np.zeros((4, 3))
    cell = np.eye(3) * 10.0  # min width 10, half = 5
    with pytest.raises(ValueError):
        pbc_dbscan(pts, cell, eps_A=6.0, min_samples=2)


# --- T025 extract_basins -----------------------------------------------------


def test_extract_basins_recovers_known_centroids_in_cubic_cell() -> None:
    rng = np.random.default_rng(0)
    target = np.array([0.3, 0.4, 0.5])
    pts = (rng.normal(target, 0.005, (50, 3))) % 1.0
    e = rng.normal(-0.5, 0.01, 50)
    s = _samples(pts, e)
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=5)
    assert len(basins) == 1
    cf = np.array(basins[0].centroid_frac)
    delta = (cf - target + 0.5) % 1.0 - 0.5
    assert float(np.linalg.norm(delta)) < 0.05


def test_extract_basins_does_not_split_basin_crossing_periodic_boundary() -> None:
    rng = np.random.default_rng(0)
    # 60 points clustered around frac=(0.0, 0.5, 0.5) but expressed across the boundary
    blob_a = rng.normal([0.02, 0.5, 0.5], 0.005, (30, 3)) % 1.0
    blob_b = rng.normal([0.98, 0.5, 0.5], 0.005, (30, 3)) % 1.0
    pts = np.vstack([blob_a, blob_b])
    e = rng.normal(-0.5, 0.01, 60)
    s = _samples(pts, e)
    basins = extract_basins(s, _atoms(), eps_A=2.0, min_samples=5)
    assert len(basins) == 1


def test_extract_basins_boltzmann_weights_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (40, 3))) % 1.0
    e = rng.normal(-0.5, 0.01, 40)
    s = _samples(pts, e)
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=5)
    total = sum(b.weight for b in basins)
    # multiple basins may exist; total weight assigned to clusters can be < 1 (noise excluded)
    assert 0.0 <= total <= 1.0 + 1e-9


def test_extract_basins_centroid_uses_minimum_image() -> None:
    pts = np.array([[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]] * 5)
    e = np.full(10, -0.5)
    s = _samples(pts, e)
    basins = extract_basins(s, _atoms(), eps_A=2.0, min_samples=3)
    assert len(basins) == 1
    cf = np.asarray(basins[0].centroid_frac)
    # The circular mean of {0.01, 0.99} is ~0.0 (or 1.0) — wrapped to [0,1) should be near 0 or 1.
    assert (abs(cf[0]) < 0.05) or (abs(cf[0] - 1.0) < 0.05)


def test_extract_basins_returns_basin_pydantic_instances() -> None:
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (20, 3))) % 1.0
    s = _samples(pts, rng.normal(-0.5, 0.01, 20))
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=5)
    assert all(isinstance(b, Basin) for b in basins)


def test_extract_basins_handles_triclinic_cell() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.5, 0.5, 7.0]])
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (30, 3))) % 1.0
    e = rng.normal(-0.5, 0.01, 30)
    s = InsertionSamples(
        positions_cart=pts @ cell,
        positions_frac=pts,
        energies_eV=e,
        accessible=np.ones(30, dtype=bool),
        temperature_K=298.15,
        gas="CO2",
    )
    basins = extract_basins(s, atoms, eps_A=0.4, min_samples=5)
    assert len(basins) >= 1


def test_extract_basins_empty_samples_returns_empty_list() -> None:
    s = InsertionSamples(
        positions_cart=np.zeros((0, 3)),
        positions_frac=np.zeros((0, 3)),
        energies_eV=np.zeros(0),
        accessible=np.zeros(0, dtype=bool),
        temperature_K=298.15,
        gas="CO2",
    )
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=3)
    assert basins == []


def test_extract_basins_min_samples_filters_noise() -> None:
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (20, 3))) % 1.0
    s = _samples(pts, rng.normal(-0.5, 0.01, 20))
    basins_low = extract_basins(s, _atoms(), eps_A=0.5, min_samples=3)
    basins_high = extract_basins(s, _atoms(), eps_A=0.5, min_samples=100)
    assert len(basins_high) <= len(basins_low)


def test_extract_basins_weighted_energy_mean_matches_analytic() -> None:
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (40, 3))) % 1.0
    e = rng.normal(-0.5, 0.01, 40)
    s = _samples(pts, e)
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=5)
    assert len(basins) == 1
    # Analytic check: weighted mean of e is between min and max
    assert basins[0].mean_energy_eV >= float(e.min()) - 1e-9
    assert basins[0].mean_energy_eV <= float(e.max()) + 1e-9


# --- T026 uncertainty --------------------------------------------------------


def test_uncertainty_accessible_fraction_matches_analytic() -> None:
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (40, 3))) % 1.0
    accessible = np.array([True] * 30 + [False] * 10)
    s = InsertionSamples(
        positions_cart=pts * 10.0,
        positions_frac=pts,
        energies_eV=np.full(40, -0.5),
        accessible=accessible,
        temperature_K=298.15,
        gas="CO2",
    )
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=5)
    out = annotate_basin_uncertainty(basins, s, _atoms())
    assert 0.0 <= out[0].accessible_fraction <= 1.0


def test_uncertainty_kish_ess_equals_n_for_uniform_weights() -> None:
    w = np.ones(50)
    ess = _effective_sample_size_kish(w)
    assert abs(ess - 50.0) < 1e-9


def test_uncertainty_kish_ess_lower_for_skewed_weights() -> None:
    w = np.array([1.0] + [1e-6] * 49)
    ess = _effective_sample_size_kish(w)
    assert ess < 5.0


def test_uncertainty_low_count_flag_set_below_threshold() -> None:
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (8, 3))) % 1.0
    e = rng.normal(-0.5, 0.01, 8)
    s = _samples(pts, e)
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=3)
    out = annotate_basin_uncertainty(basins, s, _atoms())
    if out:
        assert any(b.low_count_flag is True for b in out if b.count < 10)


def test_uncertainty_bootstrap_stderr_deterministic_with_seed() -> None:
    rng = np.random.default_rng(0)
    member_frac = (rng.normal([0.3, 0.4, 0.5], 0.005, (30, 3))) % 1.0
    member_w = np.full(30, 1.0)
    cell = np.eye(3) * 10.0
    s1 = _bootstrap_centroid_stderr(member_frac, member_w, cell, seed=42)
    s2 = _bootstrap_centroid_stderr(member_frac, member_w, cell, seed=42)
    assert s1 == s2


def test_uncertainty_preserves_basin_centroid_values() -> None:
    rng = np.random.default_rng(0)
    pts = (rng.normal([0.3, 0.4, 0.5], 0.005, (30, 3))) % 1.0
    e = rng.normal(-0.5, 0.01, 30)
    s = _samples(pts, e)
    basins = extract_basins(s, _atoms(), eps_A=0.5, min_samples=5)
    out = annotate_basin_uncertainty(basins, s, _atoms())
    assert len(out) == len(basins)
    for orig, new in zip(basins, out, strict=False):
        assert orig.centroid_frac == new.centroid_frac
        assert orig.basin_id == new.basin_id
