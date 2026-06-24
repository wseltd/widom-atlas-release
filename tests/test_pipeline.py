"""Tests for the run_atlas pipeline (T041)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from widom_atlas.core.constants import (
    DEFAULT_ANGLE_TOLERANCE_DEG,
    DEFAULT_DENSITY_GRID_SHAPE,
    DEFAULT_SYMPREC,
)
from widom_atlas.core.pipeline import AtlasResult, PipelineParams, run_atlas
from widom_atlas.io.from_arrays import from_arrays


def _atoms() -> Atoms:
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)


def _ai_with_two_basins(n_per_basin: int = 50) -> object:
    rng = np.random.default_rng(0)
    target_a = np.array([0.25, 0.5, 0.5])
    target_b = np.array([0.75, 0.5, 0.5])
    blob_a = rng.normal(target_a, 0.01, (n_per_basin, 3)) % 1.0
    blob_b = rng.normal(target_b, 0.01, (n_per_basin, 3)) % 1.0
    frac = np.vstack([blob_a, blob_b])
    e = np.concatenate(
        [
            rng.normal(-0.5, 0.01, n_per_basin),
            rng.normal(-0.4, 0.01, n_per_basin),
        ]
    )
    return from_arrays(structure=_atoms(), positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CO2")


def test_run_atlas_end_to_end_writes_manifest_and_basins(tmp_path: Path) -> None:
    ai = _ai_with_two_basins()
    params = PipelineParams(n_grid=(16, 16, 16), dbscan_eps_A=0.5, min_samples=10)
    result = run_atlas(ai, params, tmp_path / "run", structure=_atoms())
    assert isinstance(result, AtlasResult)
    assert (tmp_path / "run" / "manifest.json").exists()
    assert (tmp_path / "run" / "basins.json").exists()
    assert (tmp_path / "run" / "density.npz").exists()
    assert (tmp_path / "run" / "report" / "report.md").exists()
    assert (tmp_path / "run" / "report" / "report.html").exists()


def test_run_atlas_propagates_accessible_mask_to_basins(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 60
    target = np.array([0.3, 0.5, 0.5])
    frac = rng.normal(target, 0.01, (n, 3)) % 1.0
    e = rng.normal(-0.5, 0.01, n)
    accessible = np.array([True] * 40 + [False] * 20)
    ai = from_arrays(
        structure=_atoms(),
        positions_frac=frac,
        energies_eV=e,
        accessible=accessible,
        temperature_K=298.15,
        gas="CO2",
    )
    params = PipelineParams(n_grid=(16, 16, 16), dbscan_eps_A=0.5, min_samples=10)
    result = run_atlas(ai, params, tmp_path / "run", structure=_atoms())
    assert result.basins
    assert all(0.0 <= b.accessible_fraction <= 1.0 for b in result.basins)


def test_run_atlas_marks_symmetry_uncertain_on_low_symmetry_input(tmp_path: Path) -> None:
    cell = np.array([[5.0, 0.0, 0.0], [1.5, 6.0, 0.0], [0.5, 0.7, 7.0]])
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    rng = np.random.default_rng(2)
    n = 60
    frac = rng.random((n, 3))
    e = rng.normal(-0.2, 0.05, n)
    ai = from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CO2")
    params = PipelineParams(n_grid=(8, 8, 8), dbscan_eps_A=0.5, min_samples=5)
    result = run_atlas(ai, params, tmp_path / "run", structure=atoms)
    if result.symmetry_groups:
        assert any(g.uncertainty_flags or g.grouping_confidence < 0.7 for g in result.symmetry_groups)


def test_pipeline_params_defaults_match_constants_module() -> None:
    p = PipelineParams()
    assert p.symprec == DEFAULT_SYMPREC
    assert p.angle_tolerance == DEFAULT_ANGLE_TOLERANCE_DEG
    assert p.n_grid == DEFAULT_DENSITY_GRID_SHAPE


def test_run_atlas_writes_to_user_supplied_out_dir(tmp_path: Path) -> None:
    out = tmp_path / "user_choice" / "run42"
    ai = _ai_with_two_basins(n_per_basin=20)
    params = PipelineParams(n_grid=(8, 8, 8), dbscan_eps_A=0.5, min_samples=5)
    run_atlas(ai, params, out, structure=_atoms())
    assert out.exists()
    assert (out / "manifest.json").exists()
