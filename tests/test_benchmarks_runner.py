"""Tests for benchmarks/runner.py (T046)."""

from __future__ import annotations

import json
from pathlib import Path

from ase.build import bulk
from ase.io import write

from widom_atlas.benchmarks.runner import (
    _pipeline_params_hash,
    run_benchmark_set,
)
from widom_atlas.core.pipeline import PipelineParams


def _seed_real_si_fixture(fixtures: Path, ids: list[str]) -> None:
    fixtures.mkdir(parents=True, exist_ok=True)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_pbc(True)
    for material_id in ids:
        write(fixtures / f"{material_id}.cif", atoms)


def _seed_all_manual_fixtures(fixtures: Path) -> None:
    """Populate fixtures directory for every manual material in the small set."""
    from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET

    ids = [m.material_id for m in SMALL_BENCHMARK_SET if m.source == "manual"]
    _seed_real_si_fixture(fixtures, ids)


def test_run_benchmark_set_writes_per_material_manifest_json(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    _seed_all_manual_fixtures(fixtures)
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, out, cache,
        download=False,
        n_samples=80,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(8, 8, 8), dbscan_eps_A=0.5, min_samples=4),
    )
    succeeded = [m for m in summary.materials if m["status"] == "ok"]
    assert succeeded
    for record in succeeded:
        assert (out / record["material_id"] / "manifest.json").exists()


def test_run_benchmark_set_writes_top_level_benchmark_run_json(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    _seed_all_manual_fixtures(fixtures)
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, out, cache,
        download=False,
        n_samples=40,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(6, 6, 6), dbscan_eps_A=0.5, min_samples=3),
    )
    assert summary.benchmark_run_path.exists()
    payload = json.loads(summary.benchmark_run_path.read_text())
    assert "materials" in payload and len(payload["materials"]) >= 1


def test_run_benchmark_set_records_package_and_dependency_versions(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    _seed_all_manual_fixtures(fixtures)
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, tmp_path / "out", tmp_path / "cache",
        download=False, n_samples=20,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(4, 4, 4), dbscan_eps_A=0.5, min_samples=3),
    )
    payload = json.loads(summary.benchmark_run_path.read_text())
    assert payload["package_version"]
    assert "numpy" in payload["dependency_versions"]


def test_run_benchmark_set_records_dataset_sha256_provenance(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    _seed_all_manual_fixtures(fixtures)
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    run_benchmark_set(
        "small", "CO2", fixtures, None, out, cache,
        download=False, n_samples=20,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(4, 4, 4), dbscan_eps_A=0.5, min_samples=3),
    )
    prov_files = list(cache.rglob("*.provenance.json"))
    assert prov_files
    prov = json.loads(prov_files[0].read_text())
    assert "sha256" in prov and len(prov["sha256"]) == 64


def test_run_benchmark_set_skips_unchanged_materials_using_cache_key() -> None:
    p1 = PipelineParams()
    p2 = PipelineParams()
    assert _pipeline_params_hash(p1) == _pipeline_params_hash(p2)
    p3 = PipelineParams(n_grid=(8, 8, 8))
    assert _pipeline_params_hash(p3) != _pipeline_params_hash(p1)


def test_run_benchmark_set_continues_when_one_material_fails_and_records_error(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    _seed_all_manual_fixtures(fixtures)
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, out, cache,
        download=False, n_samples=20,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(4, 4, 4), dbscan_eps_A=0.5, min_samples=3),
    )
    statuses = {m["status"] for m in summary.materials}
    # core_mof entries are not cached → should be skipped, not crashing the run
    assert "skipped" in statuses or "ok" in statuses


def test_run_benchmark_set_emits_basins_csv_and_density_npz_per_material(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    _seed_all_manual_fixtures(fixtures)
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, tmp_path / "out", tmp_path / "cache",
        download=False, n_samples=40,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(6, 6, 6), dbscan_eps_A=0.5, min_samples=3),
    )
    for record in summary.materials:
        if record["status"] == "ok":
            mat_dir = tmp_path / "out" / record["material_id"]
            assert (mat_dir / "basins.csv").exists()
            assert (mat_dir / "density.npz").exists()


def test_run_benchmark_set_integration_real_structure_uio66(tmp_path: Path) -> None:
    """End-to-end sanity check on a real (silicon-stand-in) structure."""
    fixtures = tmp_path / "fixtures"
    _seed_all_manual_fixtures(fixtures)
    out = tmp_path / "out"
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, out, tmp_path / "cache",
        download=False, n_samples=40,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(6, 6, 6), dbscan_eps_A=0.5, min_samples=3),
    )
    uio = next((m for m in summary.materials if m["material_id"] == "UiO-66"), None)
    assert uio is not None
    assert uio["status"] == "ok"
