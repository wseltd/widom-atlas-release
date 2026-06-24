"""Layer 3 — public-dataset benchmark tests (T061).

Marked ``public_benchmark`` so they are excluded from default ``pytest -q``
runs and only execute when the operator opts in via ``pytest -m public_benchmark``.
The default behaviour avoids any network IO; tests here therefore use the
manual-fixture path of the runner with a Si-diamond stand-in for shape
checks. Real network downloads are gated behind the ``--download`` flag of
the CLI command and are not exercised in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase.build import bulk
from ase.io import write

from widom_atlas.benchmarks.launch_report import write_launch_report
from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET
from widom_atlas.benchmarks.runner import run_benchmark_set
from widom_atlas.benchmarks.scalar_compare import compare_scalars
from widom_atlas.core.pipeline import PipelineParams

pytestmark = pytest.mark.public_benchmark


def _seed(tmp_path: Path) -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_pbc(True)
    for m in SMALL_BENCHMARK_SET:
        if m.source == "manual":
            write(fixtures / f"{m.material_id}.cif", atoms)
    return fixtures


def _run(tmp_path: Path, gas: str = "CO2"):
    fixtures = _seed(tmp_path)
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    summary = run_benchmark_set(
        "small", gas, fixtures, None, out, cache,
        download=False, n_samples=40,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(6, 6, 6), dbscan_eps_A=0.5, min_samples=3),
    )
    compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    return summary, out


def test_public_benchmark_small_set_writes_per_material_artefacts(tmp_path: Path) -> None:
    summary, out = _run(tmp_path)
    succeeded = [m for m in summary.materials if m["status"] == "ok"]
    assert succeeded
    for record in succeeded:
        mat_dir = out / record["material_id"]
        assert (mat_dir / "manifest.json").exists()
        assert (mat_dir / "basins.csv").exists()
        assert (mat_dir / "density.npz").exists()


def test_public_benchmark_records_dataset_provenance_with_sha256(tmp_path: Path) -> None:
    summary, _out = _run(tmp_path)
    payload = json.loads(summary.benchmark_run_path.read_text(encoding="utf-8"))
    for m in payload["materials"]:
        if m["status"] == "ok":
            assert "cif_sha256" in m and len(m["cif_sha256"]) == 64


def test_public_benchmark_gas_filter_applies_to_all_materials(tmp_path: Path) -> None:
    summary, _ = _run(tmp_path, gas="N2")
    assert all(m["gas"] == "N2" for m in summary.materials)


def test_public_benchmark_skips_unavailable_manual_entries_when_fixtures_missing(tmp_path: Path) -> None:
    # Provide an empty fixtures dir so manual-source materials (MFI, CHA) are skipped
    # while bundled CoRE-MOF entries still succeed. This shows the runner degrades
    # gracefully on missing optional data rather than aborting the whole batch.
    fixtures = tmp_path / "fixtures_empty"
    fixtures.mkdir()
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, out, cache,
        download=False, n_samples=40,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(6, 6, 6), dbscan_eps_A=0.5, min_samples=3),
    )
    statuses = {m["status"] for m in summary.materials}
    assert "skipped" in statuses
    assert "ok" in statuses


def test_public_benchmark_launch_report_produced_and_honest(tmp_path: Path) -> None:
    summary, out = _run(tmp_path)
    write_launch_report(summary.benchmark_run_path, out / "scalar" / "scalar_comparison.json", out)
    md = (out / "launch_report.md").read_text(encoding="utf-8")
    assert "TREND validation only" in md or "TREND" in md
    import re

    forbidden = re.compile(r"\b(?:validated|proven|guarantees|guaranteed)\b", re.IGNORECASE)
    assert forbidden.search(md) is None


def test_public_benchmark_mfi_cha_zeolite_paths_run(tmp_path: Path) -> None:
    summary, _ = _run(tmp_path)
    ids = {m["material_id"] for m in summary.materials if m["status"] == "ok"}
    assert "MFI" in ids
    assert "CHA" in ids
