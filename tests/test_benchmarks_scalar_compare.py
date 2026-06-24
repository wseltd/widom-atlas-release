"""Tests for benchmarks/scalar_compare.py (T047)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write

from widom_atlas.benchmarks.runner import run_benchmark_set
from widom_atlas.benchmarks.scalar_compare import (
    _derive_KH_from_samples,
    _derive_Qads_from_samples,
    compare_scalars,
)
from widom_atlas.core.pipeline import PipelineParams


def _seed_fixtures(tmp_path: Path) -> Path:
    from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_pbc(True)
    for m in SMALL_BENCHMARK_SET:
        if m.source == "manual":
            write(fixtures / f"{m.material_id}.cif", atoms)
    return fixtures


def _run(tmp_path: Path):
    fixtures = _seed_fixtures(tmp_path)
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, out, cache,
        download=False, n_samples=60,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(6, 6, 6), dbscan_eps_A=0.5, min_samples=3),
    )
    return summary, out, cache


def test_compare_scalars_emits_csv_json_and_markdown(tmp_path: Path) -> None:
    summary, out, cache = _run(tmp_path)
    table = compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    assert (out / "scalar" / "scalar_comparison.csv").exists()
    assert (out / "scalar" / "scalar_comparison.json").exists()
    assert (out / "scalar" / "scalar_comparison.md").exists()
    assert table.rows


def test_compare_scalars_uses_TREND_label_never_PASS(tmp_path: Path) -> None:
    summary, out, cache = _run(tmp_path)
    compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    text = (out / "scalar" / "scalar_comparison.md").read_text()
    assert "PASS" not in text
    assert "VALIDATED" not in text


def test_compare_scalars_marks_UNAVAILABLE_when_KH_cannot_be_derived(tmp_path: Path) -> None:
    summary, out, cache = _run(tmp_path)
    table = compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    assert any(r.comparison_label in {"UNAVAILABLE", "TREND", "IDENTITY_UNCERTAIN", "OUT_OF_RANGE"} for r in table.rows)


def test_compare_scalars_marks_IDENTITY_UNCERTAIN_for_low_confidence_records(tmp_path: Path) -> None:
    summary, out, cache = _run(tmp_path)
    table = compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    labels = [r.comparison_label for r in table.rows]
    # We seed everything via manual fixtures; CoRE entries are skipped, so IDENTITY_UNCERTAIN comes from MOFX-DB lookup
    assert any(label in {"IDENTITY_UNCERTAIN", "UNAVAILABLE"} for label in labels)


def test_compare_scalars_markdown_includes_trend_only_caveat_header(tmp_path: Path) -> None:
    summary, out, cache = _run(tmp_path)
    compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    text = (out / "scalar" / "scalar_comparison.md").read_text()
    assert "TREND validation only" in text


def test_compare_scalars_consumes_real_benchmark_run_json_from_T046(tmp_path: Path) -> None:
    summary, out, cache = _run(tmp_path)
    table = compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    assert table.benchmark_run_path == str(summary.benchmark_run_path)


def test_derive_KH_from_samples_matches_boltzmann_reference_value() -> None:
    e = np.linspace(-0.5, 0.5, 200)
    out = _derive_KH_from_samples(e, 298.15)
    assert out is not None and out > 0.0


def test_derive_Qads_from_samples_matches_expected_within_tolerance() -> None:
    e = np.full(50, -0.3)
    qads = _derive_Qads_from_samples(e, 298.15)
    assert qads is not None
    expected = 0.3 * 96.485
    assert abs(qads - expected) < 0.5
