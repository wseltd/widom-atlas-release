"""Tests for benchmarks/launch_report.py (T048)."""

from __future__ import annotations

import json
from pathlib import Path

from ase.build import bulk
from ase.io import write

from widom_atlas.benchmarks.launch_report import (
    write_launch_report,
)
from widom_atlas.benchmarks.runner import run_benchmark_set
from widom_atlas.benchmarks.scalar_compare import compare_scalars
from widom_atlas.core.pipeline import PipelineParams


def _setup(tmp_path: Path):
    from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_pbc(True)
    for m in SMALL_BENCHMARK_SET:
        if m.source == "manual":
            write(fixtures / f"{m.material_id}.cif", atoms)
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    summary = run_benchmark_set(
        "small", "CO2", fixtures, None, out, cache,
        download=False, n_samples=40,
        samples_kind="toy_lj",
        params=PipelineParams(n_grid=(4, 4, 4), dbscan_eps_A=0.5, min_samples=3),
    )
    compare_scalars(summary.benchmark_run_path, cache, out / "scalar")
    return summary, out, cache


def test_write_launch_report_emits_markdown_html_and_json(tmp_path: Path) -> None:
    summary, out, _ = _setup(tmp_path)
    write_launch_report(summary.benchmark_run_path, out / "scalar" / "scalar_comparison.json", out)
    assert (out / "launch_report.md").exists()
    assert (out / "launch_report.html").exists()
    assert (out / "launch_report.json").exists()


def test_launch_report_markdown_omits_forbidden_words_validated_proven_guaranteed(tmp_path: Path) -> None:
    import re
    summary, out, _ = _setup(tmp_path)
    write_launch_report(summary.benchmark_run_path, out / "scalar" / "scalar_comparison.json", out)
    text = (out / "launch_report.md").read_text()
    pattern = re.compile(r"\b(?:validated|proven|guarantees|guaranteed)\b", re.IGNORECASE)
    assert pattern.search(text) is None, f"forbidden word found: {pattern.search(text).group(0)!r}"


def test_launch_report_includes_dataset_sha256_and_license_provenance(tmp_path: Path) -> None:
    summary, out, _ = _setup(tmp_path)
    write_launch_report(summary.benchmark_run_path, out / "scalar" / "scalar_comparison.json", out)
    payload = json.loads((out / "launch_report.json").read_text())
    assert "provenance" in payload
    if payload["provenance"]:
        for row in payload["provenance"]:
            assert "license" in row and "sha256" in row


def test_launch_report_summarises_TREND_label_counts(tmp_path: Path) -> None:
    summary, out, _ = _setup(tmp_path)
    write_launch_report(summary.benchmark_run_path, out / "scalar" / "scalar_comparison.json", out)
    payload = json.loads((out / "launch_report.json").read_text())
    assert "trend_counts" in payload
    for k in ("TREND", "UNAVAILABLE", "IDENTITY_UNCERTAIN", "OUT_OF_RANGE"):
        assert k in payload["trend_counts"]


def test_launch_report_html_escapes_user_derived_material_ids(tmp_path: Path) -> None:
    # Hand-craft a benchmark_run.json with an HTML-injecting material_id
    bad_run = tmp_path / "benchmark_run.json"
    bad_run.write_text(
        json.dumps(
            {
                "package_version": "0.1.0",
                "set_name": "small",
                "gas": "CO2",
                "temperature_K": 298.15,
                "materials": [
                    {
                        "material_id": "<script>alert(1)</script>",
                        "source": "manual",
                        "license": "CC BY 4.0",
                        "gas": "CO2",
                        "temperature_K": 298.15,
                        "status": "ok",
                        "cif_sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps([]), encoding="utf-8")
    write_launch_report(bad_run, scalar, tmp_path / "out")
    html = (tmp_path / "out" / "launch_report.html").read_text()
    assert "<script>alert(1)</script>" not in html


def test_launch_report_records_failed_materials_with_error_class(tmp_path: Path) -> None:
    bad_run = tmp_path / "benchmark_run.json"
    bad_run.write_text(
        json.dumps(
            {
                "package_version": "0.1.0",
                "set_name": "small",
                "gas": "CO2",
                "temperature_K": 298.15,
                "materials": [
                    {
                        "material_id": "X",
                        "source": "manual",
                        "license": "CC BY 4.0",
                        "gas": "CO2",
                        "temperature_K": 298.15,
                        "status": "failed",
                        "error_class": "RuntimeError",
                        "error_message": "boom",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps([]), encoding="utf-8")
    write_launch_report(bad_run, scalar, tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "launch_report.json").read_text())
    assert payload["totals"]["failed"] == 1
    assert payload["failed"][0]["error_class"] == "RuntimeError"


def test_launch_report_integration_consumes_T046_and_T047_outputs(tmp_path: Path) -> None:
    summary, out, _ = _setup(tmp_path)
    rep = write_launch_report(summary.benchmark_run_path, out / "scalar" / "scalar_comparison.json", out)
    assert rep.totals["attempted"] >= 1
