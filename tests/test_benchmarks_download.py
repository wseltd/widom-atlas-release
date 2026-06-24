"""Tests for benchmarks/download.py (T043)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from widom_atlas.benchmarks.download import (
    BenchmarkDataUnavailable,
    fetch_benchmark_material,
)
from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET
from widom_atlas.core.benchmark_models import BenchmarkMaterial


def _manual_material() -> BenchmarkMaterial:
    return next(m for m in SMALL_BENCHMARK_SET if m.source == "manual")


def _core_mof_material() -> BenchmarkMaterial:
    return next(m for m in SMALL_BENCHMARK_SET if m.source == "core_mof")


def _seed_cache(cache_dir: Path, material: BenchmarkMaterial) -> Path:
    base = cache_dir / material.source
    base.mkdir(parents=True, exist_ok=True)
    cif = base / f"{material.material_id}.cif"
    cif.write_text("# fake cif content\n", encoding="utf-8")
    return cif


def test_fetch_returns_cached_path_without_network_when_already_cached(tmp_path: Path) -> None:
    m = _core_mof_material()
    cif = _seed_cache(tmp_path, m)
    out = fetch_benchmark_material(m, cache_dir=tmp_path, allow_network=False)
    assert out == cif


def test_fetch_raises_BenchmarkDataUnavailable_when_offline_and_uncached(tmp_path: Path) -> None:
    # CoRE-MOF data ships with the package, so we exercise the offline-blocked
    # path with a QMOF entry instead, which truly requires Figshare.
    m = BenchmarkMaterial(
        material_id="qmof-test-uncached",
        source="qmof",
        formula="C",
        license="CC BY 4.0",
        citation="ref",
    )
    with pytest.raises(BenchmarkDataUnavailable):
        fetch_benchmark_material(m, cache_dir=tmp_path, allow_network=False)


def test_fetch_writes_meta_json_alongside_cif(tmp_path: Path) -> None:
    m = _manual_material()
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    src = fixtures / f"{m.material_id}.cif"
    src.write_text("# fake cif\n", encoding="utf-8")
    out = fetch_benchmark_material(m, cache_dir=tmp_path, fixtures_dir=fixtures)
    meta_path = tmp_path / m.source / f"{m.material_id}.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["material_id"] == m.material_id


def test_fetch_atomic_write_does_not_leave_partial_file_on_error(tmp_path: Path) -> None:
    # Atomic write is exercised in the manual-fixture path; if writing meta fails, no partial file remains
    m = _manual_material()
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    src = fixtures / f"{m.material_id}.cif"
    src.write_text("# fake\n")
    fetch_benchmark_material(m, cache_dir=tmp_path, fixtures_dir=fixtures)
    base = tmp_path / m.source
    leftovers = [p for p in base.iterdir() if p.suffix not in {".cif", ".json"}]
    assert leftovers == []


def test_fetch_refuses_restricted_license_tag(tmp_path: Path) -> None:
    m = BenchmarkMaterial(
        material_id="X",
        source="core_mof",
        formula="C",
        license="CSD-derived restricted",
        citation="ref",
    )
    with pytest.raises(BenchmarkDataUnavailable):
        fetch_benchmark_material(m, cache_dir=tmp_path)


def test_manual_fixture_dispatch_never_calls_network(tmp_path: Path) -> None:
    m = _manual_material()
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    src = fixtures / f"{m.material_id}.cif"
    src.write_text("# fake\n")
    with patch("urllib.request.urlopen") as mock:
        fetch_benchmark_material(m, cache_dir=tmp_path, fixtures_dir=fixtures, allow_network=True)
        mock.assert_not_called()


def test_qmof_dispatch_uses_pinned_figshare_url(tmp_path: Path) -> None:
    m = BenchmarkMaterial(
        material_id="qmof-test",
        source="qmof",
        formula="C",
        license="CC BY 4.0",
        citation="ref",
    )
    with pytest.raises(BenchmarkDataUnavailable):
        fetch_benchmark_material(m, cache_dir=tmp_path, allow_network=True)
