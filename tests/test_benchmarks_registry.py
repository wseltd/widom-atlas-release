"""Tests for benchmarks/registry.py (T042)."""

from __future__ import annotations

import pytest

from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET, get_benchmark_set
from widom_atlas.core.benchmark_models import BenchmarkMaterial


def test_small_set_contains_required_materials() -> None:
    ids = {m.material_id for m in SMALL_BENCHMARK_SET}
    for required in ("Mg-MOF-74", "UiO-66", "ZIF-8", "MOF-5", "MFI", "CHA"):
        assert required in ids


def test_small_set_includes_narrow_pore_and_oms_entries() -> None:
    pore_classes = {m.pore_class for m in SMALL_BENCHMARK_SET}
    assert "narrow" in pore_classes
    assert "open_metal_site" in pore_classes


def test_every_entry_has_license_tag_and_citation() -> None:
    for m in SMALL_BENCHMARK_SET:
        assert m.license
        assert m.citation


def test_get_benchmark_set_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_benchmark_set("does-not-exist")


def test_registry_entries_are_BenchmarkMaterial_instances() -> None:
    assert all(isinstance(m, BenchmarkMaterial) for m in SMALL_BENCHMARK_SET)


def test_registry_excludes_h2o_iza_bulk_and_csd_entries() -> None:
    licenses = {m.license.lower() for m in SMALL_BENCHMARK_SET}
    assert not any("csd" in l for l in licenses)
    assert not any("iza bulk" in l for l in licenses)
