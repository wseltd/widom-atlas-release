"""Tests for BenchmarkMaterial / BenchmarkRun / BenchmarkComparison (T013)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from widom_atlas.core.benchmark_models import (
    BenchmarkComparison,
    BenchmarkMaterial,
    BenchmarkRun,
)


def _material(**overrides) -> BenchmarkMaterial:
    base = {
        "material_id": "UiO-66",
        "source": "manual",
        "formula": "C48H28O32Zr6",
        "space_group": "Fm-3m",
        "cif_path": None,
        "cif_sha256": None,
        "license": "CC BY 4.0",
        "citation": "Cavka 2008",
        "pore_class": "standard",
    }
    base.update(overrides)
    return BenchmarkMaterial(**base)


def _run(material: BenchmarkMaterial | None = None, **overrides) -> BenchmarkRun:
    base = {
        "run_id": "r1",
        "material": material or _material(),
        "gas": "CO2",
        "temperature_K": 298.15,
        "n_samples": 10000,
        "atlas_report_path": Path("./benchmark-report-fixture"),
        "basins_count": 4,
        "package_version": "0.1.0",
        "timestamp": datetime.now(UTC),
    }
    base.update(overrides)
    return BenchmarkRun(**base)


def test_benchmark_material_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        BenchmarkMaterial(
            material_id="X",
            source="csd",
            formula="C",
            license="CC BY 4.0",
            citation="ref",
        )


def test_benchmark_material_validates_sha256_length() -> None:
    with pytest.raises(ValidationError):
        _material(cif_sha256="0" * 63)
    with pytest.raises(ValidationError):
        _material(cif_sha256="z" * 64)
    _material(cif_sha256="0" * 64)


def test_benchmark_run_requires_supported_gas() -> None:
    with pytest.raises(ValidationError):
        _run(gas="H2O")
    with pytest.raises(ValidationError):
        _run(gas="Ar")


def test_benchmark_comparison_validation_label_enum() -> None:
    run = _run()
    with pytest.raises(ValidationError):
        BenchmarkComparison(
            run=run,
            reference_source="none",
            validation_label="unspecified",
        )
    BenchmarkComparison(
        run=run,
        reference_source="none",
        validation_label="trend_only",
        notes="atlas-only run",
    )


def test_benchmark_models_are_frozen() -> None:
    m = _material()
    with pytest.raises(ValidationError):
        m.material_id = "Other"  # type: ignore[misc]
    r = _run(material=m)
    with pytest.raises(ValidationError):
        r.gas = "N2"  # type: ignore[misc]


def test_benchmark_comparison_handles_missing_reference() -> None:
    run = _run()
    cmp = BenchmarkComparison(
        run=run,
        reference_source="none",
        reference_id=None,
        reference_KH=None,
        computed_KH=None,
        reference_Qads=None,
        computed_Qads=None,
        trend_match=None,
        validation_label="unavailable",
        notes="no MOFX-DB scalar match",
    )
    assert cmp.validation_label == "unavailable"
    assert cmp.reference_KH is None
