"""Benchmark provenance schemas — material registry, run record, scalar comparison.

These models are used by the public-dataset benchmark layer (see
``src/widom_atlas/benchmarks/``) to record what was downloaded, hashed,
licensed, and run. ``BenchmarkComparison.validation_label`` enforces the
"TREND validation, not proof" framing from the brief: scalar comparisons
against MOFX-DB / NIST are tagged ``trend_only`` unless explicitly justified.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BenchmarkSource = Literal["core_mof", "qmof", "manual"]
BenchmarkGas = Literal["CO2", "N2", "CH4"]
PoreClass = Literal["narrow", "open_metal_site", "standard"]
ReferenceSource = Literal["mofx_db", "nist", "none"]
ValidationLabel = Literal["trend_only", "exact", "degraded", "unavailable"]

_HEX_CHARS = frozenset("0123456789abcdef")


def _is_lowercase_hex_sha256(s: str) -> bool:
    return len(s) == 64 and all(c in _HEX_CHARS for c in s)


class BenchmarkMaterial(BaseModel):
    """One curated benchmark material, with license + citation provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    material_id: str = Field(min_length=1)
    source: BenchmarkSource
    source_identifier: str | None = None
    formula: str
    space_group: str | None = None
    cif_path: Path | None = None
    cif_sha256: str | None = None
    license: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    pore_class: PoreClass | None = None
    core_mof_dataset: str | None = None

    @field_validator("cif_sha256")
    @classmethod
    def _validate_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _is_lowercase_hex_sha256(v):
            raise ValueError(
                f"cif_sha256 must be 64 lowercase hex chars; got len={len(v)}"
            )
        return v


class BenchmarkRun(BaseModel):
    """One end-to-end widom-atlas run on a benchmark material."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    material: BenchmarkMaterial
    gas: BenchmarkGas
    temperature_K: float = Field(gt=0.0)
    n_samples: int = Field(ge=0)
    atlas_report_path: Path
    basins_count: int = Field(ge=0)
    package_version: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkComparison(BaseModel):
    """Scalar comparison of widom-atlas-derived KH / Qads vs an external reference.

    The ``validation_label`` field enforces the "TREND validation, not proof"
    framing — exact-equality validation must be explicitly justified.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: BenchmarkRun
    reference_source: ReferenceSource
    reference_id: str | None = None
    reference_KH: float | None = None
    computed_KH: float | None = None
    reference_Qads: float | None = None
    computed_Qads: float | None = None
    trend_match: bool | None = None
    validation_label: ValidationLabel
    notes: str = ""
