"""Strict registry schemas (Pydantic v2, ``extra="forbid"``, ``frozen=True``).

Every entry is provenance-rich: missing DOI, license, or redistribution
status fails to parse. The schemas are versioned via ``schema_version``
strings so future revisions land cleanly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Aligned with widom_atlas.backends.schema.RedistributionStatus, but extended
# with the dataset-specific case "open_access_with_attribution_user_downloads"
# which signals that operators must download themselves.
RedistributionStatus = Literal[
    "bundled_safe",
    "user_supplied_not_bundled",
    "user_supplied_not_redistributed",
    "open_access_with_attribution",
    "open_access_user_downloads",
    "research_use_only",
    "license_unverified",
    "unknown",
]

DatasetKind = Literal[
    "structures",
    "charges",
    "force_field_parameters",
    "scalar_adsorption",
    "ml_force_field",
    "site_loaded_structures",
    "benchmark_suite",
    "service",
    "stability_metadata",
    "topology_metadata",
    "subset",
    "mixed",
]

# MOFX-DB sub-database provenance classification.
# CSD is registered as `placeholder` because the live API returns 0 entries for it.
# PCOD-syn is `experimental_or_synthetic` because the database mixes computational and synthesised
# entries; record the source field on each ingested row.
MofxdbProvenanceKind = Literal[
    "experimental",
    "hypothetical",
    "zeolite",
    "experimental_or_synthetic",
    "placeholder",
    "unknown",
]


# Closed enum for MOFX-DB force-field IDs as returned by
# https://mof.tech.northwestern.edu/forcefields.json (verified 2026-05-08).
# IDs not in this list MUST become `force_field = unknown` with a
# `unknown_mofxdb_force_field_id` warning — never silently mapped.
MOFXDB_FORCE_FIELD_ID_TO_NAME: dict[int, str] = {
    1: "UFF",
    2: "TraPPE",
    3: "Michels-Degraaff-Tenseldam with Darkrim-Levesque charges",
    4: "Darkrim-Levesque",
    5: "Hirschfelder / Talu",
    6: "Talu et al.",
    7: "TraPPE-zeo",
    8: "TraPPE-H2-3SM",
    9: "UFF (epsilon=93.08 K, sigma=3.45 A)",
}


def mofxdb_provenance_kind_from_database(database: str) -> MofxdbProvenanceKind:
    """Map a MOFX-DB sub-database name to its provenance kind.

    Used by the MOFX-DB ingester to stamp every record. Unknown names return
    ``"unknown"`` (and the ingester logs a warning).
    """
    name = database.strip().lower().replace(" ", "")
    if name in {"coremof2014", "coremof2019"}:
        return "experimental"
    if name in {"hmof", "tobacco"}:
        return "hypothetical"
    if name in {"iza", "izasc", "iza-sc"}:
        return "zeolite"
    if name in {"pcod-syn", "pcodsyn"}:
        return "experimental_or_synthetic"
    if name == "csd":
        return "placeholder"
    return "unknown"

LicenseTag = Literal[
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "GPL-2.0",
    "GPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "CC-BY-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-SA-4.0",
    "CC0-1.0",
    "ODC-BY-1.0",
    "ODC-ODbL-1.0",
    "CDLA-Sharing-1.0",
    "CDLA-Permissive-1.0",
    "CDLA-Permissive-2.0",
    "PDDL-1.0",
    "research_use",
    "public_domain_us_government",
    "unverified",
    "other_see_url",
]


class CitationEntry(BaseModel):
    """One DOI + plain-text citation. Reused across the registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doi: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, description="Plain-text citation: authors, journal, year")
    url: str | None = None


class ContentSummary(BaseModel):
    """Structured description of what the dataset contains."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_materials: int | None = Field(default=None, ge=0)
    materials_subset: list[str] = Field(default_factory=list)
    gases: list[str] = Field(default_factory=list)
    temperatures_K: list[float] = Field(default_factory=list)
    force_fields: list[str] = Field(default_factory=list)
    charge_models: list[str] = Field(default_factory=list)
    file_formats: list[str] = Field(default_factory=list)
    notes: str | None = None


class DatasetRegistryEntry(BaseModel):
    """One registered public dataset.

    The registry is provenance-only; ``samples_path`` (or ``cache_path``) is
    where the operator must place the downloaded file before
    ``widom-atlas data status`` reports it as ``present``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.4"] = "0.4"
    name: str = Field(..., min_length=1, description="Short identifier, e.g. 'CRAFTED'")
    kind: DatasetKind
    description: str = Field(..., min_length=1)
    primary_url: str = Field(..., min_length=1, description="Where the operator downloads the data")
    primary_doi: str = Field(..., min_length=1)
    license: LicenseTag
    license_url: str | None = None
    redistribution_status: RedistributionStatus
    citations: list[CitationEntry] = Field(default_factory=list, min_length=1)
    content_summary: ContentSummary
    file_format: list[str] = Field(default_factory=list)
    expected_sha256: str | None = Field(
        default=None,
        description="Operator-verified sha256 of the canonical archive; null until set",
    )
    cache_path: str | None = Field(
        default=None,
        description="Recommended local path under benchmarks/cache/<name>/ to drop the data",
    )
    last_verified: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp set by the operator after sha256 verification",
    )
    notes: str | None = None
    warnings: list[str] = Field(default_factory=list)

    # v0.4 follow-up additions
    predecessor_doi: str | None = Field(
        default=None,
        description="DOI of the canonical predecessor record (e.g. CRAFTED v1, CoRE-MOF 2019 v1) — kept for citation traceability when the primary record supersedes an older one.",
    )
    expected_md5: str | None = Field(
        default=None,
        description="Optional MD5 checksum (operator-verified). When the dataset publisher only ships MD5s (e.g. ODAC23), populate this instead of expected_sha256. data status validates whichever hash is set.",
    )
    expected_md5s: dict[str, str] = Field(
        default_factory=dict,
        description="Optional dict of {filename: md5} when the dataset is multi-file (e.g. ODAC23 has 5 separate archives).",
    )

    @field_validator("primary_doi")
    @classmethod
    def _doi_looks_real(cls, v: str) -> str:
        if not (v.startswith("10.") or v.startswith("http")):
            raise ValueError(f"primary_doi must look like a DOI ('10.x/y') or URL; got {v!r}")
        return v

    @field_validator("predecessor_doi")
    @classmethod
    def _predecessor_doi_looks_real(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (v.startswith("10.") or v.startswith("http")):
            raise ValueError(f"predecessor_doi must look like a DOI or URL; got {v!r}")
        return v


class ReferenceProvenance(BaseModel):
    """Provenance for a single literature reference value (KH or Q_ads).

    Citation + units + redistribution + temperature + pressure + measurement
    method, captured in one place so the scalar comparator can show the
    operator exactly where the reference came from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation: CitationEntry
    measurement_method: Literal[
        "experimental_isotherm",
        "experimental_neutron",
        "experimental_xrd",
        "DFT",
        "DFT_CCSD",
        "QM_MM",
        "GCMC_classical",
        "GCMC_polarisable",
        "ML_FF",
        "review_compilation",
        "other_see_notes",
    ]
    redistribution_status: RedistributionStatus
    notes: str | None = None


class ScalarReferenceEntry(BaseModel):
    """One literature reference value for one ``(material, gas, T_K)``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.4"] = "0.4"
    material_id: str = Field(..., min_length=1)
    gas: Literal["CO2", "N2", "CH4", "H2", "H2O", "Xe", "Kr", "Ar", "C2H4", "C2H6", "C3H8", "CO"]
    temperature_K: float = Field(..., gt=0)
    pressure_Pa: float | None = Field(default=None, gt=0)
    KH_value: float | None = None
    KH_units: Literal["mol/(kg*Pa)", "mmol/(g*bar)", "cm3_STP/(g*Torr)"] | None = None
    Qads_value: float | None = None
    Qads_units: Literal["kJ/mol", "kcal/mol", "K"] | None = None
    Qads_sign_convention: Literal[
        "binding_strength_positive",
        "interaction_energy_negative",
    ] = "binding_strength_positive"
    provenance: ReferenceProvenance


class SiteReferenceEntry(BaseModel):
    """One crystallographically reported gas-loaded binding site."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.4"] = "0.4"
    material_id: str = Field(..., min_length=1)
    gas: Literal["CO2", "N2", "CH4", "H2", "H2O", "Xe", "Kr", "Ar", "C2H4", "C2H6", "C3H8", "CO"]
    label: str = Field(..., min_length=1, description="e.g. 'OMS-A', 'OMS-B', 'sodalite_cage_centre'")
    centroid_frac: tuple[float, float, float] = Field(
        ..., description="Wyckoff representative in fractional coordinates (0 ≤ x < 1)"
    )
    coordination_distance_A: float | None = Field(default=None, gt=0)
    site_kind: Literal[
        "open_metal_site",
        "cage_centre",
        "channel_centre",
        "linker_pocket",
        "alpha_pocket",
        "secondary",
        "other_see_notes",
    ]
    notes: str | None = None
    provenance: ReferenceProvenance

    @field_validator("centroid_frac")
    @classmethod
    def _check_frac(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        for x in v:
            if not 0.0 <= x < 1.0:
                raise ValueError(f"centroid_frac must lie in [0, 1); got {v}")
        return v


class ThresholdSet(BaseModel):
    """One named set of validation thresholds (v0.4 / v0.5 / flagship)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    KH_relative_error_upper: float = Field(..., gt=0, description="±X (fractional) tolerated; e.g. 0.20 for 20%")
    KH_log_error_upper: float = Field(..., gt=0, description="±X log units")
    Qads_abs_error_kJmol_upper: float = Field(..., gt=0, description="±X kJ/mol")
    basin_centroid_max_distance_A: float = Field(..., gt=0)
    basin_weight_repeatability_upper: float = Field(..., gt=0, le=1.0)
    convergence_min_insertions_KH: int = Field(..., ge=1000)
    convergence_min_insertions_Qads: int = Field(..., ge=1000)
    backend_agreement_max_diff_fraction: float = Field(..., gt=0, le=1.0)
    notes: str | None = None


class ValidationThresholds(BaseModel):
    """Named sets of validation thresholds (v0.4 minimum, v0.5 broader, flagship)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.4"] = "0.4"
    sets: dict[str, ThresholdSet]


__all__ = [
    "CitationEntry",
    "ContentSummary",
    "DatasetKind",
    "DatasetRegistryEntry",
    "LicenseTag",
    "RedistributionStatus",
    "ReferenceProvenance",
    "ScalarReferenceEntry",
    "SiteReferenceEntry",
    "ThresholdSet",
    "ValidationThresholds",
]
