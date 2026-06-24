"""Strict external-sample manifest schema (v0.3).

Per ``implementation-verdict-continuation.txt`` §"Implement external sample
schema", every external sample fed into widom-atlas must arrive with
explicit provenance: which engine produced it, with what force field, what
units the energies are in, what citations it carries, and what its
redistribution status is. Units are mandatory; if any are missing the
ingest fails (no silent guessing).

This module defines the schema as Pydantic v2 models with
``model_config = ConfigDict(extra="forbid", frozen=True)`` so anything
unexpected is loud at parse time.

The schema is the contract between widom-atlas's atlas / reporting layer
and any external engine (RASPA3, OpenMM, kUPS, ML-FF, custom). It is
versioned (``sample_format_version="0.3"``) so future revisions are
explicit.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .units import ALLOWED_ENERGY_UNITS, EnergyUnit

SAMPLE_FORMAT_VERSION: Literal["0.3"] = "0.3"

ParameterMode = Literal[
    "external_samples",
    "user_supplied",
    "literature_preset",
    "neutral_lj_only",
    "parameterised_lj",
    "user_parameterised_coulomb_lj",
]

BackendCategory = Literal[
    "toy_lj",
    "parameterised_lj",
    "user_parameterised_coulomb_lj",
    "external_samples",
    "raspa3_external",
    "ml_external",
]

RedistributionStatus = Literal[
    "bundled_safe",
    "user_supplied_not_bundled",
    "user_supplied_not_redistributed",
    "open_access_with_attribution",
    "unknown",
]

ElectrostaticsTreatment = Literal[
    "Ewald",
    "Wolf",
    "PME",
    "none",
    "external_engine",
    "unknown",
]


class CitationEntry(BaseModel):
    """One DOI + role tuple. ``role`` describes which component the citation backs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal[
        "gas_model",
        "framework_lj",
        "framework_charges",
        "engine",
        "training_data",
        "validation",
        "other",
    ]
    doi: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, description="Plain-text citation incl. authors/year/journal")


class ForceFieldDescriptor(BaseModel):
    """Structured description of the force field that produced the samples.

    This is *descriptive*, not prescriptive — the engine has already done
    the work; we record what it used so the report can show provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    framework_lj: str = Field(
        ...,
        description='e.g. "UFF" / "DREIDING" / "UFF4MOF" / "user_supplied" / "unknown"',
    )
    framework_charges: str = Field(
        ...,
        description='e.g. "DDEC" / "Qeq" / "EQeq" / "PACMOF" / "user_supplied" / "none" / "unknown"',
    )
    gas_model: str = Field(
        ...,
        description='e.g. "TraPPE-CO2" / "TraPPE-N2" / "TraPPE-CH4" / "user_supplied" / "unknown"',
    )
    mixing_rules: str = Field(
        ...,
        description='e.g. "Lorentz-Berthelot" / "user_supplied" / "unknown"',
    )
    electrostatics: ElectrostaticsTreatment


class ExternalSampleManifest(BaseModel):
    """The strict v0.3 external-samples schema.

    Either lives as a JSON file paired with an `.npz`, or embedded as
    ``metadata_json`` inside the npz itself (legacy mode is auto-detected).

    All required fields fail loudly if missing or wrongly typed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_format_version: Literal["0.3"] = SAMPLE_FORMAT_VERSION
    framework: str = Field(..., min_length=1)
    gas: Literal["CO2", "N2", "CH4"]
    temperature_K: float = Field(..., gt=0)
    backend: BackendCategory
    backend_version: str = Field(default="unknown")
    n_insertions: int = Field(..., gt=0)
    random_seed: int | None = None

    # Units must be declared explicitly; downstream code converts via
    # widom_atlas.backends.units.to_eV.
    energy_unit: EnergyUnit
    distance_unit: Literal["A", "Angstrom"] = "A"

    parameter_mode: ParameterMode
    force_field: ForceFieldDescriptor
    citations: list[CitationEntry] = Field(default_factory=list)
    redistribution_status: RedistributionStatus

    # Free-form warnings (e.g. "TraPPE-CO2 mixed with UFF framework — hybrid
    # approximation; not a published validated FF"). Always present, may be
    # empty.
    warnings: list[str] = Field(default_factory=list)

    # Validity / interpretability — set by the producer based on whether
    # the run is a smoke test or a real calculator run.
    suitable_for_quantitative_interpretation: bool = False

    # Optional hashes / paths (set by ingest commands).
    samples_path: str | None = None
    samples_sha256: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("energy_unit", mode="before")
    @classmethod
    def _check_energy_unit(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(
                "energy_unit is required; one of "
                f"{ALLOWED_ENERGY_UNITS}. widom-atlas does not silently assume units."
            )
        return v

    @model_validator(mode="after")
    def _check_charge_aware_consistency(self) -> ExternalSampleManifest:
        """If the run claims charge-aware electrostatics, the FF descriptor
        must declare a non-`none` framework_charges + electrostatics field.
        """
        if self.parameter_mode == "user_parameterised_coulomb_lj":
            if self.force_field.electrostatics in ("none", "unknown"):
                raise ValueError(
                    "user_parameterised_coulomb_lj manifests must declare "
                    "force_field.electrostatics as one of "
                    "Ewald | Wolf | PME | external_engine; got "
                    f"{self.force_field.electrostatics!r}."
                )
            if self.force_field.framework_charges in ("none", "unknown"):
                raise ValueError(
                    "user_parameterised_coulomb_lj manifests must declare "
                    "force_field.framework_charges (DDEC / Qeq / EQeq / "
                    "PACMOF / user_supplied); got "
                    f"{self.force_field.framework_charges!r}."
                )
        return self


def manifest_summary(manifest: ExternalSampleManifest) -> dict[str, Any]:
    """Produce a flat dict of the manifest fields suitable for embedding into
    ``AtlasInput.metadata["external_sample_manifest"]`` and stamping into
    every ``benchmark_run.json``."""
    return manifest.model_dump(mode="json")


__all__ = [
    "SAMPLE_FORMAT_VERSION",
    "BackendCategory",
    "CitationEntry",
    "ElectrostaticsTreatment",
    "ExternalSampleManifest",
    "ForceFieldDescriptor",
    "ParameterMode",
    "RedistributionStatus",
    "manifest_summary",
]
