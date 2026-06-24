"""Tests for RobustnessReport (T011)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from widom_atlas.core.models import (
    PerturbationSpec,
    RobustnessMetrics,
    RobustnessReport,
)


def _metrics() -> RobustnessMetrics:
    return RobustnessMetrics(
        delta_ln_KH=None,
        delta_Qads_kJmol=None,
        basin_count_pristine=3,
        basin_count_perturbed=3,
        basin_count_change=0,
        basin_persistence_fraction=1.0,
        basin_splitting_count=0,
        mean_basin_displacement_A=0.05,
        accessibility_change=0.0,
        ambiguity_flags=[],
        missing_data_flags=[],
    )


def _spec() -> PerturbationSpec:
    return PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1")


def _kw(**overrides) -> dict:
    base = {
        "report_id": "r1",
        "structure_id": "UiO-66",
        "gas": "CO2",
        "temperature_K": 298.15,
        "pristine_run_id": "p1",
        "perturbations": [_spec()],
        "metrics_per_perturbation": [_metrics()],
        "summary": {"worst_delta_ln_KH": None},
        "caveats": ["toy fixture"],
    }
    base.update(overrides)
    return base


def test_robustness_report_perturbations_and_metrics_aligned() -> None:
    r = RobustnessReport(**_kw())
    assert len(r.perturbations) == len(r.metrics_per_perturbation) == 1


def test_robustness_report_rejects_misaligned_lengths() -> None:
    with pytest.raises(ValidationError):
        RobustnessReport(**_kw(perturbations=[_spec(), _spec()], metrics_per_perturbation=[_metrics()]))


def test_robustness_report_rejects_h2o_gas() -> None:
    with pytest.raises(ValidationError):
        RobustnessReport(**_kw(gas="H2O"))


def test_robustness_report_temperature_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RobustnessReport(**_kw(temperature_K=0.0))
    with pytest.raises(ValidationError):
        RobustnessReport(**_kw(temperature_K=-1.0))


def test_robustness_report_schema_version_pinned() -> None:
    r = RobustnessReport(**_kw())
    assert r.schema_version == "1"


def test_robustness_report_roundtrip_json() -> None:
    r = RobustnessReport(**_kw())
    js = r.model_dump_json()
    reloaded = RobustnessReport.model_validate_json(js)
    assert reloaded.report_id == r.report_id
    assert reloaded.gas == r.gas
    assert len(reloaded.perturbations) == 1
