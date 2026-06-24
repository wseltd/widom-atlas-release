"""Tests for RunManifest (T012)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from widom_atlas.core.models import RunManifest

_HEX = "0" * 64


def _kw(**overrides) -> dict:
    base = {
        "manifest_version": "1",
        "run_id": "r1",
        "package_version": "0.1.0",
        "python_version": "3.12.0",
        "platform": "linux",
        "dependency_versions": {"numpy": "1.26.0"},
        "structure_id": "UiO-66",
        "structure_source": "local",
        "structure_sha256": _HEX,
        "input_samples_sha256": _HEX,
        "gas": "CO2",
        "temperature_K": 298.15,
        "parameters": {"symprec": 1e-2},
        "dataset_source": "CoRE-MOF-2019",
        "dataset_license": "CC BY 4.0",
    }
    base.update(overrides)
    return base


def test_run_manifest_rejects_non_hex_sha256() -> None:
    with pytest.raises(ValidationError):
        RunManifest(**_kw(structure_sha256="z" * 64))


def test_run_manifest_rejects_short_sha256() -> None:
    with pytest.raises(ValidationError):
        RunManifest(**_kw(structure_sha256="0" * 63))


def test_run_manifest_temperature_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RunManifest(**_kw(temperature_K=0.0))


def test_run_manifest_gas_restricted_to_co2_n2_ch4() -> None:
    with pytest.raises(ValidationError):
        RunManifest(**_kw(gas="H2O"))
    with pytest.raises(ValidationError):
        RunManifest(**_kw(gas="Ar"))


def test_run_manifest_parameters_must_be_json_serialisable() -> None:
    with pytest.raises(ValidationError):
        RunManifest(**_kw(parameters={"obj": object()}))


def test_run_manifest_roundtrip_json() -> None:
    m = RunManifest(**_kw())
    js = m.model_dump_json()
    reloaded = RunManifest.model_validate_json(js)
    assert reloaded.run_id == m.run_id
    assert reloaded.structure_sha256 == _HEX


def test_run_manifest_schema_version_pinned() -> None:
    m = RunManifest(**_kw())
    assert m.manifest_version == "1"
