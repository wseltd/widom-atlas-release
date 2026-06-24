"""Tests for benchmarks/mofxdb.py (T045)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from widom_atlas.benchmarks.mofxdb import (
    MOFXDBRecord,
    _cache_path,
    load_mofxdb_scalars,
)


def test_load_mofxdb_scalars_offline_cache_hit_returns_record(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    p = _cache_path(cache, "UiO-66", "CO2")
    payload = MOFXDBRecord(
        material_id="UiO-66",
        gas="CO2",
        temperature_K=298.15,
        KH=1.2e-3,
        Qads=18.0,
        identity_confidence="high",
        license="CC BY 4.0",
        sha256="0" * 64,
        dataset_version="v1",
    ).model_dump(mode="json")
    p.write_text(json.dumps(payload), encoding="utf-8")
    rec = load_mofxdb_scalars("UiO-66", "CO2", cache)
    assert rec.KH == 1.2e-3 and rec.Qads == 18.0


def test_load_mofxdb_scalars_unknown_identity_marks_low_confidence(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    rec = load_mofxdb_scalars("Unknown-Material", "CO2", cache)
    assert rec.identity_confidence == "low"


def test_load_mofxdb_scalars_records_sha256_and_license(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    p = _cache_path(cache, "ZIF-8", "CO2")
    payload = MOFXDBRecord(
        material_id="ZIF-8",
        gas="CO2",
        temperature_K=298.15,
        KH=1.0e-4,
        Qads=20.0,
        identity_confidence="high",
        license="MIT",
        sha256="a" * 64,
        dataset_version="v2",
    ).model_dump(mode="json")
    p.write_text(json.dumps(payload), encoding="utf-8")
    rec = load_mofxdb_scalars("ZIF-8", "CO2", cache)
    assert rec.sha256 == "a" * 64
    assert rec.license == "MIT"


def test_mofxdb_record_validates_units_field() -> None:
    rec = MOFXDBRecord(material_id="UiO-66", gas="CO2", temperature_K=298.15, identity_confidence="high")
    assert rec.KH_units == "mol/(kg*Pa)"
    assert rec.Qads_units == "kJ/mol"


def test_load_mofxdb_scalars_offline_mode_does_not_perform_network_io(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    with patch("urllib.request.urlopen") as mock:
        load_mofxdb_scalars("UiO-66", "CO2", cache)
        mock.assert_not_called()
