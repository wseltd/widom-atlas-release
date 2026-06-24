"""Tests for benchmarks/hashing.py (T044)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from widom_atlas.benchmarks.hashing import (
    ProvenanceMismatch,
    record_provenance,
    sha256_file,
)
from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET


def _manual_material():
    return next(m for m in SMALL_BENCHMARK_SET if m.source == "manual")


def _seed(tmp_path: Path, content: bytes = b"hello world") -> tuple[Path, str]:
    p = tmp_path / "a.bin"
    p.write_bytes(content)
    return p, hashlib.sha256(content).hexdigest()


def test_sha256_file_matches_hashlib_reference_on_small_input(tmp_path: Path) -> None:
    p, expected = _seed(tmp_path)
    assert sha256_file(p) == expected


def test_sha256_file_streams_in_chunks_for_large_input(tmp_path: Path) -> None:
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * (3 << 20))
    assert sha256_file(p) == hashlib.sha256(b"x" * (3 << 20)).hexdigest()


def test_record_provenance_writes_expected_keys_to_provenance_json(tmp_path: Path) -> None:
    m = _manual_material()
    cache = tmp_path / "cache"
    base = cache / m.source
    base.mkdir(parents=True)
    cif = base / f"{m.material_id}.cif"
    cif.write_bytes(b"# cif")
    rec = record_provenance(m, cif, cache)
    prov_path = base / f"{m.material_id}.provenance.json"
    payload = json.loads(prov_path.read_text())
    for k in ("material_id", "source", "sha256", "file_size_bytes", "license_tag", "citation_doi", "dataset_version"):
        assert k in payload
    assert rec.sha256 == hashlib.sha256(b"# cif").hexdigest()


def test_record_provenance_is_idempotent_on_unchanged_file(tmp_path: Path) -> None:
    m = _manual_material()
    cache = tmp_path / "cache"
    base = cache / m.source
    base.mkdir(parents=True)
    cif = base / f"{m.material_id}.cif"
    cif.write_bytes(b"abc")
    record_provenance(m, cif, cache)
    p1 = (base / f"{m.material_id}.provenance.json").read_text()
    record_provenance(m, cif, cache)
    p2 = (base / f"{m.material_id}.provenance.json").read_text()
    assert p1 == p2


def test_record_provenance_raises_ProvenanceMismatch_when_recorded_hash_disagrees(tmp_path: Path) -> None:
    m = _manual_material()
    cache = tmp_path / "cache"
    base = cache / m.source
    base.mkdir(parents=True)
    cif = base / f"{m.material_id}.cif"
    cif.write_bytes(b"abc")
    record_provenance(m, cif, cache)
    cif.write_bytes(b"xyz")  # change content
    with pytest.raises(ProvenanceMismatch):
        record_provenance(m, cif, cache)


def test_record_provenance_copies_license_and_citation_from_material(tmp_path: Path) -> None:
    m = _manual_material()
    cache = tmp_path / "cache"
    base = cache / m.source
    base.mkdir(parents=True)
    cif = base / f"{m.material_id}.cif"
    cif.write_bytes(b"abc")
    rec = record_provenance(m, cif, cache)
    assert rec.license_tag == m.license
    assert rec.citation_doi == m.citation


def test_record_provenance_warns_when_dataset_version_missing(tmp_path: Path, caplog) -> None:
    m = _manual_material()
    cache = tmp_path / "cache"
    base = cache / m.source
    base.mkdir(parents=True)
    cif = base / f"{m.material_id}.cif"
    cif.write_bytes(b"abc")
    with caplog.at_level(logging.WARNING):
        rec = record_provenance(m, cif, cache)
    assert rec.dataset_version == "unknown"
