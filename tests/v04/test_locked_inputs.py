"""Tests for T002 locked_inputs module."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from widom_atlas.v04 import locked_inputs as li

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_spec_pinned_digest_matches_file() -> None:
    """V04_LOCKED_SPEC.md sha256 must equal the pinned constant."""
    path = REPO_ROOT / "V04_LOCKED_SPEC.md"
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == li.SPEC_SHA256


def test_yaml_pinned_digest_matches_file() -> None:
    """v04_case_matrix.yaml sha256 must equal the pinned constant."""
    path = REPO_ROOT / "v04_case_matrix.yaml"
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == li.CASE_MATRIX_SHA256


def test_load_locked_spec_returns_correct_text() -> None:
    spec = li.load_locked_spec(REPO_ROOT / "V04_LOCKED_SPEC.md")
    assert spec.sha256 == li.SPEC_SHA256
    assert "widom-atlas v0.4 — Locked Specification" in spec.text


def test_load_locked_case_matrix_parses_v04_2() -> None:
    matrix = li.load_locked_case_matrix(REPO_ROOT / "v04_case_matrix.yaml")
    assert matrix.sha256 == li.CASE_MATRIX_SHA256
    assert matrix.version == "v04.2"
    assert len(matrix.cases) == 6
    # 22 total branches (2026-06-01 final pivot: added 4 5c
    # reference_audited_pending_cation_cif_and_ff_lock branches:
    # 5c_NaZK5_CO2_303K, 5c_Zeolite5A_CaA_CO2_298K, 5c_Zeolite13X_NaX_CO2_273K,
    # 5c_Zeolite4A_NaA_CO2_273K. Prior 18 included 1d added 2026-05-19 pass-5 R6;
    # prior 17 with 4c + 6e added 2026-05-19 pass-1; prior 15 with 1c added
    # 2026-05-18; baseline 14).
    total = sum(len(c.get("branches", [])) for c in matrix.cases)
    assert total == 22


def test_verify_digest_raises_on_mismatch(tmp_path: Path) -> None:
    """If the file is mutated, LockedDigestMismatch must fire."""
    fake = tmp_path / "spec.md"
    fake.write_text("not the locked spec")
    with pytest.raises(li.LockedDigestMismatch):
        li.verify_digest(fake, li.SPEC_SHA256)


def test_load_locked_spec_raises_on_mutated_file(tmp_path: Path) -> None:
    fake = tmp_path / "V04_LOCKED_SPEC.md"
    fake.write_text("mutated content")
    with pytest.raises(li.LockedDigestMismatch):
        li.load_locked_spec(fake)


def test_all_22_branch_ids_present() -> None:
    """Sanity: matrix must contain every expected branch id.
    History: 14 baseline → 15 (1c added 2026-05-18) → 17 (4c + 6e added
    2026-05-19 pass-1 as refined-FF sibling branches, both deferred) →
    18 (1d Mercado 2016 Model 4 added 2026-05-19 pass-5 R6) →
    22 (4 5c replacement-scalar branches added 2026-06-01 final pivot
    as reference_audited_pending_cation_cif_and_ff_lock)."""
    matrix = li.load_locked_case_matrix(REPO_ROOT / "v04_case_matrix.yaml")
    ids: set[str] = set()
    for case in matrix.cases:
        for branch in case.get("branches", []):
            ids.add(branch["branch_id"])
    expected = {
        "1a", "1b", "1c", "1d", "2a", "2b", "3a", "3b", "4a", "4b", "4c",
        "5a", "5b",
        "5c_NaZK5_CO2_303K", "5c_Zeolite5A_CaA_CO2_298K",
        "5c_Zeolite13X_NaX_CO2_273K", "5c_Zeolite4A_NaA_CO2_273K",
        "6a", "6b", "6c", "6d", "6e",
    }
    assert ids == expected
