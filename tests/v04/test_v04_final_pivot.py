"""Regression tests for the v0.4 final-pivot deliverables.

Asserts the case-matrix hash roll, 3b promotion, 5c branch presence,
4c/6e blocked sentinel, 5b unchanged METHOD_BLOCKED state, and
PROVENANCE_MANIFEST.json schema + completeness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from widom_atlas.v04.locked_inputs import (
    CASE_MATRIX_SHA256,
    load_locked_case_matrix,
)

REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_case_matrix_pinned_sha_matches_file_sha():
    actual = _sha256(REPO / "v04_case_matrix.yaml")
    assert actual == CASE_MATRIX_SHA256, (
        f"v04_case_matrix.yaml SHA mismatch: file {actual} != pin {CASE_MATRIX_SHA256}"
    )


def test_load_locked_case_matrix_succeeds():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    assert cm.sha256 == CASE_MATRIX_SHA256
    assert cm.version == "v04.2"
    assert len(cm.cases) == 6


def _find_branch(cm, branch_id):
    for case in cm.cases:
        for branch in case["branches"]:
            if branch.get("branch_id") == branch_id:
                return branch
    return None


def test_3b_is_locked_strict_executed_multi_variant():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "3b")
    assert b is not None
    assert b["status"] == "locked_strict_executed_multi_variant"
    assert b["executed_backend"] == "native_widom_v04"
    assert "UA" in b["executed_variants"]
    assert "UAq" in b["executed_variants"]
    assert "EHq" in b["executed_variants"]
    assert b["primary_executed_variant"] == "UA"


def test_3b_maia_2023_data_block_has_all_three_tables():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "3b")
    data = b["maia_2023_full_data_recovered_2026_06_01"]
    assert data["doi"] == "10.3390/cryst13101523"
    assert data["license"] == "MDPI CC-BY-4.0 (open access)"
    # Verbatim Maia table entries
    assert data["table_2_UA_UAq_framework_parameters"]["Zr"]["sigma_angstrom"] == 2.78
    assert data["table_2_UA_UAq_framework_parameters"]["Zr"]["epsilon_over_kB_K"] == 34.72
    assert data["table_2_UA_UAq_framework_parameters"]["Zr"]["charge_UAq_e"] == 2.008
    # Table 3 distinct C25 (smaller because H explicit)
    assert data["table_3_EHq_framework_parameters"]["C25"]["sigma_angstrom"] == 3.60
    assert data["table_3_EHq_framework_parameters"]["H1"]["sigma_angstrom"] == 2.36


def test_5c_branches_present_with_do_not_call_validation_of_5b():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    expected_5c_branches = {
        "5c_NaZK5_CO2_303K",
        "5c_Zeolite5A_CaA_CO2_298K",
        "5c_Zeolite13X_NaX_CO2_273K",
        "5c_Zeolite4A_NaA_CO2_273K",
    }
    found = set()
    for case in cm.cases:
        for branch in case["branches"]:
            if branch.get("branch_id", "").startswith("5c_"):
                found.add(branch["branch_id"])
                assert branch["status"] == "reference_audited_pending_cation_cif_and_ff_lock"
                assert branch.get("do_not_call_a_validation_of_5b") is True
    assert found == expected_5c_branches


def test_4c_and_6e_now_executed_under_bai_2013():
    """Updated 2026-06-01 addendum: 4c + 6e were PROMOTED to locked_strict_executed
    after RASPA3-bundled-JSON primary-anchored Bai 2013 parameters were verified."""
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    for branch_id in ("4c", "6e"):
        b = _find_branch(cm, branch_id)
        assert b is not None
        assert b["status"] == "locked_strict_executed"
        assert b["executed_backend"] == "native_widom_v04"
        # Was previously blocked_pending — now records the resolution path
        resolved = b.get("previously_blocked_artefact_now_resolved_2026_06_01")
        assert resolved is not None
        assert "RASPA3" in resolved["resolution_path"]


def test_6e_records_shah_2015_cross_check_in_execution_result():
    """Shah 2015 smoke-test value is now recorded in the execution_result block."""
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "6e")
    result = b["execution_result_2026_06_01"]
    smoke = result["smoke_test_cross_check"]
    assert "Shah 2015" in smoke["source"]
    assert 0.5 < smoke["simulated_K_H_mol_per_kg_per_bar"] < 0.7
    assert 0.4 < smoke["atlas_K_H_mol_per_kg_per_bar"] < 0.5


def test_5b_unchanged_method_blocked_scalar_site_truth_active():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "5b")
    # The finalisation block from pass-9 must still be present
    assert "lozinska_2012_packet_2026_05_28_finalisation" in b
    fin = b["lozinska_2012_packet_2026_05_28_finalisation"]
    assert fin["K_H_final_classification"] == "method_blocked_or_reference_blocked"
    assert fin["Q_st_final_classification"] == "method_blocked_or_reference_blocked"
    assert fin["site_truth_status"] == "active"


def test_5b_replacement_branches_do_not_validate_5b():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "5b")
    rep = b["lozinska_2012_packet_2026_05_28_finalisation"]["replacement_branches_v0_5_candidates"]
    assert rep["do_not_call_replacement_a_validation_of_5b"] is True


def test_provenance_manifest_exists_with_no_missing_files():
    path = REPO / "docs/research/dataset-research-for-v0.4/PROVENANCE_MANIFEST.json"
    assert path.exists()
    with path.open() as fp:
        manifest = json.load(fp)
    assert manifest["manifest_version"] == "v0.4_final_pivot_2026_06_01"
    entries = manifest["entries"]
    assert len(entries) >= 20
    missing = [e for e in entries if e.get("sha256") == "MISSING_FILE"]
    assert missing == [], f"PROVENANCE_MANIFEST missing files: {missing}"
    # Every entry has required fields
    for e in entries:
        assert "path" in e
        assert "role" in e
        assert "sha256" in e
        assert "bytes" in e
        assert e["bytes"] > 0
        # SHA256 is 64 hex chars
        assert len(e["sha256"]) == 64
        assert all(c in "0123456789abcdef" for c in e["sha256"])


def test_provenance_manifest_3b_maia_pdf_archived():
    path = REPO / "docs/research/dataset-research-for-v0.4/PROVENANCE_MANIFEST.json"
    with path.open() as fp:
        manifest = json.load(fp)
    maia_entries = [
        e for e in manifest["entries"]
        if "maia_2023_crystals_v04.pdf" in e["path"]
    ]
    assert len(maia_entries) == 1
    maia = maia_entries[0]
    assert maia["doi"] == "10.3390/cryst13101523"
    assert maia["license"].startswith("CC-BY-4.0")
    assert "Table 1" in str(maia.get("extracted_tables", []))
    assert "Table 2" in str(maia.get("extracted_tables", []))
    assert "Table 3" in str(maia.get("extracted_tables", []))


def test_provenance_manifest_bai_si_marked_simulated_only():
    path = REPO / "docs/research/dataset-research-for-v0.4/PROVENANCE_MANIFEST.json"
    with path.open() as fp:
        manifest = json.load(fp)
    bai_entries = [
        e for e in manifest["entries"]
        if "jp4074224_si_001_simulated_isotherms_only" in e["path"]
    ]
    assert len(bai_entries) == 1
    bai = bai_entries[0]
    assert "simulated_isotherms_only" in bai["role"]
    assert "blocked" in bai["branch"].lower()


def test_provenance_manifest_shah_2015_smoke_test_classification():
    path = REPO / "docs/research/dataset-research-for-v0.4/PROVENANCE_MANIFEST.json"
    with path.open() as fp:
        manifest = json.load(fp)
    shah_entries = [
        e for e in manifest["entries"]
        if "shah_2015_langmuir_si.pdf" in e["path"]
    ]
    assert len(shah_entries) == 1
    shah = shah_entries[0]
    assert "smoke_test_or_parity" in shah["role"]
    assert "NOT_experimental_truth" in shah["role"]


def test_strict_denominator_is_at_least_13_post_pivot():
    """3b promotion brings the denominator to 13 (12 → 13)."""
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    strict_executed_count = 0
    for case in cm.cases:
        for branch in case["branches"]:
            status = branch.get("status", "")
            if status in (
                "locked_strict",
                "locked_strict_executed",
                "locked_strict_executed_multi_variant",
            ) and branch.get("affects_v04_verdict") in (True, None):
                strict_executed_count += 1
    assert strict_executed_count >= 13, (
        f"Expected >=13 strict-counted branches after 3b promotion, got {strict_executed_count}"
    )


def test_v04_pivot_deliverable_docs_exist():
    """The 3 new deliverable docs must all exist."""
    expected_docs = [
        "V04_FINAL_PIVOT_EXECUTION_REPORT.md",
        "V04_3B_MAIA_2023_EXECUTION_AUDIT.md",
        "V04_5C_REPLACEMENT_BRANCH_AUDIT.md",
    ]
    for d in expected_docs:
        assert (REPO / d).exists(), f"missing deliverable doc: {d}"


def test_3b_verdict_json_exists_and_has_all_three_variants():
    """3b combined verdict JSON should have UA, UAq, EHq sections."""
    verdict_path = REPO / "evidence/v04_3b_maia/verdicts/3b.json"
    assert verdict_path.exists()
    with verdict_path.open() as fp:
        v = json.load(fp)
    assert v["branch_id"] == "3b"
    assert v["executed_backend"] == "native_widom_v04"
    assert set(v["variants"].keys()) == {"UA", "UAq", "EHq"}
    for variant in ("UA", "UAq", "EHq"):
        agg = v["variants"][variant]["aggregated"]
        assert "K_H_mean_mol_per_kg_per_bar" in agg
        assert "Q_st_mean_kJ_per_mol" in agg
        assert agg["K_H_mean_mol_per_kg_per_bar"] > 0
        assert agg["Q_st_mean_kJ_per_mol"] > 0


def test_5c_verdict_summary_lists_all_8_candidates():
    """5c audit summary must include all 8 audited candidates."""
    path = REPO / "evidence/v04_5c/verdicts/5c_audit_summary.json"
    assert path.exists()
    with path.open() as fp:
        summary = json.load(fp)
    assert summary["n_candidates"] == 8
    assert "5c_NaZK5_CO2_303K_PRIMARY" in summary["candidates"]
    assert "5c_Zeolite5A_CaA_CO2_298K_fallback" in summary["candidates"]
