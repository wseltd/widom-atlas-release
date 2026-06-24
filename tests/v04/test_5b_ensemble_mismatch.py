"""Regression tests for the 5b Na-Rho ensemble-mismatch known open problem reclassification."""
from __future__ import annotations

from pathlib import Path

from widom_atlas.v04.locked_inputs import load_locked_case_matrix

REPO = Path(__file__).resolve().parents[2]


def _find_branch(cm, branch_id):
    for case in cm.cases:
        for b in case["branches"]:
            if b.get("branch_id") == branch_id:
                return b
    return None


def test_5b_carries_ensemble_mismatch_known_open_problem_block():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "5b")
    assert b is not None
    block = b.get("ensemble_mismatch_known_open_problem_2026_06_01")
    assert block is not None
    assert block["classification"] == "ensemble_mismatch_known_open_problem"
    assert block["excluded_from_strict_gate"] is True
    assert block["excluded_from_tier_a"] is True
    assert block["excluded_from_tier_b"] is True
    assert block["site_truth_axis_remains_active"] is True
    assert block["do_not_treat_as_force_field_bug"] is True
    assert block["do_not_treat_as_evaluator_bug"] is True


def test_5b_cites_coudert_2017_gating_mechanism():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "5b")
    block = b["ensemble_mismatch_known_open_problem_2026_06_01"]
    assert block["gating_mechanism_reference"]["doi"] == "10.1021/acs.chemmater.6b03837"
    assert "Coudert" in block["gating_mechanism_reference"]["citation"]


def test_5b_cites_witman_2018_flat_histogram_route():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "5b")
    block = b["ensemble_mismatch_known_open_problem_2026_06_01"]
    fh = block["principled_paths_to_open_state_K_H"]["flat_histogram_route"]
    assert fh["doi_a"] == "10.1021/acs.jctc.8b00534"
    assert "FEASST" in fh["open_implementation"]
    assert fh["status_for_widom_atlas"] == "research_project_post_v0_5_scope"


def test_5b_lozinska_finalisation_block_still_present():
    """The 2026-05-28 finalisation block must remain — 5b stays scalar METHOD_BLOCKED."""
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    b = _find_branch(cm, "5b")
    fin = b.get("lozinska_2012_packet_2026_05_28_finalisation")
    assert fin is not None
    assert fin["K_H_final_classification"] == "method_blocked_or_reference_blocked"
    assert fin["site_truth_status"] == "active"
