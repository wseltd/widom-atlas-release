"""Regression tests for Bai 2013 TraPPE-zeo loader + 4c/6e execution results."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from widom_atlas.v04.locked_inputs import load_locked_case_matrix
from widom_atlas.v04.native.bai_2013_trappe_zeo_loader import (
    BAI_2013_FRAMEWORK_CHARGES_E,
    BAI_2013_FRAMEWORK_LJ,
    TRAPPE_CO2_SELF_LJ,
    TRAPPE_UA_CH4_SELF_LJ,
    load_bai_2013_native_system,
)


REPO = Path(__file__).resolve().parents[2]


# ---------- Parameter table verbatim ----------------------------------
def test_bai_2013_Si_LJ_matches_raspa3_bundled_22K_2p3A():
    eps, sig = BAI_2013_FRAMEWORK_LJ["Si_zeo"]
    assert eps == 22.0
    assert sig == 2.3


def test_bai_2013_O_LJ_matches_raspa3_bundled_53K_3p3A():
    """Critical: ε_O is 53 K (Bai 2013), NOT 93 K (Calero/Garcia-Perez).
    The deep researcher's first proposal of 93 K was caught as a hybrid
    Calero-family error via the experimental cross-validation falsification
    test on 4c (RASPA3-bundled +2.05/-1.025 charges produced K_H = 2.22
    matching Maghsoudi 2013 within ±0.04; a 93 K O would over-bind)."""
    eps, sig = BAI_2013_FRAMEWORK_LJ["O_zeo"]
    assert eps == 53.0, f"Bai 2013 O ε is 53.0 K, not {eps}"
    assert sig == 3.3


def test_bai_2013_charges_are_raspa3_bundled_2p05_neg1p025():
    """Critical: Bai charges are Si=+2.05, O=-1.025 per RASPA3-bundled JSON.
    The deep researcher proposed (and then revised) Si=+1.20/-0.60 then
    Si=+1.30/-0.65; both rejected as hybrid Calero-family inferences not
    matching the RASPA3-bundled distribution which carries the verbatim
    Bai 2013 DOI source citation."""
    assert BAI_2013_FRAMEWORK_CHARGES_E["Si_zeo"] == +2.05
    assert BAI_2013_FRAMEWORK_CHARGES_E["O_zeo"] == -1.025
    # Electroneutrality per SiO2 unit
    assert BAI_2013_FRAMEWORK_CHARGES_E["Si_zeo"] + 2 * BAI_2013_FRAMEWORK_CHARGES_E["O_zeo"] == pytest.approx(0.0)


def test_trappe_co2_gas_lj_matches_record_116():
    assert TRAPPE_CO2_SELF_LJ["C_co2"] == (27.0, 2.80)
    assert TRAPPE_CO2_SELF_LJ["O_co2"] == (79.0, 3.05)


def test_trappe_ua_ch4_matches_record_1():
    assert TRAPPE_UA_CH4_SELF_LJ == (148.0, 3.73)


# ---------- Loader behavior --------------------------------------------
def test_loader_assembles_si_cha_co2_with_correct_atom_counts():
    cif = REPO / "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
    sys, meta = load_bai_2013_native_system(REPO, cif, "CO2")
    assert sys.n_framework_atoms == 108  # 36 Si + 72 O per Si-CHA primitive cell
    counts = meta["framework_atom_type_counts_per_primitive_cell"]
    assert counts["Si_zeo"] == 36
    assert counts["O_zeo"] == 72
    # Si:O = 1:2 stoichiometry
    assert counts["O_zeo"] == 2 * counts["Si_zeo"]
    # Electroneutrality
    import numpy as np
    assert abs(float(np.sum(sys.framework_charges_e))) < 1e-9


def test_loader_assembles_mfi_ch4_with_correct_atom_counts():
    cif = REPO / "docs/research/dataset-research-for-v0.4/7/MFI_iza.cif"
    sys, meta = load_bai_2013_native_system(REPO, cif, "CH4")
    assert sys.n_framework_atoms == 288  # 96 Si + 192 O per MFI primitive cell
    counts = meta["framework_atom_type_counts_per_primitive_cell"]
    assert counts["Si_zeo"] == 96
    assert counts["O_zeo"] == 192


def test_loader_rejects_unsupported_gas():
    cif = REPO / "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
    with pytest.raises(ValueError, match="unsupported gas_species"):
        load_bai_2013_native_system(REPO, cif, "H2O")


# ---------- 4c + 6e execution verdict --------------------------------
def test_4c_execution_verdict_json_exists():
    path = REPO / "evidence/v04_4c_bai_2013/verdicts/4c.json"
    assert path.exists()
    with path.open() as fp:
        v = json.load(fp)
    assert v["branch_id"] == "4c"
    assert v["executed_backend"] == "native_widom_v04"


def test_4c_passes_tier_A_strict_against_maghsoudi_2013():
    path = REPO / "evidence/v04_4c_bai_2013/verdicts/4c.json"
    with path.open() as fp:
        v = json.load(fp)
    delta_log = v["two_tier_verdict"]["delta_log10_K_H"]
    delta_Q = v["two_tier_verdict"]["delta_Q_st_kJ_per_mol"]
    assert abs(delta_log) <= 0.10, f"4c Δlog10 K_H = {delta_log} outside ±0.10 strict"
    assert abs(delta_Q) <= 2.0, f"4c ΔQ_st = {delta_Q} outside ±2.0 strict"
    assert v["two_tier_verdict"]["tier_A_regression"]["composite"] == "PASS"
    assert v["two_tier_verdict"]["tier_B_physical"]["composite"] == "PASS"
    assert v["two_tier_verdict"]["headline_disposition"] == "PHYSICAL_PASS"


def test_4c_bayesian_within_1_sigma_agreement():
    path = REPO / "evidence/v04_4c_bai_2013/verdicts/4c.json"
    with path.open() as fp:
        v = json.load(fp)
    bayes = v["bayesian_log_space_comparison"]
    assert abs(bayes["z_score"]) < 1.0
    assert bayes["classification"] == "AGREEMENT_WITHIN_1_SIGMA"


def test_6e_documents_TraPPE_zeo_under_binding_with_shah_2015_cross_check():
    path = REPO / "evidence/v04_6e_bai_2013/verdicts/6e.json"
    assert path.exists()
    with path.open() as fp:
        v = json.load(fp)
    assert v["branch_id"] == "6e"
    # K_H ~ 0.45, Shah 2015 reports ~0.60 — same under-binding regime
    K_H_atlas = v["K_H_mean_mol_per_kg_per_bar"]
    assert 0.40 < K_H_atlas < 0.50
    # Bayesian within 2-sigma despite Tier A FAIL
    bayes = v["bayesian_log_space_comparison"]
    assert bayes["classification"] == "AGREEMENT_WITHIN_2_SIGMA"


def test_4c_tail_correction_is_load_bearing():
    """4c's strict PASS depends on the analytical LJ tail correction.
    Without tail: K_H = 1.98 -> Δlog10 = -0.089 (still within ±0.10 strict).
    With tail: K_H = 2.22 -> Δlog10 = -0.039 (cleanly in ±0.10 strict).
    Either way 4c passes, but the tail correction halves the deviation."""
    path = REPO / "evidence/v04_4c_bai_2013/verdicts/4c.json"
    with path.open() as fp:
        v = json.load(fp)
    seeds = v["seeds"]
    for s in seeds:
        K_H_no_tail = s["K_H_mol_per_kg_per_bar_without_tail"]
        K_H_tail = s["K_H_mol_per_kg_per_bar_with_tail"]
        # Tail correction adds ~12% to K_H (Frenkel & Smit prediction)
        assert K_H_tail > K_H_no_tail
        ratio = K_H_tail / K_H_no_tail
        assert 1.05 < ratio < 1.20


# ---------- YAML promotion --------------------------------------------
def test_yaml_4c_promoted_to_locked_strict_executed():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    for case in cm.cases:
        for b in case["branches"]:
            if b.get("branch_id") == "4c":
                assert b["status"] == "locked_strict_executed"
                assert b["executed_backend"] == "native_widom_v04"
                assert b["affects_v04_verdict"] is True
                ff = b["force_field"]
                assert ff["framework_LJ"]["Si_zeo"]["epsilon_K"] == 22.0
                assert ff["framework_LJ"]["O_zeo"]["epsilon_K"] == 53.0  # NOT 93!
                assert ff["framework_charges"]["Si_zeo"] == +2.05
                assert ff["framework_charges"]["O_zeo"] == -1.025
                return
    pytest.fail("4c branch not found")


def test_yaml_6e_promoted_to_locked_strict_executed():
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    for case in cm.cases:
        for b in case["branches"]:
            if b.get("branch_id") == "6e":
                assert b["status"] == "locked_strict_executed"
                assert b["affects_v04_verdict"] is True
                ff = b["force_field"]
                # Same Bai 2013 framework as 4c
                assert ff["framework_LJ"]["Si_zeo"]["epsilon_K"] == 22.0
                assert ff["framework_LJ"]["O_zeo"]["epsilon_K"] == 53.0
                # Gas is TraPPE-UA CH4 not CO2
                assert b["gas"]["species"] == "CH4"
                return
    pytest.fail("6e branch not found")


def test_yaml_6e_records_all_four_rejected_deep_researcher_claims():
    """Audit trail: YAML 6e records the full chronology of deep researcher's wrong claims."""
    cm = load_locked_case_matrix(REPO / "v04_case_matrix.yaml")
    for case in cm.cases:
        for b in case["branches"]:
            if b.get("branch_id") == "6e":
                rejected = b.get("deep_research_2026_06_01_charge_correction_PROPOSED_REJECTED")
                assert rejected is not None
                proposals = rejected["proposed_corrections_in_chronological_order"]
                assert len(proposals) == 4  # 3 charge versions + 1 fictitious file path
                assert proposals[0]["Si_q"] == +1.20
                assert proposals[1]["Si_q"] == +1.30
                assert proposals[2]["Si_q"] == +1.50
                assert "GenericZeolites" in proposals[3]["fictitious_file_path"]
                assert "RASPA3" in rejected["why_rejected"]
                return
    pytest.fail("6e branch not found")
