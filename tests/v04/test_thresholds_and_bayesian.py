"""Regression tests for the two-tier threshold system and Bayesian K_H comparator."""
from __future__ import annotations

import math

from widom_atlas.v04.bayesian_comparator import (
    PER_SYSTEM_EXPERIMENTAL_KH_LOG10_STD,
    compare_K_H_in_log_space,
)
from widom_atlas.v04.thresholds import (
    TIER_A_REGRESSION,
    TIER_B_PHYSICAL_BANDS,
    compute_two_tier_verdict,
    map_case_id_to_physical_band,
)


# ---------- Tier system ----------
def test_tier_A_strict_thresholds_are_v04_2_values():
    """Tier A must remain ±0.10 Δlog10 K_H + ±2.0 kJ/mol Q_st (the historical v04.2 strict)."""
    assert TIER_A_REGRESSION.delta_log10_KH_max == 0.10
    assert TIER_A_REGRESSION.delta_Qst_kJ_per_mol_max == 2.0


def test_tier_B_OMS_bands_are_wider_than_tier_A():
    """Tier B for OMS / defective MOFs must be wider than Tier A to reflect literature scatter."""
    for key in ("case_1_mg_mof_74_co2", "case_2_hkust_1_co2", "case_3_uio66_co2"):
        b = TIER_B_PHYSICAL_BANDS[key]
        assert b.delta_log10_KH_max > TIER_A_REGRESSION.delta_log10_KH_max
        assert b.delta_Qst_kJ_per_mol_max > TIER_A_REGRESSION.delta_Qst_kJ_per_mol_max


def test_tier_B_silica_zeolite_bands_are_narrow():
    """Si-CHA and MFI are well-characterized — Tier B bands should be tight."""
    si_cha = TIER_B_PHYSICAL_BANDS["case_4_si_cha_co2"]
    mfi = TIER_B_PHYSICAL_BANDS["case_6_mfi_small_gas"]
    assert si_cha.delta_log10_KH_max <= 0.20
    assert mfi.delta_log10_KH_max <= 0.15


def test_tier_B_na_rho_is_infinite_band_ensemble_mismatch():
    """Na-Rho should NOT have a finite Tier B band — it's a method block, not a threshold question."""
    na_rho = TIER_B_PHYSICAL_BANDS["case_5_na_rho_co2"]
    assert na_rho.delta_log10_KH_max == float("inf")
    assert na_rho.delta_Qst_kJ_per_mol_max == float("inf")


def test_case_id_maps_to_physical_band():
    """Every case_id (1-6) must map to a Tier B band."""
    for case_id in "123456":
        b = map_case_id_to_physical_band(case_id)
        assert b is not None, f"case {case_id} missing Tier B band"


def test_6c_positive_control_passes_both_tiers():
    """6c MFI + Ar must pass Tier A AND Tier B (it's the positive control)."""
    v = compute_two_tier_verdict(
        case_id="6", branch_id="6c",
        K_H_mean_mol_per_kg_per_bar=0.207, K_H_reference_mol_per_kg_per_bar=0.224,
        Q_st_mean_kJ_per_mol=16.03, Q_st_reference_kJ_per_mol=17.0,
    )
    assert v.tier_A_composite == "PASS"
    assert v.tier_B_composite == "PASS"
    assert v.headline_disposition == "PHYSICAL_PASS"


def test_3b_EHq_passes_tier_B_within_uio66_scatter():
    """3b Maia EHq (K_H=2.87, Q_st=22.07 vs reference 5.14, 26.5) is within UiO-66 literature scatter."""
    v = compute_two_tier_verdict(
        case_id="3", branch_id="3b_EHq",
        K_H_mean_mol_per_kg_per_bar=2.874, K_H_reference_mol_per_kg_per_bar=5.14,
        Q_st_mean_kJ_per_mol=22.07, Q_st_reference_kJ_per_mol=26.5,
    )
    assert v.tier_B_composite == "PASS"
    assert v.headline_disposition == "PHYSICAL_PASS"


def test_2a_fails_both_tiers_honest_force_field_disagreement():
    """HKUST-1 UFF-Cu under-binds by 6× — outside even the wide HKUST-1 Tier B band."""
    v = compute_two_tier_verdict(
        case_id="2", branch_id="2a",
        K_H_mean_mol_per_kg_per_bar=1.22, K_H_reference_mol_per_kg_per_bar=7.0,
        Q_st_mean_kJ_per_mol=20.93, Q_st_reference_kJ_per_mol=30.0,
    )
    assert v.tier_A_composite == "FAIL"
    assert v.tier_B_composite == "FAIL"


def test_na_rho_case_5_gets_ensemble_mismatch_headline():
    """Case 5 (Na-Rho) must always headline as ENSEMBLE_MISMATCH regardless of K_H/Q_st."""
    v = compute_two_tier_verdict(
        case_id="5", branch_id="5b",
        K_H_mean_mol_per_kg_per_bar=10.0, K_H_reference_mol_per_kg_per_bar=10.0,
        Q_st_mean_kJ_per_mol=30.0, Q_st_reference_kJ_per_mol=30.0,
    )
    assert v.headline_disposition == "METHOD_BLOCKED_ENSEMBLE_MISMATCH"


# ---------- Bayesian comparator ----------
def test_bayesian_z_score_zero_at_perfect_agreement():
    res = compare_K_H_in_log_space(
        K_H_sim_mol_per_kg_per_bar=5.0,
        K_H_sim_seed_values_mol_per_kg_per_bar=[5.0, 5.0, 5.0],
        K_H_exp_mol_per_kg_per_bar=5.0,
        K_H_exp_log10_std=0.10,
    )
    assert abs(res.delta_log10) < 1e-9
    assert abs(res.z_score) < 1e-9
    assert res.classification == "AGREEMENT_WITHIN_1_SIGMA"


def test_bayesian_strong_disagreement_above_3_sigma():
    res = compare_K_H_in_log_space(
        K_H_sim_mol_per_kg_per_bar=1000.0,
        K_H_sim_seed_values_mol_per_kg_per_bar=[1000.0, 1001.0, 999.0],
        K_H_exp_mol_per_kg_per_bar=1.0,
        K_H_exp_log10_std=0.05,
    )
    assert res.classification == "STRONG_DISAGREEMENT_GT_3_SIGMA"
    assert abs(res.z_score) > 3.0


def test_bayesian_combines_sim_and_exp_uncertainties_in_quadrature():
    """combined std = sqrt(sim^2 + exp^2)."""
    res = compare_K_H_in_log_space(
        K_H_sim_mol_per_kg_per_bar=5.0,
        K_H_sim_seed_values_mol_per_kg_per_bar=[4.5, 5.0, 5.5],
        K_H_exp_mol_per_kg_per_bar=5.0,
        K_H_exp_log10_std=0.2,
    )
    expected = math.sqrt(res.K_H_sim_log10_std ** 2 + res.K_H_exp_log10_std ** 2)
    assert abs(res.combined_log10_std - expected) < 1e-9


def test_bayesian_p_within_band_is_higher_when_band_is_wider():
    """Wider Tier B band → higher probability of agreement (monotonic)."""
    base = dict(
        K_H_sim_mol_per_kg_per_bar=2.0,
        K_H_sim_seed_values_mol_per_kg_per_bar=[1.95, 2.0, 2.05],
        K_H_exp_mol_per_kg_per_bar=5.0,
        K_H_exp_log10_std=0.20,
    )
    res_narrow = compare_K_H_in_log_space(**base, tier_b_delta_log10_threshold=0.20)
    res_wide = compare_K_H_in_log_space(**base, tier_b_delta_log10_threshold=0.50)
    assert res_wide.p_agreement_within_tier_b_band > res_narrow.p_agreement_within_tier_b_band


def test_per_system_kh_log10_std_defaults_are_documented():
    """The literature-scatter defaults must be in [0, 1] (sane log-scale 1-sigma)."""
    for k, v in PER_SYSTEM_EXPERIMENTAL_KH_LOG10_STD.items():
        if math.isnan(v):
            continue  # case 5 is NaN by design
        assert 0.0 < v < 1.0, f"{k} default {v} out of sane range"
