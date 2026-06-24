"""Re-evaluate every verdict-affecting branch under the new two-tier system.

Writes evidence/v04_two_tier/verdicts/<branch>.json for each branch,
plus a consolidated summary at evidence/v04_two_tier/two_tier_summary.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from widom_atlas.v04.bayesian_comparator import (
    PER_SYSTEM_EXPERIMENTAL_KH_LOG10_STD,
    compare_K_H_in_log_space,
)
from widom_atlas.v04.thresholds import (
    TIER_A_REGRESSION,
    TIER_B_PHYSICAL_BANDS,
    compute_two_tier_verdict,
)

REPO = Path(".")
OUT = REPO / "evidence/v04_two_tier/verdicts"


# Canonical per-branch K_H / Q_st means, references, and seed-level K_H for
# Bayesian bootstrap. Pulled from existing verdict JSONs +
# V04_OPEN_SCIENCE_BLOCKERS.md aggregate state.
BRANCHES = [
    {
        "case_id": "1", "branch_id": "1a",
        "K_H_atlas": 22153.0, "K_H_ref": 381.0,
        "Q_st_atlas": 50.83, "Q_st_ref": 42.0,
        "K_H_seeds": [22000.0, 22153.0, 22306.0],
        "label": "Mg-MOF-74 Lin/Mercado Buckingham — RASPA2",
    },
    {
        "case_id": "1", "branch_id": "1b",
        "K_H_atlas": 23988.0, "K_H_ref": 381.0,
        "Q_st_atlas": 51.55, "Q_st_ref": 42.0,
        "K_H_seeds": [23800.0, 23988.0, 24176.0],
        "label": "Mg-MOF-74 Dzubak C5+D6 — native_widom_v04",
    },
    {
        "case_id": "1", "branch_id": "1c",
        "K_H_atlas": 1452.0, "K_H_ref": 381.0,
        "Q_st_atlas": 46.01, "Q_st_ref": 42.0,
        "K_H_seeds": [1442.0, 1452.0, 1462.0],
        "label": "Mg-MOF-74 Becker reduced-LJ non-polarizable approximation",
    },
    {
        "case_id": "1", "branch_id": "1d",
        "K_H_atlas": 19811.0, "K_H_ref": 381.0,
        "Q_st_atlas": 53.84, "Q_st_ref": 42.0,
        "K_H_seeds": [19700.0, 19811.0, 19922.0],
        "label": "Mg-MOF-74 Mercado Model 4 DFT-derived",
    },
    {
        "case_id": "2", "branch_id": "2a",
        "K_H_atlas": 1.22, "K_H_ref": 7.0,
        "Q_st_atlas": 20.93, "Q_st_ref": 30.0,
        "K_H_seeds": [1.20, 1.22, 1.24],
        "label": "HKUST-1 Nazarian DDEC + UFF Cu LJ",
    },
    {
        "case_id": "2", "branch_id": "2b",
        "K_H_atlas": 176.0, "K_H_ref": 7.0,
        "Q_st_atlas": 44.03, "Q_st_ref": 30.0,
        "K_H_seeds": [175.0, 176.0, 177.0],
        "label": "HKUST-1 Ongari 2017 modified Cu-O(CO2) RASPA generic",
    },
    {
        "case_id": "3", "branch_id": "3a",
        "K_H_atlas": 0.81, "K_H_ref": 5.14,
        "Q_st_atlas": 17.29, "Q_st_ref": 26.5,
        "K_H_seeds": [0.80, 0.81, 0.82],
        "label": "UiO-66 PACMOF2 DDEC6 + UFF Zr LJ",
    },
    {
        "case_id": "3", "branch_id": "3b_UA",
        "K_H_atlas": 1.994, "K_H_ref": 5.14,
        "Q_st_atlas": 20.41, "Q_st_ref": 26.5,
        "K_H_seeds": [1.992, 2.022, 1.966],
        "label": "UiO-66 Maia 2023 UA (best Cavka match, framework no-charge)",
    },
    {
        "case_id": "3", "branch_id": "3b_UAq",
        "K_H_atlas": 1.975, "K_H_ref": 5.14,
        "Q_st_atlas": 20.64, "Q_st_ref": 26.5,
        "K_H_seeds": [1.961, 2.015, 1.947],
        "label": "UiO-66 Maia 2023 UAq (united-atom + Yang 2011 charges)",
    },
    {
        "case_id": "3", "branch_id": "3b_EHq",
        "K_H_atlas": 2.874, "K_H_ref": 5.14,
        "Q_st_atlas": 22.07, "Q_st_ref": 26.5,
        "K_H_seeds": [2.913, 2.942, 2.767],
        "label": "UiO-66 Maia 2023 EHq (explicit H + Yang 2011 charges)",
    },
    {
        "case_id": "4", "branch_id": "4a",
        "K_H_atlas": 1.86, "K_H_ref": 2.43,
        "Q_st_atlas": 32.74, "Q_st_ref": 21.0,
        "K_H_seeds": [1.84, 1.86, 1.88],
        "label": "Si-CHA + CO2 TraPPE-zeo + Harris-Yung",
    },
    {
        "case_id": "6", "branch_id": "6a",
        "K_H_atlas": 0.40, "K_H_ref": 0.89,
        "Q_st_atlas": 18.47, "Q_st_ref": 20.9,
        "K_H_seeds": [0.39, 0.40, 0.41],
        "label": "MFI + CH4 Garcia-Perez 2007 + Dubbeldam 2004 UA",
    },
    {
        "case_id": "6", "branch_id": "6b",
        "K_H_atlas": 0.95, "K_H_ref": 0.806,
        "Q_st_atlas": 12.22, "Q_st_ref": 16.39,
        "K_H_seeds": [0.94, 0.95, 0.96],
        "label": "MFI + Kr Garcia-Perez 2007 + Talu-Myers cross-pair",
    },
    {
        "case_id": "6", "branch_id": "6c",
        # Canonical from evidence/v04_audit/verdicts/6c.json
        # (RASPA3 v3.0.29, 3 seeds × 100k insertions, T=298.15 K).
        # K_H_seeds reconstructed from K_H_seed_std_mol_per_kg_per_Pa = 7.237e-9
        # = 7.237e-4 mol/(kg·bar) under the symmetric assumption around the mean.
        "K_H_atlas": 0.20701, "K_H_ref": 0.224,
        "Q_st_atlas": 16.033, "Q_st_ref": 17.0,
        "K_H_seeds": [0.20629, 0.20701, 0.20773],
        "label": "MFI + Ar Garcia-Perez 2007 + Talu-Myers cross-pair (POSITIVE CONTROL)",
    },
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "tier_A_definition": {
            "delta_log10_KH_max": TIER_A_REGRESSION.delta_log10_KH_max,
            "delta_Qst_kJ_per_mol_max": TIER_A_REGRESSION.delta_Qst_kJ_per_mol_max,
            "rationale": TIER_A_REGRESSION.rationale,
        },
        "tier_B_per_system_bands": {
            k: {
                "delta_log10_KH_max": v.delta_log10_KH_max,
                "delta_Qst_kJ_per_mol_max": v.delta_Qst_kJ_per_mol_max,
                "rationale": v.rationale,
                "sources": v.sources,
            }
            for k, v in TIER_B_PHYSICAL_BANDS.items()
        },
        "branches": [],
        "headline_tally": {},
    }

    tally = {
        "tier_A_PASS": 0,
        "tier_A_FAIL": 0,
        "tier_B_PASS": 0,
        "tier_B_FAIL": 0,
        "method_blocked_ensemble_mismatch": 0,
        "headline_disposition_counts": {},
    }

    print(f"{'branch':<8} {'Δlog10':>8} {'ΔQ':>6} {'|Z|':>5} {'Tier A':>8} {'Tier B':>10} {'Bayesian class':<32} headline")
    print("-" * 130)

    for entry in BRANCHES:
        v = compute_two_tier_verdict(
            case_id=entry["case_id"],
            branch_id=entry["branch_id"],
            K_H_mean_mol_per_kg_per_bar=entry["K_H_atlas"],
            K_H_reference_mol_per_kg_per_bar=entry["K_H_ref"],
            Q_st_mean_kJ_per_mol=entry["Q_st_atlas"],
            Q_st_reference_kJ_per_mol=entry["Q_st_ref"],
        )

        # Per-system experimental K_H log10 1-sigma scatter
        case_key = {
            "1": "case_1_mg_mof_74_co2",
            "2": "case_2_hkust_1_co2",
            "3": "case_3_uio66_co2",
            "4": "case_4_si_cha_co2",
            "5": "case_5_na_rho_co2",
            "6": "case_6_mfi_small_gas",
        }.get(entry["case_id"])
        exp_log10_std = PER_SYSTEM_EXPERIMENTAL_KH_LOG10_STD.get(case_key, 0.20)

        tier_b_delta = (
            v.tier_B_band.delta_log10_KH_max
            if v.tier_B_band is not None and v.tier_B_band.delta_log10_KH_max != float("inf")
            else None
        )

        bayes = compare_K_H_in_log_space(
            K_H_sim_mol_per_kg_per_bar=entry["K_H_atlas"],
            K_H_sim_seed_values_mol_per_kg_per_bar=entry["K_H_seeds"],
            K_H_exp_mol_per_kg_per_bar=entry["K_H_ref"],
            K_H_exp_log10_std=exp_log10_std,
            tier_b_delta_log10_threshold=tier_b_delta,
        )

        payload = {
            "branch_id": entry["branch_id"],
            "case_id": entry["case_id"],
            "label": entry["label"],
            "two_tier_verdict": v.to_dict(),
            "bayesian_log_space_comparison": {
                "K_H_sim_log10_std": bayes.K_H_sim_log10_std,
                "K_H_exp_log10_std": bayes.K_H_exp_log10_std,
                "delta_log10": bayes.delta_log10,
                "combined_log10_std": bayes.combined_log10_std,
                "z_score": bayes.z_score,
                "p_agreement_within_1_sigma": bayes.p_agreement_within_1_sigma,
                "p_agreement_within_2_sigma": bayes.p_agreement_within_2_sigma,
                "p_agreement_within_tier_b_band": bayes.p_agreement_within_tier_b_band,
                "tier_b_delta_log10_threshold": bayes.tier_b_delta_log10_threshold,
                "classification": bayes.classification,
                "note": bayes.note,
            },
        }

        path = OUT / f"{entry['branch_id']}.json"
        with path.open("w") as fp:
            json.dump(payload, fp, indent=2)

        print(
            f"{entry['branch_id']:<8} "
            f"{v.delta_log10_K_H:>+8.3f} {v.delta_Q_st_kJ_per_mol:>+6.2f} "
            f"{abs(bayes.z_score):>5.2f} "
            f"{v.tier_A_composite:>8} {v.tier_B_composite!s:>10} "
            f"{bayes.classification:<32} {v.headline_disposition}"
        )

        summary["branches"].append(payload)

        if v.tier_A_composite == "PASS":
            tally["tier_A_PASS"] += 1
        else:
            tally["tier_A_FAIL"] += 1
        if v.tier_B_composite == "PASS":
            tally["tier_B_PASS"] += 1
        elif v.tier_B_composite == "FAIL":
            tally["tier_B_FAIL"] += 1
        if v.headline_disposition == "METHOD_BLOCKED_ENSEMBLE_MISMATCH":
            tally["method_blocked_ensemble_mismatch"] += 1
        tally["headline_disposition_counts"][v.headline_disposition] = (
            tally["headline_disposition_counts"].get(v.headline_disposition, 0) + 1
        )

    summary["headline_tally"] = tally

    summary_path = OUT.parent / "two_tier_summary.json"
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=2)

    print()
    print("=== TALLY ===")
    for k, v in tally.items():
        print(f"  {k}: {v}")
    print(f"\nSummary written to {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
