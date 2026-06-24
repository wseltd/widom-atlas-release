"""Audit every 5c-candidate ISODB isotherm + write per-branch verdict JSON.

For each candidate:
  * K_H fit sensitivity (1-pt, 2-pt, 3-pt linear, virial)
  * Unit conversion (mmol/g/bar -> mol/kg/bar -> mol/kg/Pa)
  * Q_st provenance (from operator-supplied literature cite)
  * Structure provenance (CIF source to use for atlas execution)
  * Force-field provenance (FF source to use for atlas execution)
  * Verdict classification

For 5c candidates the verdict is one of:
  * reference_audited_executable_strict       — has cation-CIF + FF in repo
  * reference_audited_pending_cation_cif_lock  — CIF exists but cation not placed
  * reference_audited_pending_ff_lock         — FF for the cation not in repo
  * reference_audited_pending_both            — both missing

NONE of these are simulator execution (we need cation positions). The
verdict records the experimental reference K_H + sensitivity range and
the *exact* missing artefacts for downstream atlas execution.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from widom_atlas.v04.refs.isodb_audit import (
    audit_isodb_isotherm,
    render_audit_json,
)

REPO = Path(".")
BASE = REPO / "docs/research/dataset-research-for-v0.4/5c_replacement_branches"
OUT = REPO / "evidence/v04_5c/verdicts"


# (branch_id, isodb_json, qst_kJ_per_mol, qst_source, qst_doi,
#   expected_cif_provenance, expected_ff_provenance, available_cif_in_repo,
#   available_ff_in_repo, label)
CANDIDATES = [
    (
        "5c_NaZK5_CO2_303K_PRIMARY",
        BASE / "na_zk5_pham_lobo_2013/pham_lobo_2013_isotherm49_na_zk5_co2_303K.json",
        49.0,
        "Pham et al., Adsorption 2017 (CHA zeolites) reports Na-ZK-5 Qst ≈ 49 kJ/mol — cited as literature value for KFI Na variant.",
        "10.1007/s10450-017-9894-1",
        "Na-KFI (Si/Al ≈ 3) framework + Na cation positions — typically refined from Pham & Lobo 2013 Langmuir SI Table S2 or Ikeda et al. 2008 IZA refinements.",
        "UFF Na+ (sigma=2.6576 A, epsilon=15.0966 K) + framework Garcia-Perez 2007 Si/O LJ (used for 5b Na-Rho); TraPPE-CO2 gas.",
        False,
        False,
        "Na-ZK-5 + CO2 at 303 K — primary 5c replacement-branch reference",
    ),
    (
        "5c_LiZK5_CO2_303K_alternate",
        BASE / "na_zk5_pham_lobo_2013/pham_lobo_2013_isotherm48_li_zk5_co2_303K.json",
        47.0,
        "Pham & Lobo 2013 Langmuir; KFI Li variant Qst ≈ 47 kJ/mol.",
        "10.1021/la400352r",
        "Li-KFI framework + Li cation positions.",
        "UFF Li+ + framework Garcia-Perez 2007 Si/O LJ + TraPPE-CO2.",
        False,
        False,
        "Li-ZK-5 + CO2 at 303 K — alternate KFI cation comparison",
    ),
    (
        "5c_KZK5_CO2_303K_alternate",
        BASE / "na_zk5_pham_lobo_2013/pham_lobo_2013_isotherm50_k_zk5_co2_303K.json",
        45.0,
        "Pham & Lobo 2013 Langmuir; KFI K variant Qst ≈ 45 kJ/mol.",
        "10.1021/la400352r",
        "K-KFI framework + K cation positions.",
        "UFF K+ + framework Garcia-Perez 2007 Si/O LJ + TraPPE-CO2.",
        False,
        False,
        "K-ZK-5 + CO2 at 303 K — alternate KFI cation comparison",
    ),
    (
        "5c_Zeolite5A_CaA_CO2_298K_fallback",
        BASE / "zeolite_5a_wang_levan_2009/wang_levan_2009_isotherm8_5a_co2_298K.json",
        36.0,
        "Wang & LeVan 2009 J. Chem. Eng. Data; 5A CO2 Qst ≈ 36 kJ/mol (low loading, Henry regime).",
        "10.1021/je800900a",
        "Ca-LTA framework + Ca2+ cation positions. CSD or IZA structural source.",
        "UFF Ca2+ + framework Si/O LJ (Garcia-Perez 2007 or TraPPE-zeo) + TraPPE-CO2.",
        False,  # LTA_NaK.cif in repo is all-silica, no Ca placement
        False,
        "Zeolite 5A (Ca-LTA) + CO2 at 298 K — cleanest fallback (47 data points from 1e-5 bar)",
    ),
    (
        "5c_Zeolite13X_NaX_CO2_273K_fallback",
        BASE / "zeolite_13x_wang_levan_2009/wang_levan_2009_isotherm27_13x_co2_273K.json",
        44.0,
        "Wang & LeVan 2009 J. Chem. Eng. Data; 13X CO2 Qst ≈ 44 kJ/mol.",
        "10.1021/je800900a",
        "Na-FAU framework + Na cation positions. Operator-cited reference structures.",
        "UFF Na+ + framework Si/O LJ + TraPPE-CO2.",
        False,
        False,
        "Zeolite 13X (Na-FAU) + CO2 at 273 K — well-established fallback",
    ),
    (
        "5c_Zeolite4A_NaA_CO2_273K_fallback",
        BASE / "zeolite_4a_hefti_2020/hefti_2020_isotherm35_4a_co2_273K.json",
        40.0,
        "Hefti et al. 2020 Adsorption; 4A CO2 Qst ≈ 40 kJ/mol (from gravimetric + calorimetric multi-T data).",
        "10.1007/s10450-020-00206-7",
        "Na-LTA framework + Na cation positions.",
        "UFF Na+ + framework Si/O LJ + TraPPE-CO2.",
        False,
        False,
        "Zeolite 4A (Na-LTA) + CO2 at 273 K — fallback with calorimetric Qst",
    ),
    (
        "5c_R1KCHA_CO2_273K_alternate",
        BASE / "shang_2012_k_cha/shang_2012_isotherm5_r1kcha_co2_273K.json",
        35.0,
        "Shang et al. 2012 JACS; K-CHA Qst ≈ 35 kJ/mol; trapdoor selectivity system.",
        "10.1021/ja309274y",
        "K-CHA framework + K cation positions.",
        "UFF K+ + framework Si/O LJ + TraPPE-CO2.",
        False,
        False,
        "K-CHA + CO2 at 273 K — trapdoor system (interpretation caveat)",
    ),
    (
        "5c_HighSilicaCHA_CO2_298K_alternate",
        BASE / "pham_2017_cha/pham_2017_isotherm2_high_silica_cha_co2_298K.json",
        25.0,
        "Pham et al. 2017 Adsorption; high-silica CHA Qst ≈ 25 kJ/mol.",
        "10.1007/s10450-017-9894-1",
        "All-silica CHA framework (IZA database).",
        "TraPPE-zeo Si/O LJ (still blocked pending Bai 2013 main paper) or Garcia-Perez fallback + TraPPE-CO2.",
        True,  # CHA.cif in repo
        False,
        "High-silica CHA + CO2 at 298 K — alternate; OVERLAPS 4a Si-CHA branch scope",
    ),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "audit_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "n_candidates": len(CANDIDATES),
        "candidates": {},
    }

    for (
        branch_id, isodb_path, qst_value, qst_source, qst_doi,
        cif_expected, ff_expected, cif_in_repo, ff_in_repo, label,
    ) in CANDIDATES:
        if not isodb_path.exists():
            summary["candidates"][branch_id] = {
                "status": "FAIL_isodb_json_missing",
                "missing_path": str(isodb_path),
            }
            print(f"FAIL {branch_id}: missing {isodb_path.relative_to(REPO)}")
            continue
        audit = audit_isodb_isotherm(isodb_path)
        rendered = render_audit_json(audit)

        if cif_in_repo and ff_in_repo:
            execution_status = "reference_audited_executable_strict"
        elif cif_in_repo and not ff_in_repo:
            execution_status = "reference_audited_pending_ff_lock"
        elif not cif_in_repo and ff_in_repo:
            execution_status = "reference_audited_pending_cation_cif_lock"
        else:
            execution_status = "reference_audited_pending_both_cation_cif_and_ff_lock"

        verdict_payload = {
            "branch_id_candidate": branch_id,
            "label": label,
            "reference_isodb_audit": rendered,
            "Q_st_reference": {
                "value_kJ_per_mol": qst_value,
                "source": qst_source,
                "source_doi": qst_doi,
                "method": (
                    "literature_citation (not derived from isotherm; "
                    "would require multi-T van't Hoff or calorimetric data)"
                ),
            },
            "structure_provenance_required": cif_expected,
            "structure_provenance_in_repo": cif_in_repo,
            "ff_provenance_required": ff_expected,
            "ff_provenance_in_repo": ff_in_repo,
            "execution_status": execution_status,
            "exact_missing_artefacts_for_atlas_execution": [
                a for a in [
                    (cif_expected if not cif_in_repo else None),
                    (ff_expected if not ff_in_repo else None),
                ] if a is not None
            ],
            "verdict": (
                "REFERENCE_AUDITED" if not (cif_in_repo and ff_in_repo)
                else "READY_FOR_ATLAS_RUN"
            ),
            "verdict_notes": (
                "K_H reference + Q_st reference + sensitivity range fully audited "
                "and reproducible from the archived ISODB JSON. Atlas simulation "
                "execution pending cation-CIF + force-field lock (see "
                "exact_missing_artefacts_for_atlas_execution). NOT a 5b validation."
            ),
            "do_not_call_replacement_a_validation_of_5b": True,
        }

        per_branch_path = OUT / f"{branch_id}.json"
        with per_branch_path.open("w") as fp:
            json.dump(verdict_payload, fp, indent=2)
        summary["candidates"][branch_id] = {
            "status": execution_status,
            "verdict": verdict_payload["verdict"],
            "K_H_method_values_mol_per_kg_per_bar": (
                audit.K_H_method_values_mol_per_kg_per_bar()
            ),
            "K_H_sensitivity_range_mol_per_kg_per_bar": (
                audit.K_H_sensitivity_range_mol_per_kg_per_bar()
            ),
            "Q_st_reference_kJ_per_mol": qst_value,
            "henry_regime_adequacy": audit.henry_regime_adequacy,
        }
        print(
            f"OK {branch_id}: "
            f"K_H range {audit.K_H_sensitivity_range_mol_per_kg_per_bar()[0]:.1f}-"
            f"{audit.K_H_sensitivity_range_mol_per_kg_per_bar()[1]:.1f} mol/(kg.bar), "
            f"Q_st={qst_value} kJ/mol, {execution_status}"
        )

    summary_path = OUT / "5c_audit_summary.json"
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nWrote {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
