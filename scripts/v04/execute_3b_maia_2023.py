"""Execute 3b UiO-66 + CO2 Maia 2023 variants (UA, UAq, EHq).

Runs the native_widom_v04 backend with the Maia 2023 force field for
each variant at T = 298 K, 3 seeds. Writes verdict JSONs under
evidence/v04_3b_maia/verdicts/.

Reference target (from 3a, Cmarik 2012 Toth fit):
  K_H = 5.14 mol/(kg.bar), Q_st = 26.5 kJ/mol
  acceptance windows: K_H [3.5, 7.5], Q_st [22.0, 32.0]

Per operator (verbatim, 2026-06-01 final pivot directive): "Run Maia's
best-match UA no-framework-charge variant first, because the paper says
it best matches CO2. Record UAq and EHq as available variants."
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from widom_atlas.v04.native.ewald import EwaldParameters
from widom_atlas.v04.native.maia_2023_loader import (
    load_3b_native_maia_2023,
)
from widom_atlas.v04.native.runner import run_native_widom

REPO = Path(".")
OUT_DIR = REPO / "evidence" / "v04_3b_maia"
VERDICTS_DIR = OUT_DIR / "verdicts"
EVIDENCE_DIR = OUT_DIR / "evidence"

CIF_PATH = REPO / "fixtures" / "v04" / "RUBTAK01_SL_DDEC.cif"

TEMPERATURE_K = 298.0
SEEDS = [0, 1, 2]
# Cost-controlled production budget for the all-CPU native runner. Per the
# native runner notes, K_H stabilises within ~50k insertions for medium-
# sized MOF cells; 80k gives a comfortable margin while keeping each
# variant under ~5 min on a single CPU thread.
N_INSERTIONS = 80_000

# Reference target (from 3a Cmarik 2012 Toth fit).
REFERENCE_K_H_MOL_PER_KG_PER_BAR = 5.14
REFERENCE_K_H_WINDOW_MIN = 3.5
REFERENCE_K_H_WINDOW_MAX = 7.5
REFERENCE_Q_ST_KJ_PER_MOL = 26.5
REFERENCE_Q_ST_WINDOW_MIN = 22.0
REFERENCE_Q_ST_WINDOW_MAX = 32.0


def aggregate_seeds(seed_results):
    """Aggregate per-seed K_H + Q_st: mean + std (sample, ddof=1)."""
    kh_list = [r["K_H_mol_per_kg_per_bar"] for r in seed_results]
    qst_list = [r["Q_st_kJ_per_mol"] for r in seed_results]
    return {
        "K_H_mean_mol_per_kg_per_bar": float(np.mean(kh_list)),
        "K_H_std_mol_per_kg_per_bar": float(np.std(kh_list, ddof=1)),
        "K_H_min": float(np.min(kh_list)),
        "K_H_max": float(np.max(kh_list)),
        "Q_st_mean_kJ_per_mol": float(np.mean(qst_list)),
        "Q_st_std_kJ_per_mol": float(np.std(qst_list, ddof=1)),
        "Q_st_min": float(np.min(qst_list)),
        "Q_st_max": float(np.max(qst_list)),
    }


def verdict_for_variant(variant: str, aggregated: dict) -> dict:
    """Apply strict thresholds to aggregated K_H + Q_st."""
    kh_mean = aggregated["K_H_mean_mol_per_kg_per_bar"]
    qst_mean = aggregated["Q_st_mean_kJ_per_mol"]
    delta_log10_KH = math.log10(kh_mean) - math.log10(REFERENCE_K_H_MOL_PER_KG_PER_BAR)
    delta_Q_st = qst_mean - REFERENCE_Q_ST_KJ_PER_MOL

    kh_passes_window = REFERENCE_K_H_WINDOW_MIN <= kh_mean <= REFERENCE_K_H_WINDOW_MAX
    qst_passes_window = REFERENCE_Q_ST_WINDOW_MIN <= qst_mean <= REFERENCE_Q_ST_WINDOW_MAX
    kh_passes_strict = abs(delta_log10_KH) <= 0.10
    qst_passes_strict = abs(delta_Q_st) <= 2.0

    return {
        "variant": variant,
        "K_H_mean_mol_per_kg_per_bar": kh_mean,
        "K_H_reference_mol_per_kg_per_bar": REFERENCE_K_H_MOL_PER_KG_PER_BAR,
        "delta_log10_K_H": delta_log10_KH,
        "Q_st_mean_kJ_per_mol": qst_mean,
        "Q_st_reference_kJ_per_mol": REFERENCE_Q_ST_KJ_PER_MOL,
        "delta_Q_st_kJ_per_mol": delta_Q_st,
        "K_H_passes_acceptance_window": kh_passes_window,
        "Q_st_passes_acceptance_window": qst_passes_window,
        "K_H_passes_strict_threshold_0p10": kh_passes_strict,
        "Q_st_passes_strict_threshold_2kJ_per_mol": qst_passes_strict,
        "verdict_strict": "PASS" if (kh_passes_strict and qst_passes_strict) else "FAIL",
        "verdict_window": "PASS" if (kh_passes_window and qst_passes_window) else "FAIL",
    }


def execute_variant(variant: str) -> dict:
    print(f"\n=== Maia 2023 variant {variant} ===")
    enable_ewald = variant in ("UAq", "EHq")
    ewald_params = (
        EwaldParameters(
            alpha_inv_angstrom=0.3,
            real_cutoff_angstrom=14.0,
            k_max_inv_angstrom=1.4,
        )
        if enable_ewald else None
    )

    seed_results = []
    t_total = 0.0
    for seed in SEEDS:
        print(f"  seed {seed}: building system...", flush=True)
        sys = load_3b_native_maia_2023(
            repo_root=REPO,
            cif_path=CIF_PATH,
            variant=variant,
            cutoff_angstrom=14.0,
        )
        t0 = time.time()
        result = run_native_widom(
            system=sys,
            temperature_K=TEMPERATURE_K,
            n_insertions=N_INSERTIONS,
            seed=seed,
            enable_ewald=enable_ewald,
            ewald_parameters=ewald_params,
        )
        dt = time.time() - t0
        t_total += dt
        kh_pa = result.K_H_mol_per_kg_per_Pa
        kh_bar = kh_pa * 1.0e5
        qst = result.Q_st_kJ_per_mol
        print(
            f"    seed {seed}: K_H = {kh_bar:.3f} mol/(kg.bar), "
            f"Q_st = {qst:.2f} kJ/mol, wall = {dt:.1f} s "
            f"(n_insertions={N_INSERTIONS})",
            flush=True,
        )
        seed_results.append({
            "seed": seed,
            "n_insertions": N_INSERTIONS,
            "K_H_mol_per_kg_per_Pa": float(kh_pa),
            "K_H_mol_per_kg_per_bar": float(kh_bar),
            "Q_st_kJ_per_mol": float(qst),
            "wall_seconds": float(dt),
            "framework_mass_kg": float(result.framework_mass_kg),
        })

    aggregated = aggregate_seeds(seed_results)
    verdict = verdict_for_variant(variant, aggregated)
    print(
        f"  → {variant} mean K_H = "
        f"{aggregated['K_H_mean_mol_per_kg_per_bar']:.3f} ± "
        f"{aggregated['K_H_std_mol_per_kg_per_bar']:.3f}, "
        f"Q_st = {aggregated['Q_st_mean_kJ_per_mol']:.2f} ± "
        f"{aggregated['Q_st_std_kJ_per_mol']:.2f}, "
        f"verdict_strict={verdict['verdict_strict']}, "
        f"verdict_window={verdict['verdict_window']}, "
        f"total_wall={t_total:.1f} s",
        flush=True,
    )

    return {
        "variant": variant,
        "n_insertions_per_seed": N_INSERTIONS,
        "seeds": SEEDS,
        "temperature_K": TEMPERATURE_K,
        "ewald_enabled": enable_ewald,
        "cutoff_angstrom": 14.0,
        "mixing_rule": "lorentz_berthelot",
        "cif_used": str(CIF_PATH.relative_to(REPO)),
        "force_field_lineage": f"Maia_2023_UiO-66_CO2_TraPPE_{variant}",
        "force_field_doi": "10.3390/cryst13101523",
        "charge_provenance_for_charged_variants": (
            "Maia Table 2/3 charges (Maia attributes to Yang 2011 JPC C, "
            "DOI 10.1021/jp202633t); applied per Maia atom-type after "
            "DDEC-bucket classification overwrites CIF charges"
        ),
        "seed_results": seed_results,
        "aggregated": aggregated,
        "verdict": verdict,
        "wall_seconds_total": t_total,
        "backend_tag": "native_widom_v04",
    }


def main():
    VERDICTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    all_variants_payload: dict = {
        "branch_id": "3b",
        "label": "UiO-66 + CO2 Maia 2023 (UA + UAq + EHq) — native_widom_v04",
        "executed_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "executed_backend": "native_widom_v04",
        "n_seeds": len(SEEDS),
        "n_insertions_per_seed": N_INSERTIONS,
        "temperature_K": TEMPERATURE_K,
        "reference": {
            "K_H_mol_per_kg_per_bar": REFERENCE_K_H_MOL_PER_KG_PER_BAR,
            "K_H_window_min": REFERENCE_K_H_WINDOW_MIN,
            "K_H_window_max": REFERENCE_K_H_WINDOW_MAX,
            "Q_st_kJ_per_mol": REFERENCE_Q_ST_KJ_PER_MOL,
            "Q_st_window_min": REFERENCE_Q_ST_WINDOW_MIN,
            "Q_st_window_max": REFERENCE_Q_ST_WINDOW_MAX,
            "source": "Cmarik_2012_Langmuir_SI_Table_S6 (same as 3a)",
            "source_doi": "10.1021/la3035352",
        },
        "variants": {},
    }

    for variant in ("UA", "UAq", "EHq"):
        payload = execute_variant(variant)
        all_variants_payload["variants"][variant] = payload
        per_variant_path = VERDICTS_DIR / f"3b_{variant}.json"
        with per_variant_path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        print(f"  → wrote {per_variant_path.relative_to(REPO)}")

    combined_path = VERDICTS_DIR / "3b.json"
    with combined_path.open("w") as fp:
        json.dump(all_variants_payload, fp, indent=2)
    print(f"\nCombined verdict: {combined_path.relative_to(REPO)}")

    # Print summary table
    print("\n=== 3b Maia 2023 verdict summary ===")
    print(f"{'variant':<6} {'K_H mean':>10} {'K_H std':>9} {'Q_st mean':>10} "
          f"{'Q_st std':>9} {'strict':>7} {'window':>7}")
    for v in ("UA", "UAq", "EHq"):
        a = all_variants_payload["variants"][v]["aggregated"]
        vd = all_variants_payload["variants"][v]["verdict"]
        print(
            f"{v:<6} {a['K_H_mean_mol_per_kg_per_bar']:>10.3f} "
            f"{a['K_H_std_mol_per_kg_per_bar']:>9.3f} "
            f"{a['Q_st_mean_kJ_per_mol']:>10.2f} "
            f"{a['Q_st_std_kJ_per_mol']:>9.2f} "
            f"{vd['verdict_strict']:>7} {vd['verdict_window']:>7}"
        )


if __name__ == "__main__":
    main()
