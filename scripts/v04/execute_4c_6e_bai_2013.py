"""Execute 4c (Si-CHA + CO2) and 6e (MFI + CH4) under Bai 2013 TraPPE-zeo.

Both branches were `blocked_pending_Bai_main_parameter_tables` until the
2026-06-01 cross-verification against RASPA3's bundled force-field JSONs
confirmed primary-anchored Si/O LJ + charges.

Targets:
  4c reference: Maghsoudi 2013 Toth K_H = 2.43 mol/(kg.bar), Q_st = 21 kJ/mol
  6e reference: Hufton 1993 K_H = 0.89 mol/(kg.bar) (chromatographic);
                Dunne 1996 Q_st = 20.9 kJ/mol (Tian-Calvet calorimetric)
  6e smoke-test cross-check: Shah 2015 simulated K_H ~ 0.60 mol/(kg.bar)
    (same TraPPE-zeo + TraPPE-UA CH4 stack)

Writes:
  evidence/v04_4c_bai_2013/verdicts/4c.json
  evidence/v04_6e_bai_2013/verdicts/6e.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from widom_atlas.v04.bayesian_comparator import compare_K_H_in_log_space
from widom_atlas.v04.native.bai_2013_trappe_zeo_loader import (
    load_bai_2013_native_system,
)
from widom_atlas.v04.native.ewald import EwaldParameters
from widom_atlas.v04.native.runner import run_native_widom
from widom_atlas.v04.native.tail_correction import total_lj_tail_for_probe
from widom_atlas.v04.thresholds import compute_two_tier_verdict

REPO = Path(".")

# IZA CIF paths (already in repo)
CIF_PATHS = {
    "Si_CHA": REPO / "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif",
    "MFI":    REPO / "docs/research/dataset-research-for-v0.4/7/MFI_iza.cif",
}

TEMPERATURE_K = 298.0
SEEDS = [0, 1, 2]
N_INSERTIONS = 60_000  # tighter budget; zeolites converge faster than MOFs


def execute_branch(
    branch_id: str,
    case_id: str,
    cif_path: Path,
    gas_species: str,
    label: str,
    K_H_ref_mol_per_kg_per_bar: float,
    Q_st_ref_kJ_per_mol: float,
    K_H_ref_source: str,
    Q_st_ref_source: str,
    smoke_test_K_H_simulated: float | None = None,
    smoke_test_source: str | None = None,
) -> dict:
    print(f"\n=== {branch_id}: {label} ===")
    sys, meta = load_bai_2013_native_system(
        repo_root=REPO,
        cif_path=cif_path,
        gas_species=gas_species,
        cutoff_angstrom=14.0,
        apply_lj_tail_correction=True,
        lj_shifted=True,
    )
    print(
        f"  framework atoms: {sys.n_framework_atoms}, supercell: {sys.supercell_replicas}"
    )
    print(f"  cell volume (supercell): {meta['cell_volume_angstrom3_supercell']:.1f} A^3")

    # Compute the analytical LJ tail correction for this probe + framework setup.
    framework_self_lj_for_tail = meta["framework_self_lj"]
    probe_self_lj_for_tail = meta["probe_self_lj"]
    tail_offset_K = total_lj_tail_for_probe(
        probe_atom_types=list(sys.probe.types),
        framework_type_counts=meta["framework_atom_type_counts_in_supercell"],
        framework_self_lj=framework_self_lj_for_tail,
        probe_self_lj=probe_self_lj_for_tail,
        cell_volume_angstrom3=meta["cell_volume_angstrom3_supercell"],
        cutoff_angstrom=14.0,
    )
    print(f"  LJ tail correction offset: {tail_offset_K:.3f} K (added to every insertion)")

    # Ewald only if probe has charges (CO2) or framework charged + probe charged.
    enable_ewald = (
        bool(np.any(sys.framework_charges_e != 0.0))
        and bool(np.any(sys.probe.charges_e != 0.0))
    )
    ewald_params = (
        EwaldParameters(
            alpha_inv_angstrom=0.3,
            real_cutoff_angstrom=14.0,
            k_max_inv_angstrom=1.4,
        )
        if enable_ewald else None
    )
    # If probe has no charges (CH4) but framework does (Si/O), we still need
    # Ewald to handle the framework-framework self-term consistency; but for
    # the Widom Δ-energy of a neutral probe, the framework-probe Coulomb
    # cross-term vanishes. We can therefore skip Ewald safely for CH4.
    if not bool(np.any(sys.probe.charges_e != 0.0)):
        enable_ewald = False
        ewald_params = None
        # But the runner refuses to run with charged framework + no Ewald.
        # Zero out the framework charges (probe doesn't see them anyway).
        sys.framework_charges_e = np.zeros_like(sys.framework_charges_e)

    seed_results = []
    t_total = 0.0
    for seed in SEEDS:
        print(f"  seed {seed}: running...", flush=True)
        t0 = time.time()
        res = run_native_widom(
            system=sys,
            temperature_K=TEMPERATURE_K,
            n_insertions=N_INSERTIONS,
            seed=seed,
            enable_ewald=enable_ewald,
            ewald_parameters=ewald_params,
        )
        dt = time.time() - t0
        t_total += dt

        # Apply analytical tail correction as a constant ΔU offset:
        # K_H scales by exp(-ΔU_tail / kT); Q_st gets the tail directly added.
        # Equivalent to multiplying K_H by exp(-tail/T) since accumulator was
        # over un-corrected energies. Cleanest: re-derive K_H from the
        # mean Boltzmann factor + tail offset.
        beta_inv_K = TEMPERATURE_K
        K_H_pa_no_tail = res.K_H_mol_per_kg_per_Pa
        K_H_pa_with_tail = K_H_pa_no_tail * math.exp(-tail_offset_K / beta_inv_K)
        K_H_bar_no_tail = K_H_pa_no_tail * 1e5
        K_H_bar_with_tail = K_H_pa_with_tail * 1e5
        Q_st_no_tail = res.Q_st_kJ_per_mol
        # Q_st correction: tail offset is in K -> kJ/mol via 8.314e-3
        Q_st_with_tail = Q_st_no_tail - tail_offset_K * 8.314e-3

        print(
            f"    seed {seed}: wall={dt:.1f}s, "
            f"K_H (no tail) = {K_H_bar_no_tail:.4f}, "
            f"K_H (with tail) = {K_H_bar_with_tail:.4f} mol/(kg.bar), "
            f"Q_st (with tail) = {Q_st_with_tail:.2f} kJ/mol",
            flush=True,
        )
        seed_results.append({
            "seed": seed,
            "n_insertions": N_INSERTIONS,
            "wall_seconds": dt,
            "K_H_mol_per_kg_per_bar_without_tail": float(K_H_bar_no_tail),
            "K_H_mol_per_kg_per_bar_with_tail": float(K_H_bar_with_tail),
            "Q_st_kJ_per_mol_without_tail": float(Q_st_no_tail),
            "Q_st_kJ_per_mol_with_tail": float(Q_st_with_tail),
        })

    K_H_with_tail = [r["K_H_mol_per_kg_per_bar_with_tail"] for r in seed_results]
    Q_st_with_tail = [r["Q_st_kJ_per_mol_with_tail"] for r in seed_results]
    K_H_mean = float(np.mean(K_H_with_tail))
    K_H_std = float(np.std(K_H_with_tail, ddof=1))
    Q_st_mean = float(np.mean(Q_st_with_tail))
    Q_st_std = float(np.std(Q_st_with_tail, ddof=1))

    print(
        f"  -> {branch_id} mean K_H = {K_H_mean:.4f} ± {K_H_std:.4f} mol/(kg.bar), "
        f"Q_st = {Q_st_mean:.2f} ± {Q_st_std:.2f} kJ/mol, wall_total = {t_total:.1f}s"
    )

    # Tier A + Tier B verdict
    v = compute_two_tier_verdict(
        case_id=case_id, branch_id=branch_id,
        K_H_mean_mol_per_kg_per_bar=K_H_mean,
        K_H_reference_mol_per_kg_per_bar=K_H_ref_mol_per_kg_per_bar,
        Q_st_mean_kJ_per_mol=Q_st_mean,
        Q_st_reference_kJ_per_mol=Q_st_ref_kJ_per_mol,
    )
    # Bayesian Z
    tier_b_delta = (
        v.tier_B_band.delta_log10_KH_max
        if v.tier_B_band is not None and v.tier_B_band.delta_log10_KH_max != float("inf")
        else None
    )
    bayes = compare_K_H_in_log_space(
        K_H_sim_mol_per_kg_per_bar=K_H_mean,
        K_H_sim_seed_values_mol_per_kg_per_bar=K_H_with_tail,
        K_H_exp_mol_per_kg_per_bar=K_H_ref_mol_per_kg_per_bar,
        K_H_exp_log10_std=None,  # use default per-system
        tier_b_delta_log10_threshold=tier_b_delta,
    )

    payload = {
        "branch_id": branch_id,
        "case_id": case_id,
        "label": label,
        "force_field": meta,
        "executed_backend": "native_widom_v04",
        "executed_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "temperature_K": TEMPERATURE_K,
        "n_insertions_per_seed": N_INSERTIONS,
        "n_seeds": len(SEEDS),
        "tail_correction_offset_K": float(tail_offset_K),
        "seeds": seed_results,
        "K_H_mean_mol_per_kg_per_bar": K_H_mean,
        "K_H_std_mol_per_kg_per_bar": K_H_std,
        "Q_st_mean_kJ_per_mol": Q_st_mean,
        "Q_st_std_kJ_per_mol": Q_st_std,
        "reference_K_H_mol_per_kg_per_bar": K_H_ref_mol_per_kg_per_bar,
        "reference_Q_st_kJ_per_mol": Q_st_ref_kJ_per_mol,
        "reference_K_H_source": K_H_ref_source,
        "reference_Q_st_source": Q_st_ref_source,
        "smoke_test_K_H_simulated": smoke_test_K_H_simulated,
        "smoke_test_source": smoke_test_source,
        "two_tier_verdict": v.to_dict(),
        "bayesian_log_space_comparison": {
            "delta_log10": bayes.delta_log10,
            "combined_log10_std": bayes.combined_log10_std,
            "z_score": bayes.z_score,
            "classification": bayes.classification,
            "p_agreement_within_tier_b_band": bayes.p_agreement_within_tier_b_band,
        },
    }

    out_dir = REPO / f"evidence/v04_{branch_id}_bai_2013/verdicts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{branch_id}.json"
    with out_path.open("w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"  wrote {out_path.relative_to(REPO)}")
    return payload


def main():
    p_4c = execute_branch(
        branch_id="4c",
        case_id="4",
        cif_path=CIF_PATHS["Si_CHA"],
        gas_species="CO2",
        label="Si-CHA + CO2 (Bai 2013 TraPPE-zeo + TraPPE-CO2, Lorentz-Berthelot, shifted-truncated, 14 A cutoff + LJ tail)",
        K_H_ref_mol_per_kg_per_bar=2.43,
        Q_st_ref_kJ_per_mol=21.0,
        K_H_ref_source="Maghsoudi 2013 Adsorption (DOI 10.1007/s10450-013-9528-1) Toth fit",
        Q_st_ref_source="Maghsoudi 2013 Adsorption (van't Hoff)",
        smoke_test_K_H_simulated=None,
        smoke_test_source=None,
    )

    p_6e = execute_branch(
        branch_id="6e",
        case_id="6",
        cif_path=CIF_PATHS["MFI"],
        gas_species="CH4",
        label="MFI + CH4 (Bai 2013 TraPPE-zeo + TraPPE-UA CH4, Lorentz-Berthelot, shifted-truncated, 14 A cutoff + LJ tail)",
        K_H_ref_mol_per_kg_per_bar=0.89,
        Q_st_ref_kJ_per_mol=20.9,
        K_H_ref_source="Hufton 1993 AIChE J (DOI 10.1002/aic.690390605) chromatographic",
        Q_st_ref_source="Dunne 1996 Langmuir (DOI 10.1021/la960495z) Tian-Calvet calorimetric",
        smoke_test_K_H_simulated=0.60,
        smoke_test_source="Shah, Tsapatsis, Siepmann 2015 Langmuir SI (DOI 10.1021/acs.langmuir.5b03015)",
    )

    print("\n=== HEADLINE ===")
    for p in (p_4c, p_6e):
        v = p["two_tier_verdict"]
        b = p["bayesian_log_space_comparison"]
        print(
            f"{p['branch_id']:<5} K_H={p['K_H_mean_mol_per_kg_per_bar']:.3f} (ref {p['reference_K_H_mol_per_kg_per_bar']:.3f}) "
            f"Q_st={p['Q_st_mean_kJ_per_mol']:.2f} (ref {p['reference_Q_st_kJ_per_mol']:.2f}) "
            f"Δlog10={v['delta_log10_K_H']:+.3f}, ΔQ={v['delta_Q_st_kJ_per_mol']:+.2f}, "
            f"Tier A {v['tier_A_regression']['composite']}, "
            f"Tier B {v['tier_B_physical']['composite'] if v['tier_B_physical'] else 'n/a'}, "
            f"|Z|={abs(b['z_score']):.2f}, {v['headline_disposition']}"
        )


if __name__ == "__main__":
    main()
