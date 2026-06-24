"""Site-truth replay driver: runs a native Widom pass on any site_truth_enabled
branch to extract the strongest-binding probe pose, then feeds it through
`extract_site_truth_verdict`.

Used in two ways:

  1. For branches whose strict-tier backend is already the native evaluator
     (currently 1b), the strongest-insertion is captured during the main
     production run — no replay needed.

  2. For branches whose strict-tier backend is RASPA3 or RASPA2 (2a, 5b, 1a),
     a separate native pass is run on the SAME framework + FF + cutoff to
     extract the strongest-binding configuration. K_H and Q_st from this
     replay are NOT used as the strict verdict — only the strongest-pose
     geometry is.

The replay deliberately uses the same Ewald parameters + cutoff +
shifted-truncated LJ convention as the strict-tier backend, so the
identified strongest pose is consistent with the simulation that produced
the K_H/Q_st verdict.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .ewald import EwaldParameters
from .runner import NativeWidomResult, StrongestInsertion, run_native_widom
from .site_truth import extract_site_truth_verdict
from .system import NativeSystem


def run_site_truth_replay(
    system: NativeSystem,
    site_truth_block: dict[str, Any],
    *,
    n_insertions: int = 50_000,
    n_seeds: int = 3,
    seed_base: int = 4242,
    enable_ewald: bool = True,
    temperature_K: float = 298.15,
) -> dict[str, Any]:
    """Run a native Widom pass on `system` and extract the site-truth verdict.

    Returns the verdict dict from `extract_site_truth_verdict` enriched with:
        replay_n_insertions, replay_n_seeds, replay_seed_base, replay_temperature_K.

    The strongest insertion across all seeds is selected. If `system.framework_charges_e`
    has any non-zero entries and `enable_ewald=False`, this will raise (matches
    the production runner's guard).
    """
    if not site_truth_block or not site_truth_block.get("enabled"):
        return {"passes_site_truth": None, "skipped": True, "reason": "site_truth not enabled"}

    ewald_params = EwaldParameters(
        alpha_inv_angstrom=0.3,
        real_cutoff_angstrom=system.energy_cutoff_angstrom,
        k_max_inv_angstrom=1.4,
    )

    all_runs: list[NativeWidomResult] = []
    strongest_overall: StrongestInsertion | None = None
    for s_idx in range(n_seeds):
        seed = seed_base + s_idx
        res = run_native_widom(
            system, temperature_K=temperature_K, n_insertions=n_insertions, seed=seed,
            enable_ewald=enable_ewald, ewald_parameters=ewald_params, batch_size=2000,
        )
        all_runs.append(res)
        if res.strongest_insertion is not None:
            if strongest_overall is None or res.strongest_insertion.U_K < strongest_overall.U_K:
                strongest_overall = res.strongest_insertion

    verdict = extract_site_truth_verdict(strongest_overall, site_truth_block)
    verdict.update({
        "replay_backend": "native_widom_v04_site_truth_replay",
        "replay_n_insertions_per_seed": n_insertions,
        "replay_n_seeds": n_seeds,
        "replay_seed_base": seed_base,
        "replay_temperature_K": temperature_K,
        "replay_K_H_298_per_seed": [r.K_H_mol_per_kg_per_Pa for r in all_runs],
        "replay_n_overlaps_per_seed": [r.n_overlaps for r in all_runs],
    })
    if all_runs:
        kh_mean = statistics.fmean([r.K_H_mol_per_kg_per_Pa for r in all_runs])
        verdict["replay_K_H_mean_mol_per_kg_per_Pa"] = kh_mean
    return verdict
