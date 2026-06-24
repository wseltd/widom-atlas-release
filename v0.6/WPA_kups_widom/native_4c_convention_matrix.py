#!/usr/bin/env python3
"""C1: B1 convention matrix. Re-run native 4c (Si-CHA + CO2, Ewald) FRESH this session, toggling the
LJ shift (shifted vs truncated) and the analytical LJ tail, to decompose the kUPS-vs-native +2.2%.
This confirms native's K_H is re-derived on the identical recipe (not quoted from the v0.4 lock) and
quantifies each convention so a small net % is not hiding a cancellation of larger ones.

shift on  = RASPA/native convention (subtract U(rc));  shift off = kUPS plain lennard_jones_energy.
tail on   = analytical LJ tail (4c recipe = ON);       tail off = bare truncation.
"""
import json, math, time
from pathlib import Path
import numpy as np
from widom_atlas.v04.native.bai_2013_trappe_zeo_loader import load_bai_2013_native_system
from widom_atlas.v04.native.ewald import EwaldParameters
from widom_atlas.v04.native.runner import run_native_widom
from widom_atlas.v04.native.tail_correction import total_lj_tail_for_probe

REPO = Path(".")
CIF = REPO / "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
T, SEED, N = 298.0, 0, 60_000

corners = {}
for shifted in (True, False):
    sysm, meta = load_bai_2013_native_system(
        repo_root=REPO, cif_path=CIF, gas_species="CO2",
        cutoff_angstrom=14.0, apply_lj_tail_correction=True, lj_shifted=shifted)
    tail_K = total_lj_tail_for_probe(
        probe_atom_types=list(sysm.probe.types),
        framework_type_counts=meta["framework_atom_type_counts_in_supercell"],
        framework_self_lj=meta["framework_self_lj"], probe_self_lj=meta["probe_self_lj"],
        cell_volume_angstrom3=meta["cell_volume_angstrom3_supercell"],
        cutoff_angstrom=14.0)
    t0 = time.time()
    res = run_native_widom(system=sysm, temperature_K=T, n_insertions=N, seed=SEED,
                           enable_ewald=True,
                           ewald_parameters=EwaldParameters(real_cutoff_angstrom=14.0))
    dt = time.time() - t0
    KH_notail = res.K_H_mol_per_kg_per_Pa * 1e5
    KH_tail = KH_notail * math.exp(-tail_K / T)
    qst_notail = res.Q_st_kJ_per_mol
    qst_tail = qst_notail - tail_K * 8.314e-3
    lab = "shifted" if shifted else "truncated"
    corners[lab] = {"tail_off": {"K_H": KH_notail, "q_st": qst_notail},
                    "tail_on": {"K_H": KH_tail, "q_st": qst_tail}, "tail_K": tail_K, "wall_s": round(dt, 1)}
    print(f"{lab:9s}: K_H tail_off={KH_notail:.4f} tail_on={KH_tail:.4f} | q_st tail_on={qst_tail:.2f} | tailK={tail_K:.2f} | {dt:.0f}s")

s_on = corners["shifted"]["tail_on"]["K_H"]; t_on = corners["truncated"]["tail_on"]["K_H"]
s_off = corners["shifted"]["tail_off"]["K_H"]; t_off = corners["truncated"]["tail_off"]["K_H"]
shift_pct_tailon = 100 * (t_on / s_on - 1)
shift_pct_tailoff = 100 * (t_off / s_off - 1)
tail_pct_shifted = 100 * (s_on / s_off - 1)
out = {
    "system": "4c Si-CHA + CO2 (TraPPE, charges, Ewald, 14A), fresh native re-run this session",
    "N_insertions": N, "seed": SEED, "T_K": T,
    "corners_K_H_mol_kg_bar": {"shifted+tail": s_on, "truncated+tail": t_on,
                               "shifted_notail": s_off, "truncated_notail": t_off},
    "native_shifted_tail_THIS_RUN": round(s_on, 4),
    "v04_lock_value": 2.2227, "lock_provenance": "native_widom_v04 backend, executed_at_utc 2026-06-01 (NOT this session)",
    "kups_value": 2.2706,
    "one_toggle_decomposition_pct": {
        "shift_off_minus_on_tailON": round(shift_pct_tailon, 1),
        "shift_off_minus_on_tailOFF": round(shift_pct_tailoff, 1),
        "tail_on_minus_off_shifted": round(tail_pct_shifted, 1),
    },
    "kups_vs_native_shifted_tail_pct": round(100 * (2.2706 / s_on - 1), 1),
    "kups_vs_native_truncated_tail_pct": round(100 * (2.2706 / t_on - 1), 1),
}
json.dump(out, open(REPO / "v0.6/WPA_kups_widom/native_4c_convention_matrix.json", "w"), indent=2)
print(json.dumps(out["one_toggle_decomposition_pct"], indent=1))
print(f"kUPS vs native(shifted+tail, this run): {out['kups_vs_native_shifted_tail_pct']:+.1f}%")
print(f"kUPS vs native(truncated+tail, kUPS's own convention): {out['kups_vs_native_truncated_tail_pct']:+.1f}%")
