#!/usr/bin/env python3
"""WPB — OOD governance demo on kUPS's shipped MLIP (MACE-MPA-0-medium).

Reuses the validated v0.5 WP3 protocol (Si-CHA + Ar, seed 0, classical UFF LJ
baseline vs MLIP scoring the SAME insertion geometries, two transparent OOD flags,
flagged-Boltzmann-weight fraction, REFUSE/PASS verdict). Here the MLIP is
MACE-MPA-0-medium — bit-identical to kUPS's HF export `CuspAI/kUPS-mace-jax`
(model card: re-export, not retraining). Discrimination question: does MACE-MPA-0
(2024 foundation model) behave better in overlap regions than MACE-MP-small (2023)?

Governance note (from kUPS's own model card): MACE-MPA-0 is "Not trained for
isolated molecules" -> scoring an adsorbate guest is out-of-domain by the authors'
own statement. Device 0 only; torch env (the JAX-in-kUPS-engine path is the extension).
"""
import argparse, hashlib, json, os
import numpy as np
from ase.io import read
from ase import Atoms

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medium-mpa-0")   # kUPS's model; or "small" for v0.5 comparison
ap.add_argument("--N", type=int, default=120)
ap.add_argument("--out", required=True)
ap.add_argument("--dispersion", action="store_true")  # MACE + D3(BJ) for the gate-2 leg
a = ap.parse_args()

CIF = "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
GUEST, T_K, KB_EV, EV_TO_KJ, SEED, CUTOFF = "Ar", 298.15, 8.617333262e-5, 96.48533212, 0, 6.0
KT = KB_EV * T_K              # eV (unused by weights; kept for reference)
RT_kJ = KB_EV * EV_TO_KJ * T_K  # = R*T in kJ/mol (2.478 @298.15) — energies are kJ/mol
UFF = {"Si": (202.29, 3.826), "O": (30.19, 3.500), "Ar": (185.0, 3.405)}

def lj_classical_U_K(by_elem):
    eps_g, sig_g = UFF[GUEST]; u = 0.0
    for elem, dists in by_elem.items():
        eps_f, sig_f = UFF.get(elem, (0.0, 3.5))
        eps = (eps_f * eps_g) ** 0.5; sig = 0.5 * (sig_f + sig_g)
        d = np.asarray(dists); d = d[(d > 1e-6) & (d <= CUTOFF)]
        sr6 = (sig / d) ** 6; u += float(np.sum(4.0 * eps * (sr6 * sr6 - sr6)))
    return u

frame = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell); inv = np.linalg.inv(cell)
fpos = frame.get_positions(); fsym = np.array(frame.get_chemical_symbols())

from mace.calculators import mace_mp
import torch
calc = mace_mp(model=a.model, device="cuda", default_dtype="float64", dispersion=a.dispersion)
ck = sorted([c for c in (__import__("glob").glob(os.path.expanduser("~/.cache/mace/**/*"), recursive=True)) if os.path.isfile(c)])
ck_sha = hashlib.sha256(open(ck[-1], "rb").read()).hexdigest()[:16] if ck else "?"
kups_zip = os.path.expanduser("~/models/kups-mace-jax/mace-mpa-0-medium_32.zip")
kups_sha = hashlib.sha256(open(kups_zip, "rb").read()).hexdigest()[:16] if os.path.exists(kups_zip) else "?"

def mace_U_kJ(cart):
    at = frame.copy(); at += Atoms(GUEST, positions=[cart]); at.calc = calc
    return (at.get_potential_energy() - e_frame - e_guest) * EV_TO_KJ
e_frame = (frame.copy(), )  # placeholder
at0 = frame.copy(); at0.calc = calc; e_frame = at0.get_potential_energy()
ag = Atoms(GUEST, positions=[[0, 0, 0]], cell=cell, pbc=True); ag.calc = calc; e_guest = ag.get_potential_energy()

rng = np.random.default_rng(SEED); rows = []
for i in range(a.N):
    cart = rng.random(3) @ cell
    dfrac = (fpos - cart) @ inv; dfrac -= np.round(dfrac); dist = np.linalg.norm(dfrac @ cell, axis=1)
    md = float(dist.min()); within = dist <= CUTOFF
    by = {}
    for s, r in zip(fsym[within], dist[within]): by.setdefault(s, []).append(r)
    ucl = lj_classical_U_K(by) * KB_EV * EV_TO_KJ
    uml = mace_U_kJ(cart)
    rows.append({"i": i, "min_dist_A": round(md, 4), "U_classical_kJ": round(ucl, 3),
                 "U_mace_kJ": round(float(uml), 3),
                 "w_mace": float(np.exp(min(-uml / RT_kJ, 700.0))) if np.isfinite(uml) else 0.0,
                 "w_capped": bool(np.isfinite(uml) and (-uml / RT_kJ) > 700.0)})

sig_min = 0.5 * (UFF['O'][1] + UFF[GUEST][1]); HARD = 0.80 * sig_min; FLOOR = -25.0
for r in rows:
    hard = r["min_dist_A"] < HARD
    anom = np.isfinite(r["U_mace_kJ"]) and (r["U_mace_kJ"] < FLOOR or r["U_mace_kJ"] < r["U_classical_kJ"] - 50.0)
    cap = bool(r.get("w_capped", False))
    r["flag_hard_overlap"] = bool(hard); r["flag_energetic_anomaly"] = bool(anom)
    r["flag_exponent_capped"] = cap; r["flagged"] = bool(hard or anom or cap)
wm = np.array([r["w_mace"] for r in rows]); fl = np.array([r["flagged"] for r in rows])
flagged_wf = float(wm[fl].sum() / wm.sum()) if wm.sum() > 0 else 0.0
KH_all = float(np.mean(wm)); KH_clean = float(np.mean(wm[~fl])) if np.any(~fl) else 0.0
Uarr = np.array([r["U_mace_kJ"] for r in rows])
screen_qst = float((KB_EV*EV_TO_KJ*T_K) - (np.sum(Uarr*wm)/wm.sum())) if wm.sum() > 0 else None
min_U_all = min(r["U_mace_kJ"] for r in rows)
if min_U_all > 0:
    verdict = "REFUSE [Class II: no-physisorption] — min U > 0 over N insertions; physisorption is established for this pair (6c Ar/silica q_st=15.7 kJ/mol >> RT). Per VERDICT_SCHEMA_V06.md"
elif flagged_wf >= 0.5:
    verdict = "REFUSE [Class I: OOD over-binding] — flagged-weight fraction >= 0.5. Per VERDICT_SCHEMA_V06.md"
else:
    verdict = "GOVERNED PASS WITH FLAGS" if fl.any() else "GOVERNED PASS"
# overlap survival test (v0.5 #20-style): deepest-weight insertion at an overlap
ov = [r for r in rows if r["min_dist_A"] < 2.0 and r["U_classical_kJ"] > 1000]
ov_examples = sorted(ov, key=lambda r: -r["w_mace"])[:3]
summary = {"system": "Si-CHA + Ar", "N": a.N, "seed": SEED, "T_K": T_K,
           "model": a.model, "dispersion_D3": bool(a.dispersion), "mace_checkpoint_sha16": ck_sha,
           "kups_mace_jax_zip_sha16": kups_sha,
           "model_card_note": "MACE-MPA-0 'Not trained for isolated molecules' (kUPS HF model card) -> scoring an adsorbate is OOD by the authors' statement",
           "n_flagged": int(fl.sum()), "flagged_fraction": round(float(fl.mean()), 4),
           "flagged_weight_fraction": round(flagged_wf, 4),
           "KH_proxy_all": round(KH_all, 4), "KH_proxy_flagged_removed": round(KH_clean, 4),
           "inflation_factor": round(KH_all / KH_clean, 1) if KH_clean > 0 else None,
           "min_U_mace_kJ": round(min(r["U_mace_kJ"] for r in rows), 2),
           "min_U_classical_kJ": round(min(r["U_classical_kJ"] for r in rows), 2),
           "screen_q_st_kJ": (round(screen_qst,2) if screen_qst is not None else None),
           "screen_q_st_note": "N=120 random insertions; diagnostic not converged; dominated by deepest-weight insertion for Class-I cases",
           "n_exponent_capped": int(sum(r.get("flag_exponent_capped", False) for r in rows)),
           "verdict_schema": "VERDICT_SCHEMA_V06.md",
           "overlap_survival_examples": ov_examples, "verdict": verdict}
json.dump(summary, open(a.out, "w"), indent=2)
json.dump(rows, open(a.out.replace(".json", "_per_insertion.json"), "w"), indent=1)
print(json.dumps(summary, indent=2))
