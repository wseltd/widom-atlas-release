#!/usr/bin/env python3
"""WPB deployment-class leg — UMA (fairchem), PERIODIC conditioning-safe same-graph protocol.

Two harness hazards, both caught and fixed before any verdict:
  (1) Per-graph conditioning. UMA conditions on a per-GRAPH global charge/spin/task embedding, so the
      naive U = E(host+Ar) - E(host) - E(Ar) (three graphs) does NOT cancel -> a task-dependent
      120-470 kJ/mol far-field offset (wpb_uma_odac_GATE1_FAILED.json). Fix: same-graph reference,
      U(r) = E(host+Ar@r) - E(host+Ar@ref), identical atom count/conditioning -> cancels.
  (2) Cluster OOD. A non-periodic cut cluster (needed for a 'far' Ar in real space) is itself OOD for
      a periodic materials model: it passes the far-field check yet returns +800..+2900 kJ at open 4 A
      sites (garbage). Fix: evaluate PERIODICALLY (in-domain) and use an OPEN-PORE same-graph
      reference. Validated: two open pore points agree to 0.08 kJ; a 0.09 A overlap is +21572 kJ.

Recipe (locked, per expert guidance): uma-s-1.1 (NOT uma-s-1 — extensivity bug); task_name is a
LOAD-BEARING recipe element (UMA carries five DFT levels of theory) — recorded explicitly. Same seeded
insertion geometries, minimum-image distances, and VERDICT_SCHEMA_V06.md as the MACE legs. Device 0.
"""
import argparse, json, os
import numpy as np
from ase.io import read
from ase import Atoms

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="uma-s-1p1")
ap.add_argument("--task", default="omat")     # load-bearing: which DFT level of theory
ap.add_argument("--N", type=int, default=120)
ap.add_argument("--out", required=True)
a = ap.parse_args()

CIF = "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
GUEST, T_K, KB_EV, EV_TO_KJ, SEED, CUTOFF = "Ar", 298.15, 8.617333262e-5, 96.48533212, 0, 6.0
RT_kJ = KB_EV * EV_TO_KJ * T_K
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

def min_dist(cart):
    df = (fpos - cart) @ inv; df -= np.round(df); return float(np.linalg.norm(df @ cell, axis=1).min())

os.environ.setdefault("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit(a.model, device="cuda")
calc = FAIRChemCalculator(predictor, task_name=a.task)

def E_host_Ar(cart):
    at = frame.copy(); at += Atoms(GUEST, positions=[cart]); at.info["charge"] = 0; at.info["spin"] = 1
    at.calc = calc; return at.get_potential_energy()

# open-pore same-graph reference (max min-dist over a grid) + a second open point for Gate 1
g = np.linspace(0.05, 0.95, 12); grid = sorted((((md_:=min_dist(np.array([x, y, z]) @ cell)), np.array([x, y, z]) @ cell)
                                               for x in g for y in g for z in g), key=lambda t: t[0])
ref_cart = grid[-1][1]; ref2_cart = grid[-4][1]
e_ref = E_host_Ar(ref_cart)
gate1 = (E_host_Ar(ref2_cart) - e_ref) * EV_TO_KJ          # two open points: must be ~0
def U_kJ(cart): return (E_host_Ar(cart) - e_ref) * EV_TO_KJ

rng = np.random.default_rng(SEED); rows = []
for i in range(a.N):
    cart = rng.random(3) @ cell                            # SAME seeded points as the MACE legs
    dfrac = (fpos - cart) @ inv; dfrac -= np.round(dfrac); dist = np.linalg.norm(dfrac @ cell, axis=1)
    md = float(dist.min()); within = dist <= CUTOFF
    by = {}
    for s, r in zip(fsym[within], dist[within]): by.setdefault(s, []).append(r)
    ucl = lj_classical_U_K(by) * KB_EV * EV_TO_KJ
    uml = U_kJ(cart)
    rows.append({"i": i, "min_dist_A": round(md, 4), "U_classical_kJ": round(ucl, 3),
                 "U_mace_kJ": round(float(uml), 3),
                 "w_mace": float(np.exp(min(-uml / RT_kJ, 700.0))) if np.isfinite(uml) else 0.0,
                 "w_capped": bool(np.isfinite(uml) and (-uml / RT_kJ) > 700.0)})

sig_min = 0.5 * (UFF['O'][1] + UFF[GUEST][1]); HARD = 0.80 * sig_min; FLOOR = -25.0
for r in rows:
    hard = r["min_dist_A"] < HARD
    anom = np.isfinite(r["U_mace_kJ"]) and (r["U_mace_kJ"] < FLOOR or r["U_mace_kJ"] < r["U_classical_kJ"] - 50.0)
    r["flag_hard_overlap"] = bool(hard); r["flag_energetic_anomaly"] = bool(anom)
    r["flag_exponent_capped"] = bool(r["w_capped"]); r["flagged"] = bool(hard or anom or r["w_capped"])
wm = np.array([r["w_mace"] for r in rows]); fl = np.array([r["flagged"] for r in rows])
flagged_wf = float(wm[fl].sum() / wm.sum()) if wm.sum() > 0 else 0.0
Uarr = np.array([r["U_mace_kJ"] for r in rows])
screen_qst = float((KB_EV*EV_TO_KJ*T_K) - (np.sum(Uarr*wm)/wm.sum())) if wm.sum() > 0 else None
min_U_all = min(r["U_mace_kJ"] for r in rows)
gate1_pass = abs(gate1) < 1.0
if not gate1_pass:
    verdict = f"WITHHELD — Gate 1 offset {gate1:.1f} kJ (open-open ref failed); do not interpret"
elif min_U_all > 0:
    verdict = "REFUSE [Class II: no-physisorption] — min U > 0 over N (vs open-pore ref). Per VERDICT_SCHEMA_V06.md"
elif flagged_wf >= 0.5:
    verdict = "REFUSE [Class I: OOD over-binding] — flagged-weight >= 0.5. Per VERDICT_SCHEMA_V06.md"
else:
    verdict = "GOVERNED PASS WITH FLAGS" if fl.any() else "GOVERNED PASS"

summary = {"system": "Si-CHA + Ar (PERIODIC, open-pore same-graph ref)", "N": a.N, "seed": SEED, "T_K": T_K,
           "model": a.model, "task_name_LOADBEARING": a.task, "engine": "fairchem-UMA",
           "protocol": "periodic same-graph: U(r)=E(host+Ar@r)-E(host+Ar@open_pore); conditioning-safe, in-domain",
           "verdict_schema": "VERDICT_SCHEMA_V06.md",
           "ref_min_dist_A": round(grid[-1][0], 2), "gate1_second_open_U_kJ": round(float(gate1), 4),
           "gate1_pass": bool(gate1_pass),
           "n_flagged": int(fl.sum()), "flagged_weight_fraction": round(flagged_wf, 4),
           "min_U_mace_kJ": round(min_U_all, 2), "min_U_classical_kJ": round(min(r["U_classical_kJ"] for r in rows), 2),
           "max_U_mace_kJ": round(max(r["U_mace_kJ"] for r in rows), 1),
           "screen_q_st_kJ": (round(screen_qst,2) if screen_qst is not None else None),
           "screen_q_st_note": "N=120 random insertions; diagnostic not converged; dominated by deepest-weight insertion for Class-I cases",
           "n_exponent_capped": int(sum(r["flag_exponent_capped"] for r in rows)), "verdict": verdict}
json.dump(summary, open(a.out, "w"), indent=2)
json.dump(rows, open(a.out.replace(".json", "_per_insertion.json"), "w"), indent=1)
print(json.dumps(summary, indent=2))
