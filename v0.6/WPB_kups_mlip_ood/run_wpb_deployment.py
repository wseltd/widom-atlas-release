#!/usr/bin/env python3
"""WPB deployment-class leg (UMA / ODAC via fairchem) — READY TO RUN.

Same seeded insertion geometries as run_wpb_ood.py (Si-CHA + Ar, seed 0, N=120), same OOD flags and
the pre-registered VERDICT_SCHEMA_V06.md, same Gate-1 offset checks. The only change is the scorer:
a fairchem deployment-class model (UMA multi-task, or the ODAC25-trained eSEN) instead of MACE.

Blocked tonight on HF gated access (facebook/UMA, facebook/ODAC25 → 403 GatedRepoError for the
authenticated account). The fix is a one-click license acceptance on the model page while logged in;
this script then runs unchanged. Run in venv-fairchem:

  HF_TOKEN=$(cat ~/.cache/huggingface/token) CUDA_VISIBLE_DEVICES=0 \
    ~/venvs/venv-fairchem/bin/python run_wpb_deployment.py \
    --model uma-s-1p1 --task odac --out wpb_uma.json
  # or: --model esen-sm-full-odac25 --task odac --out wpb_odac.json
"""
import argparse, hashlib, json, os
import numpy as np
from ase.io import read
from ase import Atoms

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)          # uma-s-1p1 | esen-sm-full-odac25 | ...
ap.add_argument("--task", default="odac")          # UMA task head; ODAC-specific models ignore
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

os.environ.setdefault("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit(a.model, device="cuda")
try:
    calc = FAIRChemCalculator(predictor, task_name=a.task)
except TypeError:
    calc = FAIRChemCalculator(predictor)

def E(atoms):
    atoms = atoms.copy(); atoms.calc = calc
    return atoms.get_potential_energy()

e_frame = E(frame)
ag = Atoms(GUEST, positions=[[0, 0, 0]], cell=cell, pbc=True); e_guest = E(ag)

# Gate-1 offset checks (same as MACE leg): isolated-Ar far must -> ~0
frame0 = read(CIF); iso = frame0.copy(); iso.set_pbc(False); ipos = iso.get_positions()
big = float(np.ptp(ipos, axis=0).max()) + 40.0
isobox = Atoms(iso.get_chemical_symbols(), positions=ipos, cell=[big, big, big], pbc=False)
e_iso = E(isobox); ag2 = Atoms(GUEST, positions=[[0, 0, 0]], cell=[big, big, big], pbc=False); e_guest_iso = E(ag2)
far = ipos.mean(axis=0) + np.array([0, 0, np.ptp(ipos[:, 2]) / 2 + 12.0])
at_far = Atoms(list(iso.get_chemical_symbols()) + [GUEST], positions=np.vstack([ipos, far]), cell=[big, big, big], pbc=False)
U_isolated_far = (E(at_far) - e_iso - e_guest_iso) * EV_TO_KJ

def U_kJ(cart):
    at = frame.copy(); at += Atoms(GUEST, positions=[cart])
    return (E(at) - e_frame - e_guest) * EV_TO_KJ

rng = np.random.default_rng(SEED); rows = []
for i in range(a.N):
    cart = rng.random(3) @ cell
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
min_U_all = min(r["U_mace_kJ"] for r in rows)
if min_U_all > 0:
    verdict = "REFUSE [Class II: no-physisorption] — min U > 0 over N. Per VERDICT_SCHEMA_V06.md"
elif flagged_wf >= 0.5:
    verdict = "REFUSE [Class I: OOD over-binding] — flagged-weight >= 0.5. Per VERDICT_SCHEMA_V06.md"
else:
    verdict = "GOVERNED PASS WITH FLAGS" if fl.any() else "GOVERNED PASS"

kups_zip = os.path.expanduser("~/models/kups-mace-jax/mace-mpa-0-medium_32.zip")
summary = {"system": "Si-CHA + Ar", "N": a.N, "seed": SEED, "T_K": T_K,
           "model": a.model, "task": a.task, "engine": "fairchem", "verdict_schema": "VERDICT_SCHEMA_V06.md",
           "gate1_U_isolated_Ar_far_kJ": round(float(U_isolated_far), 3),
           "gate1_expect": "~0 if differencing correct; constant offset => reference issue (root-cause before claims)",
           "n_flagged": int(fl.sum()), "flagged_weight_fraction": round(flagged_wf, 4),
           "min_U_mace_kJ": round(min_U_all, 2),
           "min_U_classical_kJ": round(min(r["U_classical_kJ"] for r in rows), 2),
           "n_exponent_capped": int(sum(r["flag_exponent_capped"] for r in rows)), "verdict": verdict}
json.dump(summary, open(a.out, "w"), indent=2)
json.dump(rows, open(a.out.replace(".json", "_per_insertion.json"), "w"), indent=1)
print(json.dumps(summary, indent=2))
