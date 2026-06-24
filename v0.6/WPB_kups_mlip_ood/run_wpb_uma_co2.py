#!/usr/bin/env python3
"""WPB CO2/CHA leg — tests whether UMA's odac head (trained on MOF/CO2/H2O adsorption, i.e. CO2 is
IN-distribution) is self-consistent and well-behaved where the Ar leg found it OOD.

Same periodic conditioning-safe same-graph protocol as run_wpb_uma.py. Probe = rigid linear CO2
(C-O 1.16 A), inserted at seeded points with random orientation. min_dist = closest of the three CO2
atoms to any framework atom. Classical baseline = TraPPE-CO2 LJ (C 27.0 K/2.80 A, O 79.0 K/3.05 A,
LB-mixed to framework), used only for the energetic-anomaly flag; electrostatics omitted from the
baseline (screen, not a parity run). VERDICT_SCHEMA_V06.md. Device 0, venv-fairchem.

Key comparison printed: Gate-1 open-open consistency for odac vs omat. For Ar/CHA odac failed Gate 1
(2.7 kJ at equivalent sites) while omat passed (0.19). If odac passes for CO2 (its training domain),
that is the in-domain control: the governance withhold for Ar was correct OOD detection.
"""
import argparse, json, os
import numpy as np
from ase.io import read
from ase import Atoms

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="uma-s-1p1")
ap.add_argument("--task", default="odac")
ap.add_argument("--N", type=int, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()

CIF = "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
T_K, KB_EV, EV_TO_KJ, SEED, CUTOFF = 298.15, 8.617333262e-5, 96.48533212, 0, 6.0
RT_kJ = KB_EV * EV_TO_KJ * T_K
DCO = 1.16  # C=O bond, Angstrom
# TraPPE-CO2 LJ self params (eps K, sigma A); framework UFF
LJ = {"C": (27.0, 2.80), "O_co2": (79.0, 3.05), "Si": (202.29, 3.826), "O": (30.19, 3.500)}

frame = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell); inv = np.linalg.inv(cell)
fpos = frame.get_positions(); fsym = np.array(frame.get_chemical_symbols())

def co2_atoms(center, axis):
    axis = axis / np.linalg.norm(axis)
    return ["C", "O", "O"], np.array([center, center + DCO * axis, center - DCO * axis])

def min_dist_co2(pos3):
    md = 1e9
    for p in pos3:
        df = (fpos - p) @ inv; df -= np.round(df); md = min(md, float(np.linalg.norm(df @ cell, axis=1).min()))
    return md

def lj_classical_kJ(pos3, elems3):
    u = 0.0
    for p, e in zip(pos3, elems3):
        eg, sg = LJ["O_co2"] if e == "O" else LJ["C"]
        df = (fpos - p) @ inv; df -= np.round(df); dist = np.linalg.norm(df @ cell, axis=1)
        for s in ("Si", "O"):
            dd = dist[(fsym == s) & (dist <= CUTOFF) & (dist > 1e-6)]
            ef, sf = LJ[s]; eps = (ef * eg) ** 0.5; sig = 0.5 * (sf + sg)
            sr6 = (sig / dd) ** 6; u += float(np.sum(4.0 * eps * (sr6 * sr6 - sr6)))
    return u * KB_EV * EV_TO_KJ

os.environ.setdefault("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit(a.model, device="cuda")
calc = FAIRChemCalculator(predictor, task_name=a.task)

def E(center, axis):
    el, pos = co2_atoms(center, axis)
    at = frame.copy(); at += Atoms(el, positions=pos); at.info["charge"] = 0; at.info["spin"] = 1
    at.calc = calc; return at.get_potential_energy()

def mdist(c):
    df = (fpos - c) @ inv; df -= np.round(df); return float(np.linalg.norm(df @ cell, axis=1).min())

g = np.linspace(0.05, 0.95, 12)
grid = sorted((((m := mdist(np.array([x, y, z]) @ cell)), np.array([x, y, z]) @ cell)
               for x in g for y in g for z in g), key=lambda t: t[0])
zax = np.array([0.0, 0.0, 1.0])
e_ref = E(grid[-1][1], zax)
gate1 = (E(grid[-2][1], zax) - e_ref) * EV_TO_KJ          # two equivalent open sites (same min_dist)

rng = np.random.default_rng(SEED); rows = []
for i in range(a.N):
    center = rng.random(3) @ cell
    axis = rng.normal(size=3)
    el, pos = co2_atoms(center, axis)
    md = min_dist_co2(pos)
    ucl = lj_classical_kJ(pos, el)
    uml = (E(center, axis) - e_ref) * EV_TO_KJ
    rows.append({"i": i, "min_dist_A": round(md, 4), "U_classical_kJ": round(ucl, 2),
                 "U_mace_kJ": round(float(uml), 3),
                 "w_mace": float(np.exp(min(-uml / RT_kJ, 700.0))) if np.isfinite(uml) else 0.0})

HARD = 0.80 * 0.5 * (LJ['O'][1] + LJ['O_co2'][1]); FLOOR = -40.0   # CO2 binds deeper than Ar
for r in rows:
    hard = r["min_dist_A"] < HARD
    anom = np.isfinite(r["U_mace_kJ"]) and (r["U_mace_kJ"] < FLOOR or r["U_mace_kJ"] < r["U_classical_kJ"] - 50.0)
    r["flagged"] = bool(hard or anom); r["flag_hard_overlap"] = bool(hard); r["flag_energetic_anomaly"] = bool(anom)
wm = np.array([r["w_mace"] for r in rows]); fl = np.array([r["flagged"] for r in rows])
flagged_wf = float(wm[fl].sum() / wm.sum()) if wm.sum() > 0 else 0.0
min_U_all = min(r["U_mace_kJ"] for r in rows)
gate1_pass = abs(gate1) < 1.0
if not gate1_pass:
    verdict = f"WITHHELD — Gate 1 offset {gate1:.1f} kJ (open-open ref failed); do not interpret"
elif min_U_all > 0:
    verdict = "REFUSE [Class II: no-physisorption] — min U > 0. Per VERDICT_SCHEMA_V06.md"
elif flagged_wf >= 0.5:
    verdict = "REFUSE [Class I: OOD over-binding] — flagged-weight >= 0.5. Per VERDICT_SCHEMA_V06.md"
else:
    verdict = "GOVERNED PASS WITH FLAGS (screen-pass; accuracy unassessed)" if fl.any() else "GOVERNED PASS"

summary = {"system": "Si-CHA + CO2 (rigid TraPPE geom, random orientation)", "N": a.N, "seed": SEED,
           "model": a.model, "task_name_LOADBEARING": a.task, "engine": "fairchem-UMA",
           "protocol": "periodic same-graph; CO2 probe (CO2 in-distribution for odac)",
           "verdict_schema": "VERDICT_SCHEMA_V06.md",
           "ref_min_dist_A": round(grid[-1][0], 2), "gate1_second_open_U_kJ": round(float(gate1), 4),
           "gate1_pass": bool(gate1_pass),
           "n_flagged": int(fl.sum()), "flagged_weight_fraction": round(flagged_wf, 4),
           "min_U_mace_kJ": round(min_U_all, 2), "max_U_mace_kJ": round(max(r["U_mace_kJ"] for r in rows), 1),
           "min_U_classical_kJ": round(min(r["U_classical_kJ"] for r in rows), 2), "verdict": verdict}
json.dump(summary, open(a.out, "w"), indent=2)
json.dump(rows, open(a.out.replace(".json", "_per_insertion.json"), "w"), indent=1)
print(json.dumps(summary, indent=2))
