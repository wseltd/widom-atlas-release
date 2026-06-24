#!/usr/bin/env python3
"""WP3 — MLIP-Widom governance demo with an out-of-distribution (OOD) diagnostic.

Demonstrates the v0.5 governance point: random Widom insertion deliberately
samples atomic overlaps. A classical potential returns a huge POSITIVE energy
for an overlap (exp(-beta U) -> 0, harmless). An MLIP (MACE-MP) has never seen
such overlaps and can return a spurious NEGATIVE (attractive) energy; because
K_H ~ <exp(-beta U)>, even one spurious negative U is exponentially amplified.
This script runs the same random insertions through BOTH a classical LJ baseline
and a MACE-MP calculator (the same ASE-calculator interface cusp-ai-oss/widom
uses), captures every per-insertion energy, flags OOD insertions by two
transparent rules, and reports K_H with and without the flagged insertions.

CPU only (torch 2.12.0+cpu, mace-torch 0.3.16). Single-atom Ar guest in MFI so
overlaps are geometrically unambiguous. No GPU used; device 1 untouched.
"""
import json
import os
import sys
import numpy as np
from ase.io import read
from ase import Atoms

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "outputs"))
LOGS = os.path.normpath(os.path.join(HERE, "..", "logs"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

CIF = "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
GUEST = "Ar"
T_K = 298.15
KB_EV = 8.617333262e-5
KT = KB_EV * T_K
EV_TO_KJ = 96.48533212
N_INSERT = int(os.environ.get("WP3_N", "100"))
SEED = 0
CUTOFF = 6.0
DEVICE = os.environ.get("WP3_DEVICE", "cpu")  # "cpu" (default, documented run) or "cuda"
TAG = "" if DEVICE == "cpu" else "_" + DEVICE

# UFF Lennard-Jones (epsilon in K, sigma in Angstrom) for the classical baseline.
UFF = {"Si": (202.29, 3.826), "O": (30.19, 3.500), "Ar": (185.0, 3.405)}


def lj_classical_U_K(min_dists_by_elem):
    """Classical LJ interaction energy (in Kelvin) of the Ar guest with the
    framework, summed over framework atoms within the cutoff."""
    eps_g, sig_g = UFF[GUEST]
    u = 0.0
    for elem, dists in min_dists_by_elem.items():
        eps_f, sig_f = UFF.get(elem, (0.0, 3.5))
        eps = (eps_f * eps_g) ** 0.5
        sig = 0.5 * (sig_f + sig_g)
        for r in dists:
            if r <= CUTOFF and r > 1e-6:
                sr6 = (sig / r) ** 6
                u += 4.0 * eps * (sr6 * sr6 - sr6)
    return u  # Kelvin


def main():
    rng = np.random.default_rng(SEED)
    frame = read(CIF)
    # Replicate so the cell exceeds 2*cutoff in every direction (min image).
    reps = []
    for i in range(3):
        L = np.linalg.norm(frame.cell[i])
        reps.append(max(1, int(np.ceil(2 * CUTOFF / L))))
    frame = frame.repeat(reps)
    cell = np.array(frame.cell)
    fpos = frame.get_positions()
    fsym = frame.get_chemical_symbols()
    n_frame = len(frame)

    log = open(os.path.join(LOGS, f"run{TAG}.log"), "w")
    def say(m):
        print(m); log.write(m + "\n"); log.flush()
    say(f"framework={CIF} reps={reps} n_atoms={n_frame} cell_diag={np.diag(cell)}")

    # --- MACE-MP calculator (CPU) ---
    mace_ok = True
    mace_meta = {}
    try:
        from mace.calculators import mace_mp
        import torch
        calc = mace_mp(model="small", device=DEVICE, default_dtype="float64")
        mace_meta = {"mace_torch": __import__("mace").__version__,
                     "torch": torch.__version__, "model": "mace_mp small", "device": DEVICE}
        # try to capture a checkpoint hash
        try:
            import hashlib, glob
            ck = sorted(glob.glob(os.path.expanduser("~/.cache/mace/*")) +
                        glob.glob(os.path.expanduser("~/.cache/mace/**/*"), recursive=True))
            ck = [c for c in ck if os.path.isfile(c)]
            if ck:
                h = hashlib.sha256(open(ck[-1], "rb").read()).hexdigest()[:16]
                mace_meta["checkpoint"] = os.path.basename(ck[-1])
                mace_meta["checkpoint_sha256_16"] = h
        except Exception as e:
            mace_meta["checkpoint"] = f"unhashed ({e})"
        say(f"MACE ready: {mace_meta}")
    except Exception as e:
        mace_ok = False
        say(f"MACE init FAILED: {e}")

    def mace_energy(atoms):
        atoms.calc = calc
        return atoms.get_potential_energy()  # eV

    # Reference energies for the interaction definition.
    if mace_ok:
        e_frame = mace_energy(frame.copy())
        e_guest = mace_energy(Atoms(GUEST, positions=[[0, 0, 0]], cell=cell, pbc=True))
        say(f"E_frame={e_frame:.3f} eV  E_guest={e_guest:.3f} eV")

    rows = []
    for i in range(N_INSERT):
        frac = rng.random(3)
        cart = frac @ cell
        # minimum-image distances to all framework atoms
        d = fpos - cart
        inv = np.linalg.inv(cell)
        dfrac = d @ inv
        dfrac -= np.round(dfrac)
        dmic = dfrac @ cell
        dist = np.linalg.norm(dmic, axis=1)
        min_dist = float(dist.min())
        within = dist <= CUTOFF
        by_elem = {}
        for sym, r in zip(np.array(fsym)[within], dist[within]):
            by_elem.setdefault(sym, []).append(r)
        u_cl_K = lj_classical_U_K(by_elem)
        u_cl_eV = u_cl_K * KB_EV
        rec = {"i": i, "min_dist_A": round(min_dist, 4),
               "U_classical_kJ": round(u_cl_eV * EV_TO_KJ, 3)}
        if mace_ok:
            sys_atoms = frame.copy()
            sys_atoms += Atoms(GUEST, positions=[cart])
            try:
                e_sys = mace_energy(sys_atoms)
                u_ml_eV = e_sys - e_frame - e_guest
            except Exception as e:
                u_ml_eV = float("nan")
                say(f"  insertion {i}: MACE eval failed: {e}")
            rec["U_mace_kJ"] = round(u_ml_eV * EV_TO_KJ, 3)
            rec["w_mace"] = float(np.exp(-min(u_ml_eV, 50.0) / KT)) if np.isfinite(u_ml_eV) else 0.0
        rec["w_classical"] = float(np.exp(-min(u_cl_eV, 50.0) / KT))
        rows.append(rec)
        if i % 20 == 0:
            say(f"  insertion {i}/{N_INSERT} min_dist={min_dist:.2f} U_cl={rec['U_classical_kJ']} "
                f"U_ml={rec.get('U_mace_kJ','-')}")

    # --- OOD diagnostic (two transparent flags) ---
    sig_min = 0.5 * (UFF['O'][1] + UFF[GUEST][1])
    HARD_OVERLAP_A = 0.80 * sig_min          # geometry flag
    ANOMALY_FLOOR_KJ = -25.0                 # Ar physisorption is ~ -10..-15 kJ/mol; below -25 is implausible
    for r in rows:
        hard = r["min_dist_A"] < HARD_OVERLAP_A
        anom = mace_ok and np.isfinite(r.get("U_mace_kJ", float("nan"))) and (
            r["U_mace_kJ"] < ANOMALY_FLOOR_KJ or
            (r["U_mace_kJ"] < r["U_classical_kJ"] - 50.0))
        r["flag_hard_overlap"] = bool(hard)
        r["flag_energetic_anomaly"] = bool(anom)
        r["flagged"] = bool(hard or anom)

    def kh_proxy(weights):
        w = np.array(weights, float)
        return float(np.mean(w)) if len(w) else 0.0

    summary = {
        "system": "Si-CHA (all-silica chabazite, IZA) + Ar", "temperature_K": T_K,
        "n_insertions": N_INSERT, "seed": SEED, "cutoff_A": CUTOFF,
        "guest": GUEST, "mace": mace_meta, "mace_ok": mace_ok,
        "flag_rules": {"hard_overlap_A": round(HARD_OVERLAP_A, 3),
                       "anomaly_floor_kJ": ANOMALY_FLOOR_KJ,
                       "anomaly_vs_classical_kJ": 50.0},
    }
    if mace_ok:
        wm = [r["w_mace"] for r in rows]
        wm_flag = [r["w_mace"] for r in rows if r["flagged"]]
        wm_clean = [r["w_mace"] for r in rows if not r["flagged"]]
        total_w = sum(wm); flag_w = sum(wm_flag)
        summary["n_flagged"] = sum(1 for r in rows if r["flagged"])
        summary["flagged_fraction"] = round(summary["n_flagged"] / N_INSERT, 4)
        summary["flagged_weight_fraction"] = round(flag_w / total_w, 4) if total_w > 0 else 0.0
        summary["KH_proxy_all_mace"] = kh_proxy(wm)
        summary["KH_proxy_flagged_removed_mace"] = kh_proxy(wm_clean)
        summary["min_U_mace_kJ"] = round(min((r["U_mace_kJ"] for r in rows if np.isfinite(r.get("U_mace_kJ", float('nan')))), default=float('nan')), 2)
        summary["min_U_classical_kJ"] = round(min(r["U_classical_kJ"] for r in rows), 2)
        # verdict
        if summary["flagged_weight_fraction"] >= 0.5:
            summary["verdict"] = "REFUSE — MLIP-Widom untrustworthy: flagged OOD insertions dominate the Boltzmann average"
        elif summary["n_flagged"] > 0:
            summary["verdict"] = "GOVERNED PASS WITH FLAGS — OOD insertions present but contribute a minority of the Boltzmann weight; not an experimental-validation claim"
        else:
            summary["verdict"] = "GOVERNED PASS — no OOD insertions flagged; not an experimental-validation claim"

    summary["device"] = DEVICE
    json.dump(summary, open(os.path.join(OUT, f"governance_summary{TAG}.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(OUT, f"per_insertion{TAG}.json"), "w"), indent=1)
    say(f"WROTE governance_summary{TAG}.json + per_insertion{TAG}.json")
    say("VERDICT: " + summary.get("verdict", "(mace unavailable)"))
    log.close()


if __name__ == "__main__":
    main()
