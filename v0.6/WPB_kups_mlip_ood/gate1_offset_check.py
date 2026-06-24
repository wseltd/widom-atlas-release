#!/usr/bin/env python3
"""Gate 1 (professor): rule out an energy-reference / differencing offset before any
"MACE-MPA-0 misses physisorption" claim. A uniform +floor over random insertions is the
signature of a broken U = E(host+Ar) - E(host) - E(Ar) differencing (e.g. Ar absent from
the model element set, or a constant e0 mishandling), NOT physics.

Tests (same calculator, same differencing as run_wpb_ood.py):
  T0  is Ar (Z=18) even in the model's atomic-number table?
  Ta  Ar 10 A from an ISOLATED framework (PBC off, big box): U must -> ~0.
  Tb  Ar at the largest-pore center (max min-dist over a fine grid, PBC on): confinement U.
  Tc  paired-geometry table: 3 WPB insertions, classical vs MPA-0 on identical points.
"""
import argparse, json, os
import numpy as np
from ase.io import read
from ase import Atoms

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medium-mpa-0")
ap.add_argument("--out", required=True)
a = ap.parse_args()

CIF = "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif"
GUEST, EV_TO_KJ, CUTOFF = "Ar", 96.48533212, 6.0
UFF = {"Si": (202.29, 3.826), "O": (30.19, 3.500), "Ar": (185.0, 3.405)}
KB_EV, T_K = 8.617333262e-5, 298.15

frame0 = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame0.cell[i])))) for i in range(3)]
frame = frame0.repeat(reps)
cell = np.array(frame.cell); inv = np.linalg.inv(cell)
fpos = frame.get_positions(); fsym = np.array(frame.get_chemical_symbols())

from mace.calculators import mace_mp
calc = mace_mp(model=a.model, device="cuda", default_dtype="float64")

# T0: is Ar in the model element table?
ar_z = 18
zt = None
for attr in ("z_table", "atomic_numbers"):
    m = getattr(calc, "models", [None])[0] if hasattr(calc, "models") else None
    if m is not None and hasattr(m, "atomic_numbers"):
        zt = [int(z) for z in m.atomic_numbers.detach().cpu().numpy().ravel()]; break
ar_in_model = (zt is None) or (ar_z in zt)   # None => couldn't introspect; report raw

# isolated-Ar reference (same as WPB)
ag = Atoms(GUEST, positions=[[0, 0, 0]], cell=cell, pbc=True); ag.calc = calc
e_guest = ag.get_potential_energy()
at0 = frame.copy(); at0.calc = calc; e_frame = at0.get_potential_energy()

def U_kJ_periodic(cart):
    at = frame.copy(); at += Atoms(GUEST, positions=[cart]); at.calc = calc
    return (at.get_potential_energy() - e_frame - e_guest) * EV_TO_KJ

# Ta: isolated framework (PBC off), Ar 10 A from everything in a big vacuum box
iso = frame0.copy(); iso.set_pbc(False)
ipos = iso.get_positions()
big = np.ptp(ipos, axis=0).max() + 40.0
isobox = Atoms(iso.get_chemical_symbols(), positions=ipos, cell=[big, big, big], pbc=False)
isobox.calc = calc; e_iso = isobox.get_potential_energy()
ag2 = Atoms(GUEST, positions=[[0, 0, 0]], cell=[big, big, big], pbc=False); ag2.calc = calc
e_guest_iso = ag2.get_potential_energy()
far = ipos.mean(axis=0) + np.array([0, 0, np.ptp(ipos[:, 2]) / 2 + 12.0])  # 12 A above the slab
at_far = Atoms(list(iso.get_chemical_symbols()) + [GUEST],
               positions=np.vstack([ipos, far]), cell=[big, big, big], pbc=False)
at_far.calc = calc
U_isolated_far = (at_far.get_potential_energy() - e_iso - e_guest_iso) * EV_TO_KJ
md_far = float(np.linalg.norm(ipos - far, axis=1).min())

# Tb: largest-pore center (max min-dist over a fine grid, PBC on)
g = np.linspace(0.05, 0.95, 18)
best_md, best_cart = -1, None
for x in g:
    for y in g:
        for z in g:
            cart = np.array([x, y, z]) @ cell
            df = (fpos - cart) @ inv; df -= np.round(df); d = np.linalg.norm(df @ cell, axis=1).min()
            if d > best_md: best_md, best_cart = float(d), cart
U_porecenter = U_kJ_periodic(best_cart)

# Tc: paired-geometry table from the actual WPB insertions (recompute classical here)
def lj_classical_kJ(cart):
    df = (fpos - cart) @ inv; df -= np.round(df); dist = np.linalg.norm(df @ cell, axis=1)
    eps_g, sig_g = UFF[GUEST]; u = 0.0
    for s in set(fsym):
        dd = dist[(fsym == s) & (dist <= CUTOFF) & (dist > 1e-6)]
        ef, sf = UFF.get(s, (0.0, 3.5)); eps = (ef * eps_g) ** 0.5; sig = 0.5 * (sf + sig_g)
        sr6 = (sig / dd) ** 6; u += float(np.sum(4.0 * eps * (sr6 * sr6 - sr6)))
    return u * KB_EV * EV_TO_KJ

per = json.load(open(os.path.join(os.path.dirname(a.out), "wpb_medium-mpa-0_per_insertion.json")))
open_rows = sorted(per, key=lambda r: -r["min_dist_A"])[:3]  # the 3 MOST open insertions
paired = []
rng = np.random.default_rng(0)
carts = (rng.random((120, 3)) @ cell)
for r in open_rows:
    cart = carts[r["i"]]
    paired.append({"i": r["i"], "min_dist_A": r["min_dist_A"],
                   "U_classical_kJ": round(lj_classical_kJ(cart), 2),
                   "U_mace_mpa0_kJ": round(float(U_kJ_periodic(cart)), 2)})

out = {"model": a.model, "ar_in_model_elements": ar_in_model, "model_z_table_len": (len(zt) if zt else None),
       "T0_note": "if ar_in_model is False, the +floor is NON-physical (Ar unsupported), not under-binding",
       "Ta_U_isolated_Ar_far_kJ": round(float(U_isolated_far), 3), "Ta_min_dist_A": round(md_far, 2),
       "Ta_expect": "~0 if differencing is correct; a constant != 0 means reference offset",
       "Tb_U_largest_pore_center_kJ": round(float(U_porecenter), 3), "Tb_min_dist_A": round(best_md, 2),
       "Tb_expect": "~0 to slightly negative if model sees Ar in an open cage; large + => confinement or offset",
       "Tc_paired_open_insertions": paired,
       "e_guest_eV": round(float(e_guest), 4), "e_frame_eV": round(float(e_frame), 2)}
json.dump(out, open(a.out, "w"), indent=2)
print(json.dumps(out, indent=2))
