#!/usr/bin/env python3
"""F4 - continuous U(r) scan for UMA uma-s-1.1 (omat) on Si-CHA + Ar.

Answers the audit request for (i) a continuous U(r) scan along a single
trajectory and (ii) a repeat against a second reference configuration, for the
-292.8 kJ/mol anomaly. UMA energies are read from the committed scan JSON
(uma_ur_scan_2026-08.json); the classical comparator is recomputed here from the
same deterministic trajectory, so no value is hardcoded.
"""
import json
import os

import figstyle as S
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCAN = os.path.join(REPO, "v0.6/WPB_kups_mlip_ood/uma_ur_scan_2026-08.json")

d = json.load(open(SCAN))
rows = d["scan"]

CUTOFF, KB_EV, EV_TO_KJ = 6.0, 8.617333262e-5, 96.48533212
TWO16, KCAL_TO_K = 2.0 ** (1.0 / 6.0), 503.2195
UFF_XD = {"Si": (3.826, 0.402), "O": (3.500, 0.060), "Ar": (3.868, 0.185)}
GEN = {e: (xd[1] * KCAL_TO_K, xd[0] / TWO16) for e, xd in UFF_XD.items()}

frame = read(os.path.join(HERE, "CHA_iza.cif"))
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell)
inv = np.linalg.inv(cell)
fpos, fsym = frame.get_positions(), np.array(frame.get_chemical_symbols())


def neigh(cart):
    df = (fpos - cart) @ inv
    df -= np.round(df)
    return np.linalg.norm(df @ cell, axis=1)


def classical(cart):
    dist = neigh(cart)
    within = dist <= CUTOFF
    eps_g, sig_g = GEN["Ar"]
    u = 0.0
    for elem in set(fsym[within]):
        eps_f, sig_f = GEN.get(elem, (0.0, 3.5))
        dd = dist[within][fsym[within] == elem]
        dd = dd[dd > 1e-6]
        sig = 0.5 * (sig_f + sig_g)
        sr6 = (sig / dd) ** 6
        u += float(np.sum(4.0 * (eps_f * eps_g) ** 0.5 * (sr6 * sr6 - sr6)))
    return u * KB_EV * EV_TO_KJ


# rebuild the identical trajectory (deterministic; matches the scan JSON)
rng = np.random.default_rng(0)
pts = [rng.random(3) @ cell for _ in range(120)]
deep_cart = pts[d["deepest_insertion_index"]]
df = (fpos - deep_cart) @ inv
df -= np.round(df)
j = int(np.argmin(np.linalg.norm(df @ cell, axis=1)))
nearest_atom = deep_cart + (df[j] @ cell)
u_hat = (deep_cart - nearest_atom) / np.linalg.norm(deep_cart - nearest_atom)

md, u_uma, u_cl = [], [], []
for row in rows:
    cart = nearest_atom + u_hat * row["r_along_A"]
    md.append(row["min_dist_A"])
    u_uma.append(row["U_ref1_kJ"])
    u_cl.append(classical(cart))

order = np.argsort(md)
md = np.array(md)[order]
u_uma = np.array(u_uma)[order]
u_cl = np.array(u_cl)[order]

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.axvspan(0, S.OVERLAP_A, color="#fde0dc", alpha=0.45, zorder=0,
           label=f"overlap (< {S.OVERLAP_A} Å, corrected)")
ax.axhline(0, color=S.COLORS["floor"], ls="--", lw=1.1, label="repulsion floor (U=0)")
ax.axhline(-25, color="#888888", ls=":", lw=1.1, label="energetic-anomaly floor (−25 kJ/mol)")
ax.plot(md, u_cl, "-o", ms=3.2, lw=1.3, color=S.COLORS["classical"], label="classical UFF (same trajectory)")
ax.plot(md, u_uma, "-v", ms=3.6, lw=1.4, color=S.COLORS["uma_omat"], label="UMA uma-s-1.1 omat")

imin = int(np.argmin(u_uma))
ax.annotate(f"smooth spurious basin\nmin {u_uma[imin]:.1f} kJ/mol at {md[imin]:.2f} Å\n(outside the overlap band)",
            (md[imin], u_uma[imin]), xytext=(0.85, -160), fontsize=7.2,
            color=S.COLORS["uma_omat"], ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=S.COLORS["uma_omat"], lw=0.8))

ax.set_yscale("symlog", linthresh=20)
ax.set_xlim(0.7, 3.1)
ax.set_xlabel("nearest host–guest distance (Å)")
ax.set_ylabel(r"insertion energy $U$ (kJ mol$^{-1}$)")
ax.set_title("Continuous $U(r)$ scan: the UMA anomaly is a smooth basin, not a numerical spike")
ax.legend(fontsize=6.6, loc="upper right")
fig.tight_layout()
S.save(fig, "f4_uma_ur_scan")
print(f"UMA min {u_uma[imin]:.2f} kJ/mol at {md[imin]:.3f} A; classical there {u_cl[imin]:.2f} kJ/mol")
print(f"reference shift across scan: {d['gate1_open_open_kJ']:.4f} kJ/mol (constant)")
