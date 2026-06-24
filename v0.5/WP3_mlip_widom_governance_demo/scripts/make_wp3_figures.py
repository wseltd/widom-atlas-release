#!/usr/bin/env python3
"""WP3 figures: (A) per-insertion energy vs nearest host-guest distance for the
classical baseline vs MACE-MP (the OOD failure made visible), (B) cumulative
contribution to <exp(-beta U)> sorted by weight, and (C) a governed verdict card."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "outputs"))
rows = json.load(open(os.path.join(OUT, "per_insertion.json")))
S = json.load(open(os.path.join(OUT, "governance_summary.json")))

d = np.array([r["min_dist_A"] for r in rows])
ucl = np.array([r["U_classical_kJ"] for r in rows])
uml = np.array([r["U_mace_kJ"] for r in rows])
flag = np.array([r["flagged"] for r in rows])
wml = np.array([r["w_mace"] for r in rows])
thr = S["flag_rules"]["hard_overlap_A"]

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

# Panel A: energy vs nearest host-guest distance (log-symmetric y)
axA = axes[0]
axA.axvline(thr, color="0.5", ls="--", lw=0.9)
axA.axhline(0, color="0.8", lw=0.6)
axA.scatter(d, np.clip(ucl, -40, 1e6), s=22, c="#1f77b4", label="classical LJ", alpha=0.8)
axA.scatter(d, uml, s=22, c="#d62728", marker="x", label="MACE-MP", alpha=0.9)
axA.set_yscale("symlog", linthresh=20)
axA.set_xlabel(r"nearest host-guest distance (\AA)")
axA.set_ylabel(r"insertion energy $U$ (kJ mol$^{-1}$)")
axA.set_title("A. Classical repels overlaps; MACE does not")
axA.legend(fontsize=7, loc="upper right")
axA.text(thr + 0.05, -35, "overlap\n(flagged)", fontsize=7, color="0.4")
axA.annotate("MACE: spurious low/negative\nU at overlaps (OOD)", (1.5, -10),
             xytext=(2.6, -38), fontsize=7, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

# Panel B: cumulative Boltzmann weight sorted descending
axB = axes[1]
order = np.argsort(-wml)
cum = np.cumsum(wml[order]) / wml.sum()
axB.plot(np.arange(1, len(cum) + 1), cum, color="#2ca02c", lw=1.6)
n_dom = int(np.searchsorted(cum, 0.9)) + 1
axB.axhline(0.9, color="0.6", ls=":", lw=0.8)
axB.set_xlabel("insertions ranked by Boltzmann weight")
axB.set_ylabel(r"cumulative fraction of $\langle e^{-\beta U}\rangle$")
axB.set_title("B. A few OOD insertions dominate $K_H$")
axB.text(n_dom + 1, 0.5, f"top {n_dom} insertions\n= 90% of the average",
         fontsize=7.5, color="0.25")
axB.set_ylim(0, 1.02)

# Panel C: governed verdict card
axC = axes[2]
axC.axis("off")
m = S["mace"]
lines = [
    ("GOVERNED MLIP-WIDOM VERDICT", True),
    ("", False),
    (f"system    : {S['system']}", False),
    (f"guest/T   : {S['guest']} / {S['temperature_K']} K", False),
    (f"MLIP      : {m['model']} ({m['mace_torch']}/{m['torch']})", False),
    (f"checkpoint: {m['checkpoint_sha256_16']}", False),
    (f"seed / N  : {S['seed']} / {S['n_insertions']}", False),
    ("", False),
    (f"flagged insertions     : {S['n_flagged']}/{S['n_insertions']} ({S['flagged_fraction']:.0%})", False),
    (f"flagged Boltzmann wt   : {S['flagged_weight_fraction']:.1%}", False),
    (f"K_H proxy (all)        : {S['KH_proxy_all_mace']:.2f}", False),
    (f"K_H proxy (OOD removed): {S['KH_proxy_flagged_removed_mace']:.2f}", False),
    (f"  -> inflation factor  : {S['KH_proxy_all_mace']/max(S['KH_proxy_flagged_removed_mace'],1e-9):.1f}x", False),
    ("", False),
    ("VERDICT: REFUSE", True),
    ("OOD insertions dominate the", False),
    ("Boltzmann average; MLIP-Widom", False),
    ("K_H is not trustworthy.", False),
]
y = 0.97
for txt, bold in lines:
    axC.text(0.02, y, txt, fontsize=8.2, family="monospace",
             fontweight="bold" if bold else "normal",
             color="#b00000" if txt.startswith("VERDICT") else "black",
             transform=axC.transAxes, va="top")
    y -= 0.056
axC.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="0.6", lw=1.0, transform=axC.transAxes))

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"wp3_mlip_governance.{ext}"), dpi=150, bbox_inches="tight")
print("wrote wp3_mlip_governance.pdf/.png ; top", n_dom, "insertions = 90% of <exp(-bU)>")
