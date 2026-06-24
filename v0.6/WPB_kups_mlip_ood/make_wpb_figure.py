#!/usr/bin/env python3
"""WPB figure: per-insertion energy vs nearest host-guest distance for the classical
UFF baseline vs the two MLIPs (MACE-MPA-0 = kUPS's shipped model; MACE-MP-small =
2023). Shows the two distinct failure modes: small hallucinates ATTRACTIVE overlaps
(below 0); MPA-0 is all-repulsive (above 0, no physisorption). Repulsion floor at 0."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
def load(m):
    with open(os.path.join(HERE, f"wpb_{m}_per_insertion.json")) as fh:
        return json.load(fh)
mpa = load("medium-mpa-0"); sml = load("small"); d3 = load("medium-mpa-0_D3")
d = np.array([r["min_dist_A"] for r in mpa])
ucl = np.array([r["U_classical_kJ"] for r in mpa])
umpa = np.array([r["U_mace_kJ"] for r in mpa]); usml = np.array([r["U_mace_kJ"] for r in sml])
ud3 = np.array([r["min_dist_A"] for r in d3]); ed3 = np.array([r["U_mace_kJ"] for r in d3])
try:
    uma = load("uma_omat"); duma = np.array([r["min_dist_A"] for r in uma]); euma = np.array([r["U_mace_kJ"] for r in uma])
except Exception:
    duma = euma = None

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(8.4, 6.3))
ax.axhline(0, color="#d62728", ls="--", lw=1.2, label="repulsion floor (U=0): physical wall")
ax.axvspan(0, 2.762, color="#fde0dc", alpha=0.4, zorder=0, label="overlap (< 2.762 Å)")
ax.scatter(d, np.clip(ucl, -40, 1e6), s=26, c="#1f77b4", label="classical UFF (correct: +∞ at overlap)", alpha=0.7)
ax.scatter(d, umpa, s=30, c="#2ca02c", marker="^", label="MACE-MPA-0 bare (kUPS): all-repulsive → REFUSE II", alpha=0.85)
ax.scatter(ud3, ed3, s=34, c="#ff7f0e", marker="D", label="MACE-MPA-0 + D3(BJ): shallow well → screen-pass", alpha=0.85)
ax.scatter(d, usml, s=34, c="#9467bd", marker="x", label="MACE-MP-small (2023): ATTRACTIVE overlaps → REFUSE I", alpha=0.85)
if euma is not None:
    ax.scatter(duma, euma, s=34, c="#8c564b", marker="v", label="UMA-s-1.1 omat (deployment): deep overlaps → REFUSE I", alpha=0.85)
ax.set_yscale("symlog", linthresh=20)
ax.set_xlabel("nearest host–guest distance (Å)"); ax.set_ylabel(r"insertion energy $U$ (kJ mol$^{-1}$)")
ax.set_title("WPB — governance discriminates the failure mode across 4 configurations (Si-CHA + Ar)")
ax.annotate("MACE-MP-small:\nmin −19.3 (attractive overlap)",
            (1.5, -15), xytext=(1.9, -90), fontsize=7.0, color="#9467bd", ha="left",
            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=0.8))
ax.annotate("MPA-0 bare ≥ +8.5\n(all-repulsive)",
            (4.5, 9), xytext=(5.3, 90), fontsize=7.0, color="#2ca02c", ha="left",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))
ax.annotate("+D3(BJ): min −5.03\n(well restored)",
            (4.2, -5), xytext=(5.7, -55), fontsize=7.0, color="#ff7f0e", ha="left",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8))
ax.set_xlim(0.5, 8)
ax.legend(fontsize=7.0, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
          frameon=True, borderaxespad=0.0)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"wpb_triptych.{ext}"), dpi=150, bbox_inches="tight")
print("wrote wpb_triptych.{pdf,png}")
