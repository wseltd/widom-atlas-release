#!/usr/bin/env python3
"""F1 — WPB triptych (4 configurations on Si-CHA + Ar). Every numeric value in every label/annotation
is read from the committed summary JSONs (no hardcoding — fixes the −334 stale-label bug)."""
import figstyle as S
import matplotlib.pyplot as plt
import numpy as np

# per-insertion geometry/energy (plotted points)
mpa = S.per_insertion("medium-mpa-0"); sml = S.per_insertion("small")
d3 = S.per_insertion("medium-mpa-0_D3"); uma = S.per_insertion("uma_omat")
d = np.array([r["min_dist_A"] for r in mpa]); ucl = np.array([r["U_classical_kJ"] for r in mpa])
umpa = np.array([r["U_mace_kJ"] for r in mpa]); usml = np.array([r["U_mace_kJ"] for r in sml])
ud3 = np.array([r["min_dist_A"] for r in d3]); ed3 = np.array([r["U_mace_kJ"] for r in d3])
duma = np.array([r["min_dist_A"] for r in uma]); euma = np.array([r["U_mace_kJ"] for r in uma])

# label numbers — sourced from JSON
m_sml = S.jget("wpb_small.json", "min_U_mace_kJ")
m_mpa = S.jget("wpb_medium-mpa-0.json", "min_U_mace_kJ")
m_d3 = S.jget("wpb_medium-mpa-0_D3.json", "min_U_mace_kJ")
m_uma = S.jget("wpb_uma_omat.json", "min_U_mace_kJ")
fw_sml = S.jget("wpb_small.json", "flagged_weight_fraction")

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.axhline(0, color=S.COLORS["floor"], ls="--", lw=1.2, label="repulsion floor (U=0): physical wall")
ax.axvspan(0, S.OVERLAP_A, color="#fde0dc", alpha=0.4, zorder=0, label=f"overlap (< {S.OVERLAP_A} Å)")
ax.scatter(d, np.clip(ucl, -40, 1e6), s=26, c=S.COLORS["classical"], label=S.LABELS["classical"] + " (correct: +∞ at overlap)", alpha=0.7)
ax.scatter(d, umpa, s=30, c=S.COLORS["mpa0_bare"], marker="^",
           label=f"{S.LABELS['mpa0_bare']}: min {S.fmt(m_mpa, signed=True)} → REFUSE II", alpha=0.85)
ax.scatter(ud3, ed3, s=34, c=S.COLORS["mpa0_d3"], marker="D",
           label=f"{S.LABELS['mpa0_d3']}: min {S.fmt(m_d3, signed=True)} → screen-pass", alpha=0.85)
ax.scatter(d, usml, s=34, c=S.COLORS["mace_small"], marker="x",
           label=f"{S.LABELS['mace_small']}: min {S.fmt(m_sml, signed=True)} → REFUSE I", alpha=0.85)
ax.scatter(duma, euma, s=34, c=S.COLORS["uma_omat"], marker="v",
           label=f"{S.LABELS['uma_omat']}: min {S.fmt(m_uma, signed=True)} → REFUSE I", alpha=0.85)
ax.set_yscale("symlog", linthresh=20)
ax.set_xlabel("nearest host–guest distance (Å)"); ax.set_ylabel(r"insertion energy $U$ (kJ mol$^{-1}$)")
ax.set_title("WPB — governance discriminates the failure mode across 4 configurations (Si-CHA + Ar)")
ax.annotate(f"small (2023) dives ATTRACTIVE\nat overlaps (min {S.fmt(m_sml, signed=True)}); flagged weight {S.fmt(fw_sml, 2)} → REFUSE I",
            (1.5, -15), xytext=(2.7, -38), fontsize=6.8, color=S.COLORS["mace_small"],
            arrowprops=dict(arrowstyle="->", color=S.COLORS["mace_small"], lw=0.8))
ax.annotate(f"MPA-0 bare ≥ {S.fmt(m_mpa, signed=True)} everywhere:\nno physisorption → REFUSE II",
            (4.5, 9), xytext=(4.7, 60), fontsize=6.8, color=S.COLORS["mpa0_bare"],
            arrowprops=dict(arrowstyle="->", color=S.COLORS["mpa0_bare"], lw=0.8))
ax.annotate(f"+D3(BJ) restores the well\n(min {S.fmt(m_d3, signed=True)}) → screen-pass",
            (4.2, -5), xytext=(5.3, -32), fontsize=6.8, color=S.COLORS["mpa0_d3"],
            arrowprops=dict(arrowstyle="->", color=S.COLORS["mpa0_d3"], lw=0.8))
ax.annotate(f"UMA omat erratic at overlaps\n(min {S.fmt(m_uma, signed=True)}) → REFUSE I",
            (2.65, -290), xytext=(3.2, -180), fontsize=6.8, color=S.COLORS["uma_omat"],
            arrowprops=dict(arrowstyle="->", color=S.COLORS["uma_omat"], lw=0.8))
ax.legend(fontsize=6.4, loc="upper right"); ax.set_xlim(0.5, 8)
fig.tight_layout()
S.save(fig, "f1_wpb_triptych")
print(f"label values from JSON: small={m_sml} mpa0={m_mpa} d3={m_d3} uma={m_uma} fw_small={fw_sml}")
