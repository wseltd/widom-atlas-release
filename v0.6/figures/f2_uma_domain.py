#!/usr/bin/env python3
"""F2 — "governance respects the training domain" (UMA). Panel A: the odac head on CO2 (in-domain →
physical well) vs Ar (OOD → erratic/withheld), same model head. Panel B: the task×system verdict
matrix. Every number reads from committed JSONs."""
import figstyle as S
import matplotlib.pyplot as plt
import numpy as np

co2 = S.per_insertion("uma_co2_odac"); ar = S.per_insertion("uma_odac")
dco2 = np.array([r["min_dist_A"] for r in co2]); eco2 = np.array([r["U_mace_kJ"] for r in co2])
dar = np.array([r["min_dist_A"] for r in ar]); ear = np.array([r["U_mace_kJ"] for r in ar])

# numbers from JSON
co2_min = S.jget("wpb_uma_co2_odac.json", "min_U_mace_kJ")
co2_q = S.jget("wpb_uma_co2_odac.json", "screen_q_st_kJ")
co2_anchor = S.jget("wpb_uma_co2_odac.json", "accuracy_anchor_q_st_kJ")
co2_gate1 = S.jget("wpb_uma_co2_odac.json", "gate1_second_open_U_kJ")
co2_N = S.jget("wpb_uma_co2_odac.json", "N")
ar_gate1_eq = S.jget("wpb_uma_odac.json", "gate1_second_open_U_kJ")  # periodic equivalent-sites Gate-1
omatAr_min = S.jget("wpb_uma_omat.json", "min_U_mace_kJ")
omatAr_v = S.jget("wpb_uma_omat.json", "verdict")
omatCO2_min = S.jget("wpb_uma_co2_omat.json", "min_U_mace_kJ")

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

# ── Panel A: odac head, CO2 (in-domain) vs Ar (OOD) ──
axA.axhline(0, color=S.COLORS["floor"], ls="--", lw=1.0)
axA.scatter(dco2, eco2, s=30, c=S.COLORS["uma_odac"], marker="s",
            label=f"odac · CO₂ (in-domain): well min {S.fmt(co2_min, signed=True)} → screen-pass", alpha=0.85)
axA.scatter(dar, ear, s=30, c=S.COLORS["uma_omat"], marker="v",
            label=f"odac · Ar (OOD): Gate-1 fail ({S.fmt(ar_gate1_eq)} kJ at equiv sites) → WITHHELD", alpha=0.7)
axA.axhline(co2_min, color=S.COLORS["uma_odac"], ls=":", lw=0.9)
axA.annotate(f"physical CO₂ well\nmin {S.fmt(co2_min, signed=True)} kJ/mol", (4.0, co2_min),
             xytext=(4.4, -120), fontsize=7, color=S.COLORS["uma_odac"],
             arrowprops=dict(arrowstyle="->", color=S.COLORS["uma_odac"], lw=0.8))
axA.set_yscale("symlog", linthresh=20); axA.set_xlim(0.5, 8)
axA.set_xlabel("nearest host–guest distance (Å)"); axA.set_ylabel(r"insertion energy $U$ (kJ mol$^{-1}$)")
axA.set_title(f"(A) odac head: in-domain CO₂ vs out-of-domain Ar  (N={co2_N})", fontsize=9)
axA.legend(fontsize=6.6, loc="upper right")

# ── Panel B: task × system verdict matrix ──
axB.axis("off")
cells = [
    # (col, row, task, system, color_key, line1, line2)
    (0, 1, "omat", "Ar",  "#d62728", "REFUSE I", f"min U {S.fmt(omatAr_min, signed=True)}"),
    (1, 1, "odac", "Ar",  "#7f7f7f", "WITHHELD", f"Gate-1 {S.fmt(ar_gate1_eq)} kJ"),
    (0, 0, "omat", "CO₂", "#d62728", "REFUSE II", f"min U {S.fmt(omatCO2_min, signed=True)}"),
    (1, 0, "odac", "CO₂", "#2ca02c", "screen-PASS", f"well {S.fmt(co2_min, signed=True)}, q_st {S.fmt(co2_q)}"),
]
for col, row, task, sysn, color, v1, v2 in cells:
    x, y = col, row
    axB.add_patch(plt.Rectangle((x, y), 0.96, 0.96, facecolor=color, alpha=0.22, edgecolor=color, lw=1.5))
    axB.text(x + 0.48, y + 0.74, f"{task} · {sysn}", ha="center", fontsize=8.5, weight="bold")
    axB.text(x + 0.48, y + 0.45, v1, ha="center", fontsize=9, color=color, weight="bold")
    axB.text(x + 0.48, y + 0.20, v2, ha="center", fontsize=7)
axB.set_xlim(-0.1, 2.05); axB.set_ylim(-0.1, 2.1)
axB.set_title(f"(B) governance respects the domain:\npasses in-domain (odac·CO₂, q_st {S.fmt(co2_q)} vs anchor {S.fmt(co2_anchor)}), refused/withheld elsewhere", fontsize=8)
fig.tight_layout()
S.save(fig, "f2_uma_domain")
print(f"values: co2_min={co2_min} co2_q={co2_q} anchor={co2_anchor} co2_gate1={co2_gate1} omatAr_min={omatAr_min} omatCO2_min={omatCO2_min}")
