#!/usr/bin/env python3
"""Cross-engine Widom parity. Panel A (6c MFI+Ar): K_H across engines — native+Ar-Si reproduces RASPA3
(parity); the 1.8× with-Si vs O-only gap is the silicon double-count (Finding F1). Panel B (4c CHA+CO2
convention matrix): the shift (+11.6%) and tail (+11.9%) toggles and where kUPS lands. Every number reads
from committed JSONs."""
import figstyle as S
import matplotlib.pyplot as plt
import numpy as np

P6 = "parity_6c_threeway.json"; CM = "native_4c_convention_matrix.json"
raspa = S.jget(P6, "RASPA3_locked", "K_H_mol_kg_bar")
natSi = S.jget(P6, "native_with_Si", "K_H_mol_kg_bar")
kupSi = S.jget(P6, "kups_with_Si", "K_H_mol_kg_bar")
natO = S.jget(P6, "native_O_only", "K_H_mol_kg_bar")
kupO = S.jget(P6, "kups_O_only", "K_H_mol_kg_bar")
d_par = S.jget(P6, "deltas", "native_withSi_vs_RASPA3_KH_pct")
d_si = S.jget(P6, "deltas", "Si_term_KH_inflation_native")

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.6))

# ── Panel A: 6c three-way ──
barsA = [("RASPA3\n(locked)", raspa, "#555555"),
         ("native\n+Ar-Si", natSi, S.COLORS["classical"]),
         ("kUPS\n+Ar-Si", kupSi, S.COLORS["uma_odac"]),
         ("native\nO-only", natO, "#bbbbbb"),
         ("kUPS\nO-only", kupO, "#cde")]
xs = np.arange(len(barsA))
axA.bar(xs, [b[1] for b in barsA], color=[b[2] for b in barsA], edgecolor="k", lw=0.6)
for x, (lab, v, _) in zip(xs, barsA):
    axA.text(x, v + 0.004, S.fmt(v, 3), ha="center", fontsize=7.5)
axA.axhline(raspa, color="#555555", ls="--", lw=0.8)
axA.set_xticks(xs); axA.set_xticklabels([b[0] for b in barsA], fontsize=7.5)
axA.set_ylabel(r"$K_H$ (mol kg$^{-1}$ bar$^{-1}$)")
axA.set_title(f"(A) 6c MFI+Ar: 3 engines agree on the as-locked hybrid ({S.fmt(d_par, signed=True)}%);\nthe ×{S.fmt(d_si, 2)} Ar-Si gap is the silicon double-count (F1)", fontsize=8.5)
axA.annotate("", xy=(3, natO), xytext=(1, natSi), arrowprops=dict(arrowstyle="<->", color="#888", lw=1.0))
axA.text(2.0, (natSi + natO) / 2, f"×{S.fmt(d_si, 2)}", ha="center", fontsize=8, color="#555")

# ── Panel B: 4c convention matrix ──
sh_t = S.jget(CM, "corners_K_H_mol_kg_bar", "shifted+tail")
tr_t = S.jget(CM, "corners_K_H_mol_kg_bar", "truncated+tail")
sh_n = S.jget(CM, "corners_K_H_mol_kg_bar", "shifted_notail")
tr_n = S.jget(CM, "corners_K_H_mol_kg_bar", "truncated_notail")
ku_on = S.jget(CM, "kups_tail_test", "tail_ON", "K_H_mol_kg_bar")
ku_off = S.jget(CM, "kups_tail_test", "tail_OFF", "K_H_mol_kg_bar")
shift_pct = S.jget(CM, "decomposition_+2.2pct_vs_lock", "shift")
tail_pct = S.jget(CM, "one_toggle_decomposition_pct", "tail_on_minus_off_shifted")
resid_pct = S.jget(CM, "decomposition_+2.2pct_vs_lock", "residual_truncated+tail")
barsB = [("native\nshift+tail\n(lock)", sh_t, "#555555"),
         ("native\ntrunc+tail", tr_t, "#bbbbbb"),
         ("kUPS\ntrunc+tail", ku_on, S.COLORS["uma_odac"]),
         ("native\nshift,no-tail", sh_n, "#ddd"),
         ("native\ntrunc,no-tail", tr_n, "#ccc"),
         ("kUPS\nno-tail", ku_off, "#9ecae1")]
xs = np.arange(len(barsB))
axB.bar(xs, [b[1] for b in barsB], color=[b[2] for b in barsB], edgecolor="k", lw=0.6)
for x, (lab, v, _) in zip(xs, barsB):
    axB.text(x, v + 0.02, S.fmt(v, 3), ha="center", fontsize=7)
axB.axhline(sh_t, color="#555555", ls="--", lw=0.8)
axB.set_xticks(xs); axB.set_xticklabels([b[0] for b in barsB], fontsize=6.6)
axB.set_ylabel(r"$K_H$ (mol kg$^{-1}$ bar$^{-1}$)")
axB.set_title(f"(B) 4c CHA+CO₂ (Ewald): shift {S.fmt(shift_pct, signed=True)}%, tail {S.fmt(tail_pct, signed=True)}%;\nkUPS matched-convention residual {S.fmt(resid_pct, signed=True)}% (q_st-invariant)", fontsize=8.5)
fig.tight_layout()
S.save(fig, "fp_cross_engine_parity")
print(f"6c: raspa={raspa} natSi={natSi} kupSi={kupSi} natO={natO} | 4c: sh_t={sh_t:.3f} ku_on={ku_on} shift={shift_pct} tail={tail_pct} resid={resid_pct}")
