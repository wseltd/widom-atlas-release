#!/usr/bin/env python3
"""Branch 6c four-error cancellation (Finding F1 / section 2.4). The originally-locked hybrid recipe
(0.207) "passed" only by a cancellation of four compounding errors; removing each in turn -- the silicon
double-count, the under-converged 12 A cutoff, and the non-calibration structure -- lands the coherent
single-source recipe on the corrected reference (0.200), cross-engine (native 0.200, RASPA3 0.199).
Every value reads from the committed JSON (cancellation_6c.json); nothing is hardcoded."""
import figstyle as S
import matplotlib.pyplot as plt
import numpy as np

D = S.load("cancellation_6c.json")
chain = D["chain"]
ref = D["reference"]["K_H_mol_kg_bar"]
lo, hi = D["reference"]["window"]
clean = D["clean_result"]
ss = D["structure_sensitivity"]

labels = [c["short"] for c in chain]
vals = [c["K_H_mol_kg_bar"] for c in chain]
cols = ["#8c8c8c", "#cfcfcf", "#b9c6e0", "#2ca02c"]  # hybrid(double-count) -> intermediates -> clean(=ref)

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.4, 4.5))

# corrected-reference band + line
ax.axhspan(lo, hi, color="#1a9850", alpha=0.10, zorder=0)
ax.axhline(ref, color="#1a9850", ls="--", lw=1.0, zorder=1)
ax.text(3.50, ref, f"corrected ref\n{S.fmt(ref, 3)} [{S.fmt(lo, 3)}, {S.fmt(hi, 3)}]",
        va="center", ha="left", fontsize=7.0, color="#1a7a34")

xs = np.arange(len(vals))
ax.bar(xs, vals, color=cols, edgecolor="k", lw=0.6, zorder=3, width=0.62)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.005, S.fmt(v, 3), ha="center", fontsize=8.4, zorder=4)

# the clean (final) bar carries the cross-engine parity -- placed in the right margin
ax.text(3.50, ref + 0.062,
        f"clean 6c bar: native {S.fmt(clean['native_K_H_mol_kg_bar'], 3)} /\n"
        f"RASPA3 {S.fmt(clean['raspa3_K_H_mol_kg_bar'], 3)} (parity {S.fmt(clean['parity_pct'], 1)}%)",
        ha="left", va="bottom", fontsize=6.8, color="#1a5e26")

# structure sensitivity of the converged single-source recipe
ax.text(0.02, 0.97, f"structure sensitivity (single-source):\nOlson {S.fmt(ss['Olson'], 3)} / "
        f"vK {S.fmt(ss['van_Koningsveld'], 3)} / IZA {S.fmt(ss['IZA_idealized'], 3)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color="#444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.6))

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=7.6)
ax.set_ylabel(r"$K_H$ (mol kg$^{-1}$ bar$^{-1}$)")
ax.set_ylim(0, max(max(vals), hi) * 1.30)
ax.set_xlim(-0.6, 5.2)
ax.set_title("6c rebuilt clean: four compounding errors cancel; the coherent\n"
             "single-source recipe lands on the corrected reference (F1)", fontsize=8.6)
ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
fig.tight_layout()
S.save(fig, "fp_6c_cancellation")
print(f"6c cancellation chain {vals} -> ref {ref}; clean native {clean['native_K_H_mol_kg_bar']} "
      f"raspa3 {clean['raspa3_K_H_mol_kg_bar']} parity {clean['parity_pct']}%")
