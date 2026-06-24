#!/usr/bin/env python3
"""F3a — insertion map: CHA framework (ball-and-stick, ASE) with the seeded Ar insertion points
colored by verdict flag (OOD-flagged red / clean blue). Two panels: MACE-MP-small vs UMA-omat — the
spatial version of "where the spurious weight lives" (flagged points are buried in the framework
walls). Geometries replayed from seed 0 (the WPB protocol); flags + counts read from committed JSONs."""
import figstyle as S
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms

CIF = S.CHA_CIF
CUTOFF, SEED = 6.0, 0
frame = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell)

def replay(n):
    rng = np.random.default_rng(SEED)
    return np.array([rng.random(3) @ cell for _ in range(n)])

legs = [("small", S.COLORS["mace_small"], "MACE-MP-small (2023) — REFUSE I"),
        ("uma_omat", S.COLORS["uma_omat"], "UMA uma-s-1.1 omat — REFUSE I")]
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
for ax, (name, accent, title) in zip(axes, legs):
    rows = S.per_insertion(name)
    cart = replay(len(rows))
    flagged = np.array([r["flagged"] for r in rows])
    nflag = int(S.jget(f"wpb_{name if name!='small' else 'small'}.json", "n_flagged"))
    N = int(S.jget(f"wpb_{name}.json", "N"))
    plot_atoms(frame, ax, rotation="0x,0y,0z", radii=0.42,
               colors=["#c9a96a" if s == "Si" else "#d7d2c8" for s in frame.get_chemical_symbols()])
    ax.scatter(cart[~flagged, 0], cart[~flagged, 1], s=44, c=S.COLORS["flag_clean"],
               edgecolors="k", lw=0.5, marker="o", zorder=5, label=f"clean ({int((~flagged).sum())})")
    ax.scatter(cart[flagged, 0], cart[flagged, 1], s=70, c=S.COLORS["flag_ood"],
               edgecolors="k", lw=0.6, marker="X", zorder=6, label=f"OOD-flagged ({int(flagged.sum())})")
    ax.set_title(f"{title}\n{nflag}/{N} flagged (buried in walls)", fontsize=8.5, color=accent)
    ax.legend(fontsize=7, loc="upper right"); ax.set_axis_off()
fig.suptitle("Insertion map (Si-CHA + Ar, view down c): flagged insertions sit inside the framework walls",
             fontsize=10, y=0.98)
fig.tight_layout()
S.save(fig, "f3a_insertion_map")
print(f"small flagged={int(np.array([r['flagged'] for r in S.per_insertion('small')]).sum())}, "
      f"uma flagged={int(np.array([r['flagged'] for r in S.per_insertion('uma_omat')]).sum())}")
