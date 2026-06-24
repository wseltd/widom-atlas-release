#!/usr/bin/env python3
"""F3b — pathology close-up: the single worst UMA-omat insertion — a deep spurious attraction at an
atomic overlap. Geometry replayed from seed 0; the energy and distance are read from the committed
per-insertion / summary JSONs (no hardcoding)."""
import figstyle as S
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

CIF = S.CHA_CIF
CUTOFF, SEED = 6.0, 0
frame = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell); inv = np.linalg.inv(cell)
fpos = frame.get_positions(); fsym = np.array(frame.get_chemical_symbols())

rows = S.per_insertion("uma_omat")
carts = []
rng = np.random.default_rng(SEED)
for _ in rows:
    carts.append(rng.random(3) @ cell)
carts = np.array(carts)
# worst insertion = global min U
imin = int(np.argmin([r["U_mace_kJ"] for r in rows]))
U_worst = S.jget("wpb_uma_omat.json", "min_U_mace_kJ")
ar = carts[imin]
# nearest framework atom under minimum image
df = (fpos - ar) @ inv; df -= np.round(df); disp = df @ cell; dist = np.linalg.norm(disp, axis=1)
jnear = int(np.argmin(dist)); dmin = float(dist[jnear])
near_disp = disp[jnear]                      # vector ar -> nearest image
# local cluster: framework atoms within 5 A (min-image), placed relative to Ar at origin
within = dist < 5.0
loc = disp[within]; locsym = fsym[within]

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(5.6, 5.4))
RVDW = {"Si": 1.45, "O": 1.32}     # ~van-der-Waals scale so the overlap is visible
for k, (p, s) in enumerate(zip(loc, locsym)):
    isnear = np.allclose(p, near_disp)
    ax.add_patch(plt.Circle((p[0], p[1]), RVDW[s], alpha=0.85,
                            facecolor=("#e3873a" if isnear else ("#c9a96a" if s == "Si" else "#d7d2c8")),
                            edgecolor=(S.COLORS["flag_ood"] if isnear else "k"), lw=(1.6 if isnear else 0.6), zorder=2))
ax.add_patch(plt.Circle((0, 0), 1.66, facecolor=S.COLORS["uma_omat"], alpha=0.9, edgecolor="k", lw=1.0, zorder=4))
ax.text(0, 0, "Ar", ha="center", va="center", fontsize=9, color="w", zorder=5, weight="bold")
ax.plot([0, near_disp[0]], [0, near_disp[1]], color=S.COLORS["flag_ood"], lw=2.2, zorder=5)
ax.text(near_disp[0] / 2 + 0.5, near_disp[1] / 2, f"{S.fmt(dmin, 2)} Å",
        color=S.COLORS["flag_ood"], fontsize=13, ha="left", va="center", weight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=S.COLORS["flag_ood"], lw=1.0))
ax.set_xlim(near_disp[0] - 4.5, near_disp[0] + 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal"); ax.set_axis_off()
ax.set_title(f"UMA omat pathology: spurious deep well\nU = {S.fmt(U_worst, signed=True)} kJ/mol at a "
             f"{S.fmt(dmin, 2)} Å overlap (insertion #{imin})", fontsize=9, color=S.COLORS["uma_omat"])
fig.tight_layout()
S.save(fig, "f3b_pathology")
print(f"worst insertion #{imin}: U={U_worst} kJ/mol, nearest framework {fsym[jnear]} at {dmin:.2f} A "
      f"(JSON min_dist={rows[imin]['min_dist_A']})")
