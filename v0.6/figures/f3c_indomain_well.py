#!/usr/bin/env python3
"""F3c — in-domain well: the CO2 configuration at the odac minimum-energy site in CHA (in-domain for
the DAC-trained head). Rigid CO2 geometry replayed from seed 0; well energy read from the committed
JSON. The "governance passes the model in its own training domain" image."""
import figstyle as S
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

CIF = S.CHA_CIF
CUTOFF, SEED, DCO = 6.0, 0, 1.16
frame = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell); inv = np.linalg.inv(cell)
fpos = frame.get_positions(); fsym = np.array(frame.get_chemical_symbols())

rows = S.per_insertion("uma_co2_odac")
rng = np.random.default_rng(SEED)
centers, axes = [], []
for _ in rows:
    centers.append(rng.random(3) @ cell)      # matches run_wpb_uma_co2.py draw order
    axes.append(rng.normal(size=3))
imin = int(np.argmin([r["U_mace_kJ"] for r in rows]))
U_well = S.jget("wpb_uma_co2_odac.json", "min_U_mace_kJ")
q_well = S.jget("wpb_uma_co2_odac.json", "screen_q_st_kJ")
center, axis = centers[imin], axes[imin] / np.linalg.norm(axes[imin])
co2 = [("C", center), ("O", center + DCO * axis), ("O", center - DCO * axis)]

# local framework cluster around the CO2 centroid (min image), relative to centroid
df = (fpos - center) @ inv; df -= np.round(df); disp = df @ cell; dist = np.linalg.norm(disp, axis=1)
within = dist < 6.0
loc, locsym = disp[within], fsym[within]

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(5.8, 5.4))
for p, s in zip(loc, locsym):
    ax.add_patch(plt.Circle((p[0], p[1]), 0.7 if s == "Si" else 0.6,
                            facecolor="#c9a96a" if s == "Si" else "#d7d2c8", edgecolor="k", lw=0.5, zorder=2))
cc = {"C": "#333333", "O": "#cc3322"}
pts = [(np.array(p) - center) for _, p in co2]
ax.plot([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]], color="k", lw=2.0, zorder=3)  # O=C=O backbone
for (el, _), p in zip(co2, pts):
    ax.add_patch(plt.Circle((p[0], p[1]), 0.65 if el == "O" else 0.55, facecolor=cc[el],
                            edgecolor="k", lw=0.8, zorder=4))
ax.text(0, -3.4, "CO₂ (in-domain for odac)", ha="center", fontsize=8.5, color=S.COLORS["uma_odac"])
ax.set_xlim(-5, 5); ax.set_ylim(-4.2, 4.2); ax.set_aspect("equal"); ax.set_axis_off()
ax.set_title(f"odac in-domain well: CO₂ at its minimum site in CHA\n"
             f"U = {S.fmt(U_well, signed=True)} kJ/mol, screen q_st {S.fmt(q_well)} → screen-PASS",
             fontsize=9, color=S.COLORS["uma_odac"])
fig.tight_layout()
S.save(fig, "f3c_indomain_well")
print(f"CO2 well insertion #{imin}: U={U_well}, q_st={q_well}, min_dist(JSON)={rows[imin]['min_dist_A']}")
