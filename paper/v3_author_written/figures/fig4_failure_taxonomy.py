#!/usr/bin/env python3
"""Figure 4 - Failure taxonomy (SCHEMATIC, no data values).

The four ways a calculated zero-coverage observable can lose contact with the
experiment, each with its one canonical v0.4 branch example. The causes and the
examples are taken from Sections 1, 6 and 7 and the locked branch dispositions
(Mg-MOF-74/HKUST-1 force-field error; MFI+Kr van't-Hoff reference mismatch;
cationic-zeolite 5c structural blocker; Na-Rho 5b ensemble mismatch). No numbers.

Run:  .venv/bin/python paper/v3_author_written/figures/fig4_failure_taxonomy.py
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    ("Force-field error",
     "energy model not transferable to the site",
     "Mg-MOF-74 & HKUST-1 + CO$_2$\n(open metal sites)", "#fdecea"),
    ("Reference-observable mismatch",
     "literature number is a different observable",
     "MFI + Kr\n($Q_{st}$ from a van't Hoff fit)", "#fff7e6"),
    ("Structural blocker",
     "calculation uses the wrong material state",
     "cationic zeolites (5c)\n(missing cations / structures)", "#eaf3fb"),
    ("Ensemble / method mismatch",
     "calculation samples a different partition function",
     "Na-Rho (5b)\n(rigid closed vs open trapdoor)", "#eef7ee"),
]


def main():
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    pos = [(0.3, 3.1), (5.15, 3.1), (0.3, 0.55), (5.15, 0.55)]
    w, h = 4.55, 2.25
    for (title, sub, ex, fc), (x, y) in zip(CELLS, pos):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.04",
                                    fc=fc, ec="black", lw=1.0))
        ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top",
                fontsize=10, fontweight="bold")
        sub_wrapped = textwrap.fill(sub, width=30)
        ax.text(x + w / 2, y + h - 0.66, sub_wrapped, ha="center", va="top",
                fontsize=8.0, style="italic", color="0.3", linespacing=1.2)
        ax.text(x + w / 2, y + 0.58, ex, ha="center", va="center", fontsize=8.6)
    ax.text(5.0, 0.12, "Schematic - canonical branch examples, no data values",
            ha="center", fontsize=8, style="italic", color="0.35")

    out = os.path.join(HERE, "fig4_failure_taxonomy.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
