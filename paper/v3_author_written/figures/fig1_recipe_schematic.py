#!/usr/bin/env python3
"""Figure 1 - "A Widom number is a recipe" (SCHEMATIC, no data values).

A conceptual flow: the locked recipe ingredients feed Widom insertion, whose
K_H / Q_st output is classified into exactly one of six verdicts. The ingredient
list is the one stated in Section 1; the six verdicts are the locked v0.4
classes. No numbers appear in this figure.

Run:  .venv/bin/python paper/v3_author_written/figures/fig1_recipe_schematic.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

INGREDIENTS = [
    "structure", "atom typing", "force field", "gas model", "mixing rule",
    "electrostatics", "backend", "sampling protocol", "reference observable",
]
VERDICTS = [
    "strict pass", "Tier-B physical pass", "force-field disagreement",
    "reference mismatch", "structural blocker", "ensemble / method mismatch",
]


def panel(ax, x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec="black", lw=1.0))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
            fontsize=9, fontweight="bold")
    ax.plot([x + 0.2, x + w - 0.2], [y + h - 0.58, y + h - 0.58],
            color="0.5", lw=0.6)
    top = y + h - 0.95
    step = (h - 1.25) / max(len(lines) - 1, 1) if len(lines) > 1 else 0
    for i, ln in enumerate(lines):
        ax.text(x + 0.22, top - i * step, u"• " + ln, ha="left",
                va="center", fontsize=8.2)


def main():
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    panel(ax, 0.1, 0.7, 4.1, 4.7, "Recipe (locked per branch)", INGREDIENTS, "#eef3fa")
    panel(ax, 4.55, 2.6, 2.6, 1.3, "Widom insertion",
          [r"$\langle e^{-\beta U}\rangle \rightarrow K_H,\ Q_{st}$"], "#fbf3e6")
    panel(ax, 7.95, 0.7, 3.9, 4.7, "Verdict (exactly one)", VERDICTS, "#eef7ee")

    ax.add_patch(FancyArrowPatch((4.25, 3.25), (4.5, 3.25), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.4, color="black"))
    ax.add_patch(FancyArrowPatch((7.2, 3.25), (7.9, 3.25), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.4, color="black"))
    ax.text(6.0, 0.2, "Schematic - no data values", ha="center",
            fontsize=8, style="italic", color="0.35")

    out = os.path.join(HERE, "fig1_recipe_schematic.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
