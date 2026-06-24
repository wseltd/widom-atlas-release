#!/usr/bin/env python3
"""Figure 2 - Strict result plot.

Plots the scalar comparison rows of Table 1 in the (delta log10 K_H, delta Q_st)
deviation plane and draws the Tier-A strict band (+/-0.10, +/-2.0 kJ/mol). Only
6c and 3a fall inside it: the 2/15 strict result. Numbers are read from
branch_data.csv (audited; derived from Table 1 / the verdict JSONs) and are NEVER
hand-typed in this script.

Labelling avoids collisions: the well-separated high-deviation points are tagged
with leader lines in the left panel; the dense near-origin cluster is resolved in
a right-hand zoom panel. The UiO-66 3b family appears as three force-field
variants, so the plotted rows total sixteen; the locked headline denominator
remains the fifteen-branch disposition (Table 1 / Appendix C).

Run:  .venv/bin/python paper/v3_author_written/figures/fig2_strict_result.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "branch_data.csv")

KH_BAND = 0.10
QST_BAND = 2.0
COLORS = {"le1": "#1a9850", "le2": "#4575b4", "2to3": "#fd8d3c", "gt3": "#d73027"}
CLABEL = {"le1": r"$\leq 1\sigma$", "le2": r"$\leq 2\sigma$",
          "2to3": r"$2$-$3\sigma$", "gt3": r"$>3\sigma$"}
ORDER = ["le1", "le2", "2to3", "gt3"]


def load():
    rows = [l for l in open(DATA) if not l.startswith("#")]
    out = {}
    for r in csv.DictReader(rows):
        if r["bclass"] == "blocked" or r["dlogKH"] == "":
            continue
        out[r["branch"]] = {"x": float(r["dlogKH"]), "y": float(r["dQst"]),
                            "bclass": r["bclass"], "tierA": r["tierA"]}
    return out


def strict_box(ax):
    ax.add_patch(Rectangle((-KH_BAND, -QST_BAND), 2 * KH_BAND, 2 * QST_BAND,
                           facecolor="#1a9850", alpha=0.13, edgecolor="#1a9850",
                           lw=1.2, zorder=1))
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.axvline(0, color="0.8", lw=0.6, zorder=0)


def points(ax, data):
    for d in data.values():
        edge = "black" if d["tierA"] == "PASS" else "none"
        ax.scatter(d["x"], d["y"], s=78 if d["tierA"] == "PASS" else 40,
                   c=COLORS[d["bclass"]], edgecolors=edge, linewidths=1.1, zorder=3)


def lab(ax, text, xy, off, arrow=False):
    ha = "left" if off[0] >= 0 else "right"
    ax.annotate(text, xy, textcoords="offset points", xytext=off, fontsize=7.3,
                ha=ha, va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45",
                                shrinkA=1, shrinkB=2) if arrow else None)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    d = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.0))

    # ---- Panel A: full range, label only well-separated points ----
    strict_box(ax1)
    points(ax1, d)
    lab(ax1, "2b", (d["2b"]["x"], d["2b"]["y"]), (-26, 0), arrow=True)
    lab(ax1, "1d", (d["1d"]["x"], d["1d"]["y"]), (-26, 6), arrow=True)
    lab(ax1, "1a, 1b", ((d["1a"]["x"] + d["1b"]["x"]) / 2, (d["1a"]["y"] + d["1b"]["y"]) / 2),
        (16, -2), arrow=True)
    lab(ax1, "4a", (d["4a"]["x"], d["4a"]["y"]), (12, 4))
    lab(ax1, "1c", (d["1c"]["x"], d["1c"]["y"]), (12, 0))
    # 2a/3a are no longer high-deviation: after the genuine-UFF rebuild they sit in the
    # near-origin cluster (resolved in the right panel), so they are labelled there.
    ax1.add_patch(Rectangle((-0.52, -6.8), 0.70, 11.2, fill=False, ls=(0, (4, 3)),
                            ec="0.45", lw=0.8, zorder=2))
    ax1.annotate("near-origin cluster\n(see right panel)", (0.18, -6.8),
                 textcoords="offset points", xytext=(6, -2), fontsize=7.0,
                 color="0.35", va="top")
    ax1.set_xlim(-1.0, 2.15)
    ax1.set_ylim(-11.5, 16.5)
    ax1.set_xlabel(r"$\Delta\log_{10} K_H$")
    ax1.set_ylabel(r"$\Delta Q_{st}$ (kJ mol$^{-1}$)")
    ax1.set_title("Displayed scalar comparisons", fontsize=10)
    ax1.grid(True, ls=":", lw=0.4, alpha=0.5)

    # ---- Panel B: zoom on the near-origin / strict-band region ----
    strict_box(ax2)
    points(ax2, d)
    lab(ax2, "4c", (d["4c"]["x"], d["4c"]["y"]), (10, 6))
    lab(ax2, "2a", (d["2a"]["x"], d["2a"]["y"]), (10, 2))
    lab(ax2, "3a", (d["3a"]["x"], d["3a"]["y"]), (10, 4))
    lab(ax2, "6c", (d["6c"]["x"], d["6c"]["y"]), (10, -3))
    lab(ax2, "6b", (d["6b"]["x"], d["6b"]["y"]), (9, -4))
    lab(ax2, "6e", (d["6e"]["x"], d["6e"]["y"]), (-30, 10), arrow=True)
    lab(ax2, "6a", (d["6a"]["x"], d["6a"]["y"]), (-30, -10), arrow=True)
    lab(ax2, "3b_EHq", (d["3b_EHq"]["x"], d["3b_EHq"]["y"]), (10, -6))
    lab(ax2, "3b_UA, 3b_UAq",
        ((d["3b_UA"]["x"] + d["3b_UAq"]["x"]) / 2, (d["3b_UA"]["y"] + d["3b_UAq"]["y"]) / 2),
        (-14, -12), arrow=True)
    ax2.text(0.0, 1.0, "strict band", fontsize=7.3, color="#1a9850",
             ha="center", va="center")
    ax2.set_xlim(-0.52, 0.22)
    ax2.set_ylim(-7.0, 4.6)
    ax2.set_xlabel(r"$\Delta\log_{10} K_H$")
    ax2.set_title("Strict-band region (zoom)", fontsize=10)
    ax2.grid(True, ls=":", lw=0.4, alpha=0.5)

    # ---- shared legend below, outside the axes ----
    handles = [Line2D([0], [0], marker="o", ls="", mfc=COLORS[k], mec="none",
                      ms=7, label=CLABEL[k]) for k in ORDER]
    handles.append(Line2D([0], [0], marker="o", ls="", mfc="white", mec="black",
                          mew=1.2, ms=8, label="Tier-A strict pass"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(HERE, "fig2_strict_result.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
