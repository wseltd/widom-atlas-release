#!/usr/bin/env python3
"""Figure 3 - Branch disposition.

Horizontal bar chart of the v0.4 disposition, in mutually-exclusive categories,
counted from branch_data.csv (audited). The 5c cationic-zeolite family is shown
as a separate, hatched, reference-audited (not-executed) annotation - it is NOT
a scalar-scored branch. Numbers are computed from the CSV, never hand-typed.

Run:  .venv/bin/python paper/v3_author_written/figures/fig3_branch_disposition.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "branch_data.csv")


def load():
    rows = [l for l in open(DATA) if not l.startswith("#")]
    return list(csv.DictReader(rows))


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    rows = load()
    strict = [r for r in rows if r["tierA"] == "PASS"]
    tierb_only = [r for r in rows if r["tierA"] == "FAIL" and r["tierB"] == "PASS"]
    near = [r for r in rows if r["tierB"] == "FAIL" and r["bclass"] == "le2"]
    tw = [r for r in rows if r["bclass"] == "2to3"]
    gt = [r for r in rows if r["bclass"] == "gt3"]
    blocked = [r for r in rows if r["bclass"] == "blocked"]

    # (label, count, color, hatch). 5c is an annotation, not a CSV row.
    # Branch names in the strict / Tier-B-only labels are DATA-DERIVED from the same
    # rows used for the counts, so they can never drift from the disposition (the
    # previous hard-coded "(6c, 4c)" / "(3b_EHq)" went stale after the v0.4.2 rebuild).
    strict_names = ", ".join(r["branch"] for r in strict)
    tierb_only_names = ", ".join(r["branch"] for r in tierb_only)
    cats = [
        (f"Tier-A strict pass\n({strict_names})", len(strict), "#1a9850", None),
        (f"Tier-B physical pass only\n({tierb_only_names})", len(tierb_only), "#74c476", None),
        (r"Near-miss, $\leq 2\sigma$", len(near), "#4575b4", None),
        (r"Disagreement, $2$-$3\sigma$", len(tw), "#fd8d3c", None),
        (r"Disagreement, $>3\sigma$", len(gt), "#d73027", None),
        ("Method-blocked\n(5b, not scored)", len(blocked), "#7a7a7a", None),
        ("Reference-audited 5c\n(not executed)", 4, "#bdbdbd", "////"),
    ]
    labels = [c[0] for c in cats][::-1]
    counts = [c[1] for c in cats][::-1]
    colors = [c[2] for c in cats][::-1]
    hatches = [c[3] for c in cats][::-1]

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ypos = range(len(labels))
    bars = ax.barh(list(ypos), counts, height=0.62, color=colors,
                   edgecolor="black", lw=0.7)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    for y, c in zip(ypos, counts):
        ax.text(c + 0.14, y, str(c), va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("number of branches")
    ax.set_xlim(0, max(counts) + 1.4)
    ax.set_title("v0.4 branch disposition", fontsize=10)
    ax.grid(True, axis="x", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    out = os.path.join(HERE, "fig3_branch_disposition.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", out, "| scalar rows:", len(strict) + len(tierb_only) + len(near) + len(tw) + len(gt))


if __name__ == "__main__":
    main()
