#!/usr/bin/env python3
"""Shared figure style for v0.6 — the SINGLE source of model→color mapping and JSON value loading.

Standing rule (post the −334 stale-label bug): NO numeric result value may be hardcoded in any figure
label / legend / caption. All result values render from the committed source JSON at build time via
`jget`. Fixed *parameters* (overlap threshold, T) are defined ONCE here so they too have a single
source. Import this everywhere; never re-type a color or a number.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")

# REPO derived relatively (figures/ -> v0.6 -> repo root) so this works in the repo AND the public export
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WPB = os.path.join(REPO, "v0.6/WPB_kups_mlip_ood")
WPA = os.path.join(REPO, "v0.6/WPA_kups_widom")
CHA_CIF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHA_iza.cif")  # public IZA structure, shipped

# ── Fixed model → color/marker mapping (single source, used by ALL figures) ──
COLORS = {
    "classical": "#1f77b4",        # classical UFF reference
    "mace_small": "#9467bd",       # MACE-MP-small (2023)
    "mpa0_bare": "#2ca02c",        # MACE-MPA-0 bare (kUPS shipped)
    "mpa0_d3": "#ff7f0e",          # MACE-MPA-0 + D3(BJ)
    "uma_omat": "#8c564b",         # UMA uma-s-1.1 omat
    "uma_odac": "#17becf",         # UMA uma-s-1.1 odac
    "flag_ood": "#d62728",         # OOD-flagged insertion (red)
    "flag_clean": "#1f77b4",       # clean insertion (blue)
    "floor": "#d62728",
}
MARKERS = {
    "classical": "o", "mace_small": "x", "mpa0_bare": "^",
    "mpa0_d3": "D", "uma_omat": "v", "uma_odac": "s",
}
LABELS = {  # display names (no numbers — those come from JSON)
    "classical": "classical UFF", "mace_small": "MACE-MP-small (2023)",
    "mpa0_bare": "MACE-MPA-0 bare (kUPS)", "mpa0_d3": "MACE-MPA-0 + D3(BJ)",
    "uma_omat": "UMA uma-s-1.1 omat", "uma_odac": "UMA uma-s-1.1 odac",
}

# ── Fixed parameters (single source; not results) ──
T_K = 298.15
# Overlap threshold = 0.80 * 0.5*(sigma_O + sigma_Ar), the WPB hard-overlap rule.
#
# AUDIT CORRECTION (2026-07): the as-run threshold was 2.762 A, built from
# 0.5*(3.500 + 3.405). Those lengths are not in the same convention -- 3.500 is
# UFF's x_i for oxygen, which is r_min, NOT sigma (sigma = x_i / 2**(1/6) =
# 3.118), while 3.405 is a Lennard-Jones sigma for argon. Re-derived
# self-consistently in genuine UFF the threshold is 2.626 A. Re-scoring the
# committed per-insertion evidence at 2.626 A leaves every refuse/screen-pass
# verdict unchanged but drops the UMA-omat flagged weight from 0.998 to 0.000
# (its deepest insertion sits at 2.646 A). OVERLAP_A_AS_RUN is retained so the
# published figures remain reproducible; OVERLAP_A is the corrected value.
TWO_ONE_SIXTH = 2.0 ** (1.0 / 6.0)
UFF_XI_O = 3.500                      # UFF nonbond distance x_i (= r_min), oxygen
SIGMA_O = UFF_XI_O / TWO_ONE_SIXTH    # -> 3.118 A
SIGMA_AR = 3.868 / TWO_ONE_SIXTH      # UFF argon x_i 3.868 -> 3.446 A
OVERLAP_A = round(0.80 * 0.5 * (SIGMA_O + SIGMA_AR), 3)          # 2.626 A
OVERLAP_A_AS_RUN = round(0.80 * 0.5 * (UFF_XI_O + 3.405), 3)     # 2.762 A (superseded)


def load(path):
    """Load a JSON file (absolute, or relative to WPB then WPA then REPO)."""
    for base in ("", WPB, WPA, REPO):
        p = path if os.path.isabs(path) else os.path.join(base, path)
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(path)


def jget(path, *keys, default=None):
    """Read a nested value from a committed JSON — the only sanctioned way to put a number on a figure."""
    d = load(path)
    for k in keys:
        if isinstance(d, list) or k in d:
            d = d[k]
        else:
            return default
    return d


def per_insertion(name):
    """Load a WPB per-insertion list by leg name, e.g. 'medium-mpa-0', 'small', 'uma_omat'."""
    return load(os.path.join(WPB, f"wpb_{name}_per_insertion.json"))


def fmt(v, prec=1, signed=False):
    """Format a number for a label; raises if v is None so a missing JSON key can't silently blank."""
    if v is None:
        raise ValueError("figure label value is None — JSON key missing; fix the source, don't hardcode")
    s = f"{v:+.{prec}f}" if signed else f"{v:.{prec}f}"
    return s


def save(fig, stem):
    """Save PDF+PNG pair into v0.6/figures/ and print the path."""
    out = os.path.join(REPO, "v0.6/figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"wrote figures/{stem}.{{pdf,png}}")
