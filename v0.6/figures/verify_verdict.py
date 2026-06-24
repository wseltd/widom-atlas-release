#!/usr/bin/env python3
"""Regenerate a WPB verdict from committed per-insertion data, re-applying VERDICT_SCHEMA_V06 — no GPU,
no MLIP, Python stdlib only. Confirms the shipped verdict JSONs follow from the committed evidence.

Usage:  python verify_verdict.py [leg]      # leg = small | medium-mpa-0 | medium-mpa-0_D3 | uma_omat | ...
Reads v0.6/WPB_kups_mlip_ood/wpb_<leg>_per_insertion.json and the matching summary, recomputes the
flags, flagged Boltzmann-weight fraction, min U, and the Class-I/Class-II verdict, and compares.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WPB = os.path.join(HERE, "..", "WPB_kups_mlip_ood")
T_K, R_kJ = 298.15, 8.314462618e-3
RT = R_kJ * T_K
SIG_O, SIG_AR = 3.500, 3.405
HARD = 0.80 * 0.5 * (SIG_O + SIG_AR)       # 2.762 A overlap threshold
FLOOR = -25.0

leg = sys.argv[1] if len(sys.argv) > 1 else "small"
rows = json.load(open(os.path.join(WPB, f"wpb_{leg}_per_insertion.json")))
summary = json.load(open(os.path.join(WPB, f"wpb_{leg}.json")))

ws, flags = [], []
for r in rows:
    u = r["U_mace_kJ"]
    w = math.exp(min(-u / RT, 700.0)) if u == u else 0.0          # u==u guards NaN
    hard = r["min_dist_A"] < HARD
    anom = (u < FLOOR) or (u < r["U_classical_kJ"] - 50.0)
    ws.append(w)
    flags.append(bool(hard or anom))

wsum = sum(ws)
flagged_wf = sum(w for w, f in zip(ws, flags) if f) / wsum if wsum > 0 else 0.0
min_u = min(r["U_mace_kJ"] for r in rows)
if min_u > 0:
    verdict = "REFUSE [Class II: no-physisorption]"
elif flagged_wf >= 0.5:
    verdict = "REFUSE [Class I: OOD over-binding]"
else:
    verdict = "GOVERNED PASS WITH FLAGS"

print(f"leg = {leg}   (N={len(rows)})")
print(f"  recomputed: min_U={min_u:.2f}  flagged_weight_fraction={flagged_wf:.4f}  ->  {verdict}")
print(f"  shipped   : min_U={summary['min_U_mace_kJ']}  flagged_weight_fraction={summary['flagged_weight_fraction']}")
print(f"  shipped verdict: {summary['verdict'][:60]}")
ok = abs(flagged_wf - summary["flagged_weight_fraction"]) < 0.01 and verdict.split("[")[0].strip() in summary["verdict"]
print(f"  MATCH: {'YES' if ok else 'NO — investigate'}")
sys.exit(0 if ok else 1)
