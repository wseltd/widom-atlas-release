# v0.6 figures — captions

**Standing rule:** no numeric value is hardcoded in any figure label/legend/caption — every number below
and in every figure renders from a committed JSON at build time (`figstyle.jget`). Regenerate all:
`cd v0.6/figures && for s in f1_triptych f2_uma_domain fp_cross_engine_parity f3a_insertion_map f3b_pathology f3c_indomain_well; do PYTHONPATH=. python $s.py; done`.
Shared model→color map and value loader: `figstyle.py`. Outputs are PDF+PNG pairs.

## F1 — `f1_wpb_triptych` (WPB, Si-CHA + Ar)
Governance discriminates the failure mode across four configurations. MACE-MP-small (2023) over-binds
overlaps (min -19.3 kJ/mol, flagged-weight 0.99) → REFUSE I; MACE-MPA-0 bare
is repulsive everywhere (min +8.5) → REFUSE II (no dispersion); **+D3(BJ) recovers a
physical well (min -5.0, screen q_st 5.1) → screen-PASS**; UMA omat is erratic
at overlaps (min -292.8, flagged-weight 1.00) → REFUSE I. *(Fixes the −334
stale-label bug: the UMA minimum now reads from the periodic-run JSON.)*

## F2 — `f2_uma_domain` (governance respects the training domain)
**(A)** The UMA odac head on CO₂/CHA (in-domain) finds a physical well (min -15.7
kJ/mol, Gate-1 +0.06 kJ, N=60); on Ar/CHA (out-of-domain) it is internally
inconsistent (Gate-1 -2.73 kJ at symmetry-equivalent sites) → WITHHELD. **(B)** Verdict
matrix: omat·Ar REFUSE I (min -292.8), odac·Ar WITHHELD, omat·CO₂ REFUSE II
(min +1.2), **odac·CO₂ screen-PASS** (q_st 17.7 vs anchor
21.0, −3.3 below strict).

## FP — `fp_cross_engine_parity` (cross-engine Widom parity)
**(A) 6c MFI+Ar:** native+Ar-Si K_H 0.209 reproduces RASPA3 0.207
(+1.1%); dropping the Ar-Si term (O-only 0.115) is ×1.82 low —
the load-bearing-mixing-term result. **(B) 4c CHA+CO₂ (Ewald):** convention matrix — shift
+11.6%, tail +11.9%; kUPS matched-convention residual -8.1%
(q_st-invariant, normalization-flavoured).

## F3a — `f3a_insertion_map` (where the spurious weight lives)
CHA framework (muted) with seed-0 Ar insertion points colored by verdict flag (OOD red ✕ / clean blue ●),
viewed down c. MACE-MP-small 99/120 flagged, UMA-omat 117/120 flagged — the
flagged insertions sit inside the framework walls. Geometries replayed from seed 0; flags from JSON.

## F3b — `f3b_pathology` (OOD made visceral)
The single worst UMA-omat insertion: U = -292.8 kJ/mol — a deep *spurious* attraction at a
2.65 Å
atomic overlap (Ar drawn at ~vdW scale visibly clashing the nearest O). Geometry replayed from seed 0.

## F3c — `f3c_indomain_well` (governance passes the model in-domain)
CO₂ at the odac minimum-energy site in CHA: U = -15.7 kJ/mol, screen q_st 17.7
→ screen-PASS. The CO₂ sits in the pore void (a physical physisorption site), unlike the Ar overlap
pathology — the in-domain control. Geometry replayed from seed 0.

## Note on v0.5 figures (frozen)
The pre-existing v0.5 figures (`WP1_source_paper_parity`, `WP2_convergence_curves`,
`WP3_mlip_widom_governance_demo`) predate this rule and contain **audited hardcoded numbers** (e.g. WP1
"Q_st 57.03", WP2 "12.6% → 1.5%"; WP1 reads no JSON). They live under the `v0.5-frozen` tag, so they are
**not edited here** (that would break the freeze) and **none are used in the v0.6 release** — the three
release figures (F2, F3a, FP) are all v0.6 and fully no-hardcode. Any v0.5 figure promoted to the paper
must be rebuilt under this rule first.
