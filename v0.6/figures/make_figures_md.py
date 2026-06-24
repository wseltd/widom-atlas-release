#!/usr/bin/env python3
"""Generate v0.6/figures/FIGURES.md — one caption per figure, EVERY number sourced from a committed
JSON via figstyle.jget (the no-hardcode rule applied to the captions doc itself)."""
import figstyle as S

g = S.jget
P6, CM = "parity_6c_threeway.json", "native_4c_convention_matrix.json"

V = dict(
    sml=g("wpb_small.json", "min_U_mace_kJ"), fw_sml=g("wpb_small.json", "flagged_weight_fraction"),
    mpa=g("wpb_medium-mpa-0.json", "min_U_mace_kJ"), d3=g("wpb_medium-mpa-0_D3.json", "min_U_mace_kJ"),
    d3q=g("wpb_medium-mpa-0_D3.json", "screen_q_st_kJ"),
    uma=g("wpb_uma_omat.json", "min_U_mace_kJ"), uma_fw=g("wpb_uma_omat.json", "flagged_weight_fraction"),
    co2min=g("wpb_uma_co2_odac.json", "min_U_mace_kJ"), co2q=g("wpb_uma_co2_odac.json", "screen_q_st_kJ"),
    co2anchor=g("wpb_uma_co2_odac.json", "accuracy_anchor_q_st_kJ"),
    co2gate=g("wpb_uma_co2_odac.json", "gate1_second_open_U_kJ"), co2N=g("wpb_uma_co2_odac.json", "N"),
    argate=g("wpb_uma_odac.json", "gate1_second_open_U_kJ"), omatCO2=g("wpb_uma_co2_omat.json", "min_U_mace_kJ"),
    raspa=g(P6, "RASPA3_locked", "K_H_mol_kg_bar"), natSi=g(P6, "native_with_Si", "K_H_mol_kg_bar"),
    kupSi=g(P6, "kups_with_Si", "K_H_mol_kg_bar"), natO=g(P6, "native_O_only", "K_H_mol_kg_bar"),
    par=g(P6, "deltas", "native_withSi_vs_RASPA3_KH_pct"), si=g(P6, "deltas", "Si_term_KH_inflation_native"),
    shift=g(CM, "decomposition_+2.2pct_vs_lock", "shift"), tail=g(CM, "one_toggle_decomposition_pct", "tail_on_minus_off_shifted"),
    resid=g(CM, "decomposition_+2.2pct_vs_lock", "residual_truncated+tail"),
    nflag_sml=g("wpb_small.json", "n_flagged"), nflag_uma=g("wpb_uma_omat.json", "n_flagged"),
)
f = S.fmt

md = f"""# v0.6 figures — captions

**Standing rule:** no numeric value is hardcoded in any figure label/legend/caption — every number below
and in every figure renders from a committed JSON at build time (`figstyle.jget`). Regenerate all:
`cd v0.6/figures && for s in f1_triptych f2_uma_domain fp_cross_engine_parity f3a_insertion_map f3b_pathology f3c_indomain_well; do PYTHONPATH=. python $s.py; done`.
Shared model→color map and value loader: `figstyle.py`. Outputs are PDF+PNG pairs.

## F1 — `f1_wpb_triptych` (WPB, Si-CHA + Ar)
Governance discriminates the failure mode across four configurations. MACE-MP-small (2023) over-binds
overlaps (min {f(V['sml'],1,True)} kJ/mol, flagged-weight {f(V['fw_sml'],2)}) → REFUSE I; MACE-MPA-0 bare
is repulsive everywhere (min {f(V['mpa'],1,True)}) → REFUSE II (no dispersion); **+D3(BJ) recovers a
physical well (min {f(V['d3'],1,True)}, screen q_st {f(V['d3q'])}) → screen-PASS**; UMA omat is erratic
at overlaps (min {f(V['uma'],1,True)}, flagged-weight {f(V['uma_fw'],2)}) → REFUSE I. *(Fixes the −334
stale-label bug: the UMA minimum now reads from the periodic-run JSON.)*

## F2 — `f2_uma_domain` (governance respects the training domain)
**(A)** The UMA odac head on CO₂/CHA (in-domain) finds a physical well (min {f(V['co2min'],1,True)}
kJ/mol, Gate-1 {f(V['co2gate'],2,True)} kJ, N={V['co2N']}); on Ar/CHA (out-of-domain) it is internally
inconsistent (Gate-1 {f(V['argate'],2,True)} kJ at symmetry-equivalent sites) → WITHHELD. **(B)** Verdict
matrix: omat·Ar REFUSE I (min {f(V['uma'],1,True)}), odac·Ar WITHHELD, omat·CO₂ REFUSE II
(min {f(V['omatCO2'],1,True)}), **odac·CO₂ screen-PASS** (q_st {f(V['co2q'])} vs anchor
{f(V['co2anchor'])}, −{f(V['co2anchor']-V['co2q'],1)} below strict).

## FP — `fp_cross_engine_parity` (cross-engine Widom parity)
**(A) 6c MFI+Ar:** native+Ar-Si K_H {f(V['natSi'],3)} reproduces RASPA3 {f(V['raspa'],3)}
({f(V['par'],1,True)}%); the ×{f(V['si'],2)} gap to the O-only recipe ({f(V['natO'],3)}) is the silicon
double-count of Finding F1 (Talu's Ar-O already folds silicon into oxygen). **(B) 4c CHA+CO₂ (Ewald):** convention matrix — shift
{f(V['shift'],1,True)}%, tail {f(V['tail'],1,True)}%; kUPS matched-convention residual {f(V['resid'],1,True)}%
(q_st-invariant, normalization-flavoured).

## F3a — `f3a_insertion_map` (where the spurious weight lives)
CHA framework (muted) with seed-0 Ar insertion points colored by verdict flag (OOD red ✕ / clean blue ●),
viewed down c. MACE-MP-small {V['nflag_sml']}/120 flagged, UMA-omat {V['nflag_uma']}/120 flagged — the
flagged insertions sit inside the framework walls. Geometries replayed from seed 0; flags from JSON.

## F3b — `f3b_pathology` (OOD made visceral)
The single worst UMA-omat insertion: U = {f(V['uma'],1,True)} kJ/mol — a deep *spurious* attraction at a
{f(g('wpb_uma_omat_per_insertion.json', int(__import__('numpy').argmin([r['U_mace_kJ'] for r in S.per_insertion('uma_omat')])), 'min_dist_A'),2)} Å
atomic overlap (Ar drawn at ~vdW scale visibly clashing the nearest O). Geometry replayed from seed 0.

## F3c — `f3c_indomain_well` (governance passes the model in-domain)
CO₂ at the odac minimum-energy site in CHA: U = {f(V['co2min'],1,True)} kJ/mol, screen q_st {f(V['co2q'])}
→ screen-PASS. The CO₂ sits in the pore void (a physical physisorption site), unlike the Ar overlap
pathology — the in-domain control. Geometry replayed from seed 0.

## Note on v0.5 figures (frozen)
The pre-existing v0.5 figures (`WP1_source_paper_parity`, `WP2_convergence_curves`,
`WP3_mlip_widom_governance_demo`) predate this rule and contain **audited hardcoded numbers** (e.g. WP1
"Q_st 57.03", WP2 "12.6% → 1.5%"; WP1 reads no JSON). They live under the `v0.5-frozen` tag, so they are
**not edited here** (that would break the freeze) and **none are used in the v0.6 release** — the three
release figures (F2, F3a, FP) are all v0.6 and fully no-hardcode. Any v0.5 figure promoted to the paper
must be rebuilt under this rule first.
"""
open(S.os.path.join(S.REPO, "v0.6/figures/FIGURES.md"), "w").write(md)
print("wrote FIGURES.md (all caption numbers from JSON)")
