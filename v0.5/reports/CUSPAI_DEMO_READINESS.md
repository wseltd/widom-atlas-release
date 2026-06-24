# CuspAI Demo-Readiness Summary

**Date:** 2026-06-11 · *Technical note. No company, funding, valuation, or
marketing claims.* · Pairs with the v3 manuscript and
`paper/v3_author_written/audit/cuspai_readiness/CUSPAI_TECHNICAL_ONE_PAGER.md`.

## One-sentence pitch
`cusp-ai-oss/widom` **produces** a zero-coverage Widom number from any ASE
calculator (DFT / classical FF / MLIP); **widom-atlas governs it** — it locks the
full recipe behind the number and returns a verdict, and it **refuses** numbers
that an MLIP got from physics it never learned.

## The layering (they compose, they don't compete)
- **CuspAI Widom = estimator.** `run_widom_insertion(calculator, atoms, gas, T,
  num_insertions, seed, …) → WidomInsertionResults` (K_H, heat, per-insertion
  energies, bootstrap σ, accessibility mask). It does not record or judge the recipe.
- **widom-atlas = governance/audit layer.** It does not re-estimate; it SHA-locks
  the recipe (structure, typing, charges, FF form+params, mixing, electrostatics,
  cutoff, tail, backend, sampling, reference, thresholds) and classifies the result
  into one of: Tier-A strict pass · Tier-B physical pass · force-field disagreement ·
  reference-observable mismatch · structural blocker · ensemble/method mismatch ·
  **OOD-refusal**.
- The seam is **ASE** and is implemented in the repo
  (`io/from_widom_result.py`, `backends/user_parameterised.py`, optional dep
  `widom @ git+…/Cusp-AI/widom.git`) and was exercised live (WP2 ran through it).

## What to show, in order (the demo)
1. **WP3 — the headline (refusal).** `wp3_mlip_governance.png` + the verdict card.
   A real MACE-MP Widom run on Si-CHA + Ar where **79/100 insertions are OOD and
   carry 99.3% of the Boltzmann weight**; K_H inflates ~32× (1.74 → 55.5); widom-atlas
   returns **REFUSE**. This is the value proposition in one slide: in a
   high-throughput MLIP screen, the expensive failure is a number that looks fine
   and is silently wrong — and here it is caught.
2. **WP2 — why sampling matters.** `wp2_convergence.png`. Weak binders converge by
   N≈10⁴; **open-metal-site K_H is heavy-tail-dominated** (top 2% of insertions =
   92% of K_H), so production counts are mandatory. A governed number must carry its
   convergence diagnostic, not just a point estimate. (One curve here ran through
   CuspAI Widom — the integration is live, not vaporware.)
3. **WP1 — honesty about validation.** `wp1_parity.png`. Where a source recipe is
   reproducible the atlas reproduces the **source's own** energetics (2b Ongari,
   3b Maia); where it isn't, the failure is reported as **unresolved pending
   source-paper parity**, not a proven force-field flaw. The two strict passes
   (4c, 6c) sit inside the gate box.

## What v0.4 actually shows (conservative, unchanged)
- **2/15** verdict-affecting branches pass the strict gate (MFI+Ar control 6c;
  Si-CHA+CO₂ 4c). **3/15** pass the Tier-B physical band (those two + UiO-66 3b_EHq).
- Open-metal-site MOFs (Mg-MOF-74, HKUST-1) fail under simplified pairwise recipes;
  heats are gas-model- and reference-dependent; generic-FF baselines under-bind.
- **5b** (Na-Rho) is method-blocked (rigid Widom can't represent trapdoor gating);
  **5c** cationic zeolites are reference-audited only.

## Honest gaps a CuspAI lead may probe (and the answer)
| Probe | Answer |
|---|---|
| "Did you reproduce the **source's own** number?" | By *type*: **6b** is a like-for-like **K_H scalar** reproduction (Δlog +0.02 after kPa+T); **2b** energetic, **3b** band, **1c** figure-read (Becker now vendored, charges digit-confirmed). Mercado is a real but isotherm-only paper; Nazarian/PACMOF2 are charge-only (no scalar). Not "6 confirmed scalars." |
| "Are the OMS failures just **under-sampling**?" | No — and the *direction* helps us: finite-N Widom under-samples deep wells, so OMS K_H is a **lower bound** and the over-binding verdicts are conservative; the strict passes are in the converged light-tail regime (Si-CHA by 10⁴). 2b tail audit + the gRASPA 3M sweep quantify it. |
| "Is the **1b** over-binding a real FF flaw?" | **No / UNRESOLVED.** Deterministic battery: params/form/guest/charges correct, double-counting SI-refuted; the gap is **electrostatic** (Mg⁺¹·⁵⁶ × a permissive 1.5 Å clamp ± geometry), and the run-to-run self-disagreement is **sampling variance**. **Not** presented as "Dzubak over-binds"; kept out of proof claims pending the fix (form-agnostic per-insertion floor). See `1b/FINDINGS.md`. |
| "Is CuspAI integration real or slideware?" | **Real** — ASE seam implemented; WP2 ran a sweep through `run_widom_insertion`; WP3 uses the same calculator interface. |
| "Does your governance do anything an estimator doesn't?" | **WP3** — the estimator alone returns 55.5; governance returns **REFUSE** with the OOD diagnostic. |
| "Did you touch the GPU / change v0.4?" | No display-GPU compute; **no v0.4 file modified**; gRASPA path honestly reported blocked. |

## Known residue (pre-submission, non-blocking)
- `v04_two_tier/6c.json` still carries Q_st ref 17.0 (manuscript corrected to the
  locked 15.7 → +0.33; protected JSON patch proposed, not applied).
- `6b` verdict-JSON inconsistency (atlas Q_st 20.73 vs 12.22) flagged for a future
  protected-file pass.
- `Mercado2016JPCC` bib uses "and others"; complete before journal submission.
- Four `feat/v05-*` branches (polarizable, consensus, zeolite, mlip-stub) remain
  **unmerged**, pending the operator's decision on the polarisation finding
  (`../V05_BRANCH_REVIEW.md`).

## What is NOT claimed
No general predictive power (recipes are validated, not materials); no
first-principles proof that a specific FF over/under-binds (only relative to a
stated experimental reference, parity closed for a subset); MLIP-Widom not asserted
valid in general; MACE-MP not asserted fit for adsorption; CuspAI Widom not asserted
wrong; no productised CuspAI integration — the ASE seam is implemented and exercised,
a production pilot is the next step.
