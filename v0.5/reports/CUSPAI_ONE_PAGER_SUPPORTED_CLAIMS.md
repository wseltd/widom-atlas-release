# widom-atlas × cusp-ai-oss/widom — one-pager (supported claims only)

*Technical note. No company, funding, valuation, or marketing claims. Every claim below is
backed by a saved artifact in the `v0.5/` tree; claims that are approximate, unresolved, or
caveated are labelled as such.*

## The layering
- **`cusp-ai-oss/widom` is an estimator/backend.** Given an ASE calculator (DFT, classical FF,
  or an MLIP) and an ASE `Atoms` framework, it returns a zero-coverage Henry coefficient, a heat
  of adsorption, averaged interaction energy, bootstrap uncertainties, and an accessibility mask.
- **widom-atlas is a governance/audit layer.** It does **not** re-estimate; it locks the complete
  recipe behind a number, records provenance, checks source-paper parity where possible, checks
  convergence, detects out-of-distribution (OOD) insertions, and **refuses** Boltzmann averages
  dominated by unphysical events.
- **widom-atlas does not replace `cusp-ai-oss/widom`.** The two compose; **ASE is the integration
  seam** (implemented in-repo; a `widom-atlas convergence` sweep was exercised through
  `run_widom_insertion`).

## What v0.4 conservatively shows (unchanged)
- **2 / 15** strict passes (MFI + Ar control 6c; Si-CHA + CO₂ 4c); **3 / 15** Tier-B physical
  passes (those two + UiO-66 3b_EHq). **5b** (Na-Rho) method-blocked (rigid Widom can't represent
  trapdoor gating); **5c** cationic zeolites reference-audited only.

## v0.5 evidence (each labelled by strength)
- **6c correction strengthens the strict control.** The Ar reference was corrected to
  Q_st = 15.7 kJ/mol, giving ΔQ_st = **+0.33** — the control survives correction.
- **Source-paper parity — by *type*, not lumped (exactly one is a like-for-like published scalar):**
  - **6b (MFI + Kr): confirmed like-for-like K_H scalar parity.** Talu–Myers 2001 (vendored, Table 4
    Kr row, verbatim unit **mol·kg⁻¹·kPa⁻¹**) simulated 0.00783 → 0.783 mol·kg⁻¹·bar⁻¹ → ~0.91 @298 K
    vs atlas 0.959 → **Δlog ≈ +0.02.** Estimator note: atlas *absolute* Widom K_H vs Talu *excess* B/kT
    (small gap at these weak-binder conditions). Q_st is **reference-blocked** (Kr absent from Table 5).
    Kr is Talu–Myers' — **not** García-Pérez (who has no krypton).
  - **2b (HKUST-1): energetic parity.** Atlas reproduces Ongari's open-metal-site interaction
    energy to 0.1 kJ/mol; the source publishes no K_H/Q_st scalar.
  - **3b family (UiO-66): band parity.** Atlas Q_st inside Maia's own simulated 20–27 kJ/mol band.
  - **1c (Mg-MOF-74, Becker 2017): figure-read approximate parity** (digitised, reduced model).
  - The remaining branches publish **no comparable source scalar** (charge-only methods or
    isotherm-only sources). 4c/6c stand on the **experimental** reference, not source-paper parity.
- **1b (Mg-MOF-74, Dzubak): characterized & mitigated — a new failure class, not a force-field claim
  (confirmatory closure pending).** The atlas over-predicts Dzubak's *own* simulation (Δlog K_H ≈ +1.8)
  with byte-exact parameters and an identical guest model. A per-insertion audit of the locked recipe
  showed **92.5 % of the K_H weight comes from insertions below the physical −42.55 kJ/mol DFT well**,
  concentrated in the near-Mg 2.0–2.8 Å shell. Diagnosis: a **domain-of-validity / realization gap** —
  the published exp-repulsion + point-charge table is valid only above ~2.3 Å; the load-bearing
  *unpublished* ingredient is the short-range protection convention, which random Widom insertion
  violates and GCMC does not. A **form-agnostic per-insertion energy floor** (the same OOD diagnostic
  WP3 applies to MLIPs) collapses K_H **~8×** at the physical-well floor (robust across placement:
  1 642 / 3 560 / 7 365 mol·kg⁻¹·bar⁻¹ at floors −42.55 / −47 / −52) — **closing the gap from ~60× to ~4×**
  (floored 1 642 vs Dzubak's 379, Δlog ≈ +0.64; **never a match**). ("~60×" = locked K_H vs Dzubak;
  within-run collapse 35×→4.3×, single-run K_H 13 252 = 17× per-seed variance.) **Family-wide:** the same
  native scatter on **1a (Lin/Mercado) and 1d (Mercado-Model-4)** is numerically identical (85 % weight
  sub-physical-well; 1a's RASPA2 verdict is itself ~60× over experiment) — so this is a *failure class*
  across every exotic-form Mg-MOF-74 branch, not a 1b quirk. **1b is NOT evidence that the Dzubak force
  field over-binds.** **Still pending (deferred to the meeting):** attribute the residual ~4.3× (DFT-vs-
  experimental geometry, floor choice) and lock a floored re-verdict. It is the project's clearest
  demonstration that *governance exposes an incomplete-recipe realization gap a parameter check misses.*
- **Governance rule (v0.5, motivated by 1b):** verdict schemas should **gate on convergence diagnostics**
  (per-seed relative uncertainty + tail concentration) so an under-converged, heavy-tail branch (1b's
  K_H scattered 17× across seeds) **auto-flags as do-not-trust**. (Rule designed and documented; wiring
  it into the verdict-schema code is queued.)
- **Open-metal-site convergence is heavy-tailed.** Native CPU + a GPU gRASPA Widom sweep show the
  CO₂/Mg-MOF-74 Henry coefficient **climbs +56 % from 30k→3M insertions and is not converged at
  3M**, while a light binder (Si-CHA/Ar) converges by N ≈ 10⁴. *Consequence:* finite-N OMS Henry
  coefficients are **lower bounds**, so the OMS over-binding verdicts are conservative w.r.t.
  sampling, and the strict passes live in the converged light-tail regime. (The gRASPA run used
  its own example force field at 313 K — it demonstrates convergence behaviour, **not** a validated
  atlas scalar for Mg-MOF-74.)
- **MLIP-Widom can be dominated by OOD overlaps, and widom-atlas refuses it.** A real MACE-MP
  Widom run on Si-CHA + Ar flagged 79–83 % of insertions as OOD overlaps carrying **> 99 % of the
  Boltzmann weight**; removing them changes K_H ~32–35×. Verdict: **REFUSE** (reproduced on GPU at
  5× the sample size with a pinned checkpoint hash).

## The safest single claim
> **widom-atlas governs whether a Widom number is trustworthy** — it locks the recipe, records
> provenance, checks source-paper parity and convergence where possible, detects OOD insertions,
> and refuses Boltzmann averages dominated by unphysical events.

## What is NOT claimed
No general predictive power; no first-principles "force field over-binds" claim (1b is a recipe
realization-gap, not a force-field result);
no claim that the gRASPA result validates v0.4; no claim that MLIP-Widom is generally invalid; no
claim that `cusp-ai-oss/widom` is wrong; no productised CuspAI integration (the ASE seam is
implemented and exercised; a pilot is the next step).

## Status (one consistent picture)
- **v0.4 mainline:** frozen — no atlas-produced scalar altered.
- **`repair/v04-provenance` branch (Gate 2, complete):** controlled, provenance-only commits —
  `6c.json` Q_st reference 17.0→15.7 (authority: Talu–Myers verbatim "Q_st = 15.7 kJ/mol at zero
  pressure (Dunne)"); `becker_loader` citation 2018/122,27538 → 2017/121,4659; REPAIR_LOG. The 6b
  Q_st verdict-JSON needed no edit (positive convention confirmed; Kr reference-blocked). Awaiting
  review/merge.
- **v0.5 evidence tree:** complete — WP1 parity (matrix/CSV/figure), WP2/WP3, the 1b dossier
  (`1b/FINDINGS.md` + per-insertion audit + floor sweep + 1a/1d family scatter), 6b source vendored
  (Talu–Myers 2001) + restored to confirmed K_H parity, 4c Maghsoudi anchor verified in-repo.
- **1b:** characterized & mitigated as a domain-of-validity / realization-gap governance finding;
  **confirmatory closure (residual-gap attribution + locked floored re-verdict) deferred to the meeting.**
- **Remaining, correctly deferred:** the 1b closure above; the modern-MLIP discrimination demo (by the
  *meeting*, not the first email). The gRASPA-Blackwell note is independent and can go to snurr-group now.
