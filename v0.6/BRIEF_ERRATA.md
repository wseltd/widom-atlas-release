# v0.6 Brief — errata (reality vs the April-2026 README)

Logged per the amendment. These are corrections the *repo* makes to the *brief's premises*; they
strengthen the thesis and are recorded for the audit trail (as v0.5 logged reviewer-supplied facts).

## E1 — "kUPS has NO Widom test-particle insertion" — FALSE (with nuance)
- The kUPS **source main HEAD** ships a full Widom implementation: `kups.mcmc.widom`
  (`widom_test()`, `GhostProbe`, `WidomStatistics`), `kups.application.simulations.mcmc_widom`,
  `examples/mcmc_widom.yaml`, and `analyze_mcmc` reducing to μ_ex / K_H / q_st.
- **Nuance:** the *released* PyPI package **kups 1.0.1** (what `pip install kups` gives) does **not**
  include `mcmc_widom` (no `kups_mcmc_widom` console script; module absent). The Widom code is
  **unreleased** — present in source main, zero releases, ~six weeks old, no published external
  validation. So the brief's premise was correct *for the release* and outdated *for the source*.
- **Action taken:** installed kUPS **editable from source** (`pip install -e ~/kups_src`) into
  `venv-kups` to access their actual Widom for independent validation.
- **Consequence:** WPA pivots from "build the missing Widom" (off-thesis duplicate) to **"first
  independent cross-engine validation + governance of kUPS's own brand-new Widom"** (on-thesis):
  *kUPS shipped a Widom estimator; this is, to our knowledge, its first independent cross-engine
  validation.*

## E2 — kUPS README feature list OMITS Widom — documentation gap (noted, not actioned)
- The README lists Monte Carlo as "NVT and GCMC ensembles" only; it does not mention the Widom module
  that exists in the code. This is the source of the brief's error. **Filed as a fact; a one-line docs
  note is prepared in `WPA_kups_widom/upstream_candidate/` but NOT submitted** (Onur decides).

## E3 — kUPS has built-in short-range protection — "blocking spheres" (the 1b thesis, as their API)
- `kups.potential.classical.blocking`: hard-sphere infinite-barrier exclusion to prevent overlap with
  framework atoms in porous materials; configurable per-adsorbate via `blocking_spheres:` in the host
  config. **This is the v0.5 1b lesson — "the load-bearing unpublished ingredient is the short-range
  protection convention" — made an explicit, configurable feature by CuspAI's own engineering.** It is
  **optional and per-species**, so a recipe ported without it reproduces our 1b over-binding. Blocking
  configuration is therefore demonstrably *part of the recipe and must be locked* — no longer our
  claim, their API. Drives WPA-A3 (blocking-sphere governance study) and WPB (does geometric blocking
  catch an *energetic* MLIP hallucination? hypothesis: no).

## E4 — Representability (recorded for WPA-A3 scope)
- kUPS has **no Buckingham / exotic functional forms** (LJ, Coulomb/Ewald, harmonic, Morse, MACE, UMA).
  So the v0.5 exotic-form branches **1a (Lin Buckingham), 1b (Dzubak A·exp−C/r⁵−D/r⁶), 1d (Mercado
  Buckingham)** are **NOT representable in kUPS** — recorded as a representability finding. The
  blocking-sphere governance study (A3) therefore uses OMS recipes kUPS *can* represent: **1c (Becker
  reduced LJ + fixed charges)** and **2a (UFF Cu LJ + EPM2/Nazarian charges)**.
