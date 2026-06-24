# v0.6 WPB verdict schema — v2

**Status:** authoritative rules file. The WPB scripts cite it (`verdict_schema` field in every summary
JSON) and implement exactly these rules. The refusal classes (Class I / II + exponent-cap) are
**pre-registered** — defined before any result, independent of which model triggers them. **The Gate-1
reference-consistency rule and the WITHHELD verdict are NOT pre-registered:** they were added in v2
*after* the UMA odac/Ar run exposed the need (see Changelog). This file is honest about that provenance
so a skeptic can tell pre-registered rules from post-hoc ones.

## Changelog
- **v1 (2026-06-12, before the WPB runs)** — pre-registered: per-insertion flags (hard-overlap,
  energetic-anomaly, exponent-cap auto-flag) and the two refusal classes (I: OOD over-binding, flagged
  weight ≥ 0.5; II: no-physisorption, min U > 0 or q_st ≤ RT where physisorption is established).
  Applied to MACE-MP-small, MACE-MPA-0 (bare/+D3).
- **v2 (2026-06-12, after the UMA odac/Ar run)** — **added post-hoc** in response to a real finding:
  the **Gate-1 reference-consistency rule** and the **WITHHELD verdict** (below). UMA's odac head gave
  2.73 kJ between two symmetry-equivalent open sites (a per-graph-conditioned model used out of its Ar
  domain), so a same-graph measurement could not be trusted. The rule generalizes that catch. Recorded
  here as a v2 addition — *not* something that pre-dated the result.
- Also v2: clarified GOVERNED PASS = "screen-pass; accuracy unassessed" and added the screen-q_st
  diagnostics + the odac/CO₂ accuracy note (q_st 17.7 vs anchor 21.0).

## Inputs (per run)
- N seeded random insertions (state N in every summary; default N=120, seed 0).
- For each insertion i: nearest host–guest distance `min_dist_A`, classical UFF energy `U_classical_kJ`,
  MLIP energy `U_mace_kJ` (= E(host+guest) − E(host) − E(guest), differencing verified by Gate 1),
  Boltzmann weight `w = exp(−U_mace / RT)` with **RT in kJ/mol** (RT = R·T = 2.478 kJ/mol at 298.15 K).
- `flagged_weight_fraction` = Σ_flagged w / Σ_all w.

## Per-insertion flags (OOD markers)
1. **hard-overlap:** `min_dist_A < 0.80·σ_min` (σ_min = ½(σ_O+σ_Ar) = 3.4525 Å ⇒ 2.762 Å).
2. **energetic-anomaly:** `U_mace < −25 kJ/mol` OR `U_mace < U_classical − 50 kJ/mol`.
3. **exponent-capped (auto-flag):** the weight required `exp(−U_mace/RT) > exp(700)` (i.e. a
   hallucination so deep the float overflows). **Needing the cap is itself a verdict** — auto-OOD.

An insertion is `flagged` if any of (1)(2)(3) fire.

## Refusal classes (checked in this order)
- **Class II — no-physisorption / under-binding ⇒ REFUSE.**
  Trigger: `min(U_mace) > 0` over the N insertions **OR** computed `q_st ≤ RT`, **and** the guest/host
  pair has *established physisorption* (reference well exists). For Ar/all-silica the anchor is the
  6c locked value **q_st = 15.7 kJ/mol ≫ RT = 2.478 kJ/mol** (RASPA3-locked), so a real well is known
  to exist; a model that finds none is refused. (Guard: this class does **not** fire for a genuinely
  non-adsorbing pair, by the established-physisorption precondition.)
- **Class I — OOD over-binding ⇒ REFUSE.**
  Trigger: `flagged_weight_fraction ≥ 0.5` — the flagged (overlap / anomalous / capped) insertions
  carry at least half the partition weight, so the Henry estimate is built from unphysical states.
- **GOVERNED PASS WITH FLAGS = "screen-pass; accuracy unassessed".** Flags present but neither refusal
  class triggers (a physical well exists, `min U < 0`, flagged insertions do **not** dominate the
  weight). **This certifies only that the configuration survives the OOD/physicality screen — it does
  NOT assess thermodynamic accuracy.** A screen-pass model may still under- or over-estimate the true
  K_H / q_st; validating that requires a converged run against experiment or DFT, which this screen does
  not perform.
- **GOVERNED PASS.** No flags (still screen-only, accuracy unassessed).

## Gate-1 reference-consistency rule + WITHHELD verdict (v2, post-hoc — see Changelog)
For conditioned multi-task potentials (e.g. UMA) the insertion energy is measured by a **same-graph
reference**, `U(r) = E(host+guest @ r) − E(host+guest @ open-pore ref)`. That reference must be
trustworthy, so **before any verdict** the harness runs Gate 1:
- **Reference-consistency check.** Compute the energy of **two open-pore points at the same
  nearest-neighbour distance (symmetry-equivalent sites)**. They must agree: `|ΔU| < 1.0 kJ/mol`
  (well below RT = 2.48; above numerical noise).
- **WITHHELD** (a verdict distinct from REFUSE/PASS). If Gate 1 fails (`|ΔU| ≥ 1.0`), the same-graph
  zero-point is unreliable, so **every** insertion energy inherits that error → the harness **declines
  to issue a verdict**. WITHHELD is a statement about *measurement validity for this engine*, not about
  the model's quality. It must never be reported as REFUSE or PASS.
- For non-conditioned models (MACE) the analogous Gate 1 is the far-field offset check (isolated guest
  U → 0); same principle — validate the harness before judging the model.

**Equivalence certificate (the control that licenses the rule).** On a model that *is* self-consistent,
two symmetry-equivalent sites agree to within noise: **UMA omat on Ar/CHA gave ΔU = −0.19 kJ/mol**
(`wpb_uma_omat.json`) — and UMA odac on CO₂/CHA gave +0.06. These establish that equivalent sites
*should* agree, so the **odac/Ar failure (−2.73 kJ) is a genuine inconsistency**, not an artifact of the
check. The certificate is what makes the WITHHELD verdict defensible rather than arbitrary.

**Screen q_st (N=120 random insertions — diagnostic, not converged):** MPA-0 bare −8.5 (negative ⇒
under-binding, corroborates Class II); MPA-0 +D3(BJ) **+5.1** (positive, > RT ⇒ no Class II, a real but
modest signal); MP-small 18.3 and UMA-omat 295 are *dominated by their single deepest overlap
hallucination* and are therefore unphysical artifacts, not heats of adsorption — itself a Class-I tell.

## Provenance requirements (every run)
- State **N**, seed, T, CIF, and the model.
- Pin `mace_checkpoint_sha16` and `kups_mace_jax_zip_sha16` (the HF JAX export revision).
- Record `dispersion_D3` (bool) and, if true, the D3 variant: **D3(BJ)**, xc=PBE, cutoff 40 Bohr
  (mace_mp default via torch-dftd `SumCalculator`).

## Applied to the WPB triptych (result, generated *after* the rules above)
| Model | min U (kJ/mol) | flagged-wt | capped | Class fired | Verdict |
|---|---|---|---|---|---|
| MACE-MP-small (2023) | −19.3 | 0.995 | 0 | I | REFUSE (OOD over-binding) |
| MACE-MPA-0 bare (kUPS) | +8.51 | 0.0 | 0 | II | REFUSE (no physisorption) |
| MACE-MPA-0 + D3(BJ) | −5.03 | 0.0001 | 0 | none | GOVERNED PASS WITH FLAGS |
| UMA uma-s-1.1, task=omat | −292.8 | 1.0 | 0 | I | REFUSE (OOD over-binding) |

UMA note: verdict is computed **only after** Gate 1 passes for the periodic same-graph protocol
(`gate1_second_open_U_kJ = −0.19`, |·| < 1.0). A failed Gate 1 (e.g. the naive 3-graph differencing,
or the non-periodic cluster) yields a **WITHHELD** verdict, not a refusal — the harness must prove
itself valid for an engine before it is allowed to judge that engine.
