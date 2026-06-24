# kUPS API study + reality-vs-brief corrections (v0.6)

**Date:** 2026-06-12. Source studied: `github.com/cusp-ai-oss/kups` (shallow clone, `~/kups_src`).
All facts below are read directly from the kUPS source — corrections to the brief's April-2026
README assumptions are flagged **[BRIEF SUPERSEDED]**, exactly as v0.5 flagged reviewer-supplied
facts. These corrections *strengthen* the governance thesis; they are not failures.

## Reality corrections (the repo evolved past the brief)

1. **[BRIEF SUPERSEDED] kUPS now HAS Widom test-particle insertion.** The brief said "kUPS has NO
   Widom." The repo ships a full implementation:
   - `kups.mcmc.widom`: `widom_test(key, state, propose_fn, patch_fn, log_probability_ratio_fn)` →
     per-system raw lnα; `GhostProbe` propagator wrapper; `WidomStatistics` accumulator.
   - `kups.application.simulations.mcmc_widom` + `examples/mcmc_widom.yaml` (a runnable Widom config:
     RUBTAK Zr-MOF + CO₂, 1 bar / 298 K).
   - `kups.application.mcmc.analysis.analyze_mcmc` reduces to **μ_ex (eV), K_H (Å³/eV), q_st (eV)** with
     block-averaged SEM; formulas `μ_ex = −kT ln⟨W⟩`, `q_st = kT − ⟨ΔU·W⟩/⟨W⟩` — **identical physics to
     widom-atlas.**
   - **Consequence for WPA:** the contribution is no longer "build the missing Widom." It becomes the
     stronger, more honest artifact: **validate kUPS's *own* Widom in the cross-engine parity (native /
     RASPA3 / gRASPA / kUPS) and govern it.** A standalone batched estimator on kUPS primitives is kept
     only as a fallback if their `mcmc_widom` cannot be driven with our exact recipes.

2. **[BRIEF SUPERSEDED] kUPS has built-in short-range protection — "blocking spheres."**
   `kups.potential.classical.blocking`: "hard-sphere repulsion using blocking spheres that create
   infinite energy barriers … preventing particle overlap with framework atoms in porous materials
   (zeolites, MOFs)." Configurable per-adsorbate-species via `blocking_spheres:` in the host YAML.
   - **This is exactly the 1b realization-gap lesson, as a kUPS feature.** It is the *unpublished,
     load-bearing short-range protection convention* — but in kUPS it is **optional and user-specified**.
     A user porting a MOF/zeolite force field into kUPS Widom **without** configuring blocking spheres
     reproduces our 1b over-binding. **Governance value:** widom-atlas should flag (a) whether blocking
     spheres are set for the recipe, and (b) the OOD/sub-physical-well Boltzmann-weight fraction — the
     exact diagnostic from v0.5 WP3 + the 1b audit. This is the cleanest possible bridge from our work
     to kUPS.

3. **Capabilities confirmed (match brief):** MC (NVT + GCMC: translation/rotation/reinsertion/exchange),
   MD (NVE/NVT/NPT), relaxation (FIRE/L-BFGS). Potentials: Lennard-Jones, Coulomb (Ewald), harmonic,
   Morse, MACE, UMA. No Buckingham / exotic forms (our 1b exotic-form lesson applies to porters). JAX,
   composable propagators, batched/vectorised, differentiable, CPU/GPU/TPU. Apache-2.0, Python 3.10+.

## Config schema (from `examples/mcmc_widom.yaml`)
```yaml
adsorbates: [ !import ["adsorbate/co2.yaml"] ]   # positions, symbols, charges (TraPPE CO2)
hosts:
  - cif_file: host/RUBTAK.cif
    pressure: 100_000        # Pa
    temperature: 298.15      # K
    init_adsorbates: [0]
    cell_replication: 3
    # blocking_spheres: [ - [ {center:[..], radius:..} ] ]   # OPTIONAL short-range protection
lj:    !import ["lennard_jones/trappe.yaml"]   # per-element [sigma_Å, epsilon_eV], Lorentz-Berthelot
ewald: { real_cutoff: 12.0, precision: 1.e-6 }
run:   { num_cycles, num_warmup_cycles, num_displacements_per_cycle, num_widom_per_cycle,
         translation_prob, rotation_prob, reinsertion_prob, seed }
```
- **LJ units:** `[sigma_Å, epsilon_eV]` (UFF Ar = [3.446, 0.00802 eV ≈ 93 K] confirms eV). Mixing
  Lorentz–Berthelot; cutoff 12 Å; tail correction available.
- **Adsorbate CO₂:** symbols `C_co2`,`O_co2`,`O_co2`; charges 0.7/−0.35/−0.35; C–O = 1.16 Å (TraPPE).
- **K_H unit (Å³/eV)** must be mapped to widom-atlas `mol·kg⁻¹·bar⁻¹` for parity — the conversion is a
  documented governance step (units convention), not a discrepancy.

## Plan implications
- **WPA:** drive `mcmc_widom` for the locked **6c (MFI + Ar, LJ-only — clean LJ/Widom test)** and **4c
  (Si-CHA + CO₂, TraPPE + Ewald — exercises Ewald)** recipes; convert K_H to our units; tabulate the
  four-engine parity (native / RASPA3 / gRASPA / kUPS). Any Ewald-convention difference (tinfoil/self/
  units) is documented as a governance finding.
- **WPB:** kUPS ships MACE (and UMA) — the OOD demo runs on *their* shipped MLIP inside *their* engine;
  the blocking-spheres finding frames why the refusal/discrimination layer matters even with their
  protection available-but-optional.
