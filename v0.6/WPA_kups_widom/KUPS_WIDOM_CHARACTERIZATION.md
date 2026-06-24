# WPA-A1 — Characterization of kUPS's Widom estimator

**Date:** 2026-06-12. All facts read directly from kUPS source (`~/kups_src`, main HEAD) and verified
against a live run. Convention deltas vs the widom-atlas native / RASPA path are tabulated; **deltas
are findings, not failures.**

## Estimator definition (confirmed)
- **Ghost move / lnα.** `kups.mcmc.widom.widom_test()` runs the full propose→patch→log-ratio pipeline
  for an insertion and **discards the state patch** (state untouched). It returns the **raw** lnα
  (`move_lr + log_probability_ratio`). The MC log-probability ratio (`mcmc.probability`) is
  `log p_ratio = −ΔU/(k_BT) = (U_old − U_new)/(k_BT)`. For a Widom ghost insertion U_old = 0, so
  **lnα = −βΔU = −βU_insertion**, and **W ≡ exp(lnα) = exp(−βU)** — the standard Widom Boltzmann
  factor (no fugacity/N! terms enter the Widom path; those belong to the GCMC exchange move, not used
  here).
- **Reductions** (`application.mcmc.analysis`, canonical formulas, block-averaged with delta-method SEM):
  - μ_ex = −k_BT · ln⟨W⟩   **[eV]**
  - **K_H = V · ⟨W⟩ / (k_BT)**   **[Å³/eV]**   (V = supercell volume, ⟨W⟩ dimensionless)
  - q_st = k_BT − ⟨ΔU·W⟩/⟨W⟩ = k_BT − ⟨U·exp(−βU)⟩/⟨exp(−βU)⟩   **[eV]**
  These are **identical in physics** to widom-atlas (μ_ex = −RT ln⟨e^{−βU}⟩; Q_st = RT − ⟨U e^{−βU}⟩/⟨e^{−βU}⟩).
- **Live confirmation** (RUBTAK + CO₂, 30 cyc): μ_ex −0.145 eV, K_H 6.87×10⁸ Å³/eV, q_st 0.255 eV (24.6 kJ/mol).

## The clean cross-engine metric
Because units differ, the parity (A2) compares **⟨exp(−βU)⟩** directly, which every engine produces.
For kUPS: **⟨W⟩ = K_H[Å³/eV] · k_BT[eV] / V[Å³]**. (kUPS K_H → our mol·kg⁻¹·bar⁻¹ is a separate
documented conversion: K_H[mol/kg/Pa] = ⟨W⟩·V·N_A/(M_framework·k_BT) in SI; ×10⁵ for Pa→bar.)

## Convention table (kUPS vs widom-atlas native / RASPA)
| Aspect | kUPS | widom-atlas native / RASPA | Delta / handling |
|---|---|---|---|
| Widom W | exp(−βU), ghost discard | exp(−βU) | **none** (verified) |
| Energy units | eV | K (native) / K (RASPA) | convert eV↔K (×11604.5) for comparison |
| K_H units | Å³/eV | mol·kg⁻¹·bar⁻¹ | compare ⟨exp(−βU)⟩ instead; conversion documented |
| LJ mixing | Lorentz–Berthelot from per-element self-params (`trappe.yaml`) | recipe-dependent (explicit cross-pairs for Talu/Dzubak) | **map carefully**: set self-params so LB reproduces the locked cross-pair; if a recipe is explicit-cross-only, that is a convention delta (recorded) |
| LJ cutoff / tail | configurable `cutoff`, `tail_correction: true/false` | recipe cutoff + analytical tail | match cutoff; set tail to match (or document) |
| Electrostatics | Ewald (`real_cutoff`, `precision`) | Ewald (α, real cutoff, k_max) | match real cutoff; Ewald-convention details (tinfoil/self) read from `coulomb.py` per recipe; differences recorded |
| Orientation sampling (rigid) | rotation move in the ghost pipeline | uniform random rotation | confirm uniform; recorded in A2 |
| RNG / seed | `run.seed` (JAX PRNG, deterministic) | numpy default_rng(seed) | seeds pinned per run |
| **Short-range protection** | **blocking spheres (optional, per-species)** | 1.5 Å electrostatic hard-core (native); RASPA hard wall | **the load-bearing recipe element** — A3 studies its K_H sensitivity |

## Implications for A2 / A3
- **A2 metric:** ⟨exp(−βU)⟩, target ≤1 % vs native/RASPA3. If kUPS deviates, isolate the convention
  (LJ mixing, cutoff/tail, Ewald) deterministically — do **not** tune to match.
- **A3:** blocking spheres are kUPS's explicit short-range protection — the v0.5 1b lesson as their API.
  Sweep on/off + radius on an OMS recipe kUPS can represent (1c/2a) and tabulate K_H sensitivity.
