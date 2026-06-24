# WPA-A2 — Cross-engine Widom parity (the first independent validation of kUPS's Widom)

**Date:** 2026-06-12. Device 0. kUPS run in `venv-kups` (editable from `~/kups_src`, JAX/CUDA);
native + analysis in `.venv`. Every number regenerates from `parity_6c_threeway.json`,
`native_6c*.json`, `cross_engine_parity.csv`, and the kUPS run on `configs/widom_6c_withSi.yaml`.
Per the brief: **deltas are findings, not failures.**

## System (6c, the positive control)
MFI (all-silica) + Ar, García-Pérez 2007 framework + Talu-Myers 2001 cross-pair. The exact locked
RASPA3 recipe (from `evidence/.../6c/.../force_field.json`):
- Ar–O **explicit binary** LJ **93.0 K / 3.335 Å** (Talu-Myers 2001 Table 3).
- Ar self **119.8 / 3.405**; O self **53.0 / 3.3**; Si self **22.0 / 2.3** (TraPPE-zeo, Bai 2013).
- **MixingRule Lorentz-Berthelot**, **TruncationMethod shifted**, **TailCorrections false**,
  CutOffVDW 12.0 Å, ChargeMethod None (Ar neutral). ⇒ the framework presents **both** an Ar–O term
  (explicit) **and** an Ar–Si term (LB-mixed: 51.34 K / 2.8525 Å).

## The metric
Units differ across engines (kUPS reports K_H in Å³/eV; native/RASPA3 in mol·kg⁻¹·bar⁻¹), so the
engine-agnostic comparison is **⟨exp(−βU)⟩**; K_H and Q_st are then reported in common units.
kUPS K_H[Å³/eV] → ⟨W⟩ via ⟨W⟩ = K_H·k_BT/V (k_BT = 0.025693 eV, V = 41 690 Å³ supercell).

## Result — three-way table (full recipe unless noted)
| Engine | ⟨W⟩ | K_H (mol·kg⁻¹·bar⁻¹) | Q_st (kJ/mol) |
|---|---|---|---|
| RASPA3 (locked, reference) | — | **0.2070** | **16.03** |
| native independent (Ar–O **+ Ar–Si**) | 9.54 | **0.2093** | 15.91 |
| kUPS Widom (Ar–O **+ Ar–Si**) | 11.08 | 0.2432 | 16.25 |
| native (Ar–O **only**, naive) | 5.24 | 0.1149 | 14.36 |
| kUPS (Ar–O **only**) | 5.98 | 0.1313 | — |

**Deltas:** native+Si vs RASPA3 → **K_H +1.1 %, Q_st −0.12 kJ/mol** (parity). kUPS+Si vs native →
**K_H +16.2 %, Q_st +0.22 kJ/mol vs RASPA3**. Ar–Si term inflates native K_H by **×1.82**.

## Two findings

### F1 — the Ar–Si term is a load-bearing recipe element (×1.82 on K_H)
Dropping the Si interaction — a *natural* simplification ("Si is buried, only the surface O atoms
matter") — under-predicts the Henry coefficient by **1.8×** (0.2093 → 0.1149) and Q_st by 1.55 kJ/mol.
The locked recipe includes Ar–Si via Lorentz-Berthelot from Si's TraPPE-zeo self-parameters; a port
that keeps only the *explicit* Ar–O binary silently loses it. With the full recipe the **independent
native estimator reproduces RASPA3 to 1.1 % on K_H and 0.12 kJ/mol on Q_st** — confirming both the
estimator and the recipe reconstruction. This is the v0.5 realization-gap thesis on the positive
control: *the published cross-pair table is not the complete recipe; the mixing rule applied to the
framework self-parameters is the load-bearing, easily-dropped ingredient.* (My own first native run
dropped it — caught and corrected; the −O-only row is kept as the documented failure mode.)

### F2 — kUPS's Widom validated; the +16 % K_H is a shifted-vs-truncated LJ convention (ROOT-CAUSED)
This is the headline of A2: the **first independent cross-engine validation of kUPS's brand-new,
externally-unvalidated Widom** (source main HEAD; unreleased — see `../BRIEF_ERRATA.md`). The estimator
is correct, and the one cross-engine discrepancy is now fully explained.

- **Estimator is correct.** W = exp(−βU); the μ_ex / K_H / q_st reductions match widom-atlas (A1).
- **The +16 % K_H is the LJ truncation convention — root-caused, not a normalization bug.** kUPS's
  `lennard_jones_energy` (`kups.potential.classical.lennard_jones`) is **truncated, not energy-shifted**
  (`4ε(c6²−c6)` masked at `r < cutoff`, no `−U(rc)` term); our config (`tail_correction: false`, no
  `truncation_radius`) selects exactly that plain variant. RASPA3's recipe is **`TruncationMethod:
  shifted`**, and the native estimator shifts too. The unshifted potential is **more attractive by
  U(rc) per interaction** — a near-constant per-insertion offset. Toggling the shift off in native
  confirms it:

  | native (with-Si) | ⟨W⟩ | K_H (mol·kg⁻¹·bar⁻¹) | q_st (kJ/mol) |
  |---|---|---|---|
  | **shifted** (RASPA3 convention) | 9.54 | 0.2093 | 15.91 |
  | **truncated / no-shift** (kUPS convention) | 11.35 | 0.2490 | 16.34 |
  | kUPS Widom | 11.08 | 0.2432 | 16.25 |

  The implied constant shift is **δ = RT·ln(11.35/9.54) = 0.43 kJ/mol** (predicted ≈0.35). It accounts
  for **both** observables: matching kUPS's truncated convention, native-no-shift agrees with kUPS to
  **+2.4 % on K_H (within the 3.4 % kUPS SEM) and 0.09 kJ/mol on q_st**. The earlier "K_H normalization"
  diagnosis is **superseded** (`F2_rootcause_shift.json`).
- **Reading.** Neither engine is wrong; truncated-not-shifted is a legitimate convention. But the
  **shift convention is a load-bearing recipe element worth ~16 % on K_H** — it must be locked and
  matched, exactly the realization-gap thesis. The actionable upstream point is simply: *kUPS's default
  LJ is unshifted; recipes transferred from shifted-convention engines (RASPA) need the shift matched
  or the ~16 % difference declared.* Does **not** affect WPB (energy path).

## Convergence / honesty
- kUPS: 50 cycles × 500 ghost insertions = 25 000 Widom samples, 12 blocks, K_H SEM ≈ 3.4 %
  (`henry_coefficient.sem/mean`). The +16 % delta is ~5× the SEM — a real systematic, not noise.
- native: 1–2×10⁶ uniform insertions, seed 0; ⟨W⟩ converged to <0.5 %.
- Both shifted-truncated at 12 Å, no tail (matching RASPA3). A separate test added the analytical LJ
  tail to native and moved K_H only +8 % — confirming the recipe's `TailCorrections: false` and ruling
  out tail as the kUPS gap driver. The Ar–Si term, not tail, was the 1.8× O-only gap.

## F3 — the electrostatic leg (4c, Si-CHA + CO₂, Ewald): q_st validated, K_H residual open
The 4c positive control exercises the full electrostatic path: TraPPE-CO₂ (qC +0.7, qO −0.35) in
Si-CHA with framework charges (Si +2.05, O −1.025), Lorentz-Berthelot, 14 Å cutoff + LJ tail, Ewald.
Wired into kUPS (charged P1 CIF read via `store_tags=True` → `_atom_site_charge`; kUPS's built-in
TraPPE-CO₂ adsorbate; Ewald auto-enabled on `state.is_charged`):

**q_st is validated to 0.01 kJ/mol** (kUPS 22.98 vs native 22.99) — the electrostatic energies are
right. **K_H needs the convention matrix to read honestly** (`native_4c_convention_matrix.json`, a fresh
same-session native re-run — the 2.2227 above is the v0.4 *lock*, executed 2026-06-01, **not** re-run;
the fresh native(shifted+tail) gives 2.215, reproducing the lock to 0.3 %):

| native K_H (mol·kg⁻¹·bar⁻¹), fresh | tail OFF | tail ON |
|---|---|---|
| **shifted** (RASPA/native) | 1.981 | **2.215** |
| **truncated** (kUPS's convention) | 2.210 | **2.472** |
| one-toggle | shift = **+11.6 %**, tail = **+11.9 %** | |

kUPS = **2.271**. **The +2.2 % vs the shifted lock is a *cancellation*, not clean agreement:** kUPS uses
a **truncated** LJ, so the +11.6 % shift convention pushes it above native(shifted), while a separate
**−8.2 % residual** pushes it back down. Read on the **same** convention, **kUPS is −8.2 % vs
native(truncated+tail)** (2.271 vs 2.472). (Correction to an earlier draft: the shift is **not**
negligible at 14 Å — it is +11.6 %, comparable to 6c's +16 %.)

So the honest 4c result: **q_st matches to 0.01 kJ/mol** (energies validated), and **K_H carries a
−8.1 % matched-convention residual** (truncated+tail; −11.7 % with the tail matched off too) — larger
than 6c's −2.4 %, a genuinely 4c-specific gap.

**Does kUPS apply a tail in the Widom path? — tested, YES** (`native_4c_convention_matrix.json →
kups_tail_test`). Re-running kUPS 4c with `tail_correction: false` drops K_H 7.87→6.76×10⁶ Å³/eV
(−16 %) and q_st 22.98→22.49 — so kUPS **does** apply an LJ tail to the insertion energy (a "no tail in
the Widom path" hypothesis is *falsified*), and the correct matching cell is therefore truncated **+tail**.
kUPS's tail (+16.3 %) is ~4 % *stronger* than native's (+11.9 %) — one identified contributor to the
residual; the rest is the **Ewald convention**, **CO₂ orientational sampling**, and the K_H
**normalization**. Crucially the residual is **q_st-invariant** (Δq_st ≤ 0.3 kJ/mol on matched
convention) → it is a **normalization-flavoured multiplicative scale on ⟨W⟩** (volume / insertion-count),
now cleanly isolated *after* convention-matching — the same character as 6c's −2.4 %, just larger.
μ_ex (−0.1139 eV) is the convention-independent ⟨W⟩ cross-check (kUPS reports K_H per *primitive-cell*
volume). The Ewald path runs and the energies are right; the q_st-invariant K_H residual is the open 4c
question.

## Deferred (honest scope)
- **gRASPA** column: not included; the locked reference is RASPA3/native. gRASPA high-N convergence is a
  v0.6 infrastructure item, not required for the parity conclusion.
