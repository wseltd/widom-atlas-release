# WP3 — MLIP-Widom Governance Demo (OOD Diagnostic)

**Date:** 2026-06-11 · **Compute:** CPU only (device 0/1 untouched) · **Status:** ran, real MLIP, verdict = **REFUSE**

This is the strongest single piece of CuspAI-facing evidence in v0.5: a **live
MLIP-Widom run** in which an off-the-shelf machine-learning interatomic
potential (MACE-MP) produces a Henry coefficient that is **dominated by
physically impossible insertions**, and the widom-atlas governance layer
**catches it and refuses the number**. Nothing here is simulated narrative — the
numbers below come straight from the run artifacts.

## 1. What was run

| Field | Value |
|---|---|
| System | Si-CHA (all-silica chabazite, IZA `CHA_iza.cif`) + Ar |
| Temperature | 298.15 K |
| Insertions | 100 (random, seed 0) |
| Cutoff | 6.0 Å, minimum-image (framework replicated to exceed 2×cutoff) |
| Classical baseline | UFF Lennard-Jones (Si, O, Ar), Lorentz-Berthelot mixing |
| MLIP | **MACE-MP `small`**, mace-torch 0.3.16 / torch 2.12.0+cpu, float64, CPU |
| MLIP checkpoint | `20231210mace128L0_energy_epoch249model` (sha256[:16] `2ddb079cee0e131e`) |
| Engine seam | same ASE-`Calculator` interface that `cusp-ai-oss/widom` consumes |

Both energy models scored **the same 100 random insertion points**. The
classical baseline and the MLIP differ *only* in the energy function, so any
divergence is attributable to the potential, not the sampling.

## 2. The OOD failure, made visible

Random Widom insertion deliberately samples atomic overlaps. A classical
potential returns a huge **positive** energy for an overlap, so its Boltzmann
weight `exp(-βU)` collapses to zero — harmless. An MLIP has **never seen** such
overlaps in training; MACE-MP returns a spurious **low or negative** energy
there. Because `K_H ∝ ⟨exp(-βU)⟩`, even a few spurious negatives are
exponentially amplified.

Concrete examples from `per_insertion.json` (this run):

| Insertion | nearest host–guest dist | classical U | MACE U |
|---|---|---|---|
| #20 | 1.94 Å | **+4 099 kJ/mol** (correctly repulsive) | **−8.8 kJ/mol** (spurious attraction) |
| #60 | 1.43 Å | **+209 110 kJ/mol** (hard wall) | **+2.5 kJ/mol** (no wall) |

Classical physics says "you cannot put an Ar atom 1.4 Å from a framework O";
MACE-MP says "that's fine, even slightly favourable." That is the
out-of-distribution failure in one line.

## 3. Governance verdict

Two transparent flags were applied (recorded in `governance_summary.json`):

- **Hard-overlap flag:** nearest host–guest distance `< 0.80 × σ_min = 2.762 Å`
  (σ_min from the O–Ar LB diameter). Pure geometry — no reference to the MLIP.
- **Energetic-anomaly flag:** MACE `U < −25 kJ/mol` (Ar physisorption is
  ≈ −10…−15 kJ/mol; below −25 is implausible) **or** MACE `U` more than
  50 kJ/mol below the classical `U` at the same point.

Result:

| Quantity | Value |
|---|---|
| Flagged insertions | **79 / 100** (79%) |
| **Flagged share of the Boltzmann weight** | **99.3 %** |
| K_H proxy `⟨exp(-βU)⟩`, all insertions | **55.47** |
| K_H proxy, flagged (OOD) insertions removed | **1.74** |
| Inflation from OOD insertions | **≈ 31.8 ×** |
| Concentration | **top 11 insertions = 90 %** of the average |
| **Verdict** | **REFUSE** — flagged OOD insertions dominate the Boltzmann average |

The decision rule is mechanical: *flagged-weight-fraction ≥ 0.5 → REFUSE*. Here
it is 0.993, so the MLIP-Widom K_H is rejected as untrustworthy. Removing the
flagged insertions changes the answer by **a factor of ~32**, which is itself
the proof that the raw number was governed by physics the model never learned.

## 4. Figure

`outputs/wp3_mlip_governance.{pdf,png}` — three panels:

- **A.** Insertion energy vs nearest host–guest distance. Classical LJ shoots to
  `+10⁶ kJ/mol` as the distance drops below ~3 Å; MACE-MP stays near zero or
  goes negative across the same overlap region (the OOD failure).
- **B.** Cumulative contribution to `⟨exp(-βU)⟩` ranked by weight — **11
  insertions supply 90 %** of the average, all of them flagged.
- **C.** Governed verdict card: locked system, MLIP identity + checkpoint hash,
  seed, N, flagged counts, K_H with/without OOD, and the **REFUSE** verdict.

## 5. What this does and does **not** claim

**Does:**
- Demonstrates a real MLIP (MACE-MP) running the Widom estimator and producing a
  number that is **silently wrong** without governance.
- Shows widom-atlas catching it by recipe-locked, model-agnostic diagnostics and
  **refusing**, rather than emitting a fine-looking K_H.
- Uses the **same ASE-calculator seam** `cusp-ai-oss/widom` exposes, so the
  diagnostic transfers directly onto a CuspAI MLIP-Widom pipeline.

**Does not:**
- Claim MLIP-Widom is invalid in general, or that MACE-MP is unfit for
  adsorption — only that **ungoverned** MLIP-Widom on machine-generated
  insertions is unsafe, which is exactly the high-throughput failure mode.
- Claim `cusp-ai-oss/widom` is wrong. Its built-in `min_interaction_energy =
  −1.25` floor is a coarse clip; WP3 adds geometry + anomaly flags **on top** and
  reports the flagged-weight fraction the floor alone does not surface.
- Touch the display GPU. The documented run is CPU MACE on 108-atom Si-CHA; the
  GPU reproduction below ran on **device 0 only** — device 1 (display) untouched.

## 5b. GPU reproduction (2026-06-11) — robust at 5× sample size
After the Blackwell GPU stack was enabled (torch 2.12.0+cu130, sm_120), the same
demo was re-run with MACE on **device 0** at **N = 500** (5× the CPU run), seed 0,
same checkpoint (`2ddb079cee0e131e`):

| Run | N | device | flagged | flagged Boltzmann wt | K_H(all) | K_H(OOD removed) | verdict |
|---|---|---|---|---|---|---|---|
| documented | 100 | cpu | 79 (79%) | 99.34 % | 55.47 | 1.743 | REFUSE |
| GPU | 500 | cuda | 413 (83%) | **99.50 %** | 59.88 | 1.719 | **REFUSE** |

The OOD phenomenon and verdict are **stable at 5× the sample size**; removing the
flagged insertions drops K_H ~35×. Wall-clock: **16.8 s on GPU vs ~3 min on CPU**
(~50× throughput). The display GPU stayed at its normal desktop load throughout.
Artifacts: `outputs/governance_summary_cuda.json`, `per_insertion_cuda.json`,
`logs/run_cuda.log` (the CPU artifacts are preserved unchanged).

## 6. Reproduce

```bash
# documented CPU run
.venv/bin/python v0.5/WP3_mlip_widom_governance_demo/scripts/run_mlip_widom_governed.py
# GPU run (device 0 only), scaled up
CUDA_VISIBLE_DEVICES=0 WP3_DEVICE=cuda WP3_N=500 \
  .venv/bin/python v0.5/WP3_mlip_widom_governance_demo/scripts/run_mlip_widom_governed.py
.venv/bin/python v0.5/WP3_mlip_widom_governance_demo/scripts/make_wp3_figures.py
```

Artifacts: `v0.5/WP3_mlip_widom_governance_demo/outputs/{governance_summary[_cuda].json,
per_insertion[_cuda].json, wp3_mlip_governance.pdf, wp3_mlip_governance.png}` and
`.../logs/run[_cuda].log`. Deterministic (seed 0); the MACE checkpoint hash is
recorded so the energy model is pinned. `WP3_DEVICE` (cpu|cuda) and `WP3_N` select
device and insertion count; CPU is the default so the documented run reproduces.
