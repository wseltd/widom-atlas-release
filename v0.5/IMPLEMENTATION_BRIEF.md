# v0.5 Implementation Brief (as executed)

**Recorded:** 2026-06-11. This is the working brief the `v0.5/` tree implements,
captured here so the deliverables are self-describing. It is the WP1/WP2/WP3
CuspAI-readiness scope — **distinct** from the four `feat/v05-*` polarizable/
consensus/zeolite/mlip-stub branches (`../V05_BRANCH_REVIEW.md`).

## Goal
Make widom-atlas ready for a technical conversation with CuspAI by closing the
three gaps a skeptical CuspAI reviewer is most likely to probe, **without
re-opening v0.4 science**:
1. "Did you reproduce the **source paper's own** number, or only experiment?" → **WP1**
2. "Are the strongly-binding failures just **under-sampling**?" → **WP2**
3. "Does your governance actually catch a **wrong MLIP-Widom** number?" → **WP3**

## Hard constraints (binding)
- **Do not touch v0.4** science files, the case matrix, verdict JSONs, scientific
  locks, the archived v2, or the v3 CuspAI-readiness manuscript. Everything lives
  under a new `v0.5/` tree.
- **GPU device 0 only.** The display GPU (device 1) must not be touched. No
  multi-GPU, no `cuda:1`, no `CUDA_VISIBLE_DEVICES=0,1`.
- **Verify the environment; do not assume.** Do not assume CUDA 12.8, sm_120,
  gRASPA support, MACE support, or CuspAI widom API details are correct until the
  local environment confirms them. Record corrections.
- **Do not fake any demo.** A run is only "ran" if it actually ran; pin models and
  seeds; report what failed.

## Work packages

### WP0 — Environment verification (gate)
Probe the real CPU/GPU stack: Python, RASPA2/3, native evaluator, ase/numpy,
CUDA toolkit/`nvcc`, gRASPA, PyTorch/MACE, `cusp-ai-oss/widom`. Flag every
reviewer-supplied assumption that is wrong. → `reports/ENVIRONMENT_VERIFICATION.md`.

### WP1 — Source-paper simulator parity (first deliverable / priority)
For every verdict-affecting branch, determine whether the *originating paper's own
simulated* K_H/Q_st can be reconstructed from local evidence, and if so compare it
to the atlas value. Separate "force-field over/under-binds" (parity confirmed) from
"recipe drift / unresolved" (parity discrepancy or source unavailable). Produce a
machine-readable per-branch record, a matrix, a report, and a parity figure.

### WP2 — Convergence vs insertion count
Show whether K_H/Q_st converge with insertion count and where that breaks. Intended
gRASPA-GPU high-count sweep; **CPU/native fallback if the GPU path is unavailable**.
Quantify the open-metal-site heavy-tail under-sampling risk. Produce convergence
curves and a report. No GPU result may be claimed if no GPU path exists.

### WP3 — MLIP-Widom governance demo (the headline)
Run a **real MLIP** (MACE-MP via the same ASE-calculator seam `cusp-ai-oss/widom`
consumes) through Widom insertion on a small framework, against a classical
baseline. Build an **out-of-distribution diagnostic**: random insertion samples
atomic overlaps where an MLIP can hallucinate attractive energies that dominate
`⟨exp(-βU)⟩`. Flag OOD insertions (geometry + energetic anomaly), report K_H with
and without them, and emit a governed verdict (REFUSE if flagged insertions
dominate the Boltzmann weight). Produce visuals and a report.

## Non-claims (must hold in every deliverable)
- No claim that MLIP-Widom is valid in general, or that MACE-MP is suitable for
  adsorption prediction.
- No claim that CuspAI widom is wrong.
- No claim that CuspAI integration "already exists" unless it actually runs.
- No claim that GPU acceleration was part of v0.4.

## Definition of done
Three WP reports + environment verification + a v0.5 implementation summary + a
CuspAI demo-readiness summary, each with an explicit "honest limits" section, all
under `v0.5/`, all reproducible, with the display GPU untouched.
