# widom-atlas v0.5 — CuspAI demo-readiness implementation

**Status:** built and run 2026-06-11 · **Compute:** CPU / GPU device 0 only —
the display GPU (device 1) was never touched · **v0.4 is frozen:** nothing in this
tree modifies any v0.4 science file, case matrix, verdict JSON, scientific lock,
the archived v2, or the v3 CuspAI-readiness manuscript.

## What this tree is
Three work packages that strengthen widom-atlas for a CuspAI technical
conversation, each a self-contained, reproducible deliverable:

| WP | Question | Deliverable | Headline result |
|---|---|---|---|
| **WP1** | Does the atlas reproduce each **source paper's own simulated** K_H/Q_st, not just experiment? | `WP1_source_paper_parity/` | By *type*: **1 like-for-like scalar (6b)** + 1 energetic (2b) + 3 band (3b) + 1 figure-read (1c Becker, vendored); **1b UNRESOLVED** (electrostatic/clamp, mechanism characterised — `1b/FINDINGS.md`); 8 no-scalar |
| **WP2** | Do Widom estimates **converge** with insertion count, and where does that break? | `WP2_convergence_curves/` | Weak binders converge by N≈10⁴; **OMS K_H is heavy-tail-dominated** (top 2% of insertions = 92% of K_H) → needs production counts |
| **WP3** | Can an **MLIP-Widom** run be silently wrong, and does governance catch it? | `WP3_mlip_widom_governance_demo/` | Live MACE-MP run: **79/100 insertions OOD, 99.3% of the Boltzmann weight** → verdict **REFUSE** |

Plus `reports/ENVIRONMENT_VERIFICATION.md` (what is/isn't installed) and the
roll-up reports `reports/V05_IMPLEMENTATION_SUMMARY.md` and
`reports/CUSPAI_DEMO_READINESS.md`.

## ⚠️ This is NOT the four `feat/v05-*` branches
There are **two distinct things both called "v0.5"** in this repo — do not
conflate them:

1. **This `v0.5/` directory tree** (on branch `article/v3-author-written-manuscript`)
   — the WP1/WP2/WP3 CuspAI-readiness implementation. Documents, scripts, figures,
   and run artifacts. No `src/` changes.
2. **Four unmerged feature branches** `feat/v05-polarizable-thole-and-becker`,
   `feat/v05-consensus-isotherm`, `feat/v05-zeolite-cifs`, `feat/v05-mlip-ase-stub`
   — a separate earlier effort (polarizable force fields, consensus isotherms,
   zeolite CIF retrieval, an MLIP ASE **stub**), reviewed in `../V05_BRANCH_REVIEW.md`,
   **not merged**, pending an operator decision on the polarisation finding.

**Relationship:** WP3 here is the *realized* version of the old
`feat/v05-mlip-ase-stub` idea — that branch shipped an ASE-calculator **stub**;
WP3 runs a **real MACE-MP MLIP** through the same seam and governs the result.

## Reproduce
```bash
# WP3 — MLIP-Widom governance demo (CPU MACE, ~3 min incl. model download)
.venv/bin/python v0.5/WP3_mlip_widom_governance_demo/scripts/run_mlip_widom_governed.py
.venv/bin/python v0.5/WP3_mlip_widom_governance_demo/scripts/make_wp3_figures.py

# WP2 — native CPU convergence sweep + figure
CUDA_VISIBLE_DEVICES=0 .venv/bin/python v0.5/WP2_convergence_curves/scripts/run_native_convergence.py
.venv/bin/python v0.5/WP2_convergence_curves/scripts/make_wp2_figure.py

# WP1 — parity figure (data is checked in as parity_branches.csv)
.venv/bin/python v0.5/WP1_source_paper_parity/scripts/make_wp1_parity_figure.py
```
All runs are seed-0 deterministic. WP3 pins the MACE checkpoint by sha256.

## Environment constraints honoured
- **GPU device 1 (display) untouched.** All compute is CPU or device 0.
- **gRASPA GPU path blocked** (no CUDA toolkit / `nvcc`, no gRASPA) → WP2 uses the
  native CPU fallback; no GPU result is claimed.
- **No v0.4 science file modified.** WP1 *points at* the locked v0.4 verdict JSONs
  and the read-only v3 audit; it does not copy or mutate them.

## What is NOT claimed
- MLIP-Widom is **not** asserted valid in general; MACE-MP is **not** asserted fit
  for adsorption. WP3 shows ungoverned MLIP-Widom is unsafe and governance catches it.
- `cusp-ai-oss/widom` is **not** claimed wrong. WP2/WP3 use its real ASE seam.
- GPU acceleration was **not** part of v0.4 and is not claimed here.
- See each WP report's "Honest limits" section.
