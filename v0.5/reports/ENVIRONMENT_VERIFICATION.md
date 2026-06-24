# v0.5 Environment Verification

**Date:** 2026-06-11. All checks run locally on this workstation, **device 0
only** (the display GPU, device 1, was not touched). The brief's placeholder stack
versions were superseded by the actual environment; differences are noted neutrally below.

> **UPDATE — 2026-06-11 (later same day): the GPU stack below was subsequently
> ENABLED and verified.** Every "absent/blocked" GPU item is now installed and
> running on device 0: PyTorch **2.12.0+cu130** (sm_120, real matmul + GPU MACE),
> CUDA toolkit **13.0.2** (`nvcc`, sm_120 smoke kernel), NVHPC SDK **26.3**
> (`nvc++`), and **gRASPA built + running a full GCMC on device 0** (Ar/Mg-MOF-74,
> Q_st −8.19 kJ/mol, ~1 s). Display GPU stayed untouched. See
> `../WP2_convergence_curves/gpu_graspa/GPU_GRASPA_ENABLEMENT.md`. The sections
> below record the **original as-of-check** state (before enablement).

## Headline
The **CPU stack is fully present** (RASPA2, RASPA3, native evaluator). The
**GPU compute stack the brief assumed is largely absent**: no CUDA toolkit
(`nvcc`), no gRASPA, and (at check time) no PyTorch/MACE. `cusp-ai-oss/widom` is
installed and verified working. **This — not GPU count — is the binding
constraint:** WP1 (CPU) is fully runnable; WP2's gRASPA-GPU path is **blocked**
(no compiler, no gRASPA) with a CPU/native fallback; WP3's MLIP engine is being
brought up on **CPU MACE** (the GPU/Blackwell PyTorch build is the open
question, not the GPU count).

## CPU stack
| Item | Status |
|---|---|
| Python | **3.13.9** — (brief placeholder 3.12 superseded) |
| RASPA2 (`simulate`) | ✅ `~/miniconda3/envs/raspa2/bin/simulate` |
| RASPA3 | ✅ `~/miniconda3/envs/raspa3/bin/raspa3` (v3.0.29 per v0.4 records) |
| native widom-atlas evaluator | ✅ `src/widom_atlas/evaluator/runner.py:run_widom_evaluator` |
| ase | ✅ (3.28.0) |
| numpy / pandas / scipy / matplotlib | ✅ (2.4.4 / 3.0.2 / 1.17.1 / 3.10.9) |

## GPU stack (device 0)
| Item | Status |
|---|---|
| GPU model | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (×2; device 1 is the live display GPU) |
| NVIDIA driver | **580.126.09** |
| CUDA (driver-reported) | **13.0** — the actual driver reports CUDA 13.0 (brief placeholder 12.8 superseded); no CUDA toolkit installed |
| CUDA toolkit / `nvcc` | ❌ **not installed** (`nvcc` not on PATH) → cannot compile CUDA code locally |
| compute capability | Blackwell → expected **sm_120**; **could not compile-verify** without `nvcc`/a CUDA-enabled torch |
| gRASPA | ❌ **not installed**, and cannot be built locally (no `nvcc`) — the brief assumed gRASPA buildable; the actual build path is documented in Section 8/9 |
| PyTorch sees GPU | ⏳ installing (see below); a Blackwell sm_120 + CUDA 13 GPU build is the open question |
| MACE / mace-torch | ⏳ installing CPU build (`from mace.calculators import mace_mp`) |
| **cusp-ai-oss/widom** | ✅ **installed (0.1.1) and verified working** — already ran a real convergence sweep through it (`samples_origin: cuspai_widom`) |

### cusp-ai-oss/widom API (verified)
```
run_widom_insertion(calculator: ase Calculator, structure: ase Atoms, gas: str,
  temperature: float, model_outputs_interaction_energy: bool,
  num_insertions=10000, optimize_structures=False, cutoff_distance=1.0,
  cutoff_to_com=False, min_interplanar_distance=6.0, random_seed=0,
  min_interaction_energy=-1.25) -> WidomInsertionResults
```
- **ASE-calculator-compatible:** yes (first arg is an `ase` `Calculator`).
- `WidomInsertionResults` exposes **per-insertion energies** (`interaction_energies`, `samples`), `is_valid`, `is_accessible` — so the **WP3 OOD insertion-energy diagnostic is implementable** against the real API.
- Note: the driver already carries a **`min_interaction_energy = -1.25`** floor parameter — a coarse built-in clip; the WP3 diagnostic adds the geometry/anomaly flags on top and reports flagged-weight fractions.

## Consequences for the three work packages
- **WP1 (CPU parity):** ✅ runnable (RASPA2/RASPA3/native). The binding limit is **source data**, not compute: most source papers report no simulated K_H/Q_st scalar (see the parity matrix).
- **WP2 (convergence):** GPU/gRASPA path **BLOCKED** (no `nvcc`, no gRASPA). **Fallback:** native / CuspAI-widom CPU convergence at reduced insertion counts (10⁴–10⁶ feasible; 10⁷–10⁸ not without gRASPA). No GPU result will be claimed.
- **WP3 (MLIP-Widom):** `cusp-ai-oss/widom` works; the MLIP energy engine is being brought up on **CPU MACE** (a real MLIP demo, GPU-acceleration deferred). If the CPU MACE install succeeds, the OOD diagnostic runs against a real MLIP on small systems.

## Two-GPU question (recorded)
**Not needed.** WP1 is CPU. WP2 (gRASPA Widom) and WP3 (MACE single-points) are
each single-GPU and memory-light (≪96 GB) when a GPU path exists at all; two
GPUs would only add throughput. Device 0 (95 GB free) is sufficient; the display
GPU (device 1) was left untouched. The real blocker is the missing CUDA
toolkit / gRASPA / Blackwell-torch build, not the number of cards.
