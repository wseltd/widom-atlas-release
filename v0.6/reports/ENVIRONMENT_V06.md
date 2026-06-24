# v0.6 Environment (kUPS retarget)

**Date:** 2026-06-12. **Isolation:** ALL kUPS/JAX work runs in a **separate venv**
`~/venvs/venv-kups` — it never touches the system Python or the v0.5 torch `cu130`
`.venv` (the kUPS docs warn JAX-CUDA and PyTorch-CUDA binaries conflict in one env).
**Device:** GPU 0 only (`CUDA_VISIBLE_DEVICES=0` on every command); display GPU 1 untouched.

## venv-kups — pinned versions
| Package | Version | Note |
|---|---|---|
| Python | 3.13.9 | venv at `~/venvs/venv-kups` |
| **kups** | **1.0.1** | installed **editable from source** `~/kups_src` (main HEAD) — see erratum below |
| jax / jaxlib | 0.10.1 / 0.10.1 | |
| jax-cuda12-plugin / pjrt | 0.10.1 | bundles **CUDA 12.9** runtime (independent of system CUDA 13 / torch cu130) |
| nvidia-cuda-runtime-cu12 | 12.9.79 | JAX's own CUDA libs |
| ase / numpy / scipy | 3.28.0 / 2.4.6 / 1.17.1 | |
| pymatgen | 2026.5.4 | |
| huggingface_hub | 1.19.0 | for the kUPS-mace-jax model |
| optax / pydantic / h5py | 0.2.8 / 2.13.4 / 3.16.0 | kUPS deps |

## Verification (device 0 only)
- `jax.devices()` → **`[CudaDevice(id=0)]`** — GPU 0 visible to JAX.
- `import kups` → OK; `kups.application.simulations.mcmc_widom` importable.
- **End-to-end run:** `kups_mcmc_widom` on the shipped `examples/mcmc_widom.yaml` (RUBTAK Zr-MOF
  + CO₂, reduced to 30 cycles) completed on GPU 0 in ~5 s, producing
  `WidomAnalysisResult`: μ_ex = −0.145 eV, K_H = 6.87×10⁸ Å³/eV, q_st = 0.255 eV (≈ 24.6 kJ/mol).
- HF model `CuspAI/kUPS-mace-jax` downloading to `~/models/kups-mace-jax` (for WPB; revision hash
  pinned in the WPB report).

## Isolation guarantees
- No package installed into the system Python or the v0.5 `.venv`.
- kUPS's JAX uses its own bundled CUDA 12.9 — no interaction with the v0.5 torch `cu130` / system
  CUDA-13 toolkit / NVHPC.
- All v0.6 file outputs are under `v0.6/`; the kUPS source clone is in `~/kups_src`, models in
  `~/models`, both outside the repo.
