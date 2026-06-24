# widom-atlas

A governed Widom-insertion validation harness for zero-coverage adsorption
thermodynamics (K_H, Q_st, binding-site geometry) in porous materials
(MOFs and zeolites). Pure Python + NumPy core, with optional RASPA3 and
RASPA2 external-backend integration.

License: Apache-2.0.

---

## Papers

- [widom-atlas master article](papers/widom-atlas_master.pdf)
- [kUPS governance companion report](papers/widom-atlas_kups-governance_companion.pdf)

arXiv submission is pending endorsement. The PDFs above are the current public preprint versions.

---

## Current scientific status (locked 2026-06-01)

**widom-atlas is NOT scientifically validated as a general predictive tool.**

The strict-tier denominator is **15** verdict-affecting branches; the
strict-tier numerator is **2** (the 6c MFI + Ar positive control and a new
4c Si-CHA + CO₂ branch under Bai 2013 TraPPE-zeo).

```
Tier A strict PASS:      2 / 15
  └─ 6c MFI + Ar           (positive control)
  └─ 4c Si-CHA + CO₂       (Bai 2013 TraPPE-zeo + analytical LJ tail)

Tier B physical PASS:    3 (Tier A + 3b_EHq UiO-66 Maia explicit-H)

Strict FAIL:            13 (Bayesian Z-score classified)

Method blocked:          1 (5b Na-Rho trapdoor — ensemble mismatch)

Reference audited
  pending lock:          4 (5c Na-ZK-5 / 5A / 13X / 4A)
```

| Locked artefact | SHA-256 (first 16) |
|---|---|
| `v04_case_matrix.yaml` | `b0571780a4794be0fabb2a7d47a093fcb8dab07c4355467bb2fabe1a6a085664` |
| Paper PDF (`paper/main.pdf`) | MD5 `132b7bba6fa085d3` |

Full lock record: [`V04_SCIENTIFIC_STATUS_LOCKED_2026_06_01.md`](./V04_SCIENTIFIC_STATUS_LOCKED_2026_06_01.md)
and `evidence/scientific_status_lock_2026_06_01.json`.

---

## Headline result — 4c Si-CHA + CO₂ at 298 K

The Bai 2013 TraPPE-zeo all-silica framework force field
(Si: ε/k_B = 22.0 K, σ = 2.3 Å, q = +2.05 e; O: ε/k_B = 53.0 K, σ = 3.3 Å,
q = −1.025 e) with Lorentz-Berthelot mixing, shifted-truncated LJ at 14 Å,
analytical tail correction, and native Ewald reproduces the Maghsoudi
2013 experimental K_H and Q_st on Si-CHA + CO₂ at 298 K within the original
±0.10 / ±2.0 strict thresholds:

```
K_H  = 2.22 ± 0.14 mol/(kg·bar)   vs. Maghsoudi 2013 = 2.43
Q_st = 22.99 ± 0.33 kJ/mol        vs. Maghsoudi 2013 = 21.0
Δlog10 K_H = -0.039  (PASS ±0.10 strict)
ΔQ_st     = +1.99 kJ/mol  (PASS ±2.0 strict)
Bayesian |Z| = 0.19   (AGREEMENT_WITHIN_1_SIGMA)
```

3 seeds × 60,000 Widom insertions on a 3×3×2 supercell, T = 298 K,
~80 s wall time on a single CPU core. See
[`paper/main.pdf`](./paper/main.pdf) for the formal write-up,
[`V04_FINAL_REPORT_FOR_PROFESSOR.md`](./V04_FINAL_REPORT_FOR_PROFESSOR.md)
for the disposition summary, and
[`V04_DEEP_RESEARCH_PIVOT_2026_06_01.md`](./V04_DEEP_RESEARCH_PIVOT_2026_06_01.md)
for the methodology change log.

The Bai 2013 framework parameters were verified through three independent
RASPA-distributed reference files (RASPA3 bundled JSON, RASPA2 bundled
`.def`, and the Dubbeldam 2011 `MFI_SI.cif`) without recourse to the
paywalled primary paper.

---

## Reproduce the 4c strict-tier result

```bash
git clone <widom-atlas-repo>
cd widom-atlas

# Python 3.12+; set up venv and install
python -m venv .venv
.venv/bin/pip install -e .

# RASPA3 v3.0.29 + RASPA2 v2.0.50 (conda-forge — used for cross-engine V1-V4)
conda create -n raspa3 -c conda-forge raspa3=3.0.29 -y
conda create -n raspa2 -c conda-forge raspa2=2.0.50 -y

# Execute the 4c Si-CHA + CO₂ and 6e MFI + CH₄ production runs
PYTHONPATH=src .venv/bin/python scripts/v04/execute_4c_6e_bai_2013.py

# Expected 4c output:
#   K_H  = 2.22 ± 0.14 mol/(kg·bar)
#   Q_st = 22.99 ± 0.33 kJ/mol
#   Tier A PASS, Tier B PASS, headline PHYSICAL_PASS
```

Per-branch verdict JSONs land in `evidence/v04_<branch>_bai_2013/verdicts/`.

---

## What widom-atlas v0.4 is

**A governed validation harness** built on:

- **Single SHA-256 pinned case matrix** (`v04_case_matrix.yaml`, ~2200 lines)
  specifying for every verdict-affecting branch: framework CIF, atom-type
  relabel convention, force-field lineage + functional form + per-atom-type
  parameters + charges, mixing rule, electrostatics treatment, gas model,
  experimental K_H + Q_st references with literature-scatter acceptance
  windows, validation tier, and disposition enum.
- **Three backends in parallel**:
  - Custom NumPy native evaluator supporting LJ 12-6, Buckingham, Dzubak
    `A·exp(−Br) − C/r⁵ − D/r⁶`, and RASPA generic `A·exp(−Br) − C₆/r⁶ − C₈/r⁸`
    pair potentials with native Ewald electrostatics. The Dzubak and Ongari
    forms are not expressible in RASPA3 v3.0.29's JSON pair-potential path.
  - RASPA3 v3.0.29 (conda-forge) for Ewald LJ branches.
  - RASPA2 v2.0.50 (conda-forge) for Buckingham branches.
  - ASE Calculator wrapper enabling cross-checks through the upstream
    [`cusp-ai-oss/widom`](https://github.com/cusp-ai-oss/widom) estimator.
- **Two-tier verdict system**:
  - **Tier A regression gate** (internal): historical ±0.10 Δlog₁₀ K_H +
    ±2.0 kJ/mol Q_st thresholds. Used to catch code-change-induced shifts.
  - **Tier B physical-accuracy band** (headline): per-system bands keyed to
    documented literature scatter (Park 2017, McCready 2024).
- **Bayesian log-space K_H comparator** producing per-branch Z-score
  classification: WITHIN_1σ / WITHIN_2σ / TENSION_2-3σ /
  STRONG_DISAGREEMENT_>3σ.
- **Analytical LJ + Buckingham + Dzubak + Ongari tail corrections** with
  closed-form derivations per Frenkel & Smit Ch. 3.
- **Polarizable Widom prototype infrastructure** (Thole damping + SCF
  induced dipoles + polarization-energy bookkeeping); infrastructure
  complete, kernel debug pending on 800+ atom systems.

---

## Six host-guest case systems × 22 branches

```
Case 1 Mg-MOF-74 + CO₂   1a Lin/Mercado | 1b Dzubak | 1c Becker | 1d Mercado 2016
Case 2 HKUST-1 + CO₂     2a UFF Cu | 2b Ongari 2017 modified Cu-O(CO₂)
Case 3 UiO-66 + CO₂      3a PACMOF2 + UFF Zr | 3b Maia 2023 (UA / UAq / EHq)
Case 4 CHA + CO₂         4a TraPPE-zeo + Harris-Yung | 4b Cu-SSZ-13 (deferred)
                         4c Bai 2013 TraPPE-zeo + TraPPE-CO₂ ← strict PASS
Case 5 Na-Rho + CO₂      5a exploratory | 5b trapdoor (ensemble mismatch)
                         5c × 4 reference-audited replacement-scalar candidates
Case 6 MFI + small gas   6a CH₄ Garcia-Perez | 6b Kr | 6c Ar ← positive control
                         6d numerical-only | 6e Bai 2013 + TraPPE-UA CH₄
```

---

## Repository layout

| Path | Content |
|---|---|
| `v04_case_matrix.yaml` | SHA-locked branch specification (~2200 lines) |
| `src/widom_atlas/v04/locked_inputs.py` | SHA-pinned loader (`LockedDigestMismatch` on mutation) |
| `src/widom_atlas/v04/thresholds.py` | Tier A + Tier B threshold definitions |
| `src/widom_atlas/v04/bayesian_comparator.py` | Log-space Bayesian K_H comparator |
| `src/widom_atlas/v04/native/` | NumPy Widom evaluator: pair potentials, Ewald, runner, polarizable extension, ASE wrapper, analytical tail correction |
| `src/widom_atlas/v04/native/bai_2013_trappe_zeo_loader.py` | Bai 2013 + TraPPE gas loader for 4c and 6e |
| `src/widom_atlas/v04/native/maia_2023_loader.py` | Maia 2023 UA / UAq / EHq loader for 3b |
| `scripts/v04/execute_4c_6e_bai_2013.py` | Production-run driver for the 4c strict-tier result |
| `scripts/v04/emit_scientific_status_lock.py` | Reproducible scientific-status lock emitter |
| `tests/v04/` | 312 regression tests (incl. scientific-lock cross-check) |
| `evidence/v04_<branch>/verdicts/<branch>.json` | Per-branch verdict JSONs |
| `paper/main.{tex,pdf}` | Formal paper write-up (20 pages) |
| `docs/research/dataset-research-for-v0.4/` | Archived primary literature + SHA-256 provenance manifest |

---

## Install

```bash
# Base install
python -m venv .venv
.venv/bin/pip install -e .

# Dev extras (pytest, ruff, mypy, hypothesis, pre-commit)
.venv/bin/pip install -e ".[dev]"

# External backends (optional)
conda create -n raspa3 -c conda-forge raspa3=3.0.29 -y
conda create -n raspa2 -c conda-forge raspa2=2.0.50 -y
```

Dependencies pinned in `pyproject.toml`: `numpy`, `scipy`, `pandas`,
`pydantic`, `ase`, `pymatgen`, `spglib`, `scikit-learn`, `matplotlib`,
`tqdm`, `rich`, `typer`, `jinja2`, `pyyaml`. Python 3.12+.

---

## Test

```bash
# All v0.4 regression tests
PYTHONPATH=src .venv/bin/python -m pytest tests/v04/

# Cross-engine V1-V4 native-vs-RASPA validation (requires RASPA3 + RASPA2)
PYTHONPATH=src .venv/bin/python -m pytest tests/v04/test_native_runner.py
                                          tests/v04/test_raspa3/

# Bai 2013 cross-validation against RASPA-bundled reference files
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/v04/test_bai_2013_loader_and_4c_6e_execution.py \
  tests/v04/test_trappe_official_cross_validation.py

# Linting + type checking
.venv/bin/ruff check src/widom_atlas/v04/ tests/v04/
.venv/bin/mypy src/widom_atlas/v04/native/ \
               src/widom_atlas/v04/thresholds.py \
               src/widom_atlas/v04/bayesian_comparator.py
```

312 tests on v0.4 modules including scientific-lock cross-check, all passing.

---


## Non-Goals

widom-atlas v0.4 explicitly **is not**:

- A general-purpose adsorption simulator. It does not run GCMC, MD, or
  free-energy integration; it consumes Widom-insertion data and
  produces validation-tier verdicts against primary experimental
  references.
- A predictor of K_H or Q_st for arbitrary new (framework, gas) systems.
  Only the 15 verdict-affecting case-matrix branches have governed
  dispositions; transferring any result outside that scope requires
  independent verification.
- A force-field developer. New parameter sets must come from primary
  literature with a DOI in the YAML lineage field; widom-atlas refits
  nothing.
- A polarisable-Widom production tool yet. The infrastructure exists
  but the SCF kernel diverges on production-scale supercells; the
  prototype is not load-bearing on any reported verdict.
- A solver for the Na-Rho trapdoor open-state K_H. The rigid-framework
  Widom estimator samples the closed-state partition function which is
  a different observable from the experimental open-state Langmuir
  K_H; closing this requires flat-histogram / Wang-Landau MC out of
  v0.4 scope.
- A replacement for RASPA3 or RASPA2. The native NumPy evaluator
  complements them on the specific pair-potential families they cannot
  express (Dzubak r⁻⁵, RASPA-generic r⁻⁸); the external backends
  remain the canonical Ewald path for LJ branches.

## Trade-Offs

The widom-atlas design makes the following explicit trade-offs:

- **Pure Python NumPy evaluator vs. compiled C/C++**: simplicity,
  reproducibility, and portability across CPU architectures over raw
  performance. A single 80 000-insertion Widom run on a 100-atom
  supercell takes 20–90 s wall on a single CPU core; this is
  acceptable for validation work, not production GCMC.
- **SHA-locked case matrix vs. live YAML editing**: every parameter
  change rolls a SHA-256 with an audit-trail comment in
  `locked_inputs.py`. Trades the convenience of casual edits for
  traceability and tamper detection. The hash mismatch path raises
  before any consumer reads the matrix.
- **Two-tier disposition (regression vs. physical accuracy)**: the
  strict ±0.10 Δlog₁₀ K_H thresholds are tighter than documented
  literature scatter for open-metal-site MOFs. Tier B physical-accuracy
  bands keyed to literature scatter are the headline; Tier A remains
  as an internal regression gate. The dual-disposition trades simple
  binary verdicts for a verdict + per-branch Bayesian Z-score.
- **Lorentz-Berthelot cross-pair mixing as default**: faster and
  reproducible across engines, at the cost of not capturing the
  improved fits that bespoke cross-pair tables (Talu-Myers, Mason)
  can provide. Branch-specific cross-pair tables are wired per-branch
  when the primary FF source distributes them.
- **Shifted-truncated LJ + analytical tail correction**: matches the
  RASPA3 default convention for cross-engine V1–V4 validation, at the
  cost of a small documented bias relative to the full-tail Frenkel-
  Smit convention. The analytical tail correction recovers most of
  the bias and was load-bearing on the 4c strict pass.
- **Three external backends instead of one**: triple-checking comes
  at the cost of three install paths (Python venv + RASPA3 conda
  env + RASPA2 conda env). The redundancy isolates estimator bugs
  from force-field physics.

## Limitations

The reported results are subject to the following limitations:

- The native NumPy evaluator uses double-precision IEEE-754 arithmetic
  with log-sum-exp Widom accumulation; no higher-precision accumulator
  or formal numerical-stability proof beyond the V1–V4 cross-engine
  agreement tests.
- The analytical tail correction assumes g(r) = 1 beyond the cutoff,
  which is an approximation at the framework-pore length scale; no
  formal error bound on this approximation.
- RASPA3 is pinned at v3.0.29 from conda-forge; the conda-forge package
  binary-build provenance is not separately tracked beyond the
  recorded SHA-256.
- The Tier B per-system physical-accuracy bands are derived
  heuristically from documented literature spread; a fully quantitative
  consensus-isotherm methodology (per McCready 2024) would require
  NIST ISODB replicate-isotherm integration not yet wired.
- The polarisable Widom prototype diverges in SCF iteration on
  production-scale supercells and is not used in any reported verdict.
- 5b Na-Rho scalar verdict is METHOD_BLOCKED by ensemble mismatch; site
  geometry remains active and currently FAILs.
- 5c branches are reference-audited but not atlas-executed pending
  cation-CIF + cation-FF lock for each of Na-KFI, Ca-LTA, Na-FAU, Na-LTA.

## Constraints (verbatim from the spec)

- No NVIDIA / GPU / ML / performance work.
- No threshold loosening, no branch dropping, no simulator-as-truth.
- No invented force-field parameters.
- Every force-field parameter table requires a primary-source DOI in the
  YAML lineage field.
- 5b Na-Rho stays scalar METHOD_BLOCKED + site-truth active; 5c
  replacement-scalar branches are NOT 5b validation.

---

## Historical v0.3 functionality (preserved)

The original v0.3 / v0.2 array-based atlas API for consuming Widom samples
and producing adsorption-density maps, DBSCAN basins, symmetry-equivalent
site grouping, and robustness reports is retained under
`src/widom_atlas/evaluator/`, `src/widom_atlas/io/`, and
`src/widom_atlas/benchmarks/`. The Typer CLI (`widom-atlas analyse-samples`,
`strain`, `compare`, `benchmark`, `info`) still works on the v0.3 surface.

The v0.4 work documented above is a separate, governed validation harness;
it shares the `widom_atlas` Python namespace but does not depend on the
v0.3 array-based atlas API.

### v0.3 quick example

```python
import numpy as np
from ase import Atoms
from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.core.pipeline import PipelineParams, run_atlas

atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
rng = np.random.default_rng(0)
positions_frac = rng.random((1000, 3))
energies_eV = rng.normal(-0.2, 0.05, 1000)

atlas_input = from_arrays(
    structure=atoms,
    positions_frac=positions_frac,
    energies_eV=energies_eV,
    temperature_K=298.15,
    gas="CO2",
    metadata={"src": "your-widom-campaign"},
)
params = PipelineParams(n_grid=(48, 48, 48), dbscan_eps_A=1.5, min_samples=8)
result = run_atlas(atlas_input, params, out_dir="runs/example", structure=atoms)
print("basins:", len(result.basins), "manifest:", result.manifest.run_id)
```

```bash
widom-atlas analyse-samples samples.npz \
  --structure structure.cif --gas CO2 --temperature 298.15 \
  --out runs/example
```

---

## Citation

If you use the 4c Si-CHA + CO₂ strict-tier result or the two-tier
threshold methodology, please cite the paper at `paper/main.pdf` and
record the locked CASE_MATRIX_SHA256 you reproduced against:

```
CASE_MATRIX_SHA256 = b0571780a4794be0fabb2a7d47a093fcb8dab07c4355467bb2fabe1a6a085664
PAPER_PDF_MD5      = 132b7bba6fa085d3cacb645ea6cc5a49
Lock record        : V04_SCIENTIFIC_STATUS_LOCKED_2026_06_01.md
                     evidence/scientific_status_lock_2026_06_01.json
```

---

## License

Apache-2.0 for the widom-atlas source. Per-file primary-literature
provenance recorded in `docs/research/dataset-research-for-v0.4/PROVENANCE_MANIFEST.json`
with SHA-256 + DOI + license per archived artefact.
