# widom-atlas — API reference

Apache-2.0. See [`README.md`](../README.md) for project overview and the three-layer testing strategy. See [`../implementation-verdict.txt`](../implementation-verdict.txt) for the binding spec.

## Public symbols (top-level)

| Symbol | Module | Purpose |
|---|---|---|
| `__version__` | `widom_atlas` | Package version string |
| `AtlasInput` | `widom_atlas.io` | Frozen input contract for the pipeline |
| `InsertionSamples` | `widom_atlas.core.models` | Numpy-backed validated samples |
| `Basin` | `widom_atlas.core.models` | One adsorption basin record |
| `DensityGrid` | `widom_atlas.core.models` | Boltzmann-weighted 3D density grid |
| `SymmetryGroup` | `widom_atlas.core.models` | Symmetry-equivalent basin group |
| `PerturbationSpec` | `widom_atlas.core.models` | Strain / atom-removal spec |
| `RobustnessMetrics` | `widom_atlas.core.models` | Pristine-vs-perturbed metrics |
| `RobustnessReport` | `widom_atlas.core.models` | Aggregate robustness report |
| `RunManifest` | `widom_atlas.core.models` | Run provenance |
| `BenchmarkMaterial`, `BenchmarkRun`, `BenchmarkComparison` | `widom_atlas.models` (and `widom_atlas.core.benchmark_models`) | Public-benchmark provenance |

## I/O — adapter-first foundation

```python
from widom_atlas.io import AtlasInput
from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.io.npz import save_samples_npz, from_npz
from widom_atlas.io.structure_adapters import ase_to_pymatgen, pymatgen_to_ase, get_cell_matrix
```

`from_arrays(structure=..., positions_cart=..., positions_frac=..., energies_eV=..., accessible=..., temperature_K=..., gas=..., metadata=...)` returns a validated :class:`AtlasInput`. **Numpy arrays only** — Python lists are rejected.

## PBC primitives

```python
from widom_atlas.pbc.wrap import wrap_frac, cart_to_frac, frac_to_cart
from widom_atlas.pbc.minimum_image import min_image_displacement, min_image_distance
from widom_atlas.pbc.expansion import expand_27_images, collapse_to_primary
```

Cell convention: rows are lattice vectors (ASE / pymatgen convention).

## Density

```python
from widom_atlas.density.boltzmann import boltzmann_weights, log_boltzmann_weights
from widom_atlas.density.grid import build_density_grid
from widom_atlas.density.smoothing import smooth_density
from widom_atlas.density.io import save_density_npz, load_density_npz, DENSITY_NPZ_SCHEMA_VERSION
```

`build_density_grid(samples, structure, n_grid=(48,48,48), temperature_K=None)` returns a probability-normalised :class:`DensityGrid` with periodic accumulation.

## Clustering

```python
from widom_atlas.clustering.pbc_dbscan import pbc_dbscan
from widom_atlas.clustering.basins import extract_basins
from widom_atlas.clustering.uncertainty import annotate_basin_uncertainty
```

`pbc_dbscan` runs DBSCAN over precomputed minimum-image distances.
`extract_basins` produces :class:`Basin` records using circular-mean fractional centroids so boundary-crossing basins are not split.
`annotate_basin_uncertainty` populates `accessible_fraction`, Kish-effective-sample-size-corrected `energy_stderr_eV`, bootstrap `centroid_stderr_A`, Poisson `weight_stderr`, and a low-count flag.

## Symmetry

```python
from widom_atlas.symmetry.types import FrameworkSymmetry
from widom_atlas.symmetry.spglib_ops import detect_symmetry
from widom_atlas.symmetry.match import group_equivalent_basins
from widom_atlas.symmetry.grouping import group_basins
```

`detect_symmetry(structure, symprec=1e-2, angle_tolerance=5.0)` returns a :class:`FrameworkSymmetry` with `confidence ∈ {high, medium, low, uncertain}` derived from the displacement of refined positions.
`group_basins(structure, basins, ...)` returns :class:`SymmetryGroup` records with explicit `grouping_confidence` and `uncertainty_flags ∈ {tolerance_ambiguous, low_symmetry_host, defective_structure, strained_structure, partial_match, energy_mismatch}`.

## Perturbation

```python
from widom_atlas.perturb.strain import apply_strain
from widom_atlas.perturb.defects import remove_atoms, DefectRecord
from widom_atlas.perturb.api import apply_perturbation
```

`apply_strain(structure, mode, value, axis=None)` for `affine | isotropic | uniaxial | volume_preserving`.
`remove_atoms(structure, indices)` is curated-only — no automatic defect-chemistry inference.
`apply_perturbation(atlas_input, spec_or_specs)` clears samples and records `perturbation_history` + `samples_cleared_due_to_perturbation=True` in metadata.

## Robustness

```python
from widom_atlas.robustness.atlas_metrics import compute_atlas_metrics
from widom_atlas.robustness.scalar_metrics import compute_delta_ln_KH, compute_delta_Qads, compute_scalar_metrics
from widom_atlas.robustness.compare import build_robustness_report
```

Atlas-level metrics (basin persistence, splitting, displacement, accessibility change) work even when scalar Henry/Qads are unavailable.

## Reports

```python
from widom_atlas.reports.manifest import build_manifest, write_manifest
from widom_atlas.reports.tables import write_basins_csv, write_basins_json, write_symmetry_groups_json
from widom_atlas.reports.figures import plot_density_slices, plot_basin_centroids, plot_robustness_bar
from widom_atlas.reports.markdown import render_markdown_report
from widom_atlas.reports.html import render_html_report
```

Figures use the `Agg` matplotlib backend and are headless-safe. The HTML renderer uses Jinja2 autoescape; the Markdown renderer uses StrictUndefined.

## Pipeline

```python
from widom_atlas.core.pipeline import run_atlas, PipelineParams, AtlasResult
```

`run_atlas(atlas_input, params, out_dir, structure=None)` orchestrates density → basins → symmetry → reports and writes a deterministic `manifest.json`.

## Benchmarks (Layer 3)

```python
from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET, get_benchmark_set
from widom_atlas.benchmarks.download import fetch_benchmark_material, BenchmarkDataUnavailable
from widom_atlas.benchmarks.hashing import sha256_file, record_provenance, ProvenanceRecord, ProvenanceMismatch
from widom_atlas.benchmarks.mofxdb import load_mofxdb_scalars, MOFXDBRecord
from widom_atlas.benchmarks.runner import run_benchmark_set
from widom_atlas.benchmarks.scalar_compare import compare_scalars, ScalarComparisonTable
from widom_atlas.benchmarks.launch_report import write_launch_report, LaunchReadinessReport
```

The benchmark layer is opt-in (default-CI excludes it). Scalar comparisons against MOFX-DB / NIST are TREND-labelled — never `PASS` / `VALIDATED`. License-risky data (CSD-derived, IZA bulk redistribution) is rejected at registry time.

## CLI

`widom-atlas {analyse-samples, strain, compare, benchmark, info}`. See `widom-atlas <cmd> --help`.

## Tolerance defaults

Sourced from `widom_atlas.core.constants`:

| Constant | Value |
|---|---|
| `KB_EV_PER_K` | `8.617333262e-5` |
| `EV_TO_KJMOL` | `96.48533212331001` |
| `DEFAULT_SYMPREC` | `1e-2` |
| `DEFAULT_ANGLE_TOLERANCE_DEG` | `5.0` |
| `DEFAULT_BASIN_MATCH_TOL_A` | `0.35` |
| `DEFAULT_ENERGY_MATCH_TOL_KJMOL` | `2.0` |
| `DEFAULT_DENSITY_GRID_SHAPE` | `(48, 48, 48)` |
| `ALLOWED_GASES_V1` | `{'CO2', 'N2', 'CH4'}` |
