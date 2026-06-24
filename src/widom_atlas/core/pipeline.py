"""End-to-end widom-atlas pipeline orchestrator."""

from __future__ import annotations

import json
import platform as _platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from widom_atlas.clustering.basins import extract_basins
from widom_atlas.clustering.uncertainty import annotate_basin_uncertainty
from widom_atlas.core.constants import (
    DEFAULT_ANGLE_TOLERANCE_DEG,
    DEFAULT_BASIN_MATCH_TOL_A,
    DEFAULT_DENSITY_GRID_SHAPE,
    DEFAULT_ENERGY_MATCH_TOL_KJMOL,
    DEFAULT_SYMPREC,
)
from widom_atlas.core.models import Basin, DensityGrid, RunManifest, SymmetryGroup
from widom_atlas.density.grid import build_density_grid
from widom_atlas.density.io import save_density_npz
from widom_atlas.density.smoothing import smooth_density
from widom_atlas.io.models import AtlasInput
from widom_atlas.io.npz import save_samples_npz
from widom_atlas.reports.figures import (
    plot_basin_centroids,
    plot_density_slices,
    plot_robustness_bar,
)
from widom_atlas.reports.html import render_html_report
from widom_atlas.reports.manifest import _sha256_file, write_manifest
from widom_atlas.reports.markdown import render_markdown_report
from widom_atlas.reports.tables import (
    write_basins_csv,
    write_basins_json,
    write_symmetry_groups_json,
)
from widom_atlas.symmetry.grouping import group_basins

_RECORDED_DEPS = (
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
    "pydantic-core",
    "ase",
    "pymatgen",
    "spglib",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "rich",
    "typer",
    "jinja2",
    # Optional extras (REPAIR-7): present when the `widom` / `benchmarks`
    # / `all` extra is installed; otherwise tagged as unavailable rather
    # than silently omitted.
    "widom",
    "CoRE-MOF",
)


def _dep_version(name: str) -> str:
    """Return the installed version of ``name`` or an explicit ``unavailable: …`` string.

    Returning a structured ``unavailable: …`` value (rather than the bare
    ``"unknown"``) makes the manifest contract "every recorded dep has a
    string value" — REPAIR-7.
    """
    try:
        return str(_metadata.version(name))
    except _metadata.PackageNotFoundError:
        return f"unavailable: package {name!r} not installed in this environment"


class PipelineParams(BaseModel):
    """Tunable pipeline knobs; defaults sourced from :mod:`widom_atlas.core.constants`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_grid: tuple[int, int, int] = Field(default=DEFAULT_DENSITY_GRID_SHAPE)
    grid_spacing_A: float | None = None
    smoothing_sigma_A: float = 0.0
    dbscan_eps_A: float = 1.5
    min_samples: int = 8
    symprec: float = DEFAULT_SYMPREC
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE_DEG
    basin_match_tol_A: float = DEFAULT_BASIN_MATCH_TOL_A
    energy_match_tol_kJmol: float = DEFAULT_ENERGY_MATCH_TOL_KJMOL
    temperature_K: float | None = None
    annotate_uncertainty: bool = True


@dataclass
class AtlasResult:
    """Bundle of artefacts produced by :func:`run_atlas`."""

    atlas_input: AtlasInput
    density: DensityGrid
    basins: list[Basin]
    symmetry_groups: list[SymmetryGroup]
    manifest: RunManifest
    out_dir: Path


def _build_manifest(
    atlas_input: AtlasInput,
    structure: Any,
    params: PipelineParams,
    out_dir: Path,
    sample_path: Path,
    structure_path: Path,
) -> RunManifest:
    deps = {name: _dep_version(name) for name in _RECORDED_DEPS}
    package_version = _dep_version("widom-atlas")
    rid = f"{atlas_input.structure_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    cell = np.asarray(atlas_input.cell_matrix, dtype=np.float64)
    structure_sha = _sha256_file(structure_path)
    samples_sha = _sha256_file(sample_path)
    manifest = RunManifest(
        run_id=rid,
        package_version=package_version,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=_platform.platform(),
        dependency_versions=deps,
        structure_id=atlas_input.structure_id,
        structure_source=str(structure_path),
        structure_sha256=structure_sha,
        input_samples_sha256=samples_sha,
        gas=atlas_input.gas,  # type: ignore[arg-type]
        temperature_K=atlas_input.temperature_K,
        parameters=json.loads(params.model_dump_json()),
        dataset_source=None,
        dataset_license=None,
        output_paths={
            "density_npz": str((out_dir / "density.npz").resolve()),
            "basins_csv": str((out_dir / "basins.csv").resolve()),
            "basins_json": str((out_dir / "basins.json").resolve()),
            "symmetry_groups_json": str((out_dir / "symmetry_groups.json").resolve()),
            "report_md": str((out_dir / "report" / "report.md").resolve()),
            "report_html": str((out_dir / "report" / "report.html").resolve()),
            "figures_dir": str((out_dir / "figures").resolve()),
            "cell_matrix": str(cell.tolist()),
        },
    )
    return manifest


def run_atlas(
    atlas_input: AtlasInput,
    params: PipelineParams,
    out_dir: Path,
    structure: Any | None = None,
) -> AtlasResult:
    """Execute the full atlas pipeline; return an :class:`AtlasResult` and write artefacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "report").mkdir(parents=True, exist_ok=True)

    # Persist input (used both as artefact and as hash source for the manifest).
    sample_path = out / "input_samples.npz"
    save_samples_npz(atlas_input, sample_path)

    structure_path = out / "structure_metadata.json"
    structure_path.write_text(
        json.dumps(
            {
                "structure_id": atlas_input.structure_id,
                "cell_matrix": [list(r) for r in atlas_input.cell_matrix],
                "metadata": atlas_input.metadata,
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    samples = atlas_input.samples
    structure_obj = structure if structure is not None else _LooseCellCarrier(atlas_input.cell_matrix_A)

    density = build_density_grid(samples, structure_obj, n_grid=params.n_grid, temperature_K=params.temperature_K)
    if params.smoothing_sigma_A > 0.0:
        density = smooth_density(density, sigma_A=params.smoothing_sigma_A)
    save_density_npz(density, out / "density.npz")

    basins = extract_basins(
        samples,
        structure_obj,
        eps_A=params.dbscan_eps_A,
        min_samples=params.min_samples,
        temperature_K=params.temperature_K,
    )
    if params.annotate_uncertainty and basins:
        basins = annotate_basin_uncertainty(basins, samples, structure_obj)
    write_basins_csv(basins, out / "basins.csv")
    write_basins_json(basins, out / "basins.json")

    try:
        symmetry_groups = group_basins(
            structure_obj if structure is not None else _try_atoms_from_cell(atlas_input.cell_matrix_A),
            basins,
            symprec=params.symprec,
            angle_tolerance=params.angle_tolerance,
            basin_match_tol_A=params.basin_match_tol_A,
            energy_match_tol_kJmol=params.energy_match_tol_kJmol,
        )
    except Exception:
        symmetry_groups = []
    write_symmetry_groups_json(symmetry_groups, out / "symmetry_groups.json")

    fig_density = plot_density_slices(density, out / "figures" / "density_slices.png")
    fig_basins = plot_basin_centroids(basins, structure_obj, out / "figures" / "basin_centroids.png")
    fig_robust = plot_robustness_bar({}, out / "figures" / "robustness_bar.png")

    samples_summary = {
        "n_samples": atlas_input.n_samples,
        "input_hash": atlas_input.input_hash,
        "mean_energy_eV": float(np.mean(atlas_input.energies_eV)) if atlas_input.energies_eV else None,
    }
    density_summary = {
        "shape": list(density.shape),
        "spacing_A": list(density.spacing_A),
        "smoothing_sigma_A": density.smoothing_sigma_A,
    }
    structure_metadata = {
        "structure_id": atlas_input.structure_id,
        "cell_matrix": [list(r) for r in atlas_input.cell_matrix],
    }
    context = {
        "structure_metadata": structure_metadata,
        "gas": atlas_input.gas,
        "temperature_K": atlas_input.temperature_K,
        "samples_summary": samples_summary,
        "density_summary": density_summary,
        "basins": [b.model_dump(mode="json") for b in basins],
        "symmetry_groups": [g.model_dump(mode="json") for g in symmetry_groups],
        "perturbation_summary": list(atlas_input.metadata.get("perturbation_history", [])),
        "robustness_metrics": None,
        "caveats": [
            "Toy / synthetic insertion samples are not chemically meaningful by themselves.",
            "Symmetry assignments are uncertain on defective or strained frameworks.",
        ],
        "uncertainty_notes": [],
        "figure_paths": {
            "density_slices": str(fig_density.relative_to(out / "report").parent / fig_density.name)
            if False
            else f"../figures/{fig_density.name}",
            "basin_centroids": f"../figures/{fig_basins.name}",
            "robustness_bar": f"../figures/{fig_robust.name}",
        },
    }
    render_markdown_report(context, out / "report" / "report.md")
    render_html_report(context, out / "report" / "report.html")

    manifest = _build_manifest(atlas_input, structure_obj, params, out, sample_path, structure_path)
    write_manifest(manifest, out / "manifest.json")

    # REPAIR-4: a separate config.json carries the full PipelineParams + run-level
    # metadata so downstream consumers don't have to dig into the manifest.
    config_payload = {
        "schema_version": "1",
        "widom_atlas_version": _dep_version("widom-atlas"),
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": manifest.run_id,
        "structure_id": atlas_input.structure_id,
        "gas": atlas_input.gas,
        "temperature_K": atlas_input.temperature_K,
        "pipeline_params": json.loads(params.model_dump_json()),
    }
    (out / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # CuspAI-grade hardening: emit a scalar_summary.json carrying the upstream
    # Widom scalars (henry_coefficient, heat_of_adsorption_kJmol, …) when
    # they are present in atlas_input.metadata. This is what
    # build_robustness_report reads to compute delta_ln_KH / delta_Qads.
    widom_scalars = atlas_input.metadata.get("widom_scalars")
    if isinstance(widom_scalars, dict):
        scalar_summary = {
            "schema_version": "1",
            "henry_coefficient": widom_scalars.get("henry_coefficient"),
            "henry_coefficient_std": widom_scalars.get("henry_coefficient_std"),
            "heat_of_adsorption_kJmol": widom_scalars.get("heat_of_adsorption_kJmol"),
            "heat_of_adsorption_std_kJmol": widom_scalars.get("heat_of_adsorption_std_kJmol"),
            "averaged_interaction_energy_eV": widom_scalars.get("averaged_interaction_energy_eV"),
            "atomic_density": widom_scalars.get("atomic_density"),
            "samples_origin": atlas_input.metadata.get("samples_origin", "unknown"),
            "calculator": atlas_input.metadata.get("calculator", "unknown"),
            "n_insertions_total": widom_scalars.get("n_insertions_total")
            or atlas_input.metadata.get("n_insertions_total"),
            "n_insertions_kept": widom_scalars.get("n_insertions_kept")
            or atlas_input.metadata.get("n_insertions_kept"),
        }
        (out / "scalar_summary.json").write_text(
            json.dumps(scalar_summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    # REPAIR-5: emit a robustness/metrics.json stub on every single-run analysis
    # so downstream consumers don't have to special-case its absence. The full
    # robustness/metrics.json shape is produced by widom-atlas compare; this
    # stub explicitly tags itself as not-applicable.
    (out / "robustness").mkdir(parents=True, exist_ok=True)
    robustness_stub = {
        "schema_version": "1",
        "status": "not_applicable_single_run",
        "reason": (
            "robustness comparison requires a pristine + perturbed pair; "
            "this run is a single analyse-samples and has no perturbed counterpart. "
            "Use `widom-atlas compare <pristine_run> <perturbed_run> --out <dir>` "
            "to produce a populated metrics.json."
        ),
        "run_id": manifest.run_id,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (out / "robustness" / "metrics.json").write_text(
        json.dumps(robustness_stub, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return AtlasResult(
        atlas_input=atlas_input,
        density=density,
        basins=basins,
        symmetry_groups=symmetry_groups,
        manifest=manifest,
        out_dir=out,
    )


class _LooseCellCarrier:
    """Cell-only stand-in used when the caller does not supply an ASE/pymatgen object."""

    def __init__(self, cell: np.ndarray) -> None:
        self._cell = np.asarray(cell, dtype=np.float64)

    class _Cell:
        def __init__(self, arr: np.ndarray) -> None:
            self.array = arr

    @property
    def cell(self) -> _LooseCellCarrier._Cell:
        return _LooseCellCarrier._Cell(self._cell)


def _try_atoms_from_cell(cell: np.ndarray) -> Any:
    try:
        from ase import Atoms

        return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(cell, dtype=np.float64), pbc=True)
    except Exception:
        return _LooseCellCarrier(cell)


__all__ = ["AtlasResult", "PipelineParams", "_build_manifest", "run_atlas"]
