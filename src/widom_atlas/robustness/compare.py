"""End-to-end :class:`RobustnessReport` builder from on-disk run directories."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from widom_atlas.core.constants import DEFAULT_BASIN_MATCH_TOL_A
from widom_atlas.core.models import (
    Basin,
    PerturbationSpec,
    RobustnessMetrics,
    RobustnessReport,
)
from widom_atlas.robustness.atlas_metrics import compute_atlas_metrics
from widom_atlas.robustness.scalar_metrics import compute_scalar_metrics


def _load_basins_from_run_dir(run_dir: Path) -> list[Basin]:
    p = Path(run_dir) / "basins.json"
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding="utf-8"))
    items = payload.get("basins", payload) if isinstance(payload, dict) else payload
    return [Basin(**item) for item in items]


def _load_manifest(run_dir: Path) -> dict:
    p = Path(run_dir) / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_scalar_summary(run_dir: Path) -> dict:
    p = Path(run_dir) / "scalar_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_robustness_report(
    pristine_run_dir: Path,
    perturbed_run_dir: Path,
    *,
    match_tol_A: float = DEFAULT_BASIN_MATCH_TOL_A,
) -> RobustnessReport:
    """Load pristine + perturbed run outputs and assemble a :class:`RobustnessReport`."""
    pristine_basins = _load_basins_from_run_dir(pristine_run_dir)
    perturbed_basins = _load_basins_from_run_dir(perturbed_run_dir)
    pristine_manifest = _load_manifest(pristine_run_dir)
    perturbed_manifest = _load_manifest(perturbed_run_dir)
    pristine_summary = _load_scalar_summary(pristine_run_dir)
    perturbed_summary = _load_scalar_summary(perturbed_run_dir)

    cell_list = perturbed_manifest.get("cell_matrix") or pristine_manifest.get("cell_matrix")
    if cell_list is None:
        raise ValueError(
            "neither pristine nor perturbed manifest carries a cell_matrix; cannot run PBC matching"
        )
    cell = np.asarray(cell_list, dtype=np.float64)

    atlas_metrics = compute_atlas_metrics(
        pristine_basins, perturbed_basins, cell, match_tol_A=match_tol_A
    )
    scalar_metrics = compute_scalar_metrics(pristine_summary, perturbed_summary)

    metrics_model = RobustnessMetrics(
        delta_ln_KH=scalar_metrics["delta_ln_KH"],
        delta_Qads_kJmol=scalar_metrics["delta_Qads_kJmol"],
        basin_count_pristine=int(atlas_metrics["basin_count_pristine"]),
        basin_count_perturbed=int(atlas_metrics["basin_count_perturbed"]),
        basin_count_change=int(atlas_metrics["basin_count_change"]),
        basin_persistence_fraction=float(atlas_metrics["basin_persistence_fraction"]),
        basin_splitting_count=int(atlas_metrics["basin_splitting_count"]),
        mean_basin_displacement_A=float(atlas_metrics["mean_basin_displacement_A"]),
        accessibility_change=float(atlas_metrics["accessibility_change"]),
        ambiguity_flags=list(atlas_metrics["ambiguity_flags"]),
        missing_data_flags=list(atlas_metrics["missing_data_flags"]) + list(scalar_metrics["missing_fields"]),
    )

    perturbation_record = perturbed_manifest.get("perturbation_spec")
    if perturbation_record is None:
        perturbation_record = {"kind": "isotropic", "magnitude": 0.0, "label": "unknown"}
    pspec = PerturbationSpec(**perturbation_record)

    summary = {
        "delta_ln_KH": metrics_model.delta_ln_KH,
        "delta_Qads_kJmol": metrics_model.delta_Qads_kJmol,
        "basin_persistence_fraction": metrics_model.basin_persistence_fraction,
        "mean_basin_displacement_A": metrics_model.mean_basin_displacement_A,
    }

    return RobustnessReport(
        report_id=f"{pristine_manifest.get('run_id','pristine')}_vs_{perturbed_manifest.get('run_id','perturbed')}",
        structure_id=str(perturbed_manifest.get("structure_id") or pristine_manifest.get("structure_id") or "unknown"),
        gas=str(perturbed_manifest.get("gas") or pristine_manifest.get("gas") or "CO2"),
        temperature_K=float(perturbed_manifest.get("temperature_K") or pristine_manifest.get("temperature_K") or 298.15),
        pristine_run_id=str(pristine_manifest.get("run_id", "pristine")),
        perturbations=[pspec],
        metrics_per_perturbation=[metrics_model],
        summary=summary,
        caveats=list(perturbed_manifest.get("caveats", [])) + list(pristine_manifest.get("caveats", [])),
        generated_at=datetime.now(UTC),
        schema_version="1",
    )


__all__ = ["_load_basins_from_run_dir", "build_robustness_report"]
