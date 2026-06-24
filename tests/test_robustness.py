"""Tests for robustness module: atlas metrics, scalar metrics, compare (T033–T035)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from widom_atlas.core.models import Basin, RobustnessReport
from widom_atlas.robustness.atlas_metrics import compute_atlas_metrics
from widom_atlas.robustness.compare import build_robustness_report
from widom_atlas.robustness.scalar_metrics import (
    compute_delta_ln_KH,
    compute_delta_Qads,
    compute_scalar_metrics,
)


def _basin(idx: int, frac: tuple[float, float, float], **overrides) -> Basin:
    base = {
        "basin_id": idx,
        "count": 10,
        "weight": 0.1,
        "centroid_frac": frac,
        "centroid_cart_A": (frac[0] * 10.0, frac[1] * 10.0, frac[2] * 10.0),
        "mean_energy_eV": -0.5,
        "std_energy_eV": 0.01,
        "accessible_fraction": 1.0,
        "spread_A": 0.1,
    }
    base.update(overrides)
    return Basin(**base)


# --- T033 atlas_metrics ------------------------------------------------------


def test_compute_atlas_metrics_persistence_and_splitting() -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.2, 0.5, 0.5)), _basin(1, (0.8, 0.5, 0.5))]
    perturbed = [
        _basin(0, (0.21, 0.5, 0.5)),
        _basin(1, (0.22, 0.5, 0.5)),  # second perturbed near pristine 0 → split
        _basin(2, (0.81, 0.5, 0.5)),
    ]
    m = compute_atlas_metrics(pristine, perturbed, cell, match_tol_A=0.5)
    assert m["basin_count_pristine"] == 2
    assert m["basin_count_perturbed"] == 3
    assert m["basin_persistence_fraction"] == 1.0
    assert m["basin_splitting_count"] == 1


def test_compute_atlas_metrics_pbc_minimum_image() -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.01, 0.5, 0.5))]
    perturbed = [_basin(0, (0.99, 0.5, 0.5))]  # min-image distance ~0.2 A
    m = compute_atlas_metrics(pristine, perturbed, cell, match_tol_A=0.5)
    assert m["basin_persistence_fraction"] == 1.0
    assert m["mean_basin_displacement_A"] < 0.5


def test_compute_atlas_metrics_empty_perturbed_returns_zero_persistence() -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.2, 0.5, 0.5))]
    perturbed: list[Basin] = []
    m = compute_atlas_metrics(pristine, perturbed, cell, match_tol_A=0.5)
    assert m["basin_persistence_fraction"] == 0.0


def test_compute_atlas_metrics_displacement_uses_min_image_not_naive() -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.01, 0.5, 0.5))]
    perturbed = [_basin(0, (0.99, 0.5, 0.5))]
    m = compute_atlas_metrics(pristine, perturbed, cell, match_tol_A=0.5)
    # min-image distance ~0.2 A; naive Cartesian would be ~9.8 A
    assert m["mean_basin_displacement_A"] < 1.0


# --- T034 scalar_metrics -----------------------------------------------------


def test_compute_delta_ln_KH_basic() -> None:
    out = compute_delta_ln_KH(1.0, 2.0)
    assert abs(out - np.log(2.0)) < 1e-12


def test_compute_delta_ln_KH_returns_none_on_missing() -> None:
    assert compute_delta_ln_KH(None, 1.0) is None
    assert compute_delta_ln_KH(1.0, None) is None


def test_compute_delta_ln_KH_returns_none_on_nonpositive() -> None:
    assert compute_delta_ln_KH(0.0, 1.0) is None
    assert compute_delta_ln_KH(-1.0, 2.0) is None
    assert compute_delta_ln_KH(1.0, 0.0) is None


def test_compute_delta_Qads_basic() -> None:
    assert compute_delta_Qads(20.0, 25.0) == pytest.approx(5.0)


def test_compute_scalar_metrics_graceful_degradation() -> None:
    out = compute_scalar_metrics({}, {})
    assert out["delta_ln_KH"] is None
    assert out["delta_Qads_kJmol"] is None
    assert out["degraded"] is True
    assert "henry_coefficient_pristine" in out["missing_fields"]


def test_compute_scalar_metrics_marks_missing_fields() -> None:
    out = compute_scalar_metrics({"henry_coefficient": 1.0}, {})
    assert "henry_coefficient_perturbed" in out["missing_fields"]
    assert "heat_of_adsorption_pristine_kJmol" in out["missing_fields"]


# --- T035 build_robustness_report -------------------------------------------


def _seed_run_dir(
    tmp_path: Path,
    name: str,
    basins: list[Basin],
    cell: np.ndarray,
    structure_id: str = "ToyCell",
    gas: str = "CO2",
    KH: float | None = None,
    Q_kJmol: float | None = None,
    perturbation: dict | None = None,
) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "basins.json").write_text(
        json.dumps({"basins": [b.model_dump() for b in basins]}), encoding="utf-8"
    )
    manifest = {
        "run_id": name,
        "structure_id": structure_id,
        "gas": gas,
        "temperature_K": 298.15,
        "cell_matrix": cell.tolist(),
    }
    if perturbation is not None:
        manifest["perturbation_spec"] = perturbation
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if KH is not None or Q_kJmol is not None:
        (d / "scalar_summary.json").write_text(
            json.dumps({"henry_coefficient": KH, "heat_of_adsorption_kJmol": Q_kJmol}),
            encoding="utf-8",
        )
    return d


def test_build_robustness_report_end_to_end_from_run_dirs(tmp_path: Path) -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.2, 0.5, 0.5)), _basin(1, (0.8, 0.5, 0.5))]
    perturbed = [_basin(0, (0.22, 0.5, 0.5)), _basin(1, (0.82, 0.5, 0.5))]
    pri = _seed_run_dir(tmp_path, "pristine", pristine, cell, KH=1.0, Q_kJmol=20.0)
    per = _seed_run_dir(
        tmp_path, "perturbed", perturbed, cell,
        KH=1.5, Q_kJmol=22.0,
        perturbation={"kind": "isotropic", "magnitude": 0.01, "label": "iso1"},
    )
    report = build_robustness_report(pri, per, match_tol_A=0.5)
    assert isinstance(report, RobustnessReport)
    assert report.metrics_per_perturbation[0].basin_persistence_fraction == 1.0


def test_build_robustness_report_handles_missing_scalar_summary(tmp_path: Path) -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.2, 0.5, 0.5))]
    perturbed = [_basin(0, (0.21, 0.5, 0.5))]
    pri = _seed_run_dir(tmp_path, "pristine", pristine, cell)
    per = _seed_run_dir(
        tmp_path, "perturbed", perturbed, cell,
        perturbation={"kind": "isotropic", "magnitude": 0.01, "label": "iso1"},
    )
    report = build_robustness_report(pri, per, match_tol_A=0.5)
    assert report.metrics_per_perturbation[0].delta_ln_KH is None


def test_build_robustness_report_round_trips_basins_from_json(tmp_path: Path) -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.2, 0.5, 0.5), spread_A=0.42)]
    perturbed = [_basin(0, (0.22, 0.5, 0.5), spread_A=0.55)]
    pri = _seed_run_dir(tmp_path, "pristine", pristine, cell)
    per = _seed_run_dir(
        tmp_path, "perturbed", perturbed, cell,
        perturbation={"kind": "isotropic", "magnitude": 0.01, "label": "iso1"},
    )
    report = build_robustness_report(pri, per, match_tol_A=0.5)
    assert report.metrics_per_perturbation[0].basin_count_pristine == 1


def test_build_robustness_report_merges_ambiguity_and_missing_flags(tmp_path: Path) -> None:
    cell = np.eye(3) * 10.0
    pristine = [_basin(0, (0.2, 0.5, 0.5))]
    perturbed = [_basin(0, (0.21, 0.5, 0.5)), _basin(1, (0.22, 0.5, 0.5))]
    pri = _seed_run_dir(tmp_path, "pristine", pristine, cell)
    per = _seed_run_dir(
        tmp_path, "perturbed", perturbed, cell,
        perturbation={"kind": "isotropic", "magnitude": 0.01, "label": "iso1"},
    )
    report = build_robustness_report(pri, per, match_tol_A=0.5)
    metrics = report.metrics_per_perturbation[0]
    assert any("multiple_perturbed" in f for f in metrics.ambiguity_flags)
