"""Wire the v0.4 evaluator into the full atlas pipeline (one run = one full atlas).

For each (CIF, UPF, gas, T) tuple this module:

1. Loads the CIF + UPF.
2. Runs the internal Widom evaluator (returns per-insertion centres + energies +
   K_H + Q_ads).
3. Builds an ``AtlasInput`` via ``widom_atlas.io.from_arrays`` (all energies
   converted from K to eV).
4. Calls ``widom_atlas.core.pipeline.run_atlas`` to produce density grids,
   basins, symmetry groups, and figure/report artefacts under
   ``benchmarks/results/v0.4-validation/atlas/<case_id>/``.
5. Returns an ``EndToEndCaseResult`` with the K_H/Q_ads scalars, the path to
   the atlas run directory, and the basin count + dominant centroid.

This is the path that lifts v0.4 from "schema-only" to a real atlas case.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from widom_atlas.backends.user_parameterised import UserParameterFile
from widom_atlas.core.pipeline import PipelineParams, run_atlas
from widom_atlas.evaluator.component import (
    Component,
    ch4_trappe_ua,
    co2_garcia_sanchez_2009,
    n2_trappe,
)
from widom_atlas.evaluator.runner import (
    K_BOLTZMANN_KJ_MOL_PER_K,
    WidomResult,
    load_atoms,
    run_widom_evaluator,
)
from widom_atlas.io.from_arrays import from_arrays

K_TO_EV = K_BOLTZMANN_KJ_MOL_PER_K / 96.485  # 1 K = 8.617e-5 eV (via kJ/mol → eV)
ALLOWED_ATLAS_GASES = frozenset({"CO2", "N2", "CH4"})


@dataclass(frozen=True)
class EndToEndCaseResult:
    case_id: str
    framework_name: str
    gas: str
    temperature_K: float
    structure_path: str
    upf_path: str
    n_insertions_used: int
    KH_mol_per_kg_per_Pa: float | None
    log10_KH: float | None
    Qads_kJ_per_mol: float | None
    e_lj_K_mean: float | None
    e_coul_K_mean: float | None
    atlas_run_dir: str
    n_basins: int
    dominant_basin_weight: float | None
    dominant_basin_centroid_frac: tuple[float, float, float] | None
    dominant_basin_centroid_A: tuple[float, float, float] | None
    pipeline_completed: bool
    notes: str
    warnings: list[str]


COMPONENT_FACTORIES = {
    "CO2": co2_garcia_sanchez_2009,
    "N2": n2_trappe,
    "CH4": ch4_trappe_ua,
}


def _safe_log10(x: float | None) -> float | None:
    if x is None or x <= 0:
        return None
    return float(np.log10(x))


def _energies_K_to_eV(e_K: np.ndarray) -> np.ndarray:
    return e_K * K_TO_EV


def run_one_end_to_end(
    *,
    case_id: str,
    framework_name: str,
    structure_path: Path,
    upf_path: Path,
    gas: str,
    temperature_K: float,
    n_insertions: int,
    seed: int,
    r_cut_A: float,
    grid_mode: Literal["deterministic_uniform", "stochastic_uniform"],
    out_root: Path,
    smoothing_sigma_A: float = 0.0,
    n_grid: tuple[int, int, int] = (40, 40, 40),
    dbscan_eps_A: float = 1.5,
    min_samples: int = 8,
) -> EndToEndCaseResult:
    """Run one full atlas case and return its EndToEndCaseResult."""
    if not structure_path.exists():
        return _missing_result(case_id, framework_name, gas, temperature_K, structure_path, upf_path,
                               f"structure missing: {structure_path}")
    if not upf_path.exists():
        return _missing_result(case_id, framework_name, gas, temperature_K, structure_path, upf_path,
                               f"UPF missing: {upf_path}")
    if gas not in COMPONENT_FACTORIES:
        return _missing_result(case_id, framework_name, gas, temperature_K, structure_path, upf_path,
                               f"no Component factory for gas={gas!r}; allowed: {sorted(COMPONENT_FACTORIES)}")
    if gas not in ALLOWED_ATLAS_GASES:
        return _missing_result(case_id, framework_name, gas, temperature_K, structure_path, upf_path,
                               f"gas {gas!r} not in atlas-pipeline allow-list {sorted(ALLOWED_ATLAS_GASES)}")

    upf = UserParameterFile.model_validate_json(upf_path.read_text(encoding="utf-8"))
    component: Component = COMPONENT_FACTORIES[gas]()
    atoms = load_atoms(structure_path)

    result: WidomResult = run_widom_evaluator(
        atoms=atoms,
        framework_name=framework_name,
        user_parameter_file=upf,
        component=component,
        temperature_K=temperature_K,
        n_insertions=n_insertions,
        seed=seed,
        r_cut_A=r_cut_A,
        grid_mode=grid_mode,
        framework_source_path=structure_path,
    )

    if result.status != "ok" or result.n_insertions_used == 0:
        return EndToEndCaseResult(
            case_id=case_id,
            framework_name=framework_name,
            gas=gas,
            temperature_K=temperature_K,
            structure_path=str(structure_path),
            upf_path=str(upf_path),
            n_insertions_used=result.n_insertions_used,
            KH_mol_per_kg_per_Pa=result.KH_mol_per_kg_per_Pa,
            log10_KH=_safe_log10(result.KH_mol_per_kg_per_Pa),
            Qads_kJ_per_mol=result.Qads_kJ_per_mol,
            e_lj_K_mean=None,
            e_coul_K_mean=None,
            atlas_run_dir="",
            n_basins=0,
            dominant_basin_weight=None,
            dominant_basin_centroid_frac=None,
            dominant_basin_centroid_A=None,
            pipeline_completed=False,
            notes=f"evaluator status={result.status}; warnings={'; '.join(result.warnings)}",
            warnings=list(result.warnings),
        )

    energies_eV = _energies_K_to_eV(result.e_total_K)
    atlas_input = from_arrays(
        structure=atoms,
        positions_cart=result.insertion_centres_A,
        energies_eV=energies_eV,
        temperature_K=float(temperature_K),
        gas=gas,
        metadata={
            "case_id": case_id,
            "framework_name": framework_name,
            "evaluator": "widom_atlas.evaluator (LJ+Wolf-Coulomb)",
            "n_insertions": int(result.n_insertions_used),
            "seed": int(seed),
            "r_cut_A": float(r_cut_A),
            "grid_mode": grid_mode,
            "KH_mol_per_kg_per_Pa": (
                float(result.KH_mol_per_kg_per_Pa)
                if result.KH_mol_per_kg_per_Pa is not None else None
            ),
            "Qads_kJ_per_mol": (
                float(result.Qads_kJ_per_mol)
                if result.Qads_kJ_per_mol is not None else None
            ),
            "structure_sha256": str(result.provenance.get("framework_sha256", "")),
        },
    )

    case_dir = out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    params = PipelineParams(
        n_grid=n_grid,
        smoothing_sigma_A=smoothing_sigma_A,
        dbscan_eps_A=dbscan_eps_A,
        min_samples=min_samples,
        temperature_K=float(temperature_K),
    )
    try:
        atlas_result = run_atlas(atlas_input, params, case_dir, structure=atoms)
        pipeline_completed = True
        n_basins = len(atlas_result.basins)
        dom_weight: float | None
        dom_centroid_frac: tuple[float, float, float] | None
        dom_centroid_A: tuple[float, float, float] | None
        if atlas_result.basins:
            dom = max(atlas_result.basins, key=lambda b: b.weight)
            dom_weight = float(dom.weight)
            dom_centroid_frac = (
                float(dom.centroid_frac[0]),
                float(dom.centroid_frac[1]),
                float(dom.centroid_frac[2]),
            )
            dom_centroid_A = (
                float(dom.centroid_cart_A[0]),
                float(dom.centroid_cart_A[1]),
                float(dom.centroid_cart_A[2]),
            )
        else:
            dom_weight = None
            dom_centroid_frac = None
            dom_centroid_A = None
        notes = "atlas pipeline ran end-to-end (samples → density → basins → reports)"
    except Exception as exc:
        pipeline_completed = False
        n_basins = 0
        dom_weight = None
        dom_centroid_frac = None
        dom_centroid_A = None
        notes = f"atlas pipeline crashed: {exc}"

    return EndToEndCaseResult(
        case_id=case_id,
        framework_name=framework_name,
        gas=gas,
        temperature_K=temperature_K,
        structure_path=str(structure_path),
        upf_path=str(upf_path),
        n_insertions_used=int(result.n_insertions_used),
        KH_mol_per_kg_per_Pa=(
            float(result.KH_mol_per_kg_per_Pa)
            if result.KH_mol_per_kg_per_Pa is not None else None
        ),
        log10_KH=_safe_log10(result.KH_mol_per_kg_per_Pa),
        Qads_kJ_per_mol=(
            float(result.Qads_kJ_per_mol)
            if result.Qads_kJ_per_mol is not None else None
        ),
        e_lj_K_mean=float(result.e_lj_K.mean()) if result.e_lj_K.size else None,
        e_coul_K_mean=float(result.e_coul_K.mean()) if result.e_coul_K.size else None,
        atlas_run_dir=str(case_dir),
        n_basins=n_basins,
        dominant_basin_weight=dom_weight,
        dominant_basin_centroid_frac=dom_centroid_frac,
        dominant_basin_centroid_A=dom_centroid_A,
        pipeline_completed=pipeline_completed,
        notes=notes,
        warnings=list(result.warnings),
    )


def _missing_result(
    case_id: str,
    framework_name: str,
    gas: str,
    temperature_K: float,
    structure_path: Path,
    upf_path: Path,
    note: str,
) -> EndToEndCaseResult:
    return EndToEndCaseResult(
        case_id=case_id,
        framework_name=framework_name,
        gas=gas,
        temperature_K=temperature_K,
        structure_path=str(structure_path),
        upf_path=str(upf_path),
        n_insertions_used=0,
        KH_mol_per_kg_per_Pa=None,
        log10_KH=None,
        Qads_kJ_per_mol=None,
        e_lj_K_mean=None,
        e_coul_K_mean=None,
        atlas_run_dir="",
        n_basins=0,
        dominant_basin_weight=None,
        dominant_basin_centroid_frac=None,
        dominant_basin_centroid_A=None,
        pipeline_completed=False,
        notes=note,
        warnings=[],
    )


def write_end_to_end_results(results: list[EndToEndCaseResult], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            payload: dict[str, Any] = asdict(r)
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
    return out_path


__all__ = [
    "ALLOWED_ATLAS_GASES",
    "COMPONENT_FACTORIES",
    "K_TO_EV",
    "EndToEndCaseResult",
    "run_one_end_to_end",
    "write_end_to_end_results",
]
