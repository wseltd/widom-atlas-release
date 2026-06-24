"""Per-case Widom validation runner.

Each "case" = (framework, gas, temperature, FF, charges, n_insertions, seed).
The runner:

1. Locates the framework structure file (registry-resolved or supplied path)
2. Loads the UserParameterFile for this (FF, charges) pair
3. Runs the evaluator
4. Compares the result scalar to a registry-resolved reference (NIST ISODB,
   MOFX, CRAFTED) where one is available
5. Emits a single ``CaseResult`` row consumed by the table generator

Provenance pieces collected:
- registry dataset_id and SHA-256 of the structure
- UserParameterFile MD5 / SHA-256 (the runner just records the JSON sha256)
- evaluator commit / version
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from widom_atlas.backends.user_parameterised import UserParameterFile
from widom_atlas.evaluator.component import (
    Component,
    ch4_trappe_ua,
    co2_garcia_sanchez_2009,
    kr_lj,
    n2_trappe,
)
from widom_atlas.evaluator.runner import WidomResult, load_atoms, run_widom_evaluator

CaseStatusLiteral = Literal[
    "ok",
    "structure_missing",
    "ff_missing",
    "evaluator_failed",
    "reference_missing",
    "passed_no_reference",
    "skipped",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    framework_name: str
    structure_path: Path
    gas: str
    temperature_K: float
    user_parameter_file_path: Path
    n_insertions: int
    seed: int
    r_cut_A: float
    grid_mode: Literal["deterministic_uniform", "stochastic_uniform"]
    tier: Literal["flagship", "broad", "exploratory"]
    reference_KH_mol_per_kg_per_Pa: float | None = None
    reference_Qads_kJ_per_mol: float | None = None
    reference_doi: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    tier: str
    status: CaseStatusLiteral
    framework_name: str
    gas: str
    temperature_K: float
    n_insertions_used: int
    KH_internal_mol_per_kg_per_Pa: float | None
    log10_KH_internal: float | None
    log10_KH_reference: float | None
    delta_log10_KH: float | None
    Qads_internal_kJ_per_mol: float | None
    Qads_reference_kJ_per_mol: float | None
    delta_Qads_kJ_per_mol: float | None
    threshold_log10_KH: float
    threshold_Qads_kJ_per_mol: float
    pass_log10_KH: bool
    pass_Qads: bool
    pass_overall: bool
    framework_sha256: str
    upf_sha256: str
    reference_doi: str | None
    notes: str
    warnings: list[str] = field(default_factory=list)


COMPONENT_FACTORIES: dict[str, Any] = {
    "CO2": co2_garcia_sanchez_2009,
    "N2": n2_trappe,
    "CH4": ch4_trappe_ua,
    "Kr": kr_lj,
}


THRESHOLDS_BY_TIER: dict[str, tuple[float, float]] = {
    # (delta log10 K_H, delta Q_ads kJ/mol) — strict per the v0.4 follow-up brief
    # and read-this-too.md "Recommended validation thresholds"
    "flagship": (0.10, 2.0),
    "broad": (0.20, 4.0),
    "exploratory": (0.40, 7.0),
}


def _safe_log10(x: float | None) -> float | None:
    if x is None or x <= 0:
        return None
    return math.log10(x)


def _sha256(p: Path) -> str:
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_case(spec: CaseSpec) -> CaseResult:
    thresh_log, thresh_Q = THRESHOLDS_BY_TIER[spec.tier]
    if not spec.structure_path.exists():
        return _missing_result(spec, "structure_missing", "structure file not found", thresh_log, thresh_Q)
    if not spec.user_parameter_file_path.exists():
        return _missing_result(spec, "ff_missing", "UserParameterFile not found", thresh_log, thresh_Q)
    factory = COMPONENT_FACTORIES.get(spec.gas)
    if factory is None:
        return _missing_result(
            spec, "evaluator_failed", f"no Component factory for gas {spec.gas!r}", thresh_log, thresh_Q
        )

    upf = UserParameterFile.model_validate_json(spec.user_parameter_file_path.read_text(encoding="utf-8"))
    component: Component = factory()
    try:
        atoms = load_atoms(spec.structure_path)
    except Exception as exc:
        return _missing_result(
            spec, "evaluator_failed", f"ASE failed to read structure: {exc}", thresh_log, thresh_Q
        )

    res: WidomResult = run_widom_evaluator(
        atoms=atoms,
        framework_name=spec.framework_name,
        user_parameter_file=upf,
        component=component,
        temperature_K=spec.temperature_K,
        n_insertions=spec.n_insertions,
        seed=spec.seed,
        r_cut_A=spec.r_cut_A,
        grid_mode=spec.grid_mode,
        framework_source_path=spec.structure_path,
    )

    if res.status != "ok":
        return CaseResult(
            case_id=spec.case_id,
            tier=spec.tier,
            status="evaluator_failed",
            framework_name=spec.framework_name,
            gas=spec.gas,
            temperature_K=spec.temperature_K,
            n_insertions_used=res.n_insertions_used,
            KH_internal_mol_per_kg_per_Pa=res.KH_mol_per_kg_per_Pa,
            log10_KH_internal=_safe_log10(res.KH_mol_per_kg_per_Pa),
            log10_KH_reference=_safe_log10(spec.reference_KH_mol_per_kg_per_Pa),
            delta_log10_KH=None,
            Qads_internal_kJ_per_mol=res.Qads_kJ_per_mol,
            Qads_reference_kJ_per_mol=spec.reference_Qads_kJ_per_mol,
            delta_Qads_kJ_per_mol=None,
            threshold_log10_KH=thresh_log,
            threshold_Qads_kJ_per_mol=thresh_Q,
            pass_log10_KH=False,
            pass_Qads=False,
            pass_overall=False,
            framework_sha256=_sha256(spec.structure_path),
            upf_sha256=_sha256(spec.user_parameter_file_path),
            reference_doi=spec.reference_doi,
            notes=f"evaluator returned status={res.status}: {';'.join(res.warnings)}",
            warnings=res.warnings,
        )

    log_int = _safe_log10(res.KH_mol_per_kg_per_Pa)
    log_ref = _safe_log10(spec.reference_KH_mol_per_kg_per_Pa)
    delta_log = (
        abs(log_int - log_ref) if log_int is not None and log_ref is not None else None
    )
    delta_Q = (
        abs(res.Qads_kJ_per_mol - spec.reference_Qads_kJ_per_mol)
        if res.Qads_kJ_per_mol is not None and spec.reference_Qads_kJ_per_mol is not None
        else None
    )
    if log_ref is None and spec.reference_Qads_kJ_per_mol is None:
        return CaseResult(
            case_id=spec.case_id,
            tier=spec.tier,
            status="passed_no_reference",
            framework_name=spec.framework_name,
            gas=spec.gas,
            temperature_K=spec.temperature_K,
            n_insertions_used=res.n_insertions_used,
            KH_internal_mol_per_kg_per_Pa=res.KH_mol_per_kg_per_Pa,
            log10_KH_internal=log_int,
            log10_KH_reference=None,
            delta_log10_KH=None,
            Qads_internal_kJ_per_mol=res.Qads_kJ_per_mol,
            Qads_reference_kJ_per_mol=None,
            delta_Qads_kJ_per_mol=None,
            threshold_log10_KH=thresh_log,
            threshold_Qads_kJ_per_mol=thresh_Q,
            pass_log10_KH=False,
            pass_Qads=False,
            pass_overall=False,
            framework_sha256=_sha256(spec.structure_path),
            upf_sha256=_sha256(spec.user_parameter_file_path),
            reference_doi=spec.reference_doi,
            notes="evaluator ran but no reference scalar was supplied",
            warnings=res.warnings,
        )

    pass_log = bool(delta_log is not None and delta_log <= thresh_log)
    pass_Q = bool(delta_Q is not None and delta_Q <= thresh_Q)
    return CaseResult(
        case_id=spec.case_id,
        tier=spec.tier,
        status="ok",
        framework_name=spec.framework_name,
        gas=spec.gas,
        temperature_K=spec.temperature_K,
        n_insertions_used=res.n_insertions_used,
        KH_internal_mol_per_kg_per_Pa=res.KH_mol_per_kg_per_Pa,
        log10_KH_internal=log_int,
        log10_KH_reference=log_ref,
        delta_log10_KH=delta_log,
        Qads_internal_kJ_per_mol=res.Qads_kJ_per_mol,
        Qads_reference_kJ_per_mol=spec.reference_Qads_kJ_per_mol,
        delta_Qads_kJ_per_mol=delta_Q,
        threshold_log10_KH=thresh_log,
        threshold_Qads_kJ_per_mol=thresh_Q,
        pass_log10_KH=pass_log,
        pass_Qads=pass_Q,
        pass_overall=pass_log and pass_Q,
        framework_sha256=_sha256(spec.structure_path),
        upf_sha256=_sha256(spec.user_parameter_file_path),
        reference_doi=spec.reference_doi,
        notes=spec.notes,
        warnings=res.warnings,
    )


def _missing_result(
    spec: CaseSpec, status: CaseStatusLiteral, note: str, thresh_log: float, thresh_Q: float
) -> CaseResult:
    return CaseResult(
        case_id=spec.case_id,
        tier=spec.tier,
        status=status,
        framework_name=spec.framework_name,
        gas=spec.gas,
        temperature_K=spec.temperature_K,
        n_insertions_used=0,
        KH_internal_mol_per_kg_per_Pa=None,
        log10_KH_internal=None,
        log10_KH_reference=_safe_log10(spec.reference_KH_mol_per_kg_per_Pa),
        delta_log10_KH=None,
        Qads_internal_kJ_per_mol=None,
        Qads_reference_kJ_per_mol=spec.reference_Qads_kJ_per_mol,
        delta_Qads_kJ_per_mol=None,
        threshold_log10_KH=thresh_log,
        threshold_Qads_kJ_per_mol=thresh_Q,
        pass_log10_KH=False,
        pass_Qads=False,
        pass_overall=False,
        framework_sha256=_sha256(spec.structure_path) if spec.structure_path.exists() else "",
        upf_sha256=_sha256(spec.user_parameter_file_path) if spec.user_parameter_file_path.exists() else "",
        reference_doi=spec.reference_doi,
        notes=note,
    )


def write_case_results_jsonl(results: list[CaseResult], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.__dict__, sort_keys=True) + "\n")
    return out_path


__all__ = [
    "COMPONENT_FACTORIES",
    "THRESHOLDS_BY_TIER",
    "CaseResult",
    "CaseSpec",
    "run_case",
    "write_case_results_jsonl",
]
