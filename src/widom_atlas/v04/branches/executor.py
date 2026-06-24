"""T032: locked_strict branch executor with multi-T multi-seed orchestration.

For each verdict-affecting strict branch:
1. If the branch is BLOCKED (1a Lin/Mercado, 1b Dzubak — no native RASPA3
   pair-potential JSON form), emit a BLOCKED result.
2. Otherwise: resolve fixture, run RASPA3 v3.0.29 at >=3 seeds for the
   reference temperature T_ref and at >=3 seeds for T_ref + 25 K.
   Aggregate K_H over seeds; derive Q_st via two-point van't Hoff.
3. Return a structured result carrying full evidence (per-run hashes,
   per-seed K_H, derived Q_st, manifest of seeds + insertions).
"""
from __future__ import annotations

import concurrent.futures
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cif_fixtures import resolve_fixture
from ..raspa3.input_writer import write_raspa_inputs
from ..raspa3.output_parser import RaspaParsedScalars, parse_raspa3_output
from ..raspa3.runner import run_raspa3
from ..units import KH_mol_per_kg_per_bar_to_mol_per_kg_per_Pa
from ..widom.vant_hoff import vant_hoff_two_point
from .blocked_branches import blocked_reason
from .dispatcher import BranchSpec


@dataclass
class PerRunRecord:
    temperature_K: float
    seed_label: str
    work_dir: Path
    exit_code: int
    duration_s: float
    K_H_mol_per_kg_per_Pa: float | None
    K_H_uncertainty: float | None
    widom_insertions_total: int | None
    excess_chemical_potential_K: float | None
    raspa_evidence: dict[str, Any]


@dataclass
class LockedStrictResult:
    case_id: str
    branch_id: str
    status: str  # "ran", "blocked", "raspa_error", "error"
    work_dir: Path
    # Aggregated scalars (over seeds at T_ref)
    K_H_mol_per_kg_per_Pa: float | None
    K_H_uncertainty_seedmean: float | None
    Q_st_kJ_per_mol: float | None
    Q_st_uncertainty_kJ_per_mol: float | None
    Q_st_method: str | None
    # Sample-size + seed manifest
    seeds: int
    insertions_per_seed: int | None
    aggregate_insertions: int | None
    # Per-run breakdown
    runs: list[PerRunRecord] = field(default_factory=list)
    # Reference + evidence
    reference: dict[str, Any] = field(default_factory=dict)
    blocked_info: dict[str, str] | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    # Provenance from input_writer
    derived_charges: dict[str, float] | None = None
    derivation_notes: list[str] | None = None
    binary_interactions: list[dict] | None = None
    # parsed (single representative — at T_ref, seed_0) for back-compat with verdict emitter
    parsed: RaspaParsedScalars | None = None
    raspa_evidence: dict[str, Any] = field(default_factory=dict)
    raspa_exit_code: int = 0
    raspa_duration_s: float = 0.0


def _resolve_temperature_K(branch_raw: dict) -> float:
    """Read the reference temperature with the proper precedence.

    Order: branch.temperature_K -> branch.references.K_H.temperature_K -> 298.0.
    Bug C fix: 6d explicitly carries temperature_K=87.0 at branch level even
    though it has no references block (numerical-only).
    """
    if "temperature_K" in branch_raw and branch_raw["temperature_K"] is not None:
        return float(branch_raw["temperature_K"])
    refs = branch_raw.get("references") or {}
    kh = refs.get("K_H") or {}
    return float(kh.get("temperature_K") or 298.0)


def _execute_single_run(
    branch_raw: dict, cif_abs_path: Path, work_dir: Path,
    temperature_K: float, seed_label: str, n_cycles: int, repo_root: Path,
) -> tuple[PerRunRecord, dict[str, float] | None, list[str] | None, list[dict] | None]:
    """Run RASPA3 once in `work_dir`, parse, return record + writer provenance."""
    bundle = write_raspa_inputs(
        work_dir=work_dir, branch=branch_raw, cif_abs_path=cif_abs_path,
        temperature_K=temperature_K, n_cycles=n_cycles, repo_root=repo_root,
    )
    run = run_raspa3(bundle=bundle)
    parsed: RaspaParsedScalars
    if run.output_txt_path is not None:
        parsed = parse_raspa3_output(run.output_txt_path)
    else:
        parsed = RaspaParsedScalars(None, None, None, None, None, None, None, {})
    mu_ex_raw = parsed.raw_lines.get("excess_mu_K_value")
    mu_ex_K: float | None
    try:
        mu_ex_K = float(mu_ex_raw) if mu_ex_raw is not None else None
    except (TypeError, ValueError):
        mu_ex_K = None
    rec = PerRunRecord(
        temperature_K=temperature_K,
        seed_label=seed_label,
        work_dir=work_dir,
        exit_code=run.exit_code,
        duration_s=run.duration_s,
        K_H_mol_per_kg_per_Pa=parsed.K_H_mol_per_kg_per_Pa,
        K_H_uncertainty=parsed.K_H_uncertainty,
        widom_insertions_total=parsed.widom_insertions_total,
        excess_chemical_potential_K=mu_ex_K,
        raspa_evidence={
            "raspa3_version": run.raspa3_version,
            "raspa3_sha256": run.raspa3_sha256,
            "input_sha256": bundle.sha256,
            "output_txt_path": str(run.output_txt_path) if run.output_txt_path else None,
            "stdout_path": str(run.stdout_path),
            "stderr_path": str(run.stderr_path),
        },
    )
    return rec, bundle.derived_charges, bundle.derivation_notes, bundle.binary_interactions


def execute_locked_strict(
    branch: BranchSpec, repo_root: Path, evidence_root: Path,
    n_cycles: int,
    *,
    n_seeds: int = 3,
    enable_vant_hoff_Q_st: bool = True,
    vant_hoff_delta_T_K: float = 25.0,
    max_parallel_runs: int = 6,
) -> LockedStrictResult:
    """Orchestrate multi-T multi-seed for a single branch."""
    branch_id = branch.branch_id
    case_id = branch.case_id
    work_dir_branch = evidence_root / branch_id
    work_dir_branch.mkdir(parents=True, exist_ok=True)

    blocked = blocked_reason(branch_id)
    if blocked is not None:
        return LockedStrictResult(
            case_id=case_id, branch_id=branch_id, status="blocked",
            work_dir=work_dir_branch,
            K_H_mol_per_kg_per_Pa=None, K_H_uncertainty_seedmean=None,
            Q_st_kJ_per_mol=None, Q_st_uncertainty_kJ_per_mol=None, Q_st_method=None,
            seeds=0, insertions_per_seed=None, aggregate_insertions=None,
            blocked_info=blocked,
            notes=[
                f"BLOCKED: {blocked['reason']}",
                f"Prescribed: {blocked['prescribed_form']}",
                f"Required: {blocked['required_action']}",
            ],
            reference=_extract_reference_block(branch.raw, _resolve_temperature_K(branch.raw)),
        )

    framework = branch.raw.get("framework") or {}
    relpath = framework.get("source_cif_path")
    if not relpath:
        return LockedStrictResult(
            case_id=case_id, branch_id=branch_id, status="error", work_dir=work_dir_branch,
            K_H_mol_per_kg_per_Pa=None, K_H_uncertainty_seedmean=None,
            Q_st_kJ_per_mol=None, Q_st_uncertainty_kJ_per_mol=None, Q_st_method=None,
            seeds=0, insertions_per_seed=None, aggregate_insertions=None,
            error=f"no source_cif_path for {branch_id}",
        )
    try:
        fixture = resolve_fixture(repo_root, branch_id, relpath)
    except Exception as e:
        return LockedStrictResult(
            case_id=case_id, branch_id=branch_id, status="error", work_dir=work_dir_branch,
            K_H_mol_per_kg_per_Pa=None, K_H_uncertainty_seedmean=None,
            Q_st_kJ_per_mol=None, Q_st_uncertainty_kJ_per_mol=None, Q_st_method=None,
            seeds=0, insertions_per_seed=None, aggregate_insertions=None,
            error=f"fixture: {e}",
        )

    T_ref = _resolve_temperature_K(branch.raw)
    temperatures: list[float] = [T_ref]
    if enable_vant_hoff_Q_st:
        temperatures.append(T_ref + vant_hoff_delta_T_K)

    runs: list[PerRunRecord] = []
    derived_charges: dict[str, float] | None = None
    derivation_notes: list[str] | None = None
    binary_interactions: list[dict] | None = None

    # Build the job list (cartesian product of T x seed)
    jobs: list[tuple[float, str, Path]] = []
    for T in temperatures:
        for s in range(n_seeds):
            seed_label = f"T{T:g}_seed{s}"
            work_dir = work_dir_branch / seed_label
            jobs.append((T, seed_label, work_dir))

    # Run in parallel via ThreadPoolExecutor (each RASPA3 call is its own subprocess).
    def _job(T_seed_dir: tuple[float, str, Path]) -> tuple[PerRunRecord, dict | None, list | None, list | None]:
        T, seed_label, wd = T_seed_dir
        return _execute_single_run(
            branch_raw=branch.raw, cif_abs_path=fixture.abs_path,
            work_dir=wd, temperature_K=T, seed_label=seed_label,
            n_cycles=n_cycles, repo_root=repo_root,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_runs) as ex:
        for rec, dc, dn, bi in ex.map(_job, jobs):
            runs.append(rec)
            if dc and derived_charges is None:
                derived_charges = dc
            if dn and derivation_notes is None:
                derivation_notes = dn
            if bi and binary_interactions is None:
                binary_interactions = bi

    # Aggregate K_H at T_ref across seeds (only successful runs)
    runs_at_T_ref = [r for r in runs if abs(r.temperature_K - T_ref) < 1e-6 and r.K_H_mol_per_kg_per_Pa]
    K_H_values_ref = [r.K_H_mol_per_kg_per_Pa for r in runs_at_T_ref if r.K_H_mol_per_kg_per_Pa is not None]
    K_H_mean: float | None = statistics.fmean(K_H_values_ref) if K_H_values_ref else None
    K_H_std: float | None = statistics.stdev(K_H_values_ref) if len(K_H_values_ref) > 1 else None

    Q_st: float | None = None
    Q_st_unc: float | None = None
    Q_st_method: str | None = None
    if enable_vant_hoff_Q_st and len(temperatures) >= 2:
        T_high = max(temperatures)
        runs_at_T_high = [r for r in runs if abs(r.temperature_K - T_high) < 1e-6
                          and r.K_H_mol_per_kg_per_Pa]
        K_H_values_high = [r.K_H_mol_per_kg_per_Pa for r in runs_at_T_high
                           if r.K_H_mol_per_kg_per_Pa is not None]
        if K_H_values_high and K_H_values_ref and K_H_mean is not None:
            K_H_mean_high = statistics.fmean(K_H_values_high)
            # Per-temperature standard error on the mean (SEM = std / sqrt(N))
            K_H_low_unc = (
                statistics.stdev(K_H_values_ref) / (len(K_H_values_ref) ** 0.5)
                if len(K_H_values_ref) > 1 else None
            )
            K_H_high_unc = (
                statistics.stdev(K_H_values_high) / (len(K_H_values_high) ** 0.5)
                if len(K_H_values_high) > 1 else None
            )
            try:
                vh = vant_hoff_two_point(
                    K_H_low=K_H_mean, T_low_K=T_ref,
                    K_H_high=K_H_mean_high, T_high_K=T_high,
                    K_H_low_unc=K_H_low_unc, K_H_high_unc=K_H_high_unc,
                )
                Q_st = vh.Q_st_kJ_per_mol
                Q_st_unc = vh.Q_st_uncertainty_kJ_per_mol
                Q_st_method = vh.method
            except Exception as e:
                # If van't Hoff fails (e.g., K_H_high <= 0), leave Q_st null.
                _ = e

    # Sample-size manifest
    insertions_per_seed = None
    if runs_at_T_ref:
        per_seed_counts = [r.widom_insertions_total for r in runs_at_T_ref if r.widom_insertions_total]
        insertions_per_seed = min(per_seed_counts) if per_seed_counts else None
    aggregate_insertions = sum(
        (r.widom_insertions_total or 0) for r in runs
    ) or None

    # Determine status
    successes = [r for r in runs if r.exit_code == 0]
    if not successes:
        status = "raspa_error"
    elif len(successes) < len(runs):
        status = "partial"
    else:
        status = "ran"

    representative = runs[0] if runs else None
    raspa_evidence = representative.raspa_evidence if representative else {}

    return LockedStrictResult(
        case_id=case_id, branch_id=branch_id, status=status, work_dir=work_dir_branch,
        K_H_mol_per_kg_per_Pa=K_H_mean,
        K_H_uncertainty_seedmean=K_H_std,
        Q_st_kJ_per_mol=Q_st,
        Q_st_uncertainty_kJ_per_mol=Q_st_unc,
        Q_st_method=Q_st_method,
        seeds=len(runs_at_T_ref),
        insertions_per_seed=insertions_per_seed,
        aggregate_insertions=aggregate_insertions,
        runs=runs,
        reference=_extract_reference_block(branch.raw, T_ref),
        derived_charges=derived_charges,
        derivation_notes=derivation_notes,
        binary_interactions=binary_interactions,
        parsed=RaspaParsedScalars(
            K_H_mol_per_kg_per_Pa=K_H_mean,
            K_H_uncertainty=K_H_std,
            K_H_molec_per_uc_per_Pa=None,
            Q_st_kJ_per_mol=Q_st,
            Q_st_uncertainty=Q_st_unc,
            widom_insertions_total=insertions_per_seed,
            widom_runtime_s=None,
            raw_lines={},
        ),
        raspa_evidence=raspa_evidence,
        raspa_exit_code=representative.exit_code if representative else -1,
        raspa_duration_s=sum(r.duration_s for r in runs),
    )


def _extract_reference_block(branch_raw: dict, T_ref: float) -> dict[str, Any]:
    """Pull literature K_H + Q_st reference + acceptance windows from the YAML.

    The YAML uses two different Q_st schemas:
      - Schema A: {Q_st.value, Q_st.acceptance_window_min, Q_st.acceptance_window_max}
      - Schema B (2a HKUST-1): {Q_st.low_loading_value, Q_st.value_acceptance_range}
    Both are honoured.
    """
    refs_block = branch_raw.get("references") or {}
    kh_block = refs_block.get("K_H") or {}
    q_block = refs_block.get("Q_st") or {}

    K_H_ref_per_bar = kh_block.get("value")
    K_H_ref_per_Pa = (
        KH_mol_per_kg_per_bar_to_mol_per_kg_per_Pa(K_H_ref_per_bar) if K_H_ref_per_bar else None
    )

    # Q_st: accept either schema.
    q_value = q_block.get("value")
    if q_value is None:
        q_value = q_block.get("low_loading_value")
    q_min = q_block.get("acceptance_window_min")
    q_max = q_block.get("acceptance_window_max")
    if q_min is None or q_max is None:
        rng = q_block.get("value_acceptance_range")
        if isinstance(rng, list | tuple) and len(rng) == 2:
            q_min = q_min if q_min is not None else float(rng[0])
            q_max = q_max if q_max is not None else float(rng[1])

    return {
        "K_H_value_mol_per_kg_per_bar": K_H_ref_per_bar,
        "K_H_value_mol_per_kg_per_Pa": K_H_ref_per_Pa,
        "K_H_window_min": kh_block.get("acceptance_window_min"),
        "K_H_window_max": kh_block.get("acceptance_window_max"),
        "K_H_source": kh_block.get("source"),
        "K_H_temperature_K": T_ref,
        "Q_st_value_kj_per_mol": q_value,
        "Q_st_window_min": q_min,
        "Q_st_window_max": q_max,
        "Q_st_source": q_block.get("source"),
    }


def write_branch_manifest(result: LockedStrictResult) -> Path:
    """Write a per-branch manifest JSON aggregating runs, seeds, and Q_st."""
    manifest = {
        "branch_id": result.branch_id,
        "case_id": result.case_id,
        "status": result.status,
        "K_H_mol_per_kg_per_Pa": result.K_H_mol_per_kg_per_Pa,
        "K_H_seed_std": result.K_H_uncertainty_seedmean,
        "Q_st_kJ_per_mol": result.Q_st_kJ_per_mol,
        "Q_st_uncertainty_kJ_per_mol": result.Q_st_uncertainty_kJ_per_mol,
        "Q_st_method": result.Q_st_method,
        "seeds": result.seeds,
        "insertions_per_seed": result.insertions_per_seed,
        "aggregate_insertions": result.aggregate_insertions,
        "derived_charges": result.derived_charges,
        "derivation_notes": result.derivation_notes,
        "binary_interactions": result.binary_interactions,
        "blocked_info": result.blocked_info,
        "runs": [
            {
                "temperature_K": r.temperature_K,
                "seed_label": r.seed_label,
                "exit_code": r.exit_code,
                "duration_s": r.duration_s,
                "K_H_mol_per_kg_per_Pa": r.K_H_mol_per_kg_per_Pa,
                "K_H_uncertainty": r.K_H_uncertainty,
                "widom_insertions_total": r.widom_insertions_total,
                "raspa_evidence": r.raspa_evidence,
            }
            for r in result.runs
        ],
    }
    path = result.work_dir / "branch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
