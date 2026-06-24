"""Branch-1a executor using the RASPA2 backend.

Same shape as `branches/executor.py::execute_locked_strict` but
specialised for 1a (Mg-MOF-74 Lin/Mercado Buckingham) and using the
RASPA2 binary. Runs ≥3 seeds × 2 temperatures (van't Hoff for Q_st) and
emits the same `LockedStrictResult` shape so the verdict emitter is
re-usable.
"""
from __future__ import annotations

import concurrent.futures
import statistics
from pathlib import Path

from ..branches.executor import LockedStrictResult, PerRunRecord, _extract_reference_block
from ..raspa3.output_parser import RaspaParsedScalars
from ..widom.vant_hoff import vant_hoff_two_point
from .input_writer import write_raspa2_inputs
from .output_parser import parse_raspa2_output
from .raspa2_binary import DEFAULT_RASPA2_BIN, DEFAULT_RASPA2_SHARE, verify_raspa2_binary
from .runner import run_raspa2


def execute_1a_raspa2(
    branch_raw: dict,
    cif_abs_path: Path,
    lin_mercado_pkg_dir: Path,
    evidence_root: Path,
    n_cycles: int = 5000,
    n_seeds: int = 3,
    enable_vant_hoff: bool = True,
    vant_hoff_delta_T_K: float = 25.0,
    max_parallel_runs: int = 4,
    raspa2_bin: Path = DEFAULT_RASPA2_BIN,
    raspa2_share_dir: Path = DEFAULT_RASPA2_SHARE,
) -> LockedStrictResult:
    """Orchestrate RASPA2 multi-T multi-seed for 1a."""
    case_id = "1"
    branch_id = "1a"
    work_dir_branch = evidence_root / branch_id
    work_dir_branch.mkdir(parents=True, exist_ok=True)

    # Verify the binary up-front; raises if mismatched.
    verify_raspa2_binary(bin_path=raspa2_bin, share_dir=raspa2_share_dir)

    # Reference temperature: take from YAML K_H.temperature_K (default 298.0)
    refs = branch_raw.get("references") or {}
    T_ref = float((refs.get("K_H") or {}).get("temperature_K") or 298.0)
    temperatures: list[float] = [T_ref]
    if enable_vant_hoff:
        temperatures.append(T_ref + vant_hoff_delta_T_K)

    jobs: list[tuple[float, str, Path]] = []
    for T in temperatures:
        for s in range(n_seeds):
            seed_label = f"T{T:g}_seed{s}"
            jobs.append((T, seed_label, work_dir_branch / seed_label))

    def _job(args: tuple[float, str, Path]) -> PerRunRecord:
        T, seed_label, wd = args
        # RASPA2 defaults to a fixed RNG seed unless one is supplied. Generate a
        # deterministic-but-unique seed per (T, seed_label) so the audit is
        # reproducible AND the three "seeds" actually sample independently.
        seed_int = abs(hash(seed_label)) % (2 ** 31)
        bundle = write_raspa2_inputs(
            work_dir=wd, branch=branch_raw, cif_abs_path=cif_abs_path,
            lin_mercado_pkg_dir=lin_mercado_pkg_dir,
            temperature_K=T, n_cycles=n_cycles,
            raspa2_share_dir=raspa2_share_dir,
            random_seed=seed_int,
        )
        run = run_raspa2(bundle, raspa2_share_dir=raspa2_share_dir, raspa2_bin=raspa2_bin)
        if run.output_data_path is not None:
            parsed = parse_raspa2_output(run.output_data_path)
        else:
            parsed = type("p", (), {
                "K_H_mol_per_kg_per_Pa": None, "K_H_uncertainty": None,
                "Q_st_kJ_per_mol": None, "Q_st_uncertainty": None,
                "widom_insertions_total": None, "raw_lines": {},
            })()
        return PerRunRecord(
            temperature_K=T, seed_label=seed_label, work_dir=wd,
            exit_code=run.exit_code, duration_s=run.duration_s,
            K_H_mol_per_kg_per_Pa=parsed.K_H_mol_per_kg_per_Pa,
            K_H_uncertainty=parsed.K_H_uncertainty,
            widom_insertions_total=parsed.widom_insertions_total,
            excess_chemical_potential_K=None,
            raspa_evidence={
                "raspa2_version": run.raspa2_version,
                "raspa2_bin_sha256": run.raspa2_bin_sha256,
                "raspa2_lib_sha256": run.raspa2_lib_sha256,
                "input_sha256": bundle.sha256,
                "output_data_path": str(run.output_data_path) if run.output_data_path else None,
                "stdout_path": str(run.stdout_path),
                "stderr_path": str(run.stderr_path),
                "Q_st_widom_kJ_per_mol_per_run": parsed.Q_st_kJ_per_mol,  # direct Widom Q_st from RASPA2
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_runs) as ex:
        runs = list(ex.map(_job, jobs))

    # Aggregate K_H at T_ref over seeds
    runs_T_ref = [r for r in runs if abs(r.temperature_K - T_ref) < 1e-6 and r.K_H_mol_per_kg_per_Pa]
    K_H_values = [r.K_H_mol_per_kg_per_Pa for r in runs_T_ref if r.K_H_mol_per_kg_per_Pa is not None]
    K_H_mean = statistics.fmean(K_H_values) if K_H_values else None
    K_H_std = statistics.stdev(K_H_values) if len(K_H_values) > 1 else None

    # Two paths for Q_st:
    # (1) direct Widom Q_st from each RASPA2 run, averaged over seeds at T_ref (RASPA2 emits this directly).
    # (2) van't Hoff slope from K_H(T_low) and K_H(T_high) — same as the RASPA3 path.
    qst_widom_vals = [
        r.raspa_evidence.get("Q_st_widom_kJ_per_mol_per_run")
        for r in runs_T_ref
        if r.raspa_evidence.get("Q_st_widom_kJ_per_mol_per_run") is not None
    ]
    Q_st_widom_mean = statistics.fmean(qst_widom_vals) if qst_widom_vals else None
    Q_st_widom_unc = statistics.stdev(qst_widom_vals) if len(qst_widom_vals) > 1 else None

    Q_st_vh: float | None = None
    Q_st_vh_unc: float | None = None
    if enable_vant_hoff and len(temperatures) >= 2 and K_H_mean is not None:
        T_high = max(temperatures)
        K_H_high_vals = [
            r.K_H_mol_per_kg_per_Pa for r in runs
            if abs(r.temperature_K - T_high) < 1e-6 and r.K_H_mol_per_kg_per_Pa is not None
        ]
        if K_H_high_vals:
            K_H_high_mean = statistics.fmean(K_H_high_vals)
            sem_low = (
                statistics.stdev(K_H_values) / len(K_H_values) ** 0.5
                if len(K_H_values) > 1 else None
            )
            sem_high = (
                statistics.stdev(K_H_high_vals) / len(K_H_high_vals) ** 0.5
                if len(K_H_high_vals) > 1 else None
            )
            try:
                vh = vant_hoff_two_point(
                    K_H_low=K_H_mean, T_low_K=T_ref,
                    K_H_high=K_H_high_mean, T_high_K=T_high,
                    K_H_low_unc=sem_low, K_H_high_unc=sem_high,
                )
                Q_st_vh = vh.Q_st_kJ_per_mol
                Q_st_vh_unc = vh.Q_st_uncertainty_kJ_per_mol
            except Exception:
                Q_st_vh = None

    # Prefer the direct Widom Q_st (RASPA2 emits it natively); fall back to van't Hoff if needed.
    Q_st = Q_st_widom_mean if Q_st_widom_mean is not None else Q_st_vh
    Q_st_unc = Q_st_widom_unc if Q_st_widom_unc is not None else Q_st_vh_unc
    Q_st_method = (
        "direct_widom_from_RASPA2" if Q_st_widom_mean is not None else "two_point_van_t_Hoff"
    )

    insertions_per_seed = None
    if runs_T_ref:
        counts = [r.widom_insertions_total for r in runs_T_ref if r.widom_insertions_total]
        insertions_per_seed = min(counts) if counts else None
    aggregate_insertions = sum(r.widom_insertions_total or 0 for r in runs) or None

    successes = [r for r in runs if r.exit_code == 0]
    if not successes:
        status = "raspa2_error"
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
        seeds=len(runs_T_ref),
        insertions_per_seed=insertions_per_seed,
        aggregate_insertions=aggregate_insertions,
        runs=runs,
        reference=_extract_reference_block(branch_raw, T_ref),
        parsed=RaspaParsedScalars(
            K_H_mol_per_kg_per_Pa=K_H_mean, K_H_uncertainty=K_H_std,
            K_H_molec_per_uc_per_Pa=None,
            Q_st_kJ_per_mol=Q_st, Q_st_uncertainty=Q_st_unc,
            widom_insertions_total=insertions_per_seed,
            widom_runtime_s=None, raw_lines={},
        ),
        raspa_evidence={
            **raspa_evidence,
            "Q_st_widom_seed_mean": Q_st_widom_mean,
            "Q_st_widom_seed_std": Q_st_widom_unc,
            "Q_st_vant_hoff": Q_st_vh,
            "Q_st_vant_hoff_unc": Q_st_vh_unc,
            "Q_st_method": Q_st_method,
            "backend": "RASPA2_v2.0.50_Lin_Mercado_BUCKINGHAM2_hardcore_1A",
            "seeds": len(runs_T_ref),
            "insertions_per_seed": insertions_per_seed,
            "aggregate_insertions": aggregate_insertions,
            "K_H_seed_std_mol_per_kg_per_Pa": K_H_std,
        },
        raspa_exit_code=representative.exit_code if representative else -1,
        raspa_duration_s=sum(r.duration_s for r in runs),
    )
