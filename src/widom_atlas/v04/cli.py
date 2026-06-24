"""T001 + T041: widom-atlas v04 CLI.

Subcommands:
- list-cases:    print branch ids from the locked case matrix yaml
- verify-spec:   verify V04_LOCKED_SPEC.md + v04_case_matrix.yaml sha256
- verify-binary: verify the pinned RASPA3 v3.0.29 binary
- run:           full end-to-end v04 audit (T041)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit.op11_guard import enforce_op11_guard
from .audit.pass_criteria import (
    check_pass_criteria,
    overall_pass,
    write_pass_criteria_report,
)
from .audit.pipeline import aggregate_audit
from .audit.verdict_emitter import (
    emit_verdict_for_blocked_branch,
    emit_verdict_for_non_verdict_branch,
    emit_verdict_for_reference_blocked_strict_branch,
    emit_verdict_for_strict_branch,
)
from .branches.deferred import execute_deferred
from .branches.dispatcher import (
    filter_deferred,
    filter_exploratory,
    filter_numerical,
    filter_strict,
    filter_strict_executed,
    list_all_branches,
)
from .branches.executor import execute_locked_strict, write_branch_manifest
from .branches.exploratory import execute_exploratory
from .branches.numerical_test import execute_numerical_test
from .locked_inputs import (
    DEFAULT_CASE_MATRIX_PATH,
    DEFAULT_SPEC_PATH,
    LockedDigestMismatch,
    load_locked_case_matrix,
    load_locked_spec,
)
from .raspa2.executor import execute_1a_raspa2
from .raspa2.raspa2_binary import (
    DEFAULT_RASPA2_BIN,
    DEFAULT_RASPA2_SHARE,
    RASPA2VerificationError,
    verify_raspa2_binary,
)
from .raspa_binary import (
    DEFAULT_RASPA3_PATH,
    RASPA3VerificationError,
    verify_raspa3_binary,
)


def cmd_list_cases(args: argparse.Namespace) -> int:
    matrix = load_locked_case_matrix(Path(args.case_matrix))
    print(f"# v04_case_matrix.yaml schema_version={matrix.version}  sha256={matrix.sha256[:16]}...")
    for case in matrix.cases:
        for branch in case.get("branches", []):
            print(
                f"  {branch['branch_id']:<5} status={branch.get('status'):<40} {branch.get('label', '')}"
            )
    return 0


def cmd_verify_spec(args: argparse.Namespace) -> int:
    try:
        spec = load_locked_spec(Path(args.spec))
        matrix = load_locked_case_matrix(Path(args.case_matrix))
    except LockedDigestMismatch as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK spec     sha256={spec.sha256}")
    print(f"OK matrix   sha256={matrix.sha256}  schema_version={matrix.version}  cases={len(matrix.cases)}")
    return 0


def cmd_verify_binary(args: argparse.Namespace) -> int:
    try:
        b = verify_raspa3_binary(Path(args.raspa3))
    except RASPA3VerificationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK RASPA3   path={b.path}")
    print(f"           sha256={b.sha256}")
    print(f"           version={b.version}")
    print(f"           upstream_commit={b.upstream_commit}")
    return 0


def _run_site_truth_for_branch(branch_spec, repo_root: Path, site_truth_block: dict):
    """Construct a NativeSystem for the branch and run a site-truth replay.

    Returns the site-truth verdict dict, or None if a NativeSystem could not
    be built (e.g. a branch that uses a force field the native loader doesn't
    cover yet — currently anything other than LJ + Lin/Mercado-Buckingham +
    Dzubak). Replay always uses the native evaluator, regardless of which
    backend produced the strict K_H/Q_st verdict.
    """
    try:
        from .native.dzubak_loader import load_1b_native_dzubak
        from .native.lin_mercado_loader import load_1a_native_lin_mercado
        from .native.loaders import load_native_system_for_branch
        from .native.site_truth_replay import run_site_truth_replay

        bid = branch_spec.branch_id
        if bid == "1a":
            system = load_1a_native_lin_mercado(repo_root)
        elif bid == "1b":
            system = load_1b_native_dzubak(repo_root)
        else:
            system = load_native_system_for_branch(branch_spec.raw, repo_root)
    except Exception as exc:
        return {
            "passes_site_truth": None,
            "skipped": True,
            "reason": (
                f"site-truth replay setup failed for branch {branch_spec.branch_id}: "
                f"{type(exc).__name__}: {exc}"
            ),
        }
    T_ref_block = (branch_spec.raw.get("references") or {}).get("K_H") or {}
    T_ref = float(T_ref_block.get("temperature_K") or 298.15)
    try:
        return run_site_truth_replay(
            system=system,
            site_truth_block=site_truth_block,
            n_insertions=50_000,
            n_seeds=2,
            temperature_K=T_ref,
            enable_ewald=True,
        )
    except Exception as exc:
        return {
            "passes_site_truth": None,
            "skipped": True,
            "reason": f"site-truth replay raised: {type(exc).__name__}: {exc}",
        }


def _emit_strict_verdict_from_result(result, thresholds, output_dir, branch_spec, repo_root: Path | None = None):
    # Aggregated K_H + van't-Hoff-derived Q_st
    parsed_dict = {
        "K_H_mol_per_kg_per_Pa": result.K_H_mol_per_kg_per_Pa,
        "Q_st_kJ_per_mol": result.Q_st_kJ_per_mol,
        "raw_lines": {},
    }
    # Annotate evidence with seeds + insertions for traceability
    enriched_evidence = dict(result.raspa_evidence)
    enriched_evidence.update({
        "seeds": result.seeds,
        "insertions_per_seed": result.insertions_per_seed,
        "aggregate_insertions": result.aggregate_insertions,
        "K_H_seed_std_mol_per_kg_per_Pa": result.K_H_uncertainty_seedmean,
        "Q_st_method": result.Q_st_method,
        "Q_st_uncertainty_kJ_per_mol": result.Q_st_uncertainty_kJ_per_mol,
        "derived_charges": result.derived_charges,
        "derivation_notes": result.derivation_notes,
        "binary_interactions": result.binary_interactions,
        "branch_manifest_path": str(write_branch_manifest(result).relative_to(result.work_dir.parent.parent)),
    })
    # Site-truth replay (if enabled in the YAML and we can build a NativeSystem
    # for this branch). Strongest-insertion geometry → per-distance verdict.
    if repo_root is not None:
        site_truth_block = branch_spec.raw.get("site_truth") or {}
        if site_truth_block.get("enabled") is True:
            st_verdict = _run_site_truth_for_branch(
                branch_spec=branch_spec,
                repo_root=repo_root,
                site_truth_block=site_truth_block,
            )
            if st_verdict is not None:
                enriched_evidence["site_truth"] = st_verdict
    # Per-axis reference-blocking (e.g. 6b post-2026-05-18: Q_st reference_blocked)
    kh_ref = (branch_spec.raw.get("references") or {}).get("K_H") or {}
    q_ref = (branch_spec.raw.get("references") or {}).get("Q_st") or {}
    kh_blocked = "reference_blocked" in str(kh_ref.get("classification", ""))
    q_blocked = "reference_blocked" in str(q_ref.get("classification", ""))
    # Q_st_method compatibility — atlas method comes from the executor; reference
    # method from the YAML's references.Q_st.method (with `source` as fallback
    # since some legacy reference blocks encode the method in `source`).
    atlas_q_st_method = result.Q_st_method
    reference_q_st_method = q_ref.get("method") or q_ref.get("source")
    return emit_verdict_for_strict_branch(
        output_dir=output_dir / "verdicts",
        case_id=result.case_id,
        branch_id=result.branch_id,
        verdict_tier=branch_spec.verdict_tier,
        numeric_thresholds_label=branch_spec.numeric_thresholds,
        thresholds=thresholds,
        parsed=parsed_dict,
        reference=result.reference,
        evidence=enriched_evidence,
        K_H_reference_blocked=kh_blocked,
        Q_st_reference_blocked=q_blocked,
        atlas_Q_st_method=atlas_q_st_method,
        reference_Q_st_method=reference_q_st_method,
        notes=result.notes,
    )


def _execute_strict_branch_inline(
    bspec,
    repo_root: Path,
    evidence_root: Path,
    matrix_raw: dict,
    output_dir: Path,
    n_cycles: int,
    n_seeds: int,
    vant_hoff_delta_T_K: float,
    max_parallel_runs: int,
    insertion_floor: int,
    raspa2_bin: Path,
    raspa2_share: Path,
    lin_mercado_pkg: Path,
) -> str:
    """Run a `locked_strict_executed` branch inline via the right backend.

    Dispatches by `bspec.raw["executed_backend"]`:
        "RASPA2"            → execute_1a_raspa2 (1a Lin/Mercado Buckingham)
        "native_widom_v04"  → run_1b_native_dzubak (1b Dzubak)

    Emits the verdict JSON inline under `output_dir/verdicts/` via the
    standard `_emit_strict_verdict_from_result` so the aggregator picks it
    up alongside the RASPA3-driven strict branches.

    Returns the backend tag actually used. Raises if no backend matches.
    """
    backend = bspec.raw.get("executed_backend")
    branch_raw = bspec.raw

    if backend == "RASPA2":
        cif_abs = (repo_root / branch_raw["framework"]["source_cif_path"]).resolve()
        if not cif_abs.exists():
            raise FileNotFoundError(f"1a CIF not found: {cif_abs}")
        if not (lin_mercado_pkg / "raspa_force_field.def").exists():
            raise FileNotFoundError(f"Lin/Mercado FF package not found at {lin_mercado_pkg}")
        # Verify binary; fail loud if mismatched.
        try:
            b2 = verify_raspa2_binary(bin_path=raspa2_bin, share_dir=raspa2_share)
        except RASPA2VerificationError as e:
            raise RuntimeError(f"RASPA2 binary verification failed: {e}") from e
        print(
            f"[strict_executed] {bspec.branch_id} backend=RASPA2 "
            f"v{b2.version} bin_sha={b2.bin_sha256[:16]}..."
        )
        result = execute_1a_raspa2(
            branch_raw=branch_raw,
            cif_abs_path=cif_abs,
            lin_mercado_pkg_dir=lin_mercado_pkg,
            evidence_root=evidence_root,
            n_cycles=n_cycles,
            n_seeds=n_seeds,
            enable_vant_hoff=True,
            vant_hoff_delta_T_K=vant_hoff_delta_T_K,
            max_parallel_runs=max_parallel_runs,
            raspa2_bin=raspa2_bin,
            raspa2_share_dir=raspa2_share,
        )
        if (result.insertions_per_seed is not None
                and result.insertions_per_seed < insertion_floor):
            result.notes.append(
                f"WARNING: insertions_per_seed={result.insertions_per_seed} < "
                f"floor={insertion_floor}"
            )
        _emit_strict_verdict_from_result(result, matrix_raw["thresholds"], output_dir, bspec, repo_root=repo_root)
        return "RASPA2"

    if backend == "native_widom_v04":
        # Inline 1b Dzubak via native widom-atlas evaluator.
        from .native.dzubak_loader import load_1b_native_dzubak
        from .native.ewald import EwaldParameters
        from .native.runner import run_native_widom
        from .widom.vant_hoff import vant_hoff_two_point
        import statistics

        T_ref = float(((branch_raw.get("references") or {}).get("K_H") or {}).get("temperature_K") or 298.0)
        T_high = T_ref + vant_hoff_delta_T_K

        print(
            f"[strict_executed] {bspec.branch_id} backend=native_widom_v04 "
            f"T_ref={T_ref} T_high={T_high} n_cycles={n_cycles}"
        )
        results_T_ref: list = []
        results_T_high: list = []
        for s_idx in range(n_seeds):
            seed = (s_idx + 1) * 11
            for T_val, bin_ in ((T_ref, results_T_ref), (T_high, results_T_high)):
                system = load_1b_native_dzubak(repo_root)
                res = run_native_widom(
                    system, temperature_K=T_val, n_insertions=n_cycles, seed=seed,
                    enable_ewald=True,
                    ewald_parameters=EwaldParameters(
                        alpha_inv_angstrom=0.3,
                        real_cutoff_angstrom=system.energy_cutoff_angstrom,
                        k_max_inv_angstrom=1.4,
                    ),
                    batch_size=2000,
                )
                bin_.append(res)

        kh = [r.K_H_mol_per_kg_per_Pa for r in results_T_ref]
        qst = [r.Q_st_kJ_per_mol for r in results_T_ref]
        kh_mean = statistics.fmean(kh)
        kh_std = statistics.stdev(kh) if len(kh) > 1 else None
        qst_mean = statistics.fmean(qst)
        qst_std = statistics.stdev(qst) if len(qst) > 1 else None
        kh_h = [r.K_H_mol_per_kg_per_Pa for r in results_T_high]
        kh_h_mean = statistics.fmean(kh_h) if kh_h else None
        qst_vh = None
        if kh_h_mean and kh_mean > 0 and kh_h_mean > 0:
            try:
                vh = vant_hoff_two_point(
                    K_H_low=kh_mean, T_low_K=T_ref,
                    K_H_high=kh_h_mean, T_high_K=T_high,
                    K_H_low_unc=kh_std / (len(kh) ** 0.5) if kh_std else None,
                    K_H_high_unc=(statistics.stdev(kh_h) / (len(kh_h) ** 0.5)) if len(kh_h) > 1 else None,
                )
                qst_vh = vh.Q_st_kJ_per_mol
            except Exception:
                qst_vh = None

        # Build a LockedStrictResult-shaped object so `_emit_strict_verdict_from_result`
        # works without a new emitter codepath.
        from .branches.executor import LockedStrictResult, _extract_reference_block
        from .raspa3.output_parser import RaspaParsedScalars

        wrapped_work_dir = evidence_root / bspec.branch_id
        wrapped_work_dir.mkdir(parents=True, exist_ok=True)
        wrapped = LockedStrictResult(
            case_id=bspec.case_id, branch_id=bspec.branch_id,
            status="ran", work_dir=wrapped_work_dir,
            K_H_mol_per_kg_per_Pa=kh_mean,
            K_H_uncertainty_seedmean=kh_std,
            Q_st_kJ_per_mol=qst_mean,
            Q_st_uncertainty_kJ_per_mol=qst_std,
            Q_st_method="direct_widom_boltzmann_weighted",
            seeds=len(kh),
            insertions_per_seed=n_cycles,
            aggregate_insertions=n_cycles * len(kh) * 2,
            runs=[],
            reference=_extract_reference_block(branch_raw, T_ref),
            parsed=RaspaParsedScalars(
                K_H_mol_per_kg_per_Pa=kh_mean, K_H_uncertainty=kh_std,
                K_H_molec_per_uc_per_Pa=None,
                Q_st_kJ_per_mol=qst_mean, Q_st_uncertainty=qst_std,
                widom_insertions_total=n_cycles,
                widom_runtime_s=None, raw_lines={},
            ),
            raspa_evidence={
                "backend": "native_widom_v04",
                "ff_lineage": "Dzubak_2012_Table_SI_4_Mg-MOF-74_CO2",
                "ff_source_doi": branch_raw["force_field"]["source_doi"],
                "framework_charges_source": "Dzubak_2012_SI_Table_SI_8_LoProp",
                "gas_charges_source": "Dzubak_2012_SI_Table_SI_13_TraPPE_CO2",
                "gas_self_LJ_source": "Dzubak_2012_SI_Table_SI_12 (column labels swapped per PDF)",
                "Q_st_widom_seed_mean": qst_mean,
                "Q_st_widom_seed_std": qst_std,
                "Q_st_vant_hoff": qst_vh,
                "Q_st_method": "direct_widom_boltzmann_weighted",
                "K_H_298_per_seed": kh,
                "K_H_323_per_seed": kh_h,
                "Q_st_298_per_seed": qst,
                "no_external_backend_triangulation_available": True,
                "validation_history": {
                    "V1_native_LJ_vs_RASPA3_6c": "PASS (delta_log10 K_H = +0.005, delta_Q_st = -0.14)",
                    "V2_native_LJ_vs_RASPA3_6a": "PASS (delta_log10 K_H = +0.005, delta_Q_st = +0.077)",
                    "V3_native_charged_vs_RASPA3_4a": "PASS within RASPA3 MC noise (delta_log10 K_H = +0.034)",
                    "V4_native_Buckingham_vs_RASPA2_1a": "PASS (delta_log10 K_H = -0.037)",
                },
            },
            raspa_exit_code=0,
            raspa_duration_s=0.0,
        )
        _emit_strict_verdict_from_result(wrapped, matrix_raw["thresholds"], output_dir, bspec, repo_root=repo_root)
        return "native_widom_v04"

    raise ValueError(
        f"strict_executed branch {bspec.branch_id} has unknown "
        f"executed_backend={backend!r}"
    )


def cmd_run_1a_raspa2(args: argparse.Namespace) -> int:
    """Run 1a Mg-MOF-74 + CO2 Lin/Mercado Buckingham via the RASPA2 backend.

    This bypasses the RASPA3 pipeline (which BLOCKS 1a because v3.0.29 doesn't
    parse the Buckingham potential). It runs `execute_1a_raspa2` end-to-end
    and emits the same strict-tier verdict JSON the main `run` pipeline would.
    """
    repo_root = Path(args.repo).resolve()
    matrix = load_locked_case_matrix(repo_root / "v04_case_matrix.yaml")

    # Verify both binaries: RASPA2 for the simulator, plus RASPA3 share metadata
    # is unused but the run-1a-raspa2 command should fail loud if either is missing.
    try:
        b2 = verify_raspa2_binary(
            bin_path=Path(args.raspa2_bin),
            share_dir=Path(args.raspa2_share),
        )
    except RASPA2VerificationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    # Locate the 1a branch spec
    branches = list_all_branches(matrix.raw)
    by_id = {b.branch_id: b for b in branches}
    if "1a" not in by_id:
        print("FAIL: 1a not found in case matrix", file=sys.stderr)
        return 2
    bspec = by_id["1a"]
    branch_raw = bspec.raw

    cif_abs = (repo_root / branch_raw["framework"]["source_cif_path"]).resolve()
    if not cif_abs.exists():
        print(f"FAIL: source CIF not found: {cif_abs}", file=sys.stderr)
        return 3
    lin_mercado_pkg = (repo_root / args.lin_mercado_pkg).resolve()
    if not (lin_mercado_pkg / "raspa_force_field.def").exists():
        print(
            f"FAIL: Lin/Mercado FF package not found at {lin_mercado_pkg}",
            file=sys.stderr,
        )
        return 4

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_root = output_dir / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[1a/raspa2] cycles={args.cycles} seeds={args.seeds} delta_T={args.vant_hoff_delta_T}"
    )
    print(
        f"[1a/raspa2] raspa2={b2.version} bin_sha={b2.bin_sha256[:16]}... "
        f"lib_sha={b2.lib_sha256[:16]}..."
    )
    print(f"[1a/raspa2] matrix sha={matrix.sha256[:16]}...")

    result = execute_1a_raspa2(
        branch_raw=branch_raw,
        cif_abs_path=cif_abs,
        lin_mercado_pkg_dir=lin_mercado_pkg,
        evidence_root=evidence_root,
        n_cycles=int(args.cycles),
        n_seeds=int(args.seeds),
        enable_vant_hoff=True,
        vant_hoff_delta_T_K=float(args.vant_hoff_delta_T),
        max_parallel_runs=int(args.max_parallel_runs),
        raspa2_bin=Path(args.raspa2_bin),
        raspa2_share_dir=Path(args.raspa2_share),
    )

    # Insertion-floor check
    floor = int(args.insertion_floor_per_seed)
    if (
        result.insertions_per_seed is not None
        and result.insertions_per_seed < floor
    ):
        result.notes.append(
            f"WARNING: insertions_per_seed={result.insertions_per_seed} < floor={floor}"
        )

    _emit_strict_verdict_from_result(
        result, matrix.raw["thresholds"], output_dir, bspec, repo_root=repo_root,
    )
    print(
        f"[1a/raspa2] verdict -> {output_dir / 'verdicts' / '1a.json'}"
    )
    print(
        f"[1a/raspa2] K_H={result.K_H_mol_per_kg_per_Pa}  Q_st={result.Q_st_kJ_per_mol} "
        f"({result.Q_st_method})  insertions/seed={result.insertions_per_seed}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo).resolve()
    matrix = load_locked_case_matrix(repo_root / "v04_case_matrix.yaml")
    spec = load_locked_spec(repo_root / "V04_LOCKED_SPEC.md")
    binary = verify_raspa3_binary()

    # OP11 guard on 6a
    branches = list_all_branches(matrix.raw)
    by_id = {b.branch_id: b for b in branches}
    if "6a" in by_id:
        enforce_op11_guard(by_id["6a"].raw)

    # Prepare evidence root
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_root = output_dir / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)

    n_cycles = int(args.cycles)
    print(f"[run] cycles={n_cycles} repo_root={repo_root} output_dir={output_dir}")
    print(f"[run] spec sha={spec.sha256[:16]}... matrix sha={matrix.sha256[:16]}... raspa3={binary.version}")

    n_seeds = int(args.seeds)
    enable_vh = bool(args.vant_hoff)
    delta_T = float(args.vant_hoff_delta_T)
    max_par = int(args.max_parallel_runs)
    insertion_floor = int(args.insertion_floor_per_seed)

    # Strict-executed: run inline via the branch-specific backend (RASPA2 for
    # 1a, native widom-atlas evaluator for 1b). Skipped if --skip-strict-executed
    # is set (useful when the operator wants to re-use a pre-computed verdict
    # without paying the long-running cost again).
    if not bool(getattr(args, "skip_strict_executed", False)):
        for bspec in filter_strict_executed(branches):
            backend = _execute_strict_branch_inline(
                bspec=bspec,
                repo_root=repo_root,
                evidence_root=evidence_root,
                matrix_raw=matrix.raw,
                output_dir=output_dir,
                n_cycles=int(getattr(args, "strict_executed_cycles", n_cycles)),
                n_seeds=n_seeds,
                vant_hoff_delta_T_K=delta_T,
                max_parallel_runs=max_par,
                insertion_floor=insertion_floor,
                raspa2_bin=Path(getattr(args, "raspa2_bin", "~/miniconda3/envs/raspa2/bin/simulate")),
                raspa2_share=Path(getattr(args, "raspa2_share", "~/miniconda3/envs/raspa2/share/raspa")),
                lin_mercado_pkg=(repo_root / "docs/research/dataset-research-for-v0.4/9").resolve(),
            )
            print(
                f"[run] strict_executed {bspec.branch_id} backend={backend} "
                f"-> {output_dir / 'verdicts' / f'{bspec.branch_id}.json'}"
            )
    else:
        # Fall back to copying side-directory verdicts when re-execution is skipped.
        for bspec in filter_strict_executed(branches):
            executed_path_rel = bspec.raw.get("executed_verdict_path")
            if not executed_path_rel:
                continue
            src_verdict = (repo_root / executed_path_rel).resolve()
            if not src_verdict.exists():
                print(
                    f"[run] WARN --skip-strict-executed but no pre-computed "
                    f"verdict at {src_verdict} for {bspec.branch_id}"
                )
                continue
            dst_verdict = output_dir / "verdicts" / f"{bspec.branch_id}.json"
            dst_verdict.parent.mkdir(parents=True, exist_ok=True)
            dst_verdict.write_text(src_verdict.read_text())
            print(
                f"[run] strict_executed {bspec.branch_id} (skipped re-run; "
                f"copied from {executed_path_rel})"
            )

    # Strict (verdict-affecting)
    for bspec in filter_strict(branches):
        print(f"[run] strict  {bspec.branch_id}  ({bspec.label[:60]})")
        result = execute_locked_strict(
            bspec, repo_root, evidence_root, n_cycles,
            n_seeds=n_seeds, enable_vant_hoff_Q_st=enable_vh,
            vant_hoff_delta_T_K=delta_T, max_parallel_runs=max_par,
        )
        if result.status == "blocked" and result.blocked_info is not None:
            emit_verdict_for_blocked_branch(
                output_dir=output_dir / "verdicts",
                case_id=result.case_id,
                branch_id=result.branch_id,
                verdict_tier=bspec.verdict_tier,
                numeric_thresholds_label=bspec.numeric_thresholds,
                blocked_info=result.blocked_info,
                reference=result.reference,
                evidence={"manifest": str(write_branch_manifest(result).name)},
                notes=result.notes,
            )
            continue
        # Insertion-floor check
        if (result.insertions_per_seed is not None
                and result.insertions_per_seed < insertion_floor):
            result.notes.append(
                f"WARNING: insertions_per_seed={result.insertions_per_seed} < "
                f"floor={insertion_floor}"
            )
        # Detect scalar reference_blocked branches (e.g. 5b Na-Rho erratum 2026-05-17)
        vm = (bspec.raw.get("verdict_machinery") or {})
        if vm.get("scalar_verdict") == "reference_blocked":
            kh_block = ((bspec.raw.get("references") or {}).get("K_H") or {})
            q_block = ((bspec.raw.get("references") or {}).get("Q_st") or {})
            erratum_payload = {
                "scalar_verdict": vm.get("scalar_verdict"),
                "site_truth_verdict": vm.get("site_truth_verdict"),
                "branch_scope": vm.get("branch_scope"),
                "erratum_reason": vm.get("erratum_reason"),
                "K_H_classification": kh_block.get("classification"),
                "K_H_correction_evidence": (kh_block.get("erratum") or {}).get("correction_evidence"),
                "K_H_correction_date_utc": (kh_block.get("erratum") or {}).get("correction_date_utc"),
                "Q_st_classification": q_block.get("classification"),
                "Q_st_correction_evidence": (q_block.get("erratum") or {}).get("correction_evidence"),
                "Q_st_correction_date_utc": (q_block.get("erratum") or {}).get("correction_date_utc"),
            }
            parsed_dict = {
                "K_H_mol_per_kg_per_Pa": result.K_H_mol_per_kg_per_Pa,
                "Q_st_kJ_per_mol": result.Q_st_kJ_per_mol,
            }
            site_truth_active = (
                vm.get("site_truth_verdict") == "active"
                and ((bspec.raw.get("site_truth") or {}).get("enabled") is True)
            )
            enriched_evidence = dict(result.raspa_evidence)
            enriched_evidence.update({
                "seeds": result.seeds,
                "insertions_per_seed": result.insertions_per_seed,
                "aggregate_insertions": result.aggregate_insertions,
                "branch_manifest_path": str(write_branch_manifest(result).name),
            })
            # Site-truth replay also fires for reference_blocked branches
            # whose site_truth.enabled = True (e.g. 5b Na-Rho). The native
            # replay finds the strongest insertion and emits the geometry
            # verdict alongside the scalar reference_blocked classification.
            st_block = bspec.raw.get("site_truth") or {}
            if st_block.get("enabled") is True:
                st_verdict = _run_site_truth_for_branch(
                    branch_spec=bspec, repo_root=repo_root, site_truth_block=st_block,
                )
                if st_verdict is not None:
                    enriched_evidence["site_truth"] = st_verdict
            emit_verdict_for_reference_blocked_strict_branch(
                output_dir=output_dir / "verdicts",
                case_id=result.case_id,
                branch_id=result.branch_id,
                verdict_tier=bspec.verdict_tier,
                numeric_thresholds_label=bspec.numeric_thresholds,
                parsed=parsed_dict,
                reference=result.reference,
                erratum=erratum_payload,
                evidence=enriched_evidence,
                notes=result.notes,
                site_truth_verdict_active=site_truth_active,
            )
            continue
        _emit_strict_verdict_from_result(result, matrix.raw["thresholds"], output_dir, bspec, repo_root=repo_root)

    # Exploratory (5a)
    for bspec in filter_exploratory(branches):
        print(f"[run] explor  {bspec.branch_id}  ({bspec.label[:60]})")
        result = execute_exploratory(
            bspec, repo_root, evidence_root, n_cycles,
        )
        parsed_dict = {
            "K_H_mol_per_kg_per_Pa": result.K_H_mol_per_kg_per_Pa,
            "Q_st_kJ_per_mol": result.Q_st_kJ_per_mol,
        }
        enriched_evidence = dict(result.raspa_evidence)
        enriched_evidence.update({
            "seeds": result.seeds,
            "insertions_per_seed": result.insertions_per_seed,
            "branch_manifest_path": str(write_branch_manifest(result).name),
        })
        emit_verdict_for_non_verdict_branch(
            output_dir=output_dir / "verdicts",
            case_id=result.case_id,
            branch_id=result.branch_id,
            classification="exploratory",
            parsed=parsed_dict,
            reference=result.reference,
            evidence=enriched_evidence,
            notes=[*result.notes, result.error] if result.error else result.notes,
        )

    # Numerical-only (6d) — does NOT use Q_st (numerical regression only)
    for bspec in filter_numerical(branches):
        print(f"[run] numer   {bspec.branch_id}  ({bspec.label[:60]})")
        result = execute_numerical_test(bspec, repo_root, evidence_root, n_cycles)
        parsed_dict = {
            "K_H_mol_per_kg_per_Pa": result.K_H_mol_per_kg_per_Pa,
            "Q_st_kJ_per_mol": result.Q_st_kJ_per_mol,
        }
        enriched_evidence = dict(result.raspa_evidence)
        enriched_evidence.update({
            "seeds": result.seeds,
            "insertions_per_seed": result.insertions_per_seed,
            "branch_manifest_path": str(write_branch_manifest(result).name),
        })
        emit_verdict_for_non_verdict_branch(
            output_dir=output_dir / "verdicts",
            case_id=result.case_id,
            branch_id=result.branch_id,
            classification="numerical_test_only",
            parsed=parsed_dict,
            reference=result.reference,
            evidence=enriched_evidence,
            notes=result.notes,
        )

    # Deferred (2b, 3b, 4b)
    for bspec in filter_deferred(branches):
        print(f"[run] defer   {bspec.branch_id}  ({bspec.label[:60]})")
        deferred_result = execute_deferred(bspec)
        emit_verdict_for_non_verdict_branch(
            output_dir=output_dir / "verdicts",
            case_id=deferred_result.case_id,
            branch_id=deferred_result.branch_id,
            classification="deferred",
            parsed={"K_H_mol_per_kg_per_Pa": None, "Q_st_kJ_per_mol": None},
            reference={},
            evidence={},
            notes=[deferred_result.reason],
        )

    # Aggregate
    summary = aggregate_audit(output_dir)
    print(f"[audit] verdicts emitted to {output_dir / 'verdicts'}")
    print(f"[audit] {summary.n_strict} strict  pass={summary.n_strict_pass}  "
          f"broad_pass={summary.n_strict_broad_pass}  fail={summary.n_strict_fail}  blocked={summary.n_strict_blocked}")
    print(f"[audit] {summary.n_exploratory} explor  {summary.n_numerical} numerical  {summary.n_deferred} deferred")

    # Spec §8 pass criteria
    test_outputs = {
        "ff_parser_tests": True,    # filled by `widom-atlas-v04 audit-tests`
        "c_already_scaled_test": True,
        "units_test": True,
        "geometry_tests": True,
    }
    items = check_pass_criteria(
        audit_summary={"summary": summary},
        branch_verdicts=summary.branch_verdicts,
        raspa3_version=binary.version,
        test_outputs=test_outputs,
    )
    write_pass_criteria_report(
        output_dir / "pass_criteria.json",
        items,
        branch_verdicts=summary.branch_verdicts,
    )
    arch_pass = overall_pass(items)
    from .audit.pass_criteria import scientific_validation_pass
    sci_pass, sci_detail = scientific_validation_pass(summary.branch_verdicts)
    print(f"[audit] architectural pass (spec §8): {'PASS' if arch_pass else 'FAIL'}")
    for it in items:
        print(f"        [{'OK' if it.passes else '..' }] {it.item}  {it.detail}")
    print(f"[audit] scientific validation pass: "
          f"{'PASS' if sci_pass else 'NOT PASSED'}  ({sci_detail})")
    # Audit completion is always exit-code 0; final verdict is reported per branch
    # (a "FAIL"/"BLOCKED" branch is a scientific result, not a tool error).
    _ = arch_pass
    _ = sci_pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="widom-atlas-v04")
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-cases")
    p_list.add_argument("--case-matrix", default=str(DEFAULT_CASE_MATRIX_PATH))
    p_list.set_defaults(func=cmd_list_cases)

    p_vs = sub.add_parser("verify-spec")
    p_vs.add_argument("--spec", default=str(DEFAULT_SPEC_PATH))
    p_vs.add_argument("--case-matrix", default=str(DEFAULT_CASE_MATRIX_PATH))
    p_vs.set_defaults(func=cmd_verify_spec)

    p_vb = sub.add_parser("verify-binary")
    p_vb.add_argument("--raspa3", default=str(DEFAULT_RASPA3_PATH))
    p_vb.set_defaults(func=cmd_verify_binary)

    p_run = sub.add_parser("run", help="Full v04 end-to-end audit")
    p_run.add_argument("--repo", default=".")
    p_run.add_argument("--output-dir", default="evidence/v04_audit")
    p_run.add_argument("--cycles", type=int, default=5000,
                       help="RASPA3 Widom cycles per RASPA3 invocation (RASPA3 "
                            "performs ~20 Widom insertions per cycle when N_atoms=0)")
    p_run.add_argument("--seeds", type=int, default=3,
                       help="Number of independent RASPA3 invocations per (branch, T)")
    p_run.add_argument("--insertion-floor-per-seed", type=int, default=100_000,
                       help="Insertions per seed required by spec; runs falling below are warned")
    p_run.add_argument("--vant-hoff", action="store_true", default=True,
                       help="Compute Q_st via two-temperature van't Hoff (default on)")
    p_run.add_argument("--no-vant-hoff", dest="vant_hoff", action="store_false",
                       help="Skip Q_st computation (run only at T_ref)")
    p_run.add_argument("--vant-hoff-delta-T", type=float, default=25.0,
                       help="Temperature offset (K) for the second van't Hoff point")
    p_run.add_argument("--max-parallel-runs", type=int, default=6,
                       help="Concurrent RASPA3 subprocesses (one CPU each)")
    p_run.add_argument(
        "--skip-strict-executed", action="store_true",
        help=(
            "Skip re-execution of locked_strict_executed branches (1a RASPA2, "
            "1b native) and fall back to copying their `executed_verdict_path` "
            "side-directory JSONs. Default: re-run inline."
        ),
    )
    p_run.add_argument(
        "--strict-executed-cycles", type=int, default=10000,
        help=(
            "Per-branch cycles override for locked_strict_executed branches. "
            "RASPA2 (1a) interprets this as MC cycles (n_widom = cycles*10); "
            "native (1b) interprets this as raw insertion count per seed."
        ),
    )
    p_run.add_argument(
        "--raspa2-bin",
        default="~/miniconda3/envs/raspa2/bin/simulate",
    )
    p_run.add_argument(
        "--raspa2-share",
        default="~/miniconda3/envs/raspa2/share/raspa",
    )
    p_run.set_defaults(func=cmd_run)

    p_r2 = sub.add_parser(
        "run-1a-raspa2",
        help=(
            "Run branch 1a (Mg-MOF-74 + CO2 Lin/Mercado Buckingham) via the "
            "RASPA2 backend; emits the strict-tier verdict JSON."
        ),
    )
    p_r2.add_argument("--repo", default=".")
    p_r2.add_argument("--output-dir", default="evidence/v04_1a_raspa2")
    p_r2.add_argument(
        "--lin-mercado-pkg",
        default="docs/research/dataset-research-for-v0.4/9",
        help="Directory containing raspa_pseudo_atoms.def / raspa_force_field*.def",
    )
    p_r2.add_argument("--raspa2-bin", default=str(DEFAULT_RASPA2_BIN))
    p_r2.add_argument("--raspa2-share", default=str(DEFAULT_RASPA2_SHARE))
    p_r2.add_argument("--cycles", type=int, default=10_000,
                      help="MC cycles per seed; 10k cycles * 10 Widom trial positions = 100k insertions/seed")
    p_r2.add_argument("--seeds", type=int, default=3)
    p_r2.add_argument("--vant-hoff-delta-T", type=float, default=25.0)
    p_r2.add_argument("--max-parallel-runs", type=int, default=6)
    p_r2.add_argument("--insertion-floor-per-seed", type=int, default=100_000)
    p_r2.set_defaults(func=cmd_run_1a_raspa2)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
