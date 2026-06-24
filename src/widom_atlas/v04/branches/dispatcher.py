"""T031: Branch dispatcher.

Resolves every locked branch in v04_case_matrix.yaml to the correct
executor:
- locked_strict: RaspaStrictExecutor (T032)
- exploratory_runs_but_no_verdict_impact: ExploratoryExecutor (T033)
- numerical_test_only_not_in_verdict: NumericalTestExecutor (T034)
- deferred_not_in_v04_verdict: DeferredStub (T035)
- broad_tier_pending_operator_decision: DeferredStub (skipped in v0.4)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BranchSpec:
    case_id: str
    branch_id: str
    status: str
    verdict_tier: str
    numeric_thresholds: str
    label: str
    raw: dict


def list_all_branches(matrix: dict) -> list[BranchSpec]:
    out: list[BranchSpec] = []
    for case in matrix.get("cases", []):
        for branch in case.get("branches", []):
            out.append(
                BranchSpec(
                    case_id=str(case["case_id"]),
                    branch_id=str(branch["branch_id"]),
                    status=str(branch.get("status", "unknown")),
                    verdict_tier=str(branch.get("verdict_tier", "")),
                    numeric_thresholds=str(branch.get("numeric_thresholds", "")),
                    label=str(branch.get("label", "")),
                    raw=branch,
                )
            )
    return out


def filter_strict(branches: list[BranchSpec]) -> list[BranchSpec]:
    return [b for b in branches if b.status == "locked_strict"]


def filter_strict_executed(branches: list[BranchSpec]) -> list[BranchSpec]:
    """Branches whose verdict was already executed by an external backend
    (e.g. RASPA2 for 1a Lin/Mercado). The main `run` pipeline should NOT
    re-execute these — instead it copies the pre-existing verdict JSON
    from the path recorded in `executed_verdict_path` into the main
    verdicts directory so audit aggregation includes it.

    `locked_strict_executed_multi_variant` was added 2026-06-01 for 3b
    (Maia 2023 UA/UAq/EHq) — same semantics, multiple variants per branch.
    """
    return [
        b for b in branches
        if b.status in (
            "locked_strict_executed",
            "locked_strict_executed_multi_variant",
        )
    ]


def filter_exploratory(branches: list[BranchSpec]) -> list[BranchSpec]:
    return [b for b in branches if b.status == "exploratory_runs_but_no_verdict_impact"]


def filter_numerical(branches: list[BranchSpec]) -> list[BranchSpec]:
    return [b for b in branches if b.status == "numerical_test_only_not_in_verdict"]


def filter_deferred(branches: list[BranchSpec]) -> list[BranchSpec]:
    """Branches that do NOT enter the v0.4 strict denominator. Includes the
    historical `deferred_*` statuses plus the 2026-06-01 final-pivot
    `blocked_pending_*` and `reference_audited_pending_*` statuses, which
    are semantically the same (out of the strict denominator pending an
    exact missing artefact)."""
    return [
        b for b in branches
        if b.status in (
            "deferred_not_in_v04_verdict",
            "broad_tier_pending_operator_decision",
            "deferred_partial_data_accessible_full_table_recovery_required",
            "blocked_pending_Bai_main_parameter_tables",
            "reference_audited_pending_cation_cif_and_ff_lock",
        )
    ]
