"""T033: Exploratory branch executor (5a Na-Rho closed-dehydrated).

Runs the same RASPA3 pipeline as locked_strict but its result NEVER
affects the v0.4 verdict. The audit records the outcome as
expected_physical_failure or successful-but-non-verdict.
"""
from __future__ import annotations

from pathlib import Path

from .dispatcher import BranchSpec
from .executor import LockedStrictResult, execute_locked_strict


def execute_exploratory(
    branch: BranchSpec,
    repo_root: Path,
    evidence_root: Path,
    n_cycles: int,
) -> LockedStrictResult:
    result = execute_locked_strict(branch, repo_root, evidence_root, n_cycles)
    result.status = "exploratory_" + result.status
    result.notes.append("Exploratory branch — failure does not affect v0.4 verdict.")
    return result
