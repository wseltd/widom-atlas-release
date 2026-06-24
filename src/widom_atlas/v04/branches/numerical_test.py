"""T034: numerical_test_only branch executor (6d MFI + Ar at 87 K).

A RASPA/MOFX parity fixture. Runs under the audit, output recorded in the
appendix, never affects the v0.4 verdict.
"""
from __future__ import annotations

from pathlib import Path

from .dispatcher import BranchSpec
from .executor import LockedStrictResult, execute_locked_strict


def execute_numerical_test(
    branch: BranchSpec,
    repo_root: Path,
    evidence_root: Path,
    n_cycles: int,
) -> LockedStrictResult:
    result = execute_locked_strict(branch, repo_root, evidence_root, n_cycles)
    result.status = "numerical_test_" + result.status
    result.notes.append("Numerical regression fixture only — does NOT affect v0.4 verdict.")
    return result
