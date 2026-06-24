"""T035: deferred-branch stub for 2b, 3b, 4b.

These branches are recorded scope items only — not implemented in v0.4.
The stub returns a fixed `deferred` result so the audit can record them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dispatcher import BranchSpec


@dataclass(frozen=True)
class DeferredResult:
    case_id: str
    branch_id: str
    status: str
    reason: str


def execute_deferred(branch: BranchSpec) -> DeferredResult:
    reason = str(branch.raw.get("deferred_reason") or branch.raw.get("blocked_reason") or
                 "Branch reserved for v0.5; not in v0.4 verdict.")
    return DeferredResult(
        case_id=branch.case_id,
        branch_id=branch.branch_id,
        status=branch.status,
        reason=reason,
    )
