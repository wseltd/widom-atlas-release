"""T038: Audit pipeline aggregating per-branch verdict JSONs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AuditSummary:
    output_dir: Path
    branch_verdicts: dict[str, dict]
    n_branches: int
    n_strict: int
    n_strict_pass: int
    n_strict_broad_pass: int
    n_strict_fail: int
    n_strict_blocked: int
    # Erratum 2026-05-17: scalar reference invalid but branch still in scope
    # (e.g. 5b Na-Rho — trapdoor zeolite, no Henry regime). Distinct from
    # FAIL (real disagreement) and BLOCKED (backend gap).
    n_strict_reference_blocked: int
    n_exploratory: int
    n_numerical: int
    n_deferred: int
    # 5b-style branches: scalar reclassified, site_truth still active
    n_strict_site_truth_only: int = 0
    # 6b-style branches: per-axis reference-blocked (K_H active, Q_st blocked)
    n_strict_partial_pass_ref_blocked: int = 0
    # New 2026-05-18: per-axis method-blocked (atlas Q_st_method ≠ ref Q_st_method)
    n_strict_partial_pass_method_blocked: int = 0
    n_strict_method_blocked: int = 0


def aggregate_audit(output_dir: Path) -> AuditSummary:
    verdict_dir = output_dir / "verdicts"
    branch_verdicts: dict[str, dict] = {}
    for jf in sorted(verdict_dir.glob("*.json")):
        branch_verdicts[jf.stem] = json.loads(jf.read_text())

    n_strict = 0
    n_strict_pass = 0
    n_strict_broad_pass = 0
    n_strict_fail = 0
    n_strict_blocked = 0
    n_strict_reference_blocked = 0
    n_strict_site_truth_only = 0
    n_strict_partial_pass_ref_blocked = 0
    n_strict_partial_pass_method_blocked = 0
    n_strict_method_blocked = 0
    n_exploratory = 0
    n_numerical = 0
    n_deferred = 0

    for _bid, v in branch_verdicts.items():
        if v.get("verdict_tier") in ("flagship_strict", "flagship_strict_with_ff_fallback_relaxation"):
            n_strict += 1
            verdict = v.get("verdict") or ""
            if verdict == "PASS":
                n_strict_pass += 1
            elif verdict == "BROAD_TIER_PASS_only":
                n_strict_broad_pass += 1
            elif verdict.startswith("BLOCKED"):
                n_strict_blocked += 1
            elif verdict == "REFERENCE_BLOCKED":
                n_strict_reference_blocked += 1
                if v.get("site_truth_verdict_active") is True:
                    n_strict_site_truth_only += 1
            elif verdict == "REFERENCE_OR_METHOD_BLOCKED":
                n_strict_method_blocked += 1
            elif verdict.startswith("PARTIAL_PASS_") and "REFERENCE_BLOCKED" in verdict:
                # e.g. PARTIAL_PASS_Q_st_REFERENCE_BLOCKED (6b: K_H passes, Q_st blocked)
                n_strict_partial_pass_ref_blocked += 1
            elif verdict.startswith("PARTIAL_PASS_") and "METHOD_BLOCKED" in verdict:
                n_strict_partial_pass_method_blocked += 1
            elif verdict.startswith("FAIL_") and ("REFERENCE_BLOCKED" in verdict or "METHOD_BLOCKED" in verdict):
                n_strict_fail += 1
            elif verdict == "FAIL":
                n_strict_fail += 1
        elif v.get("verdict") == "EXPLORATORY":
            n_exploratory += 1
        elif v.get("verdict") == "NUMERICAL_TEST_ONLY":
            n_numerical += 1
        elif v.get("verdict") == "DEFERRED":
            n_deferred += 1

    summary = AuditSummary(
        output_dir=output_dir,
        branch_verdicts=branch_verdicts,
        n_branches=len(branch_verdicts),
        n_strict=n_strict,
        n_strict_pass=n_strict_pass,
        n_strict_broad_pass=n_strict_broad_pass,
        n_strict_fail=n_strict_fail,
        n_strict_blocked=n_strict_blocked,
        n_strict_reference_blocked=n_strict_reference_blocked,
        n_strict_site_truth_only=n_strict_site_truth_only,
        n_strict_partial_pass_ref_blocked=n_strict_partial_pass_ref_blocked,
        n_strict_partial_pass_method_blocked=n_strict_partial_pass_method_blocked,
        n_strict_method_blocked=n_strict_method_blocked,
        n_exploratory=n_exploratory,
        n_numerical=n_numerical,
        n_deferred=n_deferred,
    )
    (output_dir / "audit_summary.json").write_text(
        json.dumps(
            {k: getattr(summary, k) if not isinstance(getattr(summary, k), Path) else str(getattr(summary, k))
             for k in ["n_branches", "n_strict", "n_strict_pass", "n_strict_broad_pass",
                       "n_strict_fail", "n_strict_blocked",
                       "n_strict_reference_blocked", "n_strict_site_truth_only",
                       "n_strict_partial_pass_ref_blocked",
                       "n_strict_partial_pass_method_blocked",
                       "n_strict_method_blocked",
                       "n_exploratory", "n_numerical", "n_deferred"]},
            indent=2,
        )
    )
    return summary
