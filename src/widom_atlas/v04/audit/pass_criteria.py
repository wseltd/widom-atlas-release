"""T039: Spec §8 PASS-criteria checker.

Enforces all 10 items from V04_LOCKED_SPEC.md §8. Returns a structured
report; the v04 audit verdict is `PASSED` only if ALL items are TRUE.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PassCriteriaResult:
    item: str
    passes: bool
    detail: str


def check_pass_criteria(
    audit_summary: dict,
    branch_verdicts: dict[str, dict],
    raspa3_version: str,
    test_outputs: dict[str, bool],
) -> list[PassCriteriaResult]:
    items: list[PassCriteriaResult] = []

    # 1: Parsers produce typed pair table with zero hand-transcribed hidden constants
    items.append(PassCriteriaResult(
        item="1_parsers_typed_no_hand_constants",
        passes=test_outputs.get("ff_parser_tests", False),
        detail="FF parser unit tests must all pass",
    ))
    # 2: C_already_scaled flag respected (Lin/Mercado)
    items.append(PassCriteriaResult(
        item="2_C_already_scaled_flag",
        passes=test_outputs.get("c_already_scaled_test", False),
        detail="C_already_scaled flag respected in BuckinghamLinMercado.energy()",
    ))
    # 3: Case 1 architecturally considers both 1a and 1b (irrespective of verdict).
    # If RASPA3 v3.0.29 cannot represent the prescribed pair potential, BLOCKED is
    # an acceptable verdict — the architectural requirement is that both branches
    # are reported. Operator directive: do NOT silently substitute LJ.
    present_1a = "1a" in branch_verdicts
    present_1b = "1b" in branch_verdicts
    v1a = branch_verdicts.get("1a", {}).get("verdict")
    v1b = branch_verdicts.get("1b", {}).get("verdict")
    items.append(PassCriteriaResult(
        item="3_case1_both_branches_considered",
        passes=present_1a and present_1b,
        detail=f"1a verdict={v1a!r}, 1b verdict={v1b!r}",
    ))
    # 4: Strict-tier via RASPA3 Ewald
    items.append(PassCriteriaResult(
        item="4_strict_tier_raspa3_ewald",
        passes=raspa3_version.startswith("3.0.29"),
        detail=f"RASPA3 version={raspa3_version}",
    ))
    # 5: Canonical units throughout
    items.append(PassCriteriaResult(
        item="5_canonical_units",
        passes=test_outputs.get("units_test", False),
        detail="Units round-trip + positive-exothermic Q_ads convention",
    ))
    # 6: Site-truth geometry self-tests pass
    items.append(PassCriteriaResult(
        item="6_site_truth_self_tests",
        passes=test_outputs.get("geometry_tests", False),
        detail="Geometry self-test runner verified pre-audit",
    ))
    # 7: Case 3 scalar-only unless Chevreau obtained
    case3_ok = True
    if "3a" in branch_verdicts:
        v = branch_verdicts["3a"]
        # Site truth must be disabled
        ref = v.get("reference") or {}
        if "site_truth_available" in ref:
            case3_ok = False
    items.append(PassCriteriaResult(
        item="7_case3_scalar_only",
        passes=case3_ok,
        detail="Case 3 enabled site_truth: False",
    ))
    # 8: Case 5 reports 5a + 5b separately, verdict on 5b only
    has_5a = "5a" in branch_verdicts
    has_5b = "5b" in branch_verdicts
    items.append(PassCriteriaResult(
        item="8_case5_split",
        passes=has_5a and has_5b,
        detail=f"5a reported={has_5a}, 5b reported={has_5b}",
    ))
    # 9: Case 6d numerical_test_only
    case6d_ok = False
    if "6d" in branch_verdicts:
        case6d_ok = branch_verdicts["6d"].get("verdict") in ("NUMERICAL_TEST_ONLY",)
    items.append(PassCriteriaResult(
        item="9_case6d_numerical_only",
        passes=case6d_ok,
        detail=f"6d verdict={branch_verdicts.get('6d',{}).get('verdict')}",
    ))
    # 10: ff_fallback uses broad_tier
    ok_10 = True
    for bid in ("2a", "3a", "5b"):
        if bid in branch_verdicts and branch_verdicts[bid].get("numeric_thresholds") != "broad_tier":
            ok_10 = False
    items.append(PassCriteriaResult(
        item="10_ff_fallback_broad_tier",
        passes=ok_10,
        detail="2a/3a/5b use broad_tier numeric thresholds",
    ))
    return items


def overall_pass(items: list[PassCriteriaResult]) -> bool:
    return all(i.passes for i in items)


def scientific_validation_pass(branch_verdicts: dict[str, dict]) -> tuple[bool, str]:
    """A v0.4 scientific PASS requires every verdict-affecting strict branch to
    have verdict PASS (or BROAD_TIER_PASS_only). BLOCKED + FAIL both disqualify
    the scientific validation.

    REFERENCE_BLOCKED branches (e.g. 5b Na-Rho per V04_REFERENCE_AUDIT.md §9.1)
    are treated separately: the scalar K_H/Q_st verdict is NOT counted, but the
    branch is reported as unresolved and the overall validation still does NOT
    pass until it is resolved (either via a defensible reference or some other
    closure). Equivalent treatment to BLOCKED for the overall pass gate.
    """
    strict_ids = ("1a", "1b", "2a", "3a", "4a", "5b", "6a", "6b", "6c")
    accepting = {"PASS", "BROAD_TIER_PASS_only"}
    failures: list[str] = []
    blocked: list[str] = []
    reference_blocked: list[str] = []
    for bid in strict_ids:
        v = branch_verdicts.get(bid, {}).get("verdict")
        if v == "BLOCKED":
            blocked.append(bid)
        elif v == "REFERENCE_BLOCKED":
            reference_blocked.append(bid)
        elif v not in accepting:
            failures.append(f"{bid}={v}")
    if not failures and not blocked and not reference_blocked:
        return True, "all 9 strict branches PASS or BROAD_TIER_PASS_only"
    parts = []
    if blocked:
        parts.append(f"BLOCKED: {','.join(blocked)}")
    if reference_blocked:
        parts.append(f"REFERENCE_BLOCKED: {','.join(reference_blocked)}")
    if failures:
        parts.append(f"failing: {','.join(failures)}")
    return False, " ; ".join(parts)


def write_pass_criteria_report(
    path: Path,
    items: list[PassCriteriaResult],
    branch_verdicts: dict[str, dict] | None = None,
) -> None:
    import json
    arch_pass = overall_pass(items)
    sci_pass: bool | None
    sci_detail: str | None
    if branch_verdicts is not None:
        sci_pass, sci_detail = scientific_validation_pass(branch_verdicts)
    else:
        sci_pass, sci_detail = None, None
    payload: dict = {
        "items": [
            {"item": i.item, "passes": i.passes, "detail": i.detail} for i in items
        ],
        "architectural_pass": arch_pass,
        "scientific_validation_pass": sci_pass,
        "scientific_validation_detail": sci_detail,
        # Back-compat: `overall_pass` was previously the architectural one.
        "overall_pass": arch_pass,
    }
    path.write_text(json.dumps(payload, indent=2))
