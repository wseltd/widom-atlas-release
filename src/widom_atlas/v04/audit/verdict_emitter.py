"""T037: Per-branch JSON verdict emitter.

Given a branch's result + reference + locked thresholds, produces one
JSON file under <output_dir>/verdicts/{branch_id}.json containing:
  branch_id, case_id, verdict, verdict_tier, numeric_thresholds,
  tier_flags, delta_log10_KH, delta_Qads_kJmol, parsed_K_H,
  parsed_Q_st, reference, evidence_paths, notes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _delta_log10(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return math.log10(a / b)


def _threshold_pair(numeric_thresholds: str, thresholds: dict) -> tuple[float, float]:
    tier_block = thresholds.get(numeric_thresholds) or thresholds.get("flagship_strict") or {}
    return (
        float(tier_block.get("delta_log10_KH_max", 0.10)),
        float(tier_block.get("delta_Qads_kJ_per_mol_max", 2.0)),
    )


def emit_verdict_for_strict_branch(
    output_dir: Path,
    case_id: str,
    branch_id: str,
    verdict_tier: str,
    numeric_thresholds_label: str,
    thresholds: dict,
    parsed: dict,            # parsed RASPA3 scalars (mol/kg/Pa, kJ/mol)
    reference: dict,         # locked literature reference + windows
    evidence: dict,          # paths + sha256
    notes: list[str] | None = None,
    K_H_reference_blocked: bool = False,
    Q_st_reference_blocked: bool = False,
    atlas_Q_st_method: str | None = None,
    reference_Q_st_method: str | None = None,
) -> Path:
    """Emit a strict-tier verdict JSON.

    Per-axis reference-blocking (2026-05-18 split-classification): if
    `K_H_reference_blocked` (resp. `Q_st_reference_blocked`) is True, the
    corresponding axis is recorded as REFERENCE_BLOCKED and does not contribute
    a PASS/FAIL decision. The overall branch verdict then becomes:
      * "PASS" if the non-blocked axes pass.
      * "PARTIAL_PASS_<axis>_REFERENCE_BLOCKED" if some axes pass and at least
        one axis is reference_blocked.
      * "FAIL" if any non-blocked axis fails.
    """
    from .q_st_compatibility import (
        COMPATIBLE, COMPATIBLE_NOTE, METHOD_BLOCKED,
        assess_compatibility,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_K_H_per_Pa = parsed.get("K_H_mol_per_kg_per_Pa")
    parsed_K_H_per_bar = parsed_K_H_per_Pa * 1e5 if parsed_K_H_per_Pa else None
    parsed_Q_st = parsed.get("Q_st_kJ_per_mol")
    ref_K_H = reference.get("K_H_value_mol_per_kg_per_bar")
    ref_Q_st = reference.get("Q_st_value_kj_per_mol")
    dKH = _delta_log10(parsed_K_H_per_bar, ref_K_H) if not K_H_reference_blocked else None

    # Q_st_method compatibility: if the atlas and reference Q_st were measured by
    # incompatible methods (e.g. atlas two-point van't Hoff vs reference
    # Clausius-Clapeyron), refuse to compute ΔQ_st and mark the axis
    # METHOD_BLOCKED with the exact missing match.
    q_method_outcome = COMPATIBLE
    q_method_note = ""
    Q_st_method_blocked = False
    if not Q_st_reference_blocked and parsed_Q_st is not None and ref_Q_st is not None:
        q_method_outcome, q_method_note = assess_compatibility(
            atlas_Q_st_method, reference_Q_st_method,
        )
        if q_method_outcome == METHOD_BLOCKED:
            Q_st_method_blocked = True

    dQ = (
        (parsed_Q_st - ref_Q_st)
        if (parsed_Q_st is not None and ref_Q_st is not None
            and not Q_st_reference_blocked and not Q_st_method_blocked)
        else None
    )
    kh_thr, q_thr = _threshold_pair(numeric_thresholds_label, thresholds)
    kh_pass = (dKH is not None) and (abs(dKH) <= kh_thr) if not K_H_reference_blocked else None
    q_pass = (
        (dQ is not None) and (abs(dQ) <= q_thr)
        if not (Q_st_reference_blocked or Q_st_method_blocked)
        else None
    )
    # Verdict logic
    Q_st_axis_blocked = Q_st_reference_blocked or Q_st_method_blocked
    if parsed_K_H_per_bar is None and parsed_Q_st is None:
        verdict = "BLOCKED_no_parsed_scalars"
    elif parsed_K_H_per_bar is None and not K_H_reference_blocked:
        verdict = "BLOCKED_no_K_H"
    elif K_H_reference_blocked and Q_st_axis_blocked:
        verdict = "REFERENCE_BLOCKED" if not Q_st_method_blocked else "REFERENCE_OR_METHOD_BLOCKED"
    elif K_H_reference_blocked:
        verdict = (
            "PARTIAL_PASS_K_H_REFERENCE_BLOCKED" if q_pass
            else "FAIL_Q_st_K_H_REFERENCE_BLOCKED"
        )
    elif Q_st_method_blocked:
        verdict = (
            "PARTIAL_PASS_Q_st_METHOD_BLOCKED" if kh_pass
            else "FAIL_K_H_Q_st_METHOD_BLOCKED"
        )
    elif Q_st_reference_blocked:
        verdict = (
            "PARTIAL_PASS_Q_st_REFERENCE_BLOCKED" if kh_pass
            else "FAIL_K_H_Q_st_REFERENCE_BLOCKED"
        )
    elif kh_pass and (q_pass or parsed_Q_st is None):
        verdict = "PASS"
    elif (
        (not kh_pass)
        and dKH is not None
        and abs(dKH) <= 0.20
        # 2026-05-19 fix: BROAD_TIER_PASS_only requires Q_st to ALSO pass strict
        # (or be axis-blocked / absent). A branch where K_H is broad-band but
        # Q_st FAILs strict is scientifically a FAIL on the Q_st axis and must
        # not be labelled as a broad-tier PASS. See test_4a_composite_verdict_*.
        and (q_pass or parsed_Q_st is None or Q_st_axis_blocked)
    ):
        verdict = "BROAD_TIER_PASS_only"
    else:
        verdict = "FAIL"
    payload: dict[str, Any] = {
        "branch_id": branch_id,
        "case_id": case_id,
        "verdict": verdict,
        "verdict_tier": verdict_tier,
        "numeric_thresholds": numeric_thresholds_label,
        "thresholds_used": {
            "delta_log10_KH_max": kh_thr,
            "delta_Qads_kJ_per_mol_max": q_thr,
        },
        "parsed_K_H_mol_per_kg_per_Pa": parsed_K_H_per_Pa,
        "parsed_K_H_mol_per_kg_per_bar": parsed_K_H_per_bar,
        "parsed_Q_st_kJ_per_mol": parsed_Q_st,
        "delta_log10_K_H": dKH,
        "delta_Q_st_kJ_per_mol": dQ,
        "passes_K_H": kh_pass,
        "passes_Q_st": q_pass,
        "Q_st_method_compatibility": {
            "atlas_method": atlas_Q_st_method,
            "reference_method": reference_Q_st_method,
            "outcome": q_method_outcome,
            "note": q_method_note,
            "method_blocked": Q_st_method_blocked,
        },
        "reference": reference,
        "evidence": evidence,
        "notes": (notes or []) + (
            [f"Q_st_method_compatibility: {q_method_note}"]
            if q_method_note else []
        ),
    }
    out_path = output_dir / f"{branch_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def emit_verdict_for_non_verdict_branch(
    output_dir: Path,
    case_id: str,
    branch_id: str,
    classification: str,    # "exploratory" | "numerical_test_only" | "deferred"
    parsed: dict,
    reference: dict,
    evidence: dict,
    notes: list[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "branch_id": branch_id,
        "case_id": case_id,
        "verdict": classification.upper(),  # EXPLORATORY/NUMERICAL_TEST_ONLY/DEFERRED
        "verdict_tier": "not_applicable",
        "numeric_thresholds": "not_applicable",
        "affects_v04_verdict": False,
        "parsed_K_H_mol_per_kg_per_Pa": parsed.get("K_H_mol_per_kg_per_Pa"),
        "parsed_Q_st_kJ_per_mol": parsed.get("Q_st_kJ_per_mol"),
        "reference": reference,
        "evidence": evidence,
        "notes": notes or [],
    }
    out_path = output_dir / f"{branch_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def emit_verdict_for_reference_blocked_strict_branch(
    output_dir: Path,
    case_id: str,
    branch_id: str,
    verdict_tier: str,
    numeric_thresholds_label: str,
    parsed: dict,
    reference: dict,
    erratum: dict,
    evidence: dict | None = None,
    notes: list[str] | None = None,
    site_truth_verdict_active: bool = False,
) -> Path:
    """Strict branch whose SCALAR K_H + Q_st reference has been reclassified
    as REFERENCE_BLOCKED (e.g. 5b Na-Rho per V04_REFERENCE_AUDIT.md §9.1).

    Distinct from BLOCKED (backend gap, no simulation possible) and FAIL
    (simulation done, real disagreement with verified reference). The branch
    REMAINS IN SCOPE — site_truth verdict may still be active.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_K_H_per_Pa = parsed.get("K_H_mol_per_kg_per_Pa")
    parsed_K_H_per_bar = parsed_K_H_per_Pa * 1e5 if parsed_K_H_per_Pa else None
    payload: dict[str, Any] = {
        "branch_id": branch_id,
        "case_id": case_id,
        "verdict": "REFERENCE_BLOCKED",
        "verdict_tier": verdict_tier,
        "numeric_thresholds": numeric_thresholds_label,
        "affects_v04_scalar_verdict": False,
        "branch_scope": "retained_in_v04",
        "site_truth_verdict_active": bool(site_truth_verdict_active),
        "erratum": erratum,
        # Simulator outputs retained for traceability
        "parsed_K_H_mol_per_kg_per_Pa": parsed_K_H_per_Pa,
        "parsed_K_H_mol_per_kg_per_bar": parsed_K_H_per_bar,
        "parsed_Q_st_kJ_per_mol": parsed.get("Q_st_kJ_per_mol"),
        "reference": reference,
        "evidence": evidence or {},
        "notes": notes or [],
    }
    out_path = output_dir / f"{branch_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def emit_verdict_for_blocked_branch(
    output_dir: Path,
    case_id: str,
    branch_id: str,
    verdict_tier: str,
    numeric_thresholds_label: str,
    blocked_info: dict[str, str],
    reference: dict,
    evidence: dict | None = None,
    notes: list[str] | None = None,
) -> Path:
    """A scientifically-blocked verdict-affecting branch.

    Used for 1a (Lin/Mercado) and 1b (Dzubak) where the RASPA3 v3.0.29 JSON
    force-field format cannot represent the prescribed pair potential. This
    is NOT a FAIL or a PASS; it is a deferred-pending-backend verdict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "branch_id": branch_id,
        "case_id": case_id,
        "verdict": "BLOCKED",
        "verdict_tier": verdict_tier,
        "numeric_thresholds": numeric_thresholds_label,
        "affects_v04_verdict": True,
        "blocked_reason": blocked_info.get("reason", ""),
        "prescribed_form": blocked_info.get("prescribed_form", ""),
        "required_action": blocked_info.get("required_action", ""),
        "parsed_K_H_mol_per_kg_per_Pa": None,
        "parsed_Q_st_kJ_per_mol": None,
        "reference": reference,
        "evidence": evidence or {},
        "notes": notes or [],
    }
    out_path = output_dir / f"{branch_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
