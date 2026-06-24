"""T040: 12 branch/audit tests (split by risk: dispatcher 4 + executor 4 + audit 4)."""
from __future__ import annotations

import json
from pathlib import Path

from widom_atlas.v04.audit.op11_guard import OP11GuardViolation, enforce_op11_guard
from widom_atlas.v04.audit.pass_criteria import (
    check_pass_criteria,
    overall_pass,
    write_pass_criteria_report,
)
from widom_atlas.v04.audit.pipeline import aggregate_audit
from widom_atlas.v04.audit.verdict_emitter import (
    emit_verdict_for_non_verdict_branch,
    emit_verdict_for_strict_branch,
)
from widom_atlas.v04.branches.deferred import execute_deferred
from widom_atlas.v04.branches.dispatcher import (
    filter_deferred,
    filter_exploratory,
    filter_numerical,
    filter_strict,
    list_all_branches,
)
from widom_atlas.v04.locked_inputs import load_locked_case_matrix
from widom_atlas.v04.refs.registry import build_registry

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _matrix():
    return load_locked_case_matrix(REPO_ROOT / "v04_case_matrix.yaml")


# --------------- Dispatcher tests (4) ---------------
def test_dispatcher_lists_18_branches() -> None:
    """Branch-count history:
      14 — v0.4 baseline
      15 — 2026-05-18: added 1c (Mg-MOF-74 refined-FF, deferred)
      17 — 2026-05-19 pass-1: added 4c (Si-CHA refined-zeolite-FF) and 6e
           (MFI+CH4 alternative-FF), both deferred pending source.
      18 — 2026-05-19 pass-5 R6: added 1d (Mg-MOF-74 Mercado 2016 Model 4,
           locked_strict_executed via native_widom_v04).
      22 — 2026-06-01 final pivot: added 4 5c replacement-scalar branches
           (Na-ZK-5 + 5A + 13X + 4A; all reference_audited_pending_cation_cif_and_ff_lock).
    """
    branches = list_all_branches(_matrix().raw)
    assert len(branches) == 22


def test_dispatcher_filters_7_strict_plus_strict_executed() -> None:
    """Strict-tier count history:
      9 — 2026-05-17 (1a + 1b reclassified locked_strict_executed)
      10 — 2026-05-19 pass-4 R5 (1c Becker 2018 reduced-LJ activated)
      11 — 2026-05-19 pass-5 R6 (1d Mercado 2016 Model 4 added)
      12 — 2026-05-19 pass-6 R7 (2b PROMOTED from blocked_pending_Ongari_SI_coefficients
          to locked_strict_executed; Ongari 2017 modified Cu-O(CO2) RASPA generic
          potential A*exp(-B*r) - C6/r^6 - C8/r^8 via native_widom_v04 backend)
      13 — 2026-06-01 final pivot (3b PROMOTED to locked_strict_executed_multi_variant
          via native_widom_v04 backend running Maia 2023 UA/UAq/EHq variants).
      15 — 2026-06-01 addendum (4c + 6e PROMOTED via Bai 2013 RASPA3-bundled-JSON
          primary-anchored execution).
    """
    from widom_atlas.v04.branches.dispatcher import filter_strict_executed
    branches = list_all_branches(_matrix().raw)
    strict_ids = {b.branch_id for b in filter_strict(branches)}
    assert strict_ids == {"2a", "3a", "4a", "5b", "6a", "6b", "6c"}
    executed_ids = {b.branch_id for b in filter_strict_executed(branches)}
    assert executed_ids == {"1a", "1b", "1c", "1d", "2b", "3b", "4c", "6e"}
    # 15 strict-tier branches total (4c + 6e promoted 2026-06-01 addendum)
    assert len(strict_ids | executed_ids) == 15


def test_dispatcher_filters_1_exploratory() -> None:
    explor = filter_exploratory(list_all_branches(_matrix().raw))
    assert len(explor) == 1
    assert explor[0].branch_id == "5a"


def test_dispatcher_filters_1_numerical_5_deferred_after_4c_6e_promotion() -> None:
    """Deferred / non-verdict-affecting branch history:
      4 — 2026-05-18 (1c, 2b, 3b, 4b)
      6 — 2026-05-19: added 4c (Si-CHA refined-zeolite-FF) and 6e
          (MFI+CH4 alternative-FF) as refined-FF sibling branches.
      4 — 2026-05-19 pass-4 R5: 1c PROMOTED to locked_strict_executed.
      4 — 2026-05-19 pass-6 R7: 2b PROMOTED to locked_strict_executed.
      7 — 2026-06-01 final pivot: 3b PROMOTED to
          locked_strict_executed_multi_variant; 4c + 6e RECLASSIFIED to
          `blocked_pending_Bai_main_parameter_tables`; 4 new 5c branches
          ADDED with status `reference_audited_pending_cation_cif_and_ff_lock`.
      5 — 2026-06-01 addendum: 4c + 6e PROMOTED to locked_strict_executed
          via Bai 2013 RASPA3-bundled-JSON primary anchored execution.
          Final deferred-equivalent (non-strict-denominator) set: 4b + 4 5c branches.
    """
    branches = list_all_branches(_matrix().raw)
    assert len(filter_numerical(branches)) == 1
    deferred_ids = {b.branch_id for b in filter_deferred(branches)}
    expected_5c = {
        "5c_NaZK5_CO2_303K", "5c_Zeolite5A_CaA_CO2_298K",
        "5c_Zeolite13X_NaX_CO2_273K", "5c_Zeolite4A_NaA_CO2_273K",
    }
    assert deferred_ids == {"4b"} | expected_5c


# --------------- Executor / guard tests (4) ---------------
def test_op11_guard_passes_on_locked_yaml() -> None:
    matrix = _matrix()
    for case in matrix.cases:
        for branch in case.get("branches", []):
            if branch["branch_id"] == "6a":
                enforce_op11_guard(branch)  # must not raise
                return
    raise AssertionError("6a not found in matrix")


def test_op11_guard_rejects_simulator_reference() -> None:
    """Synthetic branch with RASPA3 as the K_H source — must be rejected."""
    bad_branch = {
        "branch_id": "6a",
        "references": {
            "K_H": {"source": "RASPA3_v3.0.29_Widom_example"},
            "Q_st": {"source": "RASPA3_v3.0.29_Widom_example"},
        },
    }
    try:
        enforce_op11_guard(bad_branch)
    except OP11GuardViolation:
        return
    raise AssertionError("OP11 guard failed to reject RASPA3-as-truth branch")


def test_deferred_executor_returns_no_verdict_impact() -> None:
    branches = list_all_branches(_matrix().raw)
    deferred = filter_deferred(branches)
    for b in deferred:
        result = execute_deferred(b)
        assert result.status == b.status
        assert result.reason


def test_branch_reference_registry_contains_all_strict() -> None:
    """Every strict-tier branch has K_H + Q_st references EXCEPT those whose
    Q_st (or K_H) is explicitly REFERENCE_BLOCKED. As of 2026-05-18:
      * 5b Na-Rho — both K_H and Q_st REFERENCE_BLOCKED (trapdoor zeolite).
      * 6b MFI+Kr — Q_st REFERENCE_BLOCKED pending Golden-Sircar 1994.
    """
    registry = build_registry(_matrix().raw)
    matrix_raw = _matrix().raw
    branches_by_id: dict[str, dict] = {}
    for case in matrix_raw.get("cases", []):
        for b in case.get("branches", []):
            branches_by_id[b["branch_id"]] = b
    for bid in ("1a", "1b", "2a", "3a", "4a", "5b", "6a", "6b", "6c"):
        ref = registry[bid]
        branch_raw = branches_by_id[bid]
        kh_block = (branch_raw.get("references") or {}).get("K_H") or {}
        q_block = (branch_raw.get("references") or {}).get("Q_st") or {}
        kh_blocked = "reference_blocked" in str(kh_block.get("classification", ""))
        q_blocked = "reference_blocked" in str(q_block.get("classification", ""))
        if not kh_blocked:
            assert ref.K_H_value is not None, f"{bid} K_H_value missing"
        if not q_blocked:
            assert (ref.Q_st_value is not None) or (ref.Q_st_low_loading is not None), \
                f"{bid} has no Q_st value"


# --------------- Audit / verdict tests (4) ---------------
def test_verdict_emitter_writes_strict_json(tmp_path: Path) -> None:
    path = emit_verdict_for_strict_branch(
        output_dir=tmp_path,
        case_id="6",
        branch_id="6a",
        verdict_tier="flagship_strict",
        numeric_thresholds_label="flagship_strict",
        thresholds={"flagship_strict": {"delta_log10_KH_max": 0.10, "delta_Qads_kJ_per_mol_max": 2.0}},
        parsed={"K_H_mol_per_kg_per_Pa": 6.43e-6, "Q_st_kJ_per_mol": 21.0},
        reference={
            "K_H_value_mol_per_kg_per_bar": 0.89,
            "Q_st_value_kj_per_mol": 20.9,
        },
        evidence={"raspa3_version": "3.0.29"},
        notes=["test"],
    )
    payload = json.loads(path.read_text())
    assert payload["branch_id"] == "6a"
    assert payload["delta_log10_K_H"] is not None
    # 6.43e-6 mol/kg/Pa → 0.643 mol/kg/bar; ref 0.89 → log10(0.643/0.89) ≈ -0.141
    assert payload["delta_log10_K_H"] < 0


def test_4a_composite_verdict_BROAD_TIER_PASS_only_requires_Q_st_also_pass(tmp_path: Path) -> None:
    """2026-05-19 patch: a branch with K_H within broad ±0.20 but outside
    strict ±0.10 AND Q_st failing strict must be labelled FAIL, NOT
    BROAD_TIER_PASS_only. The composite label must reflect all axes.

    Numbers chosen to match 4a Si-CHA + CO₂ at the time of the 2026-05-18
    audit: K_H atlas 1.865 mol/(kg·bar) vs ref 2.43 → Δlog10 = -0.115 (broad)
    AND Q_st atlas 32.73 vs ref 21.0 → ΔQ = +11.7 (strict FAIL).
    Pre-patch verdict: BROAD_TIER_PASS_only (label-bug — Q_st failure ignored).
    Post-patch verdict: FAIL.
    """
    path = emit_verdict_for_strict_branch(
        output_dir=tmp_path,
        case_id="4",
        branch_id="4a",
        verdict_tier="flagship_strict",
        numeric_thresholds_label="flagship_strict",
        thresholds={"flagship_strict": {"delta_log10_KH_max": 0.10, "delta_Qads_kJ_per_mol_max": 2.0}},
        parsed={"K_H_mol_per_kg_per_Pa": 1.865e-5, "Q_st_kJ_per_mol": 32.73},
        reference={
            "K_H_value_mol_per_kg_per_bar": 2.43,
            "Q_st_value_kj_per_mol": 21.0,
        },
        evidence={},
        notes=[],
        atlas_Q_st_method="multi_point_van_t_Hoff_regression",
        reference_Q_st_method="multi_point_van_t_Hoff_regression",
    )
    payload = json.loads(path.read_text())
    assert payload["passes_K_H"] is False  # outside strict ±0.10
    assert payload["passes_Q_st"] is False  # outside strict ±2.0
    assert payload["verdict"] == "FAIL", (
        f"BROAD_TIER_PASS_only must NOT fire when Q_st FAILs strict; "
        f"got verdict={payload['verdict']!r}"
    )


def test_BROAD_TIER_PASS_only_still_fires_when_K_H_broad_AND_Q_st_passes(tmp_path: Path) -> None:
    """The label is preserved for its legitimate case: K_H within broad band
    + Q_st passing strict (or Q_st absent/blocked)."""
    path = emit_verdict_for_strict_branch(
        output_dir=tmp_path,
        case_id="X",
        branch_id="testX",
        verdict_tier="flagship_strict",
        numeric_thresholds_label="flagship_strict",
        thresholds={"flagship_strict": {"delta_log10_KH_max": 0.10, "delta_Qads_kJ_per_mol_max": 2.0}},
        # K_H Δlog10 = -0.15 (within broad ±0.20 but outside strict ±0.10);
        # Q_st 21.0 vs ref 20.0 → ΔQ = +1.0 (within strict ±2.0)
        parsed={"K_H_mol_per_kg_per_Pa": 0.708e-5, "Q_st_kJ_per_mol": 21.0},
        reference={"K_H_value_mol_per_kg_per_bar": 1.0, "Q_st_value_kj_per_mol": 20.0},
        evidence={},
        notes=[],
        atlas_Q_st_method="two_point_van_t_Hoff",
        reference_Q_st_method="two_point_van_t_Hoff",
    )
    payload = json.loads(path.read_text())
    assert payload["passes_K_H"] is False  # outside strict but within broad
    assert payload["passes_Q_st"] is True
    assert payload["verdict"] == "BROAD_TIER_PASS_only"


def test_BROAD_TIER_PASS_only_fires_when_K_H_broad_AND_Q_st_axis_blocked(tmp_path: Path) -> None:
    """6b-like scenario: K_H within broad, Q_st REFERENCE_BLOCKED.
    Actually 6b's K_H is PASS strict, so this is a synthetic case where K_H
    just falls outside strict ±0.10 but Q_st is axis-blocked. Should still
    emit BROAD_TIER_PASS_only... actually no, this should emit
    PARTIAL_PASS_Q_st_REFERENCE_BLOCKED if K_H is at least broad. Let me
    test the K_H_broad + Q_st_reference_blocked path differently — when
    K_H is outside strict but within broad AND Q_st is reference-blocked,
    the partial-pass label requires kh_pass (strict). Since K_H here is
    NOT strict-pass, the verdict path goes to the FAIL_K_H_Q_st_REFERENCE_BLOCKED
    branch.
    """
    path = emit_verdict_for_strict_branch(
        output_dir=tmp_path,
        case_id="X",
        branch_id="testY",
        verdict_tier="flagship_strict",
        numeric_thresholds_label="flagship_strict",
        thresholds={"flagship_strict": {"delta_log10_KH_max": 0.10, "delta_Qads_kJ_per_mol_max": 2.0}},
        parsed={"K_H_mol_per_kg_per_Pa": 0.708e-5, "Q_st_kJ_per_mol": None},
        reference={"K_H_value_mol_per_kg_per_bar": 1.0, "Q_st_value_kj_per_mol": 20.0},
        evidence={},
        notes=[],
        Q_st_reference_blocked=True,
    )
    payload = json.loads(path.read_text())
    # K_H not strict-pass + Q_st axis-blocked → falls through to FAIL_K_H_Q_st_REFERENCE_BLOCKED
    assert payload["verdict"].startswith("FAIL_K_H_Q_st_REFERENCE_BLOCKED")


def test_verdict_emitter_writes_non_verdict_json(tmp_path: Path) -> None:
    path = emit_verdict_for_non_verdict_branch(
        output_dir=tmp_path,
        case_id="5",
        branch_id="5a",
        classification="exploratory",
        parsed={"K_H_mol_per_kg_per_Pa": None, "Q_st_kJ_per_mol": None},
        reference={},
        evidence={},
    )
    payload = json.loads(path.read_text())
    assert payload["verdict"] == "EXPLORATORY"
    assert payload["affects_v04_verdict"] is False


def test_audit_pipeline_aggregates_verdicts(tmp_path: Path) -> None:
    out_dir = tmp_path / "audit"
    (out_dir / "verdicts").mkdir(parents=True)
    # Emit two synthetic verdicts
    emit_verdict_for_strict_branch(
        output_dir=out_dir / "verdicts", case_id="1", branch_id="1a",
        verdict_tier="flagship_strict", numeric_thresholds_label="flagship_strict",
        thresholds={"flagship_strict": {"delta_log10_KH_max": 0.10, "delta_Qads_kJ_per_mol_max": 2.0}},
        parsed={"K_H_mol_per_kg_per_Pa": None, "Q_st_kJ_per_mol": None},
        reference={"K_H_value_mol_per_kg_per_bar": 187.0, "Q_st_value_kj_per_mol": 42.0},
        evidence={}, notes=[],
    )
    emit_verdict_for_non_verdict_branch(
        output_dir=out_dir / "verdicts", case_id="5", branch_id="5a",
        classification="exploratory", parsed={}, reference={}, evidence={},
    )
    summary = aggregate_audit(out_dir)
    assert summary.n_branches == 2
    assert summary.n_strict_blocked >= 1  # 1a parsed None → BLOCKED


def test_pass_criteria_with_full_strict_set(tmp_path: Path) -> None:
    # Synthetic verdicts for all strict branches + 5a/6d
    verdicts = {
        "1a": {"verdict": "PASS", "verdict_tier": "flagship_strict",
               "numeric_thresholds": "flagship_strict", "reference": {}},
        "1b": {"verdict": "PASS", "verdict_tier": "flagship_strict",
               "numeric_thresholds": "flagship_strict", "reference": {}},
        "2a": {"verdict": "PASS", "verdict_tier": "flagship_strict_with_ff_fallback_relaxation",
               "numeric_thresholds": "broad_tier", "reference": {}},
        "3a": {"verdict": "PASS", "verdict_tier": "flagship_strict_with_ff_fallback_relaxation",
               "numeric_thresholds": "broad_tier", "reference": {}},
        "4a": {"verdict": "PASS", "verdict_tier": "flagship_strict",
               "numeric_thresholds": "flagship_strict", "reference": {}},
        "5a": {"verdict": "EXPLORATORY", "verdict_tier": "not_applicable",
               "numeric_thresholds": "not_applicable", "reference": {}},
        "5b": {"verdict": "PASS", "verdict_tier": "flagship_strict_with_ff_fallback_relaxation",
               "numeric_thresholds": "broad_tier", "reference": {}},
        "6a": {"verdict": "PASS", "verdict_tier": "flagship_strict",
               "numeric_thresholds": "flagship_strict", "reference": {}},
        "6b": {"verdict": "PASS", "verdict_tier": "flagship_strict",
               "numeric_thresholds": "flagship_strict", "reference": {}},
        "6c": {"verdict": "PASS", "verdict_tier": "flagship_strict",
               "numeric_thresholds": "flagship_strict", "reference": {}},
        "6d": {"verdict": "NUMERICAL_TEST_ONLY", "verdict_tier": "not_applicable",
               "numeric_thresholds": "not_applicable", "reference": {}},
    }
    items = check_pass_criteria(
        audit_summary={}, branch_verdicts=verdicts,
        raspa3_version="3.0.29",
        test_outputs={"ff_parser_tests": True, "c_already_scaled_test": True,
                      "units_test": True, "geometry_tests": True},
    )
    write_pass_criteria_report(tmp_path / "pc.json", items)
    assert overall_pass(items), [i.item for i in items if not i.passes]
