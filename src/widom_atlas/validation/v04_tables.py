"""Generate the 9 v0.4 validation-suite tables.

The release-gate audit requires exactly nine machine-readable JSON tables
under ``benchmarks/results/v0.4-validation/tables/``:

  T1 — flagship_case_results.json        (6 flagship cases, full breakdown)
  T2 — broad_coverage_summary.json       (broad-tier per-(MOF, gas, T) summary)
  T3 — exploratory_coverage_summary.json (exploratory-tier summary)
  T4 — convergence_evidence.json         (n_insertion ladder for each flagship case)
  T5 — charge_sensitivity.json           (DDEC6 / EQeq / PACMAN for one MOF)
  T6 — site_match_summary.json           (gas-loaded CIF site comparisons)
  T7 — provenance_inventory.json         (every dataset_id used + sha256 + license)
  T8 — registry_status.json              (registry datasets present/absent at run time)
  T9 — mofxdb_simin_parity.json          (the 5 deterministic MOFX parity rows)

This module accepts the case-runner outputs + parity rows and produces
all nine in one pass. Each table is wrapped in a small envelope:

  {
    "schema_version": "0.4",
    "table_id": "T1",
    "title": "...",
    "generated_at_utc": "...",
    "rows": [...],
    "notes": "..."
  }
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from widom_atlas.evaluator.parity import ParityRow

from .case_runner import CaseResult

_SCHEMA = "0.4"


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _envelope(table_id: str, title: str, rows: list[dict[str, Any]], notes: str = "") -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA,
        "table_id": table_id,
        "title": title,
        "generated_at_utc": _now_utc(),
        "rows": rows,
        "notes": notes,
        "n_rows": len(rows),
    }


def _flatten_case(c: CaseResult) -> dict[str, Any]:
    return {
        "case_id": c.case_id,
        "tier": c.tier,
        "status": c.status,
        "framework": c.framework_name,
        "gas": c.gas,
        "temperature_K": c.temperature_K,
        "n_insertions_used": c.n_insertions_used,
        "log10_KH_internal": c.log10_KH_internal,
        "log10_KH_reference": c.log10_KH_reference,
        "delta_log10_KH": c.delta_log10_KH,
        "Qads_internal_kJ_per_mol": c.Qads_internal_kJ_per_mol,
        "Qads_reference_kJ_per_mol": c.Qads_reference_kJ_per_mol,
        "delta_Qads_kJ_per_mol": c.delta_Qads_kJ_per_mol,
        "threshold_log10_KH": c.threshold_log10_KH,
        "threshold_Qads_kJ_per_mol": c.threshold_Qads_kJ_per_mol,
        "pass_log10_KH": c.pass_log10_KH,
        "pass_Qads": c.pass_Qads,
        "pass_overall": c.pass_overall,
        "framework_sha256": c.framework_sha256,
        "upf_sha256": c.upf_sha256,
        "reference_doi": c.reference_doi,
        "notes": c.notes,
        "warnings": c.warnings,
    }


def _flatten_parity(p: ParityRow) -> dict[str, Any]:
    return {**p.__dict__}


def t1_flagship_case_results(cases: list[CaseResult]) -> dict[str, Any]:
    rows = [_flatten_case(c) for c in cases if c.tier == "flagship"]
    return _envelope("T1", "Flagship case results", rows)


def t2_broad_coverage_summary(cases: list[CaseResult]) -> dict[str, Any]:
    rows = [_flatten_case(c) for c in cases if c.tier == "broad"]
    return _envelope("T2", "Broad-tier coverage summary", rows)


def t3_exploratory_coverage_summary(cases: list[CaseResult]) -> dict[str, Any]:
    rows = [_flatten_case(c) for c in cases if c.tier == "exploratory"]
    return _envelope("T3", "Exploratory-tier coverage summary", rows)


def t4_convergence_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _envelope("T4", "Convergence evidence (n_insertion ladder)", rows)


def t5_charge_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _envelope("T5", "Charge-scheme sensitivity (DDEC6 / EQeq / PACMAN)", rows)


def t6_site_match_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _envelope("T6", "Gas-loaded CIF site match summary", rows)


def t7_provenance_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _envelope("T7", "Provenance inventory (datasets used)", rows)


def t8_registry_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _envelope("T8", "Registry status at run time", rows)


def t9_mofxdb_simin_parity(parity_rows: list[ParityRow]) -> dict[str, Any]:
    rows = [_flatten_parity(p) for p in parity_rows if p.kind == "mofxdb_simin"]
    return _envelope("T9", "MOFX-DB simin parity (deterministic 5)", rows)


def write_all_tables(
    out_dir: Path,
    *,
    cases: list[CaseResult],
    parity_rows: list[ParityRow],
    convergence_rows: list[dict[str, Any]],
    charge_sensitivity_rows: list[dict[str, Any]],
    site_match_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
    registry_status_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write all 9 tables and return the mapping table_id → output path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, dict[str, Any]] = {
        "T1": t1_flagship_case_results(cases),
        "T2": t2_broad_coverage_summary(cases),
        "T3": t3_exploratory_coverage_summary(cases),
        "T4": t4_convergence_evidence(convergence_rows),
        "T5": t5_charge_sensitivity(charge_sensitivity_rows),
        "T6": t6_site_match_summary(site_match_rows),
        "T7": t7_provenance_inventory(provenance_rows),
        "T8": t8_registry_status(registry_status_rows),
        "T9": t9_mofxdb_simin_parity(parity_rows),
    }
    paths: dict[str, Path] = {}
    for tid, payload in tables.items():
        out = out_dir / f"{tid}_{payload['title'].lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        paths[tid] = out
    return paths


__all__ = [
    "t1_flagship_case_results",
    "t2_broad_coverage_summary",
    "t3_exploratory_coverage_summary",
    "t4_convergence_evidence",
    "t5_charge_sensitivity",
    "t6_site_match_summary",
    "t7_provenance_inventory",
    "t8_registry_status",
    "t9_mofxdb_simin_parity",
    "write_all_tables",
]
