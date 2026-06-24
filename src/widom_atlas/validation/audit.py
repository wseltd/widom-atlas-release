"""Final-audit assembler.

Reads the 9 tables + parity outcome and writes a single
``FINAL_V04_VALIDATION_AUDIT.md`` with one of four allowed verdicts:

- PASS
- IMPLEMENTED BUT CASE COVERAGE INCOMPLETE
- EVALUATOR PARITY FAILED
- FAIL

Verdict logic:
1. If the parity gate fails (raspa3 not pass AND not skipped, OR
   <4/5 MOFX simin rows pass) → "EVALUATOR PARITY FAILED".
2. If parity passes (or is skipped-with-internal-only) AND all six
   flagship cases attempted, AND ≥4 flagship cases pass within their
   tier threshold → "PASS".
3. If parity passes but flagship coverage incomplete (some
   structure_missing / ff_missing) → "IMPLEMENTED BUT CASE COVERAGE
   INCOMPLETE".
4. Anything else → "FAIL".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from widom_atlas.evaluator.parity import ParityRow, assess_parity_outcome

from .case_runner import CaseResult

VerdictLiteral = Literal[
    "PASS",
    "IMPLEMENTED BUT CASE COVERAGE INCOMPLETE",
    "EVALUATOR PARITY FAILED",
    "FAIL",
]


def derive_verdict(
    *,
    cases: list[CaseResult],
    parity_rows: list[ParityRow],
    flagship_min_pass: int = 4,
) -> tuple[VerdictLiteral, dict[str, Any]]:
    parity_outcome = assess_parity_outcome(parity_rows)

    flagship = [c for c in cases if c.tier == "flagship"]
    flagship_n = len(flagship)
    flagship_passed = sum(1 for c in flagship if c.pass_overall)
    flagship_passed_or_no_ref = sum(
        1 for c in flagship if c.pass_overall or c.status == "passed_no_reference"
    )
    flagship_evaluator_ran = sum(
        1 for c in flagship if c.status in ("ok", "passed_no_reference")
    )
    flagship_missing_inputs = sum(
        1 for c in flagship if c.status in ("structure_missing", "ff_missing")
    )

    # A MOFX simin row counts as "uncovered" (rather than "failed") when the
    # internal evaluator side could not be run because per-record FF/structure
    # caches were not staged. This is the verdict-relevant failure mode in a
    # build where the operator did not pre-stage the FF/structure inputs.
    mofx_rows = [r for r in parity_rows if r.kind == "mofxdb_simin"]
    mofx_uncovered = sum(
        1 for r in mofx_rows if r.log10_KH_internal is None and "FF/structure cache" in r.notes
    )
    mofx_actual_failures = sum(
        1
        for r in mofx_rows
        if (not r.pass_overall)
        and not (r.log10_KH_internal is None and "FF/structure cache" in r.notes)
    )

    summary = {
        "parity_outcome": parity_outcome,
        "flagship_n": flagship_n,
        "flagship_evaluator_ran": flagship_evaluator_ran,
        "flagship_passed_overall": flagship_passed,
        "flagship_passed_or_no_reference": flagship_passed_or_no_ref,
        "flagship_missing_inputs": flagship_missing_inputs,
        "flagship_min_pass": flagship_min_pass,
        "mofx_uncovered_due_to_missing_inputs": mofx_uncovered,
        "mofx_actual_failures": mofx_actual_failures,
        "mofx_total_rows": len(mofx_rows),
    }

    parity_actually_failed = (
        mofx_actual_failures > 0
        or (
            not parity_outcome["raspa3_pass"]
            and not parity_outcome["raspa3_skipped"]
        )
    )
    parity_uncovered = (
        not parity_actually_failed
        and (parity_outcome["mofxdb_pass_count"] < parity_outcome["mofxdb_required"])
    )

    if parity_actually_failed:
        return "EVALUATOR PARITY FAILED", summary

    if parity_uncovered or (flagship_missing_inputs > 0 and flagship_evaluator_ran < flagship_n):
        return "IMPLEMENTED BUT CASE COVERAGE INCOMPLETE", summary

    if flagship_passed >= flagship_min_pass:
        return "PASS", summary

    return "FAIL", summary


def render_audit_markdown(
    *,
    cases: list[CaseResult],
    parity_rows: list[ParityRow],
    table_paths: dict[str, Path],
    extra_notes: str = "",
) -> str:
    verdict, summary = derive_verdict(cases=cases, parity_rows=parity_rows)
    parity_outcome = summary["parity_outcome"]

    lines: list[str] = []
    lines.append("# FINAL_V04_VALIDATION_AUDIT")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}_")
    lines.append("")
    lines.append("## Parity gate (Phase C)")
    lines.append("")
    lines.append(f"- RASPA3 reference parity passed: **{parity_outcome['raspa3_pass']}**")
    lines.append(f"- RASPA3 reference skipped (no raspa3 binary): **{parity_outcome['raspa3_skipped']}**")
    lines.append(
        f"- MOFX-DB simin parity passed: **{parity_outcome['mofxdb_pass_count']}/{parity_outcome['mofxdb_required']}** "
        f"required → {parity_outcome['mofxdb_pass']}"
    )
    lines.append(f"- Total parity rows recorded: {parity_outcome['n_rows']}")
    lines.append(f"- Overall parity pass: **{parity_outcome['overall_pass']}**")
    lines.append("")
    lines.append("## Flagship-case roll-up (Phase D, tier=flagship)")
    lines.append("")
    lines.append(f"- Cases attempted: {summary['flagship_n']}")
    lines.append(f"- Cases the evaluator ran on: {summary['flagship_evaluator_ran']}")
    lines.append(f"- Cases passing tier threshold: {summary['flagship_passed_overall']}")
    lines.append(
        f"- Cases passing or with no reference (evaluator ran): {summary['flagship_passed_or_no_reference']}"
    )
    lines.append(f"- Cases with missing structure or FF inputs: {summary['flagship_missing_inputs']}")
    lines.append(f"- Required pass count for verdict PASS: ≥ {summary['flagship_min_pass']}")
    lines.append("")
    lines.append("### Per-flagship-case detail")
    lines.append("")
    lines.append("| case_id | framework | gas | T_K | status | log10_KH (int / ref / Δ) | Q_ads kJ/mol (int / ref / Δ) | pass |")
    lines.append("|---------|-----------|-----|-----|--------|---------------------------|-------------------------------|------|")
    for c in cases:
        if c.tier != "flagship":
            continue
        log_part = f"{_fmt(c.log10_KH_internal)} / {_fmt(c.log10_KH_reference)} / {_fmt(c.delta_log10_KH)}"
        q_part = (
            f"{_fmt(c.Qads_internal_kJ_per_mol)} / {_fmt(c.Qads_reference_kJ_per_mol)} "
            f"/ {_fmt(c.delta_Qads_kJ_per_mol)}"
        )
        lines.append(
            f"| {c.case_id} | {c.framework_name} | {c.gas} | {c.temperature_K:g} | "
            f"{c.status} | {log_part} | {q_part} | {c.pass_overall} |"
        )
    lines.append("")
    lines.append("## Tables (machine-readable, schema_version=0.4)")
    lines.append("")
    for tid, path in sorted(table_paths.items()):
        lines.append(f"- {tid}: `{path.as_posix()}`")
    lines.append("")
    lines.append("## Verdict definitions")
    lines.append("")
    lines.append("- **PASS**: parity gate green AND ≥4/6 flagship cases pass the tier threshold.")
    lines.append("- **IMPLEMENTED BUT CASE COVERAGE INCOMPLETE**: parity green but some flagship inputs missing.")
    lines.append("- **EVALUATOR PARITY FAILED**: parity gate not green (raspa3 not pass+not skipped, or <4/5 MOFX).")
    lines.append("- **FAIL**: anything else.")
    if extra_notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(extra_notes)
    lines.append("")
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def write_audit(out_path: Path, markdown: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


__all__ = ["VerdictLiteral", "derive_verdict", "render_audit_markdown", "write_audit"]
