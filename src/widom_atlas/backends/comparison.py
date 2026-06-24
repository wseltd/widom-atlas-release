"""Backend comparison report.

Per ``implementation-verdict-continuation.txt`` §"Required comparison
outputs": compare ``parameterised_lj`` / ``user_parameterised_coulomb_lj``
/ ``external_samples`` / ``raspa3_external`` runs side-by-side on the
same material/gas, showing whether the heavier backend improved over the
LJ-only baseline.

Each comparison source can be:

- A benchmark run directory (``run_benchmark_set`` output) — contains
  ``benchmark_run.json`` plus per-material atlas runs.
- A convergence run directory (``run_convergence_study`` output) —
  contains ``convergence_summary.json`` with per-N steps.
- A scalar-only RASPA3 sidecar JSON written by
  :func:`widom_atlas.backends.raspa3_ingest.write_scalar_only_sidecar`.

The report is a Markdown + JSON pair. The JSON payload is
machine-readable; the Markdown is meant for the audit doc.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_RowKind = Literal["benchmark_material", "convergence_step", "scalar_only"]


@dataclass(frozen=True)
class ComparisonRow:
    """One row in a backend-comparison table.

    Each backend × material × gas yields one row. ``baseline_label`` and
    ``baseline_KH`` / ``baseline_Qads`` are filled when the row is the
    LJ-only reference for the same material so an "improved?" column can
    be written without extra plumbing.
    """

    backend_label: str
    backend_category: str
    material: str
    gas: str
    temperature_K: float | None
    n_insertions: int | None
    KH: float | None
    KH_std: float | None
    Qads_kJmol: float | None
    Qads_std_kJmol: float | None
    accessible_fraction: float | None
    n_basins: int | None
    dominant_basin_weight: float | None
    dominant_centroid_frac: tuple[float, float, float] | None
    site_match_distance_A: float | None
    site_match_confidence: str | None
    scalar_reference_KH: float | None
    scalar_reference_Qads_kJmol: float | None
    scalar_reference_DOI: str | None
    KH_log_ratio_vs_reference: float | None
    Qads_delta_kJmol_vs_reference: float | None
    warnings: list[str]
    source_kind: _RowKind
    source_path: str
    extra: dict[str, Any]


def _safe_log_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return float(math.log(a / b))


def _row_from_benchmark_aggregate(
    aggregate_path: Path,
    material_record: dict[str, Any],
    scalar_match: dict[str, Any] | None = None,
) -> ComparisonRow:
    md = material_record
    backend_label = md.get("backend") or md.get("calculator") or "unknown"
    backend_category = md.get("backend") or "parameterised_lj"
    centroid: tuple[float, float, float] | None = None
    cf = md.get("dominant_centroid_frac")
    if isinstance(cf, list) and len(cf) == 3:
        centroid = (float(cf[0]), float(cf[1]), float(cf[2]))
    KH = md.get("henry_coefficient")
    Qads = md.get("heat_of_adsorption_kJmol")
    KH_ref = scalar_match.get("KH_ref") if scalar_match else None
    Qads_ref = scalar_match.get("Qads_ref_kJmol") if scalar_match else None
    return ComparisonRow(
        backend_label=str(backend_label),
        backend_category=str(backend_category),
        material=str(md.get("material_id", "unknown")),
        gas=str(md.get("gas", "unknown")),
        temperature_K=md.get("temperature_K"),
        n_insertions=None,  # not always recorded per-material
        KH=KH,
        KH_std=None,
        Qads_kJmol=Qads,
        Qads_std_kJmol=None,
        accessible_fraction=None,
        n_basins=md.get("basins_count"),
        dominant_basin_weight=None,
        dominant_centroid_frac=centroid,
        site_match_distance_A=None,
        site_match_confidence=None,
        scalar_reference_KH=KH_ref,
        scalar_reference_Qads_kJmol=Qads_ref,
        scalar_reference_DOI=scalar_match.get("source_url") if scalar_match else None,
        KH_log_ratio_vs_reference=_safe_log_ratio(KH, KH_ref),
        Qads_delta_kJmol_vs_reference=(
            (Qads - Qads_ref) if (Qads is not None and Qads_ref is not None) else None
        ),
        warnings=list(md.get("warnings") or []),
        source_kind="benchmark_material",
        source_path=str(aggregate_path),
        extra={
            "samples_origin": md.get("samples_origin"),
            "calculator": md.get("calculator"),
            "status": md.get("status"),
            "license": md.get("license"),
        },
    )


def _row_from_convergence_step(
    summary_path: Path,
    summary: dict[str, Any],
    step: dict[str, Any],
) -> ComparisonRow:
    bs = step.get("basins_summary") or {}
    centroid: tuple[float, float, float] | None = None
    cf = bs.get("dominant_centroid_frac")
    if isinstance(cf, list) and len(cf) == 3:
        centroid = (float(cf[0]), float(cf[1]), float(cf[2]))
    return ComparisonRow(
        backend_label=str(step.get("calculator") or summary.get("backend") or "unknown"),
        backend_category=str(summary.get("backend_name") or "parameterised_lj"),
        material=str(summary.get("material_id", "unknown")),
        gas=str(summary.get("gas", "unknown")),
        temperature_K=summary.get("temperature_K"),
        n_insertions=int(step.get("n_insertions", 0)) or None,
        KH=step.get("KH"),
        KH_std=step.get("KH_std"),
        Qads_kJmol=step.get("Qads_kJmol"),
        Qads_std_kJmol=step.get("Qads_std_kJmol"),
        accessible_fraction=step.get("accessible_fraction"),
        n_basins=bs.get("n_basins"),
        dominant_basin_weight=bs.get("dominant_weight"),
        dominant_centroid_frac=centroid,
        site_match_distance_A=None,
        site_match_confidence=None,
        scalar_reference_KH=None,
        scalar_reference_Qads_kJmol=None,
        scalar_reference_DOI=None,
        KH_log_ratio_vs_reference=None,
        Qads_delta_kJmol_vs_reference=None,
        warnings=[],
        source_kind="convergence_step",
        source_path=str(summary_path),
        extra={
            "thresholds": summary.get("thresholds"),
            "verdict": summary.get("verdict"),
            "step_runtime_s": step.get("runtime_s"),
        },
    )


def _row_from_scalar_sidecar(sidecar_path: Path) -> ComparisonRow:
    payload = json.loads(Path(sidecar_path).read_text())
    return ComparisonRow(
        backend_label=f"raspa3_external ({payload.get('raspa_version', 'unknown')})",
        backend_category="raspa3_external",
        material=str(payload.get("framework", "unknown")),
        gas=str(payload.get("gas", "unknown")),
        temperature_K=payload.get("temperature_K"),
        n_insertions=payload.get("n_insertions"),
        KH=payload.get("henry_coefficient_mol_per_kg_per_Pa"),
        KH_std=None,
        Qads_kJmol=payload.get("heat_of_adsorption_kJ_per_mol"),
        Qads_std_kJmol=None,
        accessible_fraction=None,
        n_basins=None,
        dominant_basin_weight=None,
        dominant_centroid_frac=None,
        site_match_distance_A=None,
        site_match_confidence=None,
        scalar_reference_KH=None,
        scalar_reference_Qads_kJmol=None,
        scalar_reference_DOI=None,
        KH_log_ratio_vs_reference=None,
        Qads_delta_kJmol_vs_reference=None,
        warnings=list(payload.get("warnings") or []),
        source_kind="scalar_only",
        source_path=str(sidecar_path),
        extra={
            "force_field_label": payload.get("force_field_label"),
            "framework_charge_source": payload.get("framework_charge_source"),
            "gas_model": payload.get("gas_model"),
            "atlas_input": payload.get("atlas_input", False),
        },
    )


def collect_rows_from_path(path: Path) -> list[ComparisonRow]:
    """Auto-detect the source kind and collect comparison rows."""
    p = Path(path)
    if p.is_dir():
        bench = p / "benchmark_run.json"
        if bench.exists():
            agg = json.loads(bench.read_text())
            scalars_p = p / "scalar_comparison" / "scalar_comparison.json"
            scalars: list[dict[str, Any]] = []
            if scalars_p.exists():
                scalars = json.loads(scalars_p.read_text()) or []
            scalar_by_mat = {
                (str(r.get("material_id")), str(r.get("gas"))): r for r in scalars
            }
            rows: list[ComparisonRow] = []
            for m in agg.get("materials", []):
                key = (str(m.get("material_id")), str(m.get("gas")))
                rows.append(_row_from_benchmark_aggregate(bench, m, scalar_by_mat.get(key)))
            return rows
        conv = p / "convergence_summary.json"
        if conv.exists():
            summary = json.loads(conv.read_text())
            return [_row_from_convergence_step(conv, summary, s) for s in summary.get("steps", [])]
        raise ValueError(f"compare-backends: directory {p} has neither benchmark_run.json nor convergence_summary.json")
    if p.suffix == ".json":
        return [_row_from_scalar_sidecar(p)]
    raise ValueError(f"compare-backends: unsupported source {p}")


def write_comparison_report(
    sources: list[Path],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Build a JSON + Markdown comparison report from N source paths."""
    rows: list[ComparisonRow] = []
    for s in sources:
        rows.extend(collect_rows_from_path(Path(s)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON payload
    json_payload = [
        {
            "backend_label": r.backend_label,
            "backend_category": r.backend_category,
            "material": r.material,
            "gas": r.gas,
            "temperature_K": r.temperature_K,
            "n_insertions": r.n_insertions,
            "KH": r.KH,
            "KH_std": r.KH_std,
            "Qads_kJmol": r.Qads_kJmol,
            "Qads_std_kJmol": r.Qads_std_kJmol,
            "accessible_fraction": r.accessible_fraction,
            "n_basins": r.n_basins,
            "dominant_basin_weight": r.dominant_basin_weight,
            "dominant_centroid_frac": list(r.dominant_centroid_frac) if r.dominant_centroid_frac else None,
            "site_match_distance_A": r.site_match_distance_A,
            "site_match_confidence": r.site_match_confidence,
            "scalar_reference_KH": r.scalar_reference_KH,
            "scalar_reference_Qads_kJmol": r.scalar_reference_Qads_kJmol,
            "scalar_reference_DOI": r.scalar_reference_DOI,
            "KH_log_ratio_vs_reference": r.KH_log_ratio_vs_reference,
            "Qads_delta_kJmol_vs_reference": r.Qads_delta_kJmol_vs_reference,
            "warnings": r.warnings,
            "source_kind": r.source_kind,
            "source_path": r.source_path,
            "extra": r.extra,
        }
        for r in rows
    ]
    out_json = out_dir / "backend_comparison.json"
    out_json.write_text(json.dumps(json_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    # Markdown: group by (material, gas)
    grouped: dict[tuple[str, str], list[ComparisonRow]] = {}
    for r in rows:
        grouped.setdefault((r.material, r.gas), []).append(r)

    md_lines = [
        "# widom-atlas — backend comparison",
        "",
        f"Sources ({len(sources)}):",
    ]
    for s in sources:
        md_lines.append(f"- `{s}`")
    md_lines.append("")
    md_lines.append(
        "Comparison rows below are grouped by (material, gas). "
        "`improved?` is computed against the **LJ-only** baseline within the same group "
        "(the `parameterised_lj` or `toy_lj` row, whichever is present); a row is "
        "flagged `yes` only when |Q_ads delta vs reference| or |KH log_ratio vs reference| "
        "is strictly smaller than the LJ-only baseline's. `—` means no comparable "
        "reference / baseline available."
    )
    md_lines.append("")

    for (mat, gas), rs in sorted(grouped.items()):
        md_lines.append(f"## {mat} + {gas}")
        md_lines.append("")
        # Find LJ-only baseline within the group
        baseline = next((r for r in rs if r.backend_category in ("toy_lj", "parameterised_lj")), None)
        b_KH_lr = baseline.KH_log_ratio_vs_reference if baseline else None
        b_dQ = baseline.Qads_delta_kJmol_vs_reference if baseline else None
        md_lines.append(
            "| backend | n | KH (mol/(kg·Pa)) | KH log_ratio | Q_ads (kJ/mol) | ΔQ_ads | dominant centroid (frac) | improved? | source | warnings |"
        )
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            improved = "—"
            if baseline is not None and r is not baseline:
                cur_lr = (
                    abs(r.KH_log_ratio_vs_reference) if r.KH_log_ratio_vs_reference is not None else None
                )
                cur_dq = (
                    abs(r.Qads_delta_kJmol_vs_reference)
                    if r.Qads_delta_kJmol_vs_reference is not None
                    else None
                )
                base_lr = abs(b_KH_lr) if b_KH_lr is not None else None
                base_dq = abs(b_dQ) if b_dQ is not None else None
                votes = []
                if cur_lr is not None and base_lr is not None:
                    votes.append(cur_lr < base_lr)
                if cur_dq is not None and base_dq is not None:
                    votes.append(cur_dq < base_dq)
                if votes:
                    improved = "yes" if all(votes) else "no" if not any(votes) else "partial"
            cf = r.dominant_centroid_frac
            cf_s = f"({cf[0]:.3f}, {cf[1]:.3f}, {cf[2]:.3f})" if cf else "—"
            warn_s = f"{len(r.warnings)} warning(s)" if r.warnings else "—"
            md_lines.append(
                f"| {r.backend_category} ({r.backend_label[:60]}) "
                f"| {r.n_insertions or '—'} "
                f"| {r.KH if r.KH is None else format(r.KH, '.3e')} "
                f"| {r.KH_log_ratio_vs_reference if r.KH_log_ratio_vs_reference is None else format(r.KH_log_ratio_vs_reference, '+.2f')} "
                f"| {r.Qads_kJmol if r.Qads_kJmol is None else format(r.Qads_kJmol, '+.2f')} "
                f"| {r.Qads_delta_kJmol_vs_reference if r.Qads_delta_kJmol_vs_reference is None else format(r.Qads_delta_kJmol_vs_reference, '+.2f')} "
                f"| {cf_s} "
                f"| {improved} "
                f"| `{Path(r.source_path).name}` "
                f"| {warn_s} |"
            )
        md_lines.append("")
    out_md = out_dir / "backend_comparison.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out_json, out_md


__all__ = [
    "ComparisonRow",
    "collect_rows_from_path",
    "write_comparison_report",
]
