"""Launch-readiness report (Markdown + HTML + JSON) — explicit honesty about caveats."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, ConfigDict, Field

_FORBIDDEN_CLAIM_WORDS = ("validated", "proven", "guarantees", "guaranteed")
_FORBIDDEN_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _FORBIDDEN_CLAIM_WORDS) + r")\b",
    re.IGNORECASE,
)

# REPAIR-6: explicit exclusion blocks rendered into every launch report so the
# launch-readiness document is self-contained — operators reading only the
# launch report (not the README) still see the v1 guard rails.
_LAUNCH_REPORT_SECTIONS: dict[str, list[str]] = {
    "excluded_tools": [
        "LLMs / Kosmos-style agents / Nemotron / Gemma — out of scope for v1.",
        "ALCHEMI — not a v1 dependency.",
        "Materials Studio (commercial) — never a v1 dependency.",
        "OVITO Pro (commercial) — never a v1 dependency.",
        "Mayavi — excluded; rendering is matplotlib only.",
        "LAMMPS / Quantum ESPRESSO / CP2K — external comparators only, not v1 dependencies.",
        "PorousMaterials.jl — Julia, not a Python dependency.",
        "EQeq / Qeq automatic charge assignment — excluded in v1.",
        "ML potentials as core dependencies — excluded; outputs may be analysed but the engines are not deps.",
    ],
    "h2o_exclusion": [
        "v1 allowed gases: CO2, N2, CH4 only.",
        "H2O is excluded because force-field + charge handling for water is not yet explicit in this codebase.",
        "AtlasInput, MOFXDBRecord and related schemas reject 'H2O' at construction time.",
    ],
    "automatic_defect_engine_exclusion": [
        "Curated atom-removal only via `widom_atlas.perturb.remove_atoms`; explicit indices required.",
        "No linker-vacancy generation, no missing-node enumeration, no functional-group substitution, no random defect sampling in v1.",
        "Automatic MOF defect chemistry is excluded because v1 cannot guarantee chemical correctness of generated defects.",
    ],
    "commercial_dependency_exclusion": [
        "No commercial dependencies are required by this package.",
        "Materials Studio, OVITO Pro, RASPA Pro and similar commercial tools are not redistributed.",
        "Optional comparator integration via documented external CLIs only — no commercial code is bundled.",
    ],
    "license_risky_dataset_exclusion": [
        "CSD-derived raw datasets are not bundled or fetched (CSD redistribution is restricted).",
        "IZA zeolite database bulk-redistribution is unclear (verdict §6); IZA CIFs are not committed and not auto-fetched.",
        "Any benchmark material whose license is non-permissive is rejected at registry / fetch time.",
    ],
    "what_this_report_does_not_certify": [
        "This report does NOT constitute proof of quantitative accuracy.",
        "Scalar comparisons against MOFX-DB / NIST / literature are TREND-only — they tell you whether the package and the reference produce comparable values, not that they agree to within a stated error.",
        "Symmetry assignments on defective, strained, or low-symmetry hosts are reported with explicit grouping_confidence and uncertainty_flags; do not interpret a high group_confidence on a perturbed framework as a chemistry claim.",
        "Henry-coefficient and isosteric-heat estimators in this package are heuristic Boltzmann-weighted summaries of insertion samples; they are not equivalent to a properly converged GCMC or DFT-level free-energy calculation.",
    ],
    "toy_lj_caveat": [
        "Per implementation-verdict.txt §13.J, the v1 smoke-test calculator is ASE Lennard-Jones.",
        "Toy Lennard-Jones outputs sample the cell topology but do NOT capture chemistry-specific dispersion, electrostatics, or open-metal-site interactions.",
        "Reports built from toy LJ samples are tagged `synthetic_toy_lj` in metadata; they exercise the pipeline but do not establish chemistry.",
    ],
}


class LaunchReadinessReport(BaseModel):
    """Pydantic record of the launch report payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package_version: str
    set_name: str
    gas: str
    temperature_K: float = Field(gt=0.0)
    totals: dict[str, int]
    trend_counts: dict[str, int]
    provenance: list[dict[str, Any]]
    failed: list[dict[str, Any]]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _build_provenance_table(materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in materials:
        if m.get("status") not in {"ok", "failed"}:
            continue
        rows.append(
            {
                "material_id": m.get("material_id", ""),
                "source": m.get("source", ""),
                "license": m.get("license", ""),
                "sha256": m.get("cif_sha256", ""),
                "dataset_version": m.get("dataset_version", "unknown"),
            }
        )
    return rows


def _summarise_scalar_labels(scalar_payload: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in scalar_payload:
        label = str(row.get("comparison_label", "UNAVAILABLE"))
        counter[label] += 1
    for required in ("TREND", "UNAVAILABLE", "IDENTITY_UNCERTAIN", "OUT_OF_RANGE"):
        counter.setdefault(required, 0)
    return dict(counter)


def _md_env() -> Environment:
    return Environment(
        loader=PackageLoader("widom_atlas.benchmarks", "templates"),
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def _html_env() -> Environment:
    return Environment(
        loader=PackageLoader("widom_atlas.benchmarks", "templates"),
        autoescape=select_autoescape(
            enabled_extensions=("html", "html.j2"),
            default_for_string=True,
            default=True,
        ),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def _render_markdown(context: dict[str, Any], out_path: Path) -> Path:
    env = _md_env()
    text = env.get_template("launch_report.md.j2").render(**context)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _render_html(context: dict[str, Any], out_path: Path) -> Path:
    env = _html_env()
    text = env.get_template("launch_report.html.j2").render(**context)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_launch_report(
    benchmark_run_path: Path,
    scalar_comparison_path: Path,
    output_dir: Path,
) -> LaunchReadinessReport:
    """Render ``launch_report.{md,html,json}`` from benchmark + scalar comparison outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads(Path(benchmark_run_path).read_text(encoding="utf-8"))
    scalar_payload = json.loads(Path(scalar_comparison_path).read_text(encoding="utf-8"))

    materials = aggregate.get("materials", [])
    totals = {
        "attempted": len(materials),
        "succeeded": sum(1 for m in materials if m.get("status") == "ok"),
        "failed": sum(1 for m in materials if m.get("status") == "failed"),
        "skipped": sum(1 for m in materials if m.get("status") == "skipped"),
    }
    failed = [m for m in materials if m.get("status") == "failed"]
    provenance = _build_provenance_table(materials)
    trend_counts = _summarise_scalar_labels(scalar_payload)

    report = LaunchReadinessReport(
        package_version=str(aggregate.get("package_version", "0.0.0")),
        set_name=str(aggregate.get("set_name", "small")),
        gas=str(aggregate.get("gas", "CO2")),
        temperature_K=float(aggregate.get("temperature_K", 298.15)),
        totals=totals,
        trend_counts=trend_counts,
        provenance=provenance,
        failed=failed,
    )

    context = {
        "package_version": report.package_version,
        "set_name": report.set_name,
        "gas": report.gas,
        "temperature_K": report.temperature_K,
        "totals": report.totals,
        "trend_counts": report.trend_counts,
        "provenance": report.provenance,
        "failed": report.failed,
        "exclusions": _LAUNCH_REPORT_SECTIONS,
    }

    md_path = output_dir / "launch_report.md"
    html_path = output_dir / "launch_report.html"
    json_path = output_dir / "launch_report.json"
    _render_markdown(context, md_path)
    _render_html(context, html_path)
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2),
        encoding="utf-8",
    )

    md_text = md_path.read_text(encoding="utf-8")
    match = _FORBIDDEN_PATTERN.search(md_text)
    if match is not None:
        raise ValueError(
            f"launch report contains forbidden overclaim word {match.group(0)!r}; "
            "rewrite to 'TREND validation' framing"
        )
    return report


__all__ = [
    "LaunchReadinessReport",
    "_build_provenance_table",
    "_render_html",
    "_render_markdown",
    "_summarise_scalar_labels",
    "write_launch_report",
]
