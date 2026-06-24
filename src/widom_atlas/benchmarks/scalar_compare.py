"""TREND-only scalar comparison of widom-atlas-derived KH/Qads vs MOFX-DB references."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from widom_atlas.benchmarks.mofxdb import MOFXDBRecord, load_mofxdb_scalars
from widom_atlas.core.constants import EV_TO_KJMOL, KB_EV_PER_K

ComparisonLabel = Literal["TREND", "UNAVAILABLE", "IDENTITY_UNCERTAIN", "OUT_OF_RANGE"]
ComparisonVerdict = Literal[
    "within_range",          # high identity + |log_ratio_KH| < 2 AND |Qads_delta| < 5 kJ/mol
    "trend_match",           # high identity + finite-positive both sides, but outside the within-range band
    "out_of_range",          # non-physical values (NaN / Inf / non-positive KH)
    "unavailable",           # at least one side missing
    "identity_uncertain",    # identity not high
]

# REPAIR-CV: explicit numerical bands used to map a TREND row to a finer-grained verdict.
WITHIN_RANGE_LOG_RATIO_KH = 2.0      # |log(KH_atlas / KH_ref)| < 2 → factor of <7×
WITHIN_RANGE_QADS_DELTA_KJMOL = 5.0  # |Qads_atlas - Qads_ref| < 5 kJ/mol


class ScalarComparisonRow(BaseModel):
    """One row of the TREND-labelled comparison table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    material_id: str
    gas: str
    temperature_K: float = Field(gt=0.0)
    KH_atlas: float | None = None
    KH_ref: float | None = None
    KH_units: str = "mol/(kg*Pa)"
    KH_log_ratio: float | None = None
    Qads_atlas_kJmol: float | None = None
    Qads_ref_kJmol: float | None = None
    Qads_units: str = "kJ/mol"
    Qads_delta_kJmol: float | None = None
    unit_conversion_path: str = "atlas: eV → kJ/mol via 96.485332… (NIST CODATA); KH: dimensionless atlas heuristic, ref in mol/(kg*Pa)"
    source: str = ""
    source_url: str | None = None
    citation: str | None = None
    identity_confidence: str
    comparison_label: ComparisonLabel
    comparison_verdict: ComparisonVerdict
    notes: str = ""


class ScalarComparisonTable(BaseModel):
    """Bundle of comparison rows + the manifest paths consumed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: list[ScalarComparisonRow]
    benchmark_run_path: str
    output_dir: str


def _derive_KH_from_samples(energies_eV: np.ndarray, temperature_K: float) -> float | None:
    """Heuristic Boltzmann estimator: ``<exp(-E/kT)>`` over insertion samples.

    Returns ``None`` when fewer than 10 samples are accessible (would-be noise).
    """
    e = np.asarray(energies_eV, dtype=np.float64)
    if e.size < 10:
        return None
    beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
    logw = -beta * e
    log_mean = float(np.log(np.mean(np.exp(logw - np.max(logw))))) + float(np.max(logw))
    return float(np.exp(log_mean))


def _derive_Qads_from_samples(energies_eV: np.ndarray, temperature_K: float) -> float | None:
    """Boltzmann-weighted mean energy → estimate of -<E_int>."""
    e = np.asarray(energies_eV, dtype=np.float64)
    if e.size < 10:
        return None
    beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
    logw = -beta * e
    logw = logw - np.max(logw)
    w = np.exp(logw)
    if w.sum() <= 0.0:
        return None
    w_norm = w / w.sum()
    mean_e_eV = float(np.sum(w_norm * e))
    return float(-mean_e_eV * EV_TO_KJMOL)


def _verdict_from(
    KH_atlas: float | None,
    KH_ref: float | None,
    KH_log_ratio: float | None,
    Qads_atlas: float | None,
    Qads_ref: float | None,
    Qads_delta: float | None,
    identity_confidence: str,
    label: ComparisonLabel,
) -> ComparisonVerdict:
    """Refine the ComparisonLabel into the four-level ComparisonVerdict (REPAIR-CV)."""
    if label == "IDENTITY_UNCERTAIN":
        return "identity_uncertain"
    if label == "UNAVAILABLE":
        return "unavailable"
    if label == "OUT_OF_RANGE":
        return "out_of_range"
    # label == "TREND"; decide within_range vs trend_match using documented bands.
    log_ok = (
        KH_log_ratio is not None
        and abs(float(KH_log_ratio)) < WITHIN_RANGE_LOG_RATIO_KH
    )
    qads_ok = (
        Qads_delta is not None
        and abs(float(Qads_delta)) < WITHIN_RANGE_QADS_DELTA_KJMOL
    )
    # Require both KH and Qads to be within the band for "within_range";
    # if one side is missing we cannot promote past "trend_match".
    if log_ok and qads_ok:
        return "within_range"
    return "trend_match"


def _label_comparison(
    KH_atlas: float | None,
    KH_ref: float | None,
    Qads_atlas: float | None,
    Qads_ref: float | None,
    identity_confidence: str,
) -> ComparisonLabel:
    """Label a single material/gas scalar-comparison row.

    Semantics (REPAIR-2):

    - ``IDENTITY_UNCERTAIN`` — the structural identity between atlas and ref
      is not high; the comparison is not meaningful at all.
    - ``UNAVAILABLE`` — at least one side could not produce a value.
    - ``OUT_OF_RANGE`` — values are present but non-physical (NaN, ±inf, or
      non-positive KH that would break ``log``).
    - ``TREND`` — identity high AND both KH (or Qads) values finite & physical.
      The magnitude / direction of disagreement is reported separately as
      ``KH_log_ratio`` and ``Qads_delta_kJmol`` on the row, NOT collapsed
      into the label. The package never claims agreement; ``TREND`` only
      asserts the comparison is meaningful enough to inspect.
    """
    if identity_confidence == "low":
        return "IDENTITY_UNCERTAIN"
    if KH_atlas is None and Qads_atlas is None:
        return "UNAVAILABLE"
    if KH_ref is None and Qads_ref is None:
        return "UNAVAILABLE"

    def _physical(x: float | None) -> bool:
        return x is not None and math.isfinite(x)

    if KH_atlas is not None and not _physical(KH_atlas):
        return "OUT_OF_RANGE"
    if KH_ref is not None and not _physical(KH_ref):
        return "OUT_OF_RANGE"
    if Qads_atlas is not None and not _physical(Qads_atlas):
        return "OUT_OF_RANGE"
    if Qads_ref is not None and not _physical(Qads_ref):
        return "OUT_OF_RANGE"
    # KH must be strictly positive for log to be defined; either side ≤ 0
    # means we cannot place this on a log axis even though identity is high.
    for k in (KH_atlas, KH_ref):
        if k is not None and k <= 0.0:
            return "OUT_OF_RANGE"

    return "TREND"


def _read_basins_for_material(material_dir: Path) -> tuple[np.ndarray, float] | None:
    samples_path = material_dir / "input_samples.npz"
    if not samples_path.exists():
        return None
    with np.load(samples_path, allow_pickle=False) as f:
        e = np.asarray(f["energies_eV"], dtype=np.float64)
        T = float(f["temperature_K"])
    return e, T


def compare_scalars(
    benchmark_run_path: Path,
    mofxdb_cache_dir: Path,
    output_dir: Path,
) -> ScalarComparisonTable:
    """Produce ``scalar_comparison.csv``/``.json``/``.md`` from a benchmark run aggregate."""
    benchmark_run_path = Path(benchmark_run_path)
    mofxdb_cache_dir = Path(mofxdb_cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(benchmark_run_path.read_text(encoding="utf-8"))
    materials = payload.get("materials", [])
    rows: list[ScalarComparisonRow] = []
    benchmark_run_dir = benchmark_run_path.parent

    for entry in materials:
        material_id = str(entry["material_id"])
        gas = str(entry["gas"])
        temperature_K = float(entry["temperature_K"])
        ref: MOFXDBRecord = load_mofxdb_scalars(material_id, gas, mofxdb_cache_dir)

        KH_atlas: float | None = None
        Qads_atlas: float | None = None
        if entry.get("status") == "ok":
            mat_dir = benchmark_run_dir / material_id
            sample_data = _read_basins_for_material(mat_dir)
            if sample_data is not None:
                e, T = sample_data
                KH_atlas = _derive_KH_from_samples(e, T)
                Qads_atlas = _derive_Qads_from_samples(e, T)

        log_ratio: float | None = None
        if KH_atlas is not None and ref.KH is not None and ref.KH > 0.0 and KH_atlas > 0.0:
            log_ratio = math.log(KH_atlas / ref.KH)
        Qads_delta: float | None = None
        if Qads_atlas is not None and ref.Qads is not None:
            Qads_delta = Qads_atlas - ref.Qads

        label = _label_comparison(KH_atlas, ref.KH, Qads_atlas, ref.Qads, ref.identity_confidence)
        verdict = _verdict_from(
            KH_atlas, ref.KH, log_ratio, Qads_atlas, ref.Qads, Qads_delta,
            ref.identity_confidence, label,
        )
        rows.append(
            ScalarComparisonRow(
                material_id=material_id,
                gas=gas,
                temperature_K=temperature_K,
                KH_atlas=KH_atlas,
                KH_ref=ref.KH,
                KH_units=ref.KH_units,
                KH_log_ratio=log_ratio,
                Qads_atlas_kJmol=Qads_atlas,
                Qads_ref_kJmol=ref.Qads,
                Qads_units=ref.Qads_units,
                Qads_delta_kJmol=Qads_delta,
                source=ref.source,
                source_url=ref.source_url,
                citation=ref.citation,
                identity_confidence=ref.identity_confidence,
                comparison_label=label,
                comparison_verdict=verdict,
                notes="atlas KH derived via heuristic Boltzmann <exp(-βE)> on insertion samples; toy LJ overestimates real CO2-MOF binding by orders of magnitude. TREND-only.",
            )
        )

    df = pd.DataFrame([r.model_dump() for r in rows])
    csv_path = output_dir / "scalar_comparison.csv"
    json_path = output_dir / "scalar_comparison.json"
    md_path = output_dir / "scalar_comparison.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps([r.model_dump() for r in rows], sort_keys=True, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Scalar comparison (TREND validation only — not proof of quantitative accuracy)",
        "",
        f"_Within-range bands_: |log(KH_atlas / KH_ref)| < {WITHIN_RANGE_LOG_RATIO_KH:.1f} (≈ factor of e^{WITHIN_RANGE_LOG_RATIO_KH:.0f}) **and** |Qads_atlas - Qads_ref| < {WITHIN_RANGE_QADS_DELTA_KJMOL:.1f} kJ/mol → `within_range`. "
        "Identity high + finite-positive but outside both bands → `trend_match`. Non-physical values → `out_of_range`. "
        "Identity not high → `identity_uncertain`. Either side missing → `unavailable`. The package never claims `validated` / `agreed`.",
        "",
        "| material | gas | T (K) | KH_atlas | KH_ref | KH_units | log(KH_atlas/KH_ref) | Q_atlas (kJ/mol) | Q_ref (kJ/mol) | ΔQ (kJ/mol) | identity | label | verdict | source / DOI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    def _f(value: float | None, fmt: str) -> str:
        return format(value, fmt) if value is not None else "—"

    for r in rows:
        kh_atlas = _f(r.KH_atlas, ".3e")
        kh_ref = _f(r.KH_ref, ".3e")
        log_ratio = _f(r.KH_log_ratio, ".3f")
        q_atlas = _f(r.Qads_atlas_kJmol, ".2f")
        q_ref = _f(r.Qads_ref_kJmol, ".2f")
        q_delta = _f(r.Qads_delta_kJmol, ".2f")
        src = r.source_url if r.source_url else "—"
        if r.citation:
            src = f"{src}<br>_{r.citation[:80]}_"
        md_lines.append(
            f"| {r.material_id} | {r.gas} | {r.temperature_K:.2f} "
            f"| {kh_atlas} | {kh_ref} | {r.KH_units} | {log_ratio} "
            f"| {q_atlas} | {q_ref} | {q_delta} "
            f"| {r.identity_confidence} | {r.comparison_label} | "
            f"`{r.comparison_verdict}` | {src} |"
        )
    md_lines.extend([
        "",
        "## Caveats",
        "",
        "- The atlas-side `KH_atlas` is a heuristic Boltzmann estimator over the insertion samples — not a properly converged GCMC integral, and not in absolute mol/(kg*Pa) units; do not interpret as `KH_ref`-equivalent without unit reconciliation.",
        "- Reference values are point literature estimates; isotherms vary with activation, defects, and temperature. Treat as TREND only.",
        "- Toy ASE Lennard-Jones (verdict §G) is too soft for real CO2-MOF energetics by orders of magnitude — `trend_match` and `out_of_range` rows are EXPECTED at v1, not failures of the comparison plumbing.",
    ])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return ScalarComparisonTable(
        rows=rows,
        benchmark_run_path=str(benchmark_run_path),
        output_dir=str(output_dir),
    )


__all__ = [
    "ScalarComparisonRow",
    "ScalarComparisonTable",
    "_derive_KH_from_samples",
    "_derive_Qads_from_samples",
    "_label_comparison",
    "compare_scalars",
]
