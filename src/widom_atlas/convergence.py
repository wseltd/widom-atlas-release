"""Widom convergence study driver.

Run a sweep over insertion counts on a single material/gas, capture the
scalars / basin shape / runtime per N, and emit explicit pass-or-fail
threshold checks. This is the "convergence study" deliverable the CuspAI-
grade audit requires — single short Widom runs are smoke tests, not
quantitative interpretation.

Acceptance criteria (REPAIR-CV thresholds, configurable):

- ``rel_KH_uncertainty_threshold`` (default 0.30)  — relative std/mean of KH
  at the largest N must be ≤ this for ``KH_converged`` = True.
- ``centroid_drift_threshold_A`` (default 0.5)    — minimum-image distance
  between the dominant-basin centroid at the last two N must be ≤ this for
  ``basin_centroid_converged`` = True.
- ``dominant_weight_change_threshold`` (default 0.05) — change in the
  dominant-basin weight between the last two N must be ≤ this for
  ``basin_weight_converged`` = True.

When a threshold is not met the study reports ``not_converged`` for that
metric. The summary JSON never claims convergence it has not earned.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from widom_atlas.backends import BackendName, get_backend
from widom_atlas.benchmarks.download import (
    fetch_benchmark_material,
)
from widom_atlas.benchmarks.hashing import record_provenance
from widom_atlas.benchmarks.registry import SMALL_BENCHMARK_SET
from widom_atlas.benchmarks.runner import _generate_atlas_input, _load_atoms_from_cif
from widom_atlas.core.benchmark_models import BenchmarkMaterial
from widom_atlas.core.pipeline import PipelineParams, run_atlas
from widom_atlas.pbc.minimum_image import min_image_distance

_LOGGER = logging.getLogger(__name__)

WIDOM_BACKEND_LABEL = "CuspAI Widom (run_widom_insertion)"

# Default acceptance thresholds for the convergence verdict.
DEFAULT_REL_KH_UNCERTAINTY = 0.30
DEFAULT_CENTROID_DRIFT_A = 0.5
DEFAULT_DOMINANT_WEIGHT_CHANGE = 0.05


def _resolve_material(material_id: str) -> BenchmarkMaterial:
    for m in SMALL_BENCHMARK_SET:
        if m.material_id == material_id:
            return m
    raise ValueError(
        f"unknown material_id={material_id!r}; "
        f"known: {[m.material_id for m in SMALL_BENCHMARK_SET]}"
    )


def _summarise_basins(basins: list, cell: np.ndarray) -> dict[str, Any]:
    """Reduce a basin list to its dominant + top-3 summary for convergence tracking."""
    if not basins:
        return {
            "n_basins": 0,
            "dominant_weight": None,
            "dominant_centroid_frac": None,
            "top3_weights": [],
        }
    sorted_b = sorted(basins, key=lambda b: -b.weight)
    dom = sorted_b[0]
    return {
        "n_basins": len(basins),
        "dominant_weight": float(dom.weight),
        "dominant_centroid_frac": list(dom.centroid_frac),
        "dominant_count": int(dom.count),
        "dominant_mean_energy_eV": float(dom.mean_energy_eV),
        "dominant_spread_A": float(dom.spread_A),
        "dominant_accessible_fraction": float(dom.accessible_fraction),
        "top3_weights": [float(b.weight) for b in sorted_b[:3]],
    }


def _basin_centroid_drift_A(
    earlier: dict[str, Any], later: dict[str, Any], cell: np.ndarray
) -> float | None:
    """Minimum-image distance between dominant centroids, in Å. ``None`` if either is empty."""
    a = earlier.get("dominant_centroid_frac")
    b = later.get("dominant_centroid_frac")
    if a is None or b is None:
        return None
    return float(min_image_distance(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), cell))


@dataclass
class ConvergenceStep:
    """One insertion-count step in a convergence sweep."""

    n_insertions: int
    runtime_s: float
    seed: int
    KH: float | None
    KH_std: float | None
    Qads_kJmol: float | None
    Qads_std_kJmol: float | None
    accessible_fraction: float
    n_accessible: int
    n_total: int
    basins_summary: dict[str, Any]
    backend: str = WIDOM_BACKEND_LABEL
    calculator: str = ""
    output_dir: str = ""


@dataclass
class ConvergenceVerdict:
    """Per-metric pass/fail verdict against documented thresholds."""

    KH_relative_uncertainty: float | None
    KH_converged: bool
    centroid_drift_A: float | None
    basin_centroid_converged: bool
    dominant_weight_change: float | None
    basin_weight_converged: bool
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> str:
        if self.KH_converged and self.basin_centroid_converged and self.basin_weight_converged:
            return "converged"
        if self.KH_relative_uncertainty is None and not self.basin_centroid_converged:
            return "insufficient_data"
        return "not_converged"


def _verdict(
    steps: list[ConvergenceStep],
    cell: np.ndarray,
    rel_kh_threshold: float,
    centroid_threshold_A: float,
    dominant_weight_threshold: float,
) -> ConvergenceVerdict:
    notes: list[str] = []
    last = steps[-1] if steps else None
    rel_kh = None
    KH_converged = False
    if last is not None and last.KH is not None and last.KH > 0 and last.KH_std is not None:
        rel_kh = float(last.KH_std / last.KH)
        KH_converged = rel_kh <= rel_kh_threshold
        if not KH_converged:
            notes.append(
                f"KH relative uncertainty at N={last.n_insertions} is {rel_kh:.3f} "
                f"> threshold {rel_kh_threshold:.3f}; not converged."
            )
    elif last is not None:
        notes.append(
            f"KH or KH_std missing at N={last.n_insertions}; cannot evaluate KH convergence."
        )

    centroid_drift = None
    centroid_converged = False
    if len(steps) >= 2:
        centroid_drift = _basin_centroid_drift_A(
            steps[-2].basins_summary, steps[-1].basins_summary, cell
        )
        if centroid_drift is None:
            notes.append(
                f"Dominant centroid missing at N={steps[-2].n_insertions} or "
                f"N={steps[-1].n_insertions}; cannot evaluate centroid drift."
            )
        else:
            centroid_converged = centroid_drift <= centroid_threshold_A
            if not centroid_converged:
                notes.append(
                    f"Dominant-basin centroid drift between N={steps[-2].n_insertions} "
                    f"and N={steps[-1].n_insertions} is {centroid_drift:.3f} Å > threshold "
                    f"{centroid_threshold_A:.3f} Å; not converged."
                )

    weight_change = None
    weight_converged = False
    if len(steps) >= 2:
        w_prev = steps[-2].basins_summary.get("dominant_weight")
        w_last = steps[-1].basins_summary.get("dominant_weight")
        if w_prev is None or w_last is None:
            notes.append("Dominant-basin weight missing; cannot evaluate weight convergence.")
        else:
            weight_change = abs(float(w_last) - float(w_prev))
            weight_converged = weight_change <= dominant_weight_threshold
            if not weight_converged:
                notes.append(
                    f"Dominant-basin weight change is {weight_change:.4f} > threshold "
                    f"{dominant_weight_threshold:.4f}; not converged."
                )

    return ConvergenceVerdict(
        KH_relative_uncertainty=rel_kh,
        KH_converged=KH_converged,
        centroid_drift_A=centroid_drift,
        basin_centroid_converged=centroid_converged,
        dominant_weight_change=weight_change,
        basin_weight_converged=weight_converged,
        notes=notes,
    )


def run_convergence_study(
    material_id: str,
    gas: str,
    insertion_counts: list[int],
    output_dir: Path,
    *,
    cache_dir: Path,
    structures_dir: Path | None = None,
    temperature_K: float = 298.15,
    seed: int = 0,
    pipeline_params: PipelineParams | None = None,
    rel_kh_uncertainty_threshold: float = DEFAULT_REL_KH_UNCERTAINTY,
    centroid_drift_threshold_A: float = DEFAULT_CENTROID_DRIFT_A,
    dominant_weight_change_threshold: float = DEFAULT_DOMINANT_WEIGHT_CHANGE,
    backend_name: BackendName = "parameterised_lj",
    external_samples_path: Path | None = None,
    external_manifest_path: Path | None = None,
    user_parameter_file: Path | None = None,
    allow_neutral_fallback: bool = False,
) -> dict[str, Any]:
    """Run a Widom convergence sweep on one material + gas at the given insertion counts.

    Writes per-N atlas runs under ``output_dir/N_<n>/`` and a top-level
    ``convergence_summary.json`` + ``convergence_report.md`` with the
    explicit threshold verdict.
    """
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    material = _resolve_material(material_id)
    if not insertion_counts:
        raise ValueError("insertion_counts must be non-empty")
    if any(n <= 0 for n in insertion_counts):
        raise ValueError(f"insertion_counts must all be positive: {insertion_counts}")

    cif_path = fetch_benchmark_material(
        material,
        cache_dir=cache_dir,
        allow_network=False,
        fixtures_dir=structures_dir,
    )
    provenance = record_provenance(material, cif_path, cache_dir)
    atoms = _load_atoms_from_cif(cif_path)
    cell = np.asarray(atoms.get_cell().array, dtype=np.float64)

    if pipeline_params is None:
        pipeline_params = PipelineParams(n_grid=(24, 24, 24), dbscan_eps_A=2.0, min_samples=4)

    backend = get_backend(
        backend_name,
        external_samples_path=external_samples_path,
        external_manifest_path=external_manifest_path,
        user_parameter_file=user_parameter_file,
        allow_neutral_fallback=allow_neutral_fallback,
    )
    steps: list[ConvergenceStep] = []
    started = datetime.now(UTC)
    for n in sorted(set(int(x) for x in insertion_counts)):
        step_dir = output_dir / f"N_{n:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        atlas_input = _generate_atlas_input(
            material=material,
            atoms=atoms,
            gas=gas,
            temperature_K=temperature_K,
            n_samples=n,
            seed=seed,
            samples_kind="widom",
            backend=backend,
        )
        result = run_atlas(atlas_input, pipeline_params, step_dir, structure=atoms)
        runtime_s = time.perf_counter() - t0

        scalars = atlas_input.metadata.get("widom_scalars") or {}
        accessible_arr = np.asarray(atlas_input.accessible, dtype=bool)
        n_accessible = int(accessible_arr.sum())
        calculator_label = atlas_input.metadata.get("calculator", "")
        steps.append(
            ConvergenceStep(
                n_insertions=n,
                runtime_s=runtime_s,
                seed=seed,
                KH=scalars.get("henry_coefficient"),
                KH_std=scalars.get("henry_coefficient_std"),
                Qads_kJmol=scalars.get("heat_of_adsorption_kJmol"),
                Qads_std_kJmol=scalars.get("heat_of_adsorption_std_kJmol"),
                accessible_fraction=float(n_accessible / max(1, accessible_arr.size)),
                n_accessible=n_accessible,
                n_total=int(accessible_arr.size),
                basins_summary=_summarise_basins(result.basins, cell),
                calculator=str(calculator_label),
                output_dir=str(step_dir.resolve()),
            )
        )
    finished = datetime.now(UTC)

    verdict = _verdict(
        steps,
        cell,
        rel_kh_uncertainty_threshold,
        centroid_drift_threshold_A,
        dominant_weight_change_threshold,
    )

    summary = {
        "schema_version": "1",
        "material_id": material_id,
        "material_source_identifier": material.source_identifier,
        "material_dataset_version": provenance.dataset_version,
        "material_sha256": provenance.sha256,
        "gas": gas,
        "temperature_K": float(temperature_K),
        "insertion_counts": [int(n) for n in sorted(set(int(x) for x in insertion_counts))],
        "seed": seed,
        "backend": WIDOM_BACKEND_LABEL,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "backend_name": backend_name,
        "calculator": steps[-1].calculator if steps else "",
        "thresholds": {
            "rel_KH_uncertainty": rel_kh_uncertainty_threshold,
            "centroid_drift_A": centroid_drift_threshold_A,
            "dominant_weight_change": dominant_weight_change_threshold,
        },
        "verdict": {
            "overall": verdict.overall,
            "KH_relative_uncertainty": verdict.KH_relative_uncertainty,
            "KH_converged": verdict.KH_converged,
            "centroid_drift_A": verdict.centroid_drift_A,
            "basin_centroid_converged": verdict.basin_centroid_converged,
            "dominant_weight_change": verdict.dominant_weight_change,
            "basin_weight_converged": verdict.basin_weight_converged,
            "notes": list(verdict.notes),
        },
        "steps": [
            {
                "n_insertions": s.n_insertions,
                "runtime_s": round(s.runtime_s, 3),
                "seed": s.seed,
                "KH": s.KH,
                "KH_std": s.KH_std,
                "Qads_kJmol": s.Qads_kJmol,
                "Qads_std_kJmol": s.Qads_std_kJmol,
                "accessible_fraction": s.accessible_fraction,
                "n_accessible": s.n_accessible,
                "n_total": s.n_total,
                "basins_summary": s.basins_summary,
                "backend": s.backend,
                "calculator": s.calculator,
                "output_dir": s.output_dir,
            }
            for s in steps
        ],
    }

    (output_dir / "convergence_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _write_report_md(summary, output_dir / "convergence_report.md")
    return summary


def _fmt(value: float | None, spec: str) -> str:
    if value is None:
        return "—"
    return format(value, spec)


def _write_report_md(summary: dict[str, Any], path: Path) -> None:
    """Render a Markdown table + verdict block."""
    sha_short = summary["material_sha256"][:16]
    lines = [
        "# widom-atlas — Widom convergence study",
        "",
        f"**Material:** {summary['material_id']}  (sha256 `{sha_short}…`)",
        f"**Source dataset:** {summary['material_dataset_version']}",
        f"**Gas:** {summary['gas']}",
        f"**Temperature (K):** {summary['temperature_K']}",
        f"**Backend:** {summary['backend']}",
        f"**Calculator:** {summary['calculator']}",
        f"**Random seed:** {summary['seed']}",
        f"**Started / finished (UTC):** {summary['started_at']} → {summary['finished_at']}",
        "",
        "## Per-N results",
        "",
        "| N | runtime (s) | KH | KH std | rel_unc | Q_ads (kJ/mol) | Q_ads std | accessible | n_basins | dominant_w | dominant_centroid (frac) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary["steps"]:
        kh = s.get("KH")
        kh_std = s.get("KH_std")
        rel_unc = f"{kh_std / kh:.3f}" if kh and kh_std and kh > 0 else "—"
        bs = s["basins_summary"] or {}
        cent = bs.get("dominant_centroid_frac")
        cent_str = f"({cent[0]:.3f}, {cent[1]:.3f}, {cent[2]:.3f})" if cent else "—"
        kh_s = _fmt(kh, ".3e")
        kh_std_s = _fmt(kh_std, ".3e")
        q_s = _fmt(s.get("Qads_kJmol"), ".3f")
        q_std_s = _fmt(s.get("Qads_std_kJmol"), ".3f")
        dom_w = bs.get("dominant_weight")
        dom_w_s = _fmt(dom_w, ".3f")
        n_basins = bs.get("n_basins", 0)
        lines.append(
            f"| {s['n_insertions']} | {s['runtime_s']:.2f} | {kh_s} | {kh_std_s} | "
            f"{rel_unc} | {q_s} | {q_std_s} | "
            f"{s['n_accessible']}/{s['n_total']} | {n_basins} | {dom_w_s} | {cent_str} |"
        )

    v = summary["verdict"]
    th = summary["thresholds"]
    rel_kh_s = _fmt(v.get("KH_relative_uncertainty"), ".4f")
    drift_val = v.get("centroid_drift_A")
    drift_s = f"{drift_val:.4f} Å" if drift_val is not None else "n/a"
    weight_val = v.get("dominant_weight_change")
    weight_s = _fmt(weight_val, ".4f") if weight_val is not None else "n/a"
    lines += [
        "",
        "## Convergence verdict",
        "",
        f"**Overall:** `{v['overall']}`",
        "",
        "| metric | value | threshold | converged? |",
        "|---|---|---|---|",
        f"| KH relative uncertainty | {rel_kh_s} | ≤ {th['rel_KH_uncertainty']:.3f} | "
        f"{'YES' if v['KH_converged'] else 'NO'} |",
        f"| dominant basin centroid drift between last two N | {drift_s} | "
        f"≤ {th['centroid_drift_A']:.3f} Å | "
        f"{'YES' if v['basin_centroid_converged'] else 'NO'} |",
        f"| dominant basin weight change between last two N | {weight_s} | "
        f"≤ {th['dominant_weight_change']:.4f} | "
        f"{'YES' if v['basin_weight_converged'] else 'NO'} |",
        "",
    ]
    if v["notes"]:
        lines += ["## Notes", ""]
        for n in v["notes"]:
            lines.append(f"- {n}")
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Small insertion counts (≤ ~1000) are smoke tests, not quantitative interpretation.",
        "- The smoke-test calculator is ASE Lennard-Jones (verdict §G); LJ overestimates real "
        "CO2-MOF binding by orders of magnitude. Use this convergence sweep to verify that the "
        "package itself stabilises — not to interpret the numbers as chemistry.",
        "- For chemistry-level interpretation, swap in a parameterised force field or ML "
        "potential and rerun the same convergence pipeline. The pipeline is calculator-agnostic.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_CENTROID_DRIFT_A",
    "DEFAULT_DOMINANT_WEIGHT_CHANGE",
    "DEFAULT_REL_KH_UNCERTAINTY",
    "WIDOM_BACKEND_LABEL",
    "ConvergenceStep",
    "ConvergenceVerdict",
    "run_convergence_study",
]
