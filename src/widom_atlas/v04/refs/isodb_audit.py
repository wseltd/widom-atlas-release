"""ISODB CO2 isotherm reference audit + K_H sensitivity analysis for 5c branches.

For every 5c-candidate ISODB JSON in
docs/research/dataset-research-for-v0.4/5c_replacement_branches/ this
module:

  1. Loads the isotherm (pressure / loading rows).
  2. Verifies units (mmol/g + bar declared in JSON).
  3. Converts to mol/kg + Pa (the atlas's internal K_H units).
  4. Fits K_H via four methods:
       - 1-point slope (q[0] / p[0])
       - 2-point linear regression through origin (lowest 2 points)
       - 3-point linear regression through origin (lowest 3 points)
       - virial-expansion fit q/p = K_H + B*q + C*q^2 (low-loading half)
  5. Reports per-method K_H + the range (sensitivity).
  6. Tags Henry-regime adequacy by checking q_first / q_high_estimate.

Q_st provenance and FF provenance are not derived here — they're recorded
in the per-branch YAML / audit-report deliverable. Q_st always comes from
the operator-supplied literature cite, never from the isotherm itself
(2-pt van't Hoff requires multi-T data, which only some ISODB rows provide).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Unit conversion constants
BAR_TO_PA: float = 1.0e5
MMOL_PER_G_TO_MOL_PER_KG: float = 1.0  # 1 mmol/g = 1 mol/kg by mass identity


@dataclass
class HenryFit:
    """A single K_H fit method's result."""

    method: str
    K_H_mmol_per_g_per_bar: float
    K_H_mol_per_kg_per_bar: float
    K_H_mol_per_kg_per_Pa: float
    n_points_used: int
    p_max_used_bar: float
    notes: list[str] = field(default_factory=list)


@dataclass
class IsothermAudit:
    """Audit of one ISODB isotherm JSON: parsed + multiple K_H fits."""

    source_file: Path
    doi: str
    adsorbent: str
    adsorbate: str
    temperature_K: float
    n_points: int
    p_units_in: str
    q_units_in: str
    pressure_bar: list[float]
    loading_mmol_per_g: list[float]
    fits: list[HenryFit] = field(default_factory=list)
    henry_regime_adequacy: str = ""

    def K_H_method_values_mol_per_kg_per_bar(self) -> dict[str, float]:
        return {f.method: f.K_H_mol_per_kg_per_bar for f in self.fits}

    def K_H_sensitivity_range_mol_per_kg_per_bar(self) -> tuple[float, float]:
        values = [f.K_H_mol_per_kg_per_bar for f in self.fits]
        return (min(values), max(values))


def _parse_isotherm_data(d: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return parallel lists (pressure, loading) from an ISODB JSON dict."""
    pressures: list[float] = []
    loadings: list[float] = []
    rows = d.get("isotherm_data", [])
    if not rows:
        raise ValueError("isotherm_data missing or empty")
    for r in rows:
        p = r.get("pressure")
        q = r.get("total_adsorption")
        if q is None:
            sd = r.get("species_data") or []
            if sd and sd[0].get("adsorption") is not None:
                q = sd[0]["adsorption"]
        if p is None or q is None:
            continue
        pressures.append(float(p))
        loadings.append(float(q))
    # Sort by pressure ascending
    paired = sorted(zip(pressures, loadings, strict=True), key=lambda x: x[0])
    pressures = [p for p, _ in paired]
    loadings = [q for _, q in paired]
    return pressures, loadings


def _linear_through_origin(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope through origin: K = sum(x_i y_i) / sum(x_i^2)."""
    num = sum(x * y for x, y in zip(xs, ys, strict=True))
    den = sum(x * x for x in xs)
    return num / den if den > 0 else float("nan")


def _virial_K_H(
    pressures: list[float], loadings: list[float], n_use: int,
) -> tuple[float, list[str]]:
    """Virial-expansion fit q/p = K_H + B*q for the low-loading half.

    Plotting (q/p) vs q and extrapolating to q=0 yields K_H. Uses the
    lowest n_use pressure points; returns (K_H, notes).
    """
    notes: list[str] = []
    used = list(zip(pressures, loadings, strict=True))[:n_use]
    qs = [q for _, q in used]
    qopps = [q / p for p, q in used]
    if not qs or all(p == 0 for p, _ in used):
        notes.append("virial_skipped_zero_pressure")
        return float("nan"), notes
    # Linear regression: qopps = K_H + B * qs
    mean_q = sum(qs) / len(qs)
    mean_y = sum(qopps) / len(qopps)
    num = sum((q - mean_q) * (y - mean_y) for q, y in zip(qs, qopps, strict=True))
    den = sum((q - mean_q) ** 2 for q in qs)
    B = num / den if den > 0 else 0.0
    K_H = mean_y - B * mean_q
    return K_H, notes


def audit_isodb_isotherm(path: Path) -> IsothermAudit:
    """Audit a single ISODB JSON file."""
    with path.open() as fp:
        d = json.load(fp)

    p_units = d.get("pressureUnits", "")
    q_units = d.get("adsorptionUnits", "")
    if p_units != "bar":
        raise ValueError(f"unsupported pressure units {p_units!r} in {path}")
    if q_units != "mmol/g":
        raise ValueError(f"unsupported adsorption units {q_units!r} in {path}")

    adsorbent = d.get("adsorbent", {}).get("name", "UNKNOWN")
    adsorbates = d.get("adsorbates", [])
    adsorbate = adsorbates[0].get("name", "UNKNOWN") if adsorbates else "UNKNOWN"
    T = float(d.get("temperature", 0.0))
    doi = d.get("DOI", "UNKNOWN")

    pressures, loadings = _parse_isotherm_data(d)
    n = len(pressures)

    audit = IsothermAudit(
        source_file=path,
        doi=doi,
        adsorbent=adsorbent,
        adsorbate=adsorbate,
        temperature_K=T,
        n_points=n,
        p_units_in=p_units,
        q_units_in=q_units,
        pressure_bar=pressures,
        loading_mmol_per_g=loadings,
    )

    # Fit 1: 1-point slope
    if n >= 1 and pressures[0] > 0:
        K1 = loadings[0] / pressures[0]
        audit.fits.append(
            HenryFit(
                method="1pt_slope",
                K_H_mmol_per_g_per_bar=K1,
                K_H_mol_per_kg_per_bar=K1 * MMOL_PER_G_TO_MOL_PER_KG,
                K_H_mol_per_kg_per_Pa=K1 / BAR_TO_PA,
                n_points_used=1,
                p_max_used_bar=pressures[0],
                notes=[],
            )
        )

    # Fit 2: 2-point linear regression through origin
    if n >= 2:
        K2 = _linear_through_origin(pressures[:2], loadings[:2])
        audit.fits.append(
            HenryFit(
                method="2pt_origin_linear",
                K_H_mmol_per_g_per_bar=K2,
                K_H_mol_per_kg_per_bar=K2 * MMOL_PER_G_TO_MOL_PER_KG,
                K_H_mol_per_kg_per_Pa=K2 / BAR_TO_PA,
                n_points_used=2,
                p_max_used_bar=pressures[1],
                notes=[],
            )
        )

    # Fit 3: 3-point linear regression through origin
    if n >= 3:
        K3 = _linear_through_origin(pressures[:3], loadings[:3])
        audit.fits.append(
            HenryFit(
                method="3pt_origin_linear",
                K_H_mmol_per_g_per_bar=K3,
                K_H_mol_per_kg_per_bar=K3 * MMOL_PER_G_TO_MOL_PER_KG,
                K_H_mol_per_kg_per_Pa=K3 / BAR_TO_PA,
                n_points_used=3,
                p_max_used_bar=pressures[2],
                notes=[],
            )
        )

    # Fit 4: virial fit on lowest half of points
    if n >= 4:
        n_use = max(2, n // 2)
        Kv, vnotes = _virial_K_H(pressures, loadings, n_use)
        audit.fits.append(
            HenryFit(
                method="virial_lowhalf",
                K_H_mmol_per_g_per_bar=Kv,
                K_H_mol_per_kg_per_bar=Kv * MMOL_PER_G_TO_MOL_PER_KG,
                K_H_mol_per_kg_per_Pa=Kv / BAR_TO_PA,
                n_points_used=n_use,
                p_max_used_bar=pressures[n_use - 1],
                notes=vnotes,
            )
        )

    # Henry-regime adequacy diagnostic: q[0] / q_sat_estimate
    q_max = max(loadings) if loadings else 0.0
    if q_max > 0 and loadings:
        q0_over_qmax = loadings[0] / q_max
        if q0_over_qmax < 0.05:
            audit.henry_regime_adequacy = (
                f"adequate (q[0]/q_max = {q0_over_qmax:.3f} < 0.05)"
            )
        elif q0_over_qmax < 0.15:
            audit.henry_regime_adequacy = (
                f"borderline (q[0]/q_max = {q0_over_qmax:.3f} in [0.05, 0.15])"
            )
        else:
            audit.henry_regime_adequacy = (
                f"inadequate (q[0]/q_max = {q0_over_qmax:.3f} >= 0.15 — first "
                f"point is far past Henry regime)"
            )

    return audit


def render_audit_json(audit: IsothermAudit) -> dict[str, Any]:
    """Convert IsothermAudit to a JSON-serialisable dict."""
    return {
        "source_file": str(audit.source_file),
        "doi": audit.doi,
        "adsorbent": audit.adsorbent,
        "adsorbate": audit.adsorbate,
        "temperature_K": audit.temperature_K,
        "n_points": audit.n_points,
        "p_units_in_json": audit.p_units_in,
        "q_units_in_json": audit.q_units_in,
        "unit_conversion": {
            "bar_to_Pa": BAR_TO_PA,
            "mmol_per_g_to_mol_per_kg": MMOL_PER_G_TO_MOL_PER_KG,
        },
        "isotherm_data": [
            {
                "p_bar": p,
                "p_Pa": p * BAR_TO_PA,
                "q_mmol_per_g": q,
                "q_mol_per_kg": q * MMOL_PER_G_TO_MOL_PER_KG,
            }
            for p, q in zip(audit.pressure_bar, audit.loading_mmol_per_g, strict=True)
        ],
        "K_H_fits": [
            {
                "method": f.method,
                "K_H_mmol_per_g_per_bar": f.K_H_mmol_per_g_per_bar,
                "K_H_mol_per_kg_per_bar": f.K_H_mol_per_kg_per_bar,
                "K_H_mol_per_kg_per_Pa": f.K_H_mol_per_kg_per_Pa,
                "n_points_used": f.n_points_used,
                "p_max_used_bar": f.p_max_used_bar,
                "notes": f.notes,
            }
            for f in audit.fits
        ],
        "K_H_sensitivity": {
            "min_mol_per_kg_per_bar": audit.K_H_sensitivity_range_mol_per_kg_per_bar()[0],
            "max_mol_per_kg_per_bar": audit.K_H_sensitivity_range_mol_per_kg_per_bar()[1],
            "ratio_max_over_min": (
                audit.K_H_sensitivity_range_mol_per_kg_per_bar()[1]
                / audit.K_H_sensitivity_range_mol_per_kg_per_bar()[0]
                if audit.K_H_sensitivity_range_mol_per_kg_per_bar()[0] > 0
                else float("inf")
            ),
        },
        "henry_regime_adequacy": audit.henry_regime_adequacy,
    }
