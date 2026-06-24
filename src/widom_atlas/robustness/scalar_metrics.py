"""Scalar robustness metrics: delta_ln_KH and delta_Qads with graceful degradation."""

from __future__ import annotations

import math
from typing import Any


def compute_delta_ln_KH(
    KH_pristine: float | None, KH_perturbed: float | None
) -> float | None:
    """Return ``log(KH_perturbed / KH_pristine)`` or ``None`` if either side is missing/non-positive."""
    if KH_pristine is None or KH_perturbed is None:
        return None
    if not math.isfinite(KH_pristine) or not math.isfinite(KH_perturbed):
        return None
    if KH_pristine <= 0.0 or KH_perturbed <= 0.0:
        return None
    return math.log(KH_perturbed / KH_pristine)


def compute_delta_Qads(
    Q_pristine_kJmol: float | None, Q_perturbed_kJmol: float | None
) -> float | None:
    """Return ``Qads_perturbed - Qads_pristine`` (kJ/mol) or ``None`` if either is missing/non-finite."""
    if Q_pristine_kJmol is None or Q_perturbed_kJmol is None:
        return None
    if not math.isfinite(Q_pristine_kJmol) or not math.isfinite(Q_perturbed_kJmol):
        return None
    return float(Q_perturbed_kJmol) - float(Q_pristine_kJmol)


def compute_scalar_metrics(
    pristine_summary: dict[str, Any], perturbed_summary: dict[str, Any]
) -> dict[str, Any]:
    """Compute delta_ln_KH and delta_Qads from per-run scalar summaries.

    Each summary may contain ``henry_coefficient`` and ``heat_of_adsorption_kJmol``
    or ``None`` / be absent. Missing fields are reported via ``missing_fields`` and
    ``degraded`` is set to True when at least one scalar could not be computed.
    """
    KH_pri = pristine_summary.get("henry_coefficient")
    KH_per = perturbed_summary.get("henry_coefficient")
    Q_pri = pristine_summary.get("heat_of_adsorption_kJmol")
    Q_per = perturbed_summary.get("heat_of_adsorption_kJmol")

    delta_ln_KH = compute_delta_ln_KH(KH_pri, KH_per)
    delta_Qads = compute_delta_Qads(Q_pri, Q_per)

    missing: list[str] = []
    if KH_pri is None:
        missing.append("henry_coefficient_pristine")
    if KH_per is None:
        missing.append("henry_coefficient_perturbed")
    if Q_pri is None:
        missing.append("heat_of_adsorption_pristine_kJmol")
    if Q_per is None:
        missing.append("heat_of_adsorption_perturbed_kJmol")

    degraded = bool(delta_ln_KH is None or delta_Qads is None)
    return {
        "delta_ln_KH": delta_ln_KH,
        "delta_Qads_kJmol": delta_Qads,
        "missing_fields": missing,
        "degraded": degraded,
    }


__all__ = ["compute_delta_Qads", "compute_delta_ln_KH", "compute_scalar_metrics"]
