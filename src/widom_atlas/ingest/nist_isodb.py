"""NIST ISODB REST ingester (public-domain US-government data).

Endpoint (verified):

- ``https://adsorption.nist.gov/isodb/api/isotherms.json`` — paginated index
- ``https://adsorption.nist.gov/isodb/api/isotherm/<filename>.json`` — record

Each isotherm record carries ``DOI``, ``adsorbates``, ``adsorbent``,
``isotherm_data`` (pressure / loading), ``temperature``, ``pressureUnits``,
``adsorptionUnits``. From the lowest-pressure points we estimate the Henry-regime
slope for v0.4 scalar-truth ingestion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NistIsodbScalar:
    """One Henry-regime + Q_ads estimator distilled from a NIST ISODB isotherm."""

    isotherm_id: str
    material: str
    gas: str
    temperature_K: float
    KH_estimator_mol_per_kg_per_Pa: float | None
    KH_estimator_method: str
    Qads_estimator_kJ_per_mol: float | None
    n_points_used_for_KH: int
    pressure_min_Pa: float | None
    pressure_max_Pa: float | None
    source_doi: str | None
    warnings: list[str] = field(default_factory=list)


def _to_pa(value: float, units: str) -> float:
    u = (units or "").lower().strip()
    if u in {"pa", "pascal", "n/m^2"}:
        return float(value)
    if u == "kpa":
        return float(value) * 1e3
    if u in {"bar"}:
        return float(value) * 1e5
    if u in {"atm"}:
        return float(value) * 101325.0
    if u in {"mmhg", "torr"}:
        return float(value) * 133.322
    raise ValueError(f"unsupported NIST ISODB pressure unit {units!r}")


def _to_mol_per_kg(value: float, units: str) -> float:
    u = (units or "").lower().strip()
    if u in {"mol/kg", "mol kg-1"}:
        return float(value)
    if u in {"mmol/g", "mmol g-1"}:
        return float(value)  # equal to mol/kg
    if u in {"cm3(stp)/g", "cm3 stp/g"}:
        return float(value) / 22414.0  # rough STP conversion
    raise ValueError(f"unsupported NIST ISODB loading unit {units!r}")


def parse_nist_isotherm(payload: dict[str, Any]) -> NistIsodbScalar:
    """Parse one NIST ISODB isotherm JSON record into a Henry-regime scalar."""
    iso_id = str(payload.get("filename", payload.get("id", "unknown")))
    adsorbates = payload.get("adsorbates") or []
    gas = "unknown"
    if adsorbates:
        gas = str(adsorbates[0].get("formula") or adsorbates[0].get("name") or "unknown")
    adsorbent = payload.get("adsorbent") or {}
    material = str(adsorbent.get("name") or adsorbent.get("hashkey") or "unknown")
    try:
        T_K = float(payload.get("temperature") or 0.0)
    except (TypeError, ValueError):
        T_K = 0.0

    pressure_units = payload.get("pressureUnits") or ""
    loading_units = payload.get("adsorptionUnits") or ""
    data = payload.get("isotherm_data") or []

    p_pa: list[float] = []
    n_mol_per_kg: list[float] = []
    warnings: list[str] = []
    for pt in data[: 5]:  # use the 5 lowest-pressure points
        try:
            p_native = float(pt.get("pressure"))
            n_native = pt.get("total_adsorption")
            if n_native is None:
                continue
            p_pa.append(_to_pa(p_native, pressure_units))
            n_mol_per_kg.append(_to_mol_per_kg(float(n_native), loading_units))
        except (TypeError, ValueError) as exc:
            warnings.append(f"point parse error: {exc}")
            continue
    KH = None
    method = "no_data"
    if len(p_pa) >= 2:
        # Linear fit through origin: K_H ≈ slope of n vs p
        num = sum(pi * ni for pi, ni in zip(p_pa, n_mol_per_kg, strict=True))
        den = sum(pi * pi for pi in p_pa)
        if den > 0:
            KH = num / den
            method = f"linear_through_origin_n={len(p_pa)}"
        else:
            warnings.append("zero pressure denominator")
    elif len(p_pa) == 1 and p_pa[0] > 0:
        KH = n_mol_per_kg[0] / p_pa[0]
        method = "single_point_through_origin"
        warnings.append("only one low-P point — KH is order-of-magnitude only")

    return NistIsodbScalar(
        isotherm_id=iso_id,
        material=material,
        gas=gas,
        temperature_K=T_K,
        KH_estimator_mol_per_kg_per_Pa=KH,
        KH_estimator_method=method,
        Qads_estimator_kJ_per_mol=None,  # NIST ISODB stores enthalpy separately, not in isotherm files
        n_points_used_for_KH=len(p_pa),
        pressure_min_Pa=min(p_pa) if p_pa else None,
        pressure_max_Pa=max(p_pa) if p_pa else None,
        source_doi=payload.get("DOI"),
        warnings=warnings,
    )


def fetch_nist_isotherm(filename: str, *, timeout_s: float = 60.0) -> dict[str, Any]:
    """Live fetch one NIST ISODB isotherm record."""
    import urllib.request

    url = f"https://adsorption.nist.gov/isodb/api/isotherm/{filename}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "widom-atlas/0.4"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read())
    if isinstance(data, dict):
        return data
    raise ValueError(f"NIST ISODB returned non-dict for {filename}: {type(data).__name__}")


def write_record_to_cache(payload: dict[str, Any], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = payload.get("filename") or payload.get("id") or "unknown"
    out = cache_dir / f"{fname}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


__all__ = [
    "NistIsodbScalar",
    "fetch_nist_isotherm",
    "parse_nist_isotherm",
    "write_record_to_cache",
]
