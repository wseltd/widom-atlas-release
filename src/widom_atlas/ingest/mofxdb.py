"""MOFX-DB REST ingester (Snurr group, Northwestern).

Endpoints (verified live in ``docs/research/dataset-research-for-v0.4/agents-notebook.md``):

- ``https://mof.tech.northwestern.edu/mofs.json?page=N`` — paginated index
- ``https://mof.tech.northwestern.edu/mofs/<id>.json`` — full per-MOF record
- ``https://mof.tech.northwestern.edu/databases.json`` — sub-database catalogue
- ``https://mof.tech.northwestern.edu/forcefields.json`` — force-field IDs

The full per-MOF record schema (verified 2026-05-08) carries these keys:

::

   id, mofid, mofkey, hashkey, name, void_fraction, surface_area_m2g,
   surface_area_m2cm3, pld, lcd, pxrd, pore_size_distribution,
   database, batch_number, elements, cif, url, adsorbates, heats,
   isotherms, mofdb_version

For every ``heats[*]`` entry, this ingester extracts the embedded
``simin`` string verbatim — a full RASPA-style ``simulation.input``
block — together with the source MOFX record id, force-field id,
component names, and units. The simin is hashed with sha256 so the
parity test (Phase C) can refer to a stable reference.

Per the v0.4 follow-up brief: ``hMOF`` and ``Tobacco`` records are
classified as ``hypothetical`` and rejected by the validation runner
unless ``--include-hypothetical`` is set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from widom_atlas.data_registry.schema import (
    MOFXDB_FORCE_FIELD_ID_TO_NAME,
    MofxdbProvenanceKind,
    mofxdb_provenance_kind_from_database,
)
from widom_atlas.ingest.raspa3_ff import hash_simin_string


@dataclass(frozen=True)
class MofxdbSiminRecord:
    """One ``heats[*]`` entry distilled to the parity-test contract."""

    mofx_record_id: int
    mofx_database: str
    provenance_kind: MofxdbProvenanceKind
    framework_name: str
    component_names: list[str]
    force_field_id: int | None
    force_field_name: str
    gas: str
    temperature_K: float
    KH_value: float | None
    KH_units: str | None
    Qads_value: float | None
    Qads_units: str | None
    simin_text: str
    simin_sha256: str
    source_doi: str | None
    warnings: list[str] = field(default_factory=list)


def _iter_heats(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("heats", []) or [])


def _iter_isotherms(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("isotherms", []) or [])


def _force_field_name(record: dict[str, Any]) -> tuple[int | None, str, list[str]]:
    """Return (id, name, warnings). MOFX-DB heats records expose force-field
    by name in ``adsorbent_forcefield``; isotherm records sometimes carry an
    integer id. We preserve the verbatim string and try to resolve it back
    to the closed enum.
    """
    warnings: list[str] = []
    explicit_id = record.get("force_field_id")
    if explicit_id is not None:
        try:
            fid_int = int(explicit_id)
        except (TypeError, ValueError):
            fid_int = None
        if fid_int is not None:
            fname = MOFXDB_FORCE_FIELD_ID_TO_NAME.get(
                fid_int, str(record.get("force_field") or "unknown")
            )
            return fid_int, fname, warnings
    raw_name = (
        record.get("force_field")
        or record.get("adsorbent_forcefield")
        or ""
    )
    raw_name = (raw_name or "").strip()
    if not raw_name:
        return None, "unknown", ["unknown_mofxdb_force_field_id"]
    for fid, fname in MOFXDB_FORCE_FIELD_ID_TO_NAME.items():
        if fname.lower() == raw_name.lower() or raw_name.lower() in fname.lower():
            return fid, fname, warnings
    warnings.append("unknown_mofxdb_force_field_id")
    return None, raw_name, warnings


def _gas_name(record: dict[str, Any]) -> str:
    adsorbates = record.get("adsorbates") or []
    if not adsorbates:
        return "unknown"
    first = adsorbates[0]
    formula = (first.get("formula") or "").strip()
    if formula:
        return formula
    name = (first.get("name") or "").strip()
    return name or "unknown"


def _simin_components(simin: str) -> list[str]:
    """Extract Component MoleculeName tokens from a simin string (best-effort)."""
    out: list[str] = []
    for line in simin.splitlines():
        s = line.strip()
        if s.startswith("Component "):
            tokens = s.split()
            for j, t in enumerate(tokens):
                if t == "MoleculeName" and j + 1 < len(tokens):
                    out.append(tokens[j + 1])
    return out


def parse_mofxdb_record(record: dict[str, Any]) -> dict[str, Any]:
    """Distill one MOFX-DB MOF record into a structured dict.

    Returns ``{mof_id, name, database, provenance_kind, void_fraction,
    pld, lcd, surface_area_m2g, cif, mofdb_version, n_heats, n_isotherms,
    simin_records: [MofxdbSiminRecord, ...]}``.
    """
    db_name = record.get("database", {}) or {}
    if isinstance(db_name, dict):
        database_str = db_name.get("name") or "unknown"
    else:
        database_str = str(db_name) if db_name else "unknown"
    prov = mofxdb_provenance_kind_from_database(database_str)

    simin_records: list[MofxdbSiminRecord] = []
    for kind, sub in [("heat", _iter_heats(record)), ("isotherm", _iter_isotherms(record))]:
        for h in sub:
            simin_text = h.get("simin") or ""
            if not simin_text or not isinstance(simin_text, str):
                continue
            fid, fname, ff_warnings = _force_field_name(h)
            adsorbates = h.get("adsorbates") or h.get("adsorbate") or [{}]
            first_ads = adsorbates[0] if adsorbates else {}
            gas = first_ads.get("formula") or first_ads.get("name") or "unknown"
            try:
                T_K = float(h.get("temperature") or 0.0)
            except (TypeError, ValueError):
                T_K = 0.0
            framework_name = ((h.get("adsorbent") or {}).get("name") or record.get("name") or "unknown")
            Q_value_raw = h.get("value")
            try:
                Q_value: float | None = float(Q_value_raw) if Q_value_raw is not None else None
            except (TypeError, ValueError):
                Q_value = None
            KH_value = None
            KH_units = None
            if kind == "isotherm":
                iso_data = h.get("isotherm_data") or []
                if iso_data:
                    p0 = iso_data[0].get("pressure")
                    n0 = (iso_data[0].get("total_adsorption") or
                          (iso_data[0].get("species_data") or [{}])[0].get("adsorption"))
                    try:
                        if p0 is not None and n0 is not None and float(p0) > 0:
                            KH_value = float(n0) / float(p0)
                            p_units = h.get("pressureUnits") or ""
                            n_units = h.get("adsorptionUnits") or ""
                            KH_units = f"({n_units})/({p_units})"
                    except (TypeError, ValueError):
                        pass
            Q_units_raw = h.get("units") or h.get("adsorptionUnits")
            Q_units = Q_units_raw if isinstance(Q_units_raw, str) else None
            ff_warnings_kind = [*ff_warnings, f"mofxdb_kind={kind}"]
            simin_records.append(
                MofxdbSiminRecord(
                    mofx_record_id=int(h.get("id", 0) or 0),
                    mofx_database=database_str,
                    provenance_kind=prov,
                    framework_name=str(framework_name),
                    component_names=_simin_components(simin_text),
                    force_field_id=fid,
                    force_field_name=fname,
                    gas=str(gas),
                    temperature_K=T_K,
                    KH_value=KH_value if kind == "isotherm" else None,
                    KH_units=KH_units,
                    Qads_value=Q_value if kind == "heat" else None,
                    Qads_units=Q_units if kind == "heat" else None,
                    simin_text=simin_text,
                    simin_sha256=hash_simin_string(simin_text),
                    source_doi=h.get("DOI") or h.get("doi"),
                    warnings=ff_warnings_kind,
                )
            )

    return {
        "mof_id": record.get("id"),
        "name": record.get("name"),
        "database": database_str,
        "provenance_kind": prov,
        "void_fraction": record.get("void_fraction"),
        "pld": record.get("pld"),
        "lcd": record.get("lcd"),
        "surface_area_m2g": record.get("surface_area_m2g"),
        "cif": record.get("cif"),
        "mofdb_version": record.get("mofdb_version"),
        "n_heats": len(_iter_heats(record)),
        "n_isotherms": len(_iter_isotherms(record)),
        "simin_records": simin_records,
    }


def fetch_mofxdb_record(mof_id: int, *, timeout_s: float = 60.0) -> dict[str, Any]:
    """Live fetch ``https://mof.tech.northwestern.edu/mofs/<id>.json``."""
    import urllib.request

    url = f"https://mof.tech.northwestern.edu/mofs/{mof_id}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "widom-atlas/0.4"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def fetch_mofxdb_databases(*, timeout_s: float = 30.0) -> list[dict[str, Any]]:
    import urllib.request

    url = "https://mof.tech.northwestern.edu/databases.json"
    req = urllib.request.Request(url, headers={"User-Agent": "widom-atlas/0.4"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return list(json.loads(r.read()))


def write_record_to_cache(record: dict[str, Any], cache_dir: Path) -> Path:
    """Persist a fetched MOFX-DB record to ``cache_dir/<mof_id>.json``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mof_id = record.get("id")
    out = cache_dir / f"{mof_id}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return out


def select_deterministic_simin_records(
    records: list[dict[str, Any]] | list[MofxdbSiminRecord],
    *,
    n: int,
    seed: int,
    exclude_hypothetical: bool = True,
    require_distinct_force_fields: bool = True,
) -> list[MofxdbSiminRecord]:
    """Select N records deterministically for the parity gate.

    Per the v0.4 follow-up brief §"Phase C parity gate is now stronger":

    - Excludes hMOF and Tobacco records by default (provenance_kind = hypothetical).
    - Where possible, picks records with distinct force_field_id values.
    - Deterministic given the seed.
    - Source records may be (a) full MOFX MOF records (dicts), (b) already-distilled
      ``parse_mofxdb_record`` outputs (dicts with ``simin_records`` key), or (c) a
      flat ``list[MofxdbSiminRecord]``.
    """
    import random

    rng = random.Random(seed)

    pool: list[MofxdbSiminRecord] = []
    for rec in records:
        if isinstance(rec, MofxdbSiminRecord):
            sr_iter: list[MofxdbSiminRecord] = [rec]
        elif isinstance(rec, dict) and "simin_records" in rec:
            sr_iter = list(rec["simin_records"])
        else:
            sr_iter = list(parse_mofxdb_record(rec)["simin_records"])
        for sr in sr_iter:
            if exclude_hypothetical and sr.provenance_kind == "hypothetical":
                continue
            pool.append(sr)
    if not pool:
        return []
    rng.shuffle(pool)
    if not require_distinct_force_fields:
        return pool[:n]

    seen_ff: set[int | None] = set()
    out: list[MofxdbSiminRecord] = []
    leftovers: list[MofxdbSiminRecord] = []
    for sr in pool:
        if sr.force_field_id in seen_ff:
            leftovers.append(sr)
            continue
        out.append(sr)
        seen_ff.add(sr.force_field_id)
        if len(out) >= n:
            break
    # Fill from leftovers if we ran out of unique FF IDs.
    if len(out) < n:
        out.extend(leftovers[: n - len(out)])
    return out[:n]


__all__ = [
    "MofxdbSiminRecord",
    "fetch_mofxdb_databases",
    "fetch_mofxdb_record",
    "parse_mofxdb_record",
    "select_deterministic_simin_records",
    "write_record_to_cache",
]
