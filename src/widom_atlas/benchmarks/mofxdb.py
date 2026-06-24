"""MOFX-DB / NIST scalar-reference loader, cache-first.

Lookup order:

1. ``cache_dir/mofxdb/<sha256>.json`` — operator-supplied cache (highest priority).
2. ``importlib.resources.files("widom_atlas.benchmarks") / "data" / "mofxdb_cache" / "<sha256>.json"``
   — bundled curated literature values shipped with the package (REPAIR-2).
3. Empty record at the configured identity confidence — degrades gracefully.

Network access is intentionally not implemented in v1. Identity mapping
between widom-atlas material_ids and MOFX-DB names is explicit so the
comparison layer can label matches as TREND only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

IdentityConfidence = Literal["high", "medium", "low"]

# Curated identity allowlist: (material_id, gas) -> (mofxdb_id, confidence)
_IDENTITY_TABLE: Final[dict[tuple[str, str], tuple[str, IdentityConfidence]]] = {
    ("UiO-66", "CO2"): ("uio-66", "high"),
    ("UiO-66", "CH4"): ("uio-66", "high"),
    ("ZIF-8", "CO2"): ("zif-8", "high"),
    ("ZIF-8", "CH4"): ("zif-8", "high"),
    ("ZIF-8", "N2"): ("zif-8", "high"),
    ("MOF-5", "CO2"): ("mof-5", "high"),
    ("Mg-MOF-74", "CO2"): ("mg-mof-74", "high"),
    ("Mg-MOF-74", "N2"): ("mg-mof-74", "medium"),
    ("MFI", "CO2"): ("mfi", "medium"),
    ("CHA", "CO2"): ("cha", "medium"),
}


class MOFXDBRecord(BaseModel):
    """One scalar reference record from MOFX-DB / NIST."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    material_id: str
    gas: str
    temperature_K: float = Field(gt=0.0)
    KH: float | None = None
    KH_units: str = "mol/(kg*Pa)"
    Qads: float | None = None
    Qads_units: str = "kJ/mol"
    source: str = "literature"
    source_url: str | None = None
    citation: str | None = None
    dataset_version: str = "unknown"
    sha256: str | None = None
    license: str = "verify_terms_per_download"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    identity_confidence: IdentityConfidence = "low"


def _identity_lookup(material_id: str, gas: str) -> tuple[str | None, IdentityConfidence]:
    pair = _IDENTITY_TABLE.get((material_id, gas))
    if pair is None:
        return None, "low"
    return pair


def _cache_key(material_id: str, gas: str) -> str:
    """SHA256 of ``"<material_id>|<gas>"`` — used as the cache filename stem."""
    return hashlib.sha256(f"{material_id}|{gas}".encode()).hexdigest()


def _cache_path(cache_dir: Path, material_id: str, gas: str) -> Path:
    """Resolve the operator-supplied cache file path. Always creates ``mofxdb/`` if needed."""
    base = Path(cache_dir) / "mofxdb"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_cache_key(material_id, gas)}.json"


def _bundled_cache_payload(material_id: str, gas: str) -> dict | None:
    """Look up a bundled curated literature record shipped with the package."""
    key = _cache_key(material_id, gas)
    try:
        bundled = resources.files("widom_atlas.benchmarks").joinpath("data", "mofxdb_cache", f"{key}.json")
        if not bundled.is_file():
            return None
        return json.loads(bundled.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def load_mofxdb_scalars(
    material_id: str,
    gas: str,
    cache_dir: Path,
) -> MOFXDBRecord:
    """Look up cached MOFX-DB / literature scalars.

    Operator cache (under ``cache_dir/mofxdb/``) wins over the bundled
    package cache. If neither resolves, a low-confidence empty record is
    returned so the comparison layer can degrade to ``UNAVAILABLE``.
    """
    mofxdb_id, confidence = _identity_lookup(material_id, gas)
    cache_p = _cache_path(cache_dir, material_id, gas)
    if cache_p.exists():
        payload = json.loads(cache_p.read_text(encoding="utf-8"))
        return MOFXDBRecord.model_validate(payload)

    bundled = _bundled_cache_payload(material_id, gas)
    if bundled is not None:
        return MOFXDBRecord.model_validate(bundled)

    return MOFXDBRecord(
        material_id=material_id,
        gas=gas,
        temperature_K=298.15,
        KH=None,
        Qads=None,
        identity_confidence=confidence,
        source_url=f"mofxdb://{mofxdb_id}" if mofxdb_id else None,
        dataset_version="unknown",
        sha256=None,
    )


__all__ = [
    "MOFXDBRecord",
    "_bundled_cache_payload",
    "_cache_key",
    "_cache_path",
    "_identity_lookup",
    "load_mofxdb_scalars",
]
