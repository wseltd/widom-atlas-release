"""Loaders + lookup helpers for the YAML registry under ``data/``.

All public APIs return validated Pydantic models from
:mod:`widom_atlas.data_registry.schema`. The YAML files are loaded via
``importlib.resources`` so they ship with the installed package.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    DatasetRegistryEntry,
    ScalarReferenceEntry,
    SiteReferenceEntry,
    ThresholdSet,
    ValidationThresholds,
)

_DATA_PACKAGE = "widom_atlas.data_registry.data"


def _load_yaml(name: str) -> Any:
    text = resources.files(_DATA_PACKAGE).joinpath(name).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def list_datasets() -> list[DatasetRegistryEntry]:
    """Return every registered dataset, validated."""
    raw = _load_yaml("datasets.yaml")
    entries = raw.get("datasets") or []
    return [DatasetRegistryEntry.model_validate(e) for e in entries]


def load_dataset(name: str) -> DatasetRegistryEntry:
    """Look up one dataset by ``name`` (e.g. ``"CRAFTED"``)."""
    for d in list_datasets():
        if d.name == name:
            return d
    available = [d.name for d in list_datasets()]
    raise KeyError(f"unknown dataset {name!r}; registered: {available}")


def list_scalar_references() -> list[ScalarReferenceEntry]:
    raw = _load_yaml("scalars.yaml")
    entries = raw.get("scalars") or []
    return [ScalarReferenceEntry.model_validate(e) for e in entries]


def load_scalar_reference(
    material_id: str,
    gas: str,
    temperature_K: float | None = None,
) -> list[ScalarReferenceEntry]:
    """Look up scalar references for a ``(material, gas, T)`` triple.

    Returns *all* matching entries (multiple references per pair are
    allowed; the operator picks the one whose measurement_method fits).
    """
    out = [
        s for s in list_scalar_references()
        if s.material_id == material_id and s.gas == gas and (
            temperature_K is None or abs(s.temperature_K - temperature_K) < 1e-6
        )
    ]
    return out


def list_site_references() -> list[SiteReferenceEntry]:
    raw = _load_yaml("sites.yaml")
    entries = raw.get("sites") or []
    return [SiteReferenceEntry.model_validate(e) for e in entries]


def load_site_reference(
    material_id: str,
    gas: str,
    label: str | None = None,
) -> list[SiteReferenceEntry]:
    """Look up site references for ``(material, gas[, label])``."""
    out = [
        s for s in list_site_references()
        if s.material_id == material_id and s.gas == gas and (label is None or s.label == label)
    ]
    return out


def load_validation_thresholds() -> ValidationThresholds:
    """Load the named-threshold-set table."""
    raw = _load_yaml("thresholds.yaml")
    return ValidationThresholds.model_validate(raw)


def load_threshold_set(name: str) -> ThresholdSet:
    """Look up one named threshold set, e.g. ``"v0_4_minimum"``."""
    vt = load_validation_thresholds()
    if name not in vt.sets:
        raise KeyError(
            f"unknown threshold set {name!r}; registered: {list(vt.sets.keys())}"
        )
    return vt.sets[name]


# ----------------------------------------------------------------------
# Local-cache verification helpers
# ----------------------------------------------------------------------

def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_of(path: Path) -> str:
    # MD5 here is a checksum for download-integrity verification only,
    # NOT a cryptographic hash. ODAC23 publishes file MD5s; we mirror them.
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_status(
    entry: DatasetRegistryEntry,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Report whether a dataset's ``cache_path`` is present locally and verified.

    Returns a dict with ``{name, present, cache_path, expected_sha256,
    expected_md5, actual_sha256, actual_md5, verified, note}``.

    Hash verification policy (v0.4): a dataset may declare ``expected_sha256``
    OR ``expected_md5`` (or both). The status is ``verified=True`` when at
    least one declared hash matches and no declared hash mismatches. If
    only ``expected_md5s`` (per-file dict) is declared, ``verified`` is set
    when the cache_path points at a directory whose files match every
    declared md5; missing files in the dict are fine.
    """
    base_template: dict[str, Any] = {
        "name": entry.name,
        "present": False,
        "cache_path": None,
        "expected_sha256": entry.expected_sha256,
        "expected_md5": entry.expected_md5,
        "actual_sha256": None,
        "actual_md5": None,
        "verified": False,
        "note": "",
    }
    if not entry.cache_path:
        base_template["note"] = "registry entry has no cache_path; nothing to check on disk"
        return base_template
    base = Path(repo_root or ".").resolve() / entry.cache_path
    if not base.exists():
        base_template["cache_path"] = str(base)
        base_template["note"] = "cache_path missing — operator has not downloaded yet"
        return base_template
    if base.is_file():
        actual_sha = _sha256_of(base)
        actual_md5 = _md5_of(base) if (entry.expected_md5 or entry.expected_sha256 is None) else None
        sha_ok = entry.expected_sha256 is not None and actual_sha == entry.expected_sha256
        md5_ok = entry.expected_md5 is not None and actual_md5 == entry.expected_md5
        any_declared = entry.expected_sha256 is not None or entry.expected_md5 is not None
        sha_mismatch = entry.expected_sha256 is not None and actual_sha != entry.expected_sha256
        md5_mismatch = entry.expected_md5 is not None and actual_md5 != entry.expected_md5
        verified = (sha_ok or md5_ok) and not (sha_mismatch or md5_mismatch)
        if not any_declared:
            note = "single file (no expected hash declared)"
        elif sha_mismatch and md5_mismatch:
            note = "sha256 AND md5 mismatch"
        elif sha_mismatch:
            note = "sha256 mismatch"
        elif md5_mismatch:
            note = "md5 mismatch"
        else:
            note = "single file, hash verified"
        return {
            **base_template,
            "present": True,
            "cache_path": str(base),
            "actual_sha256": actual_sha,
            "actual_md5": actual_md5,
            "verified": verified,
            "note": note,
        }
    # Directory case — when entry.expected_md5s is set, validate per-file md5s.
    contents = sorted(p.name for p in base.iterdir() if p.is_file())
    if entry.expected_md5s:
        present_files = {p.name: p for p in base.iterdir() if p.is_file()}
        all_match = True
        any_present = False
        per_file_status: dict[str, str] = {}
        for fname, expected_md5 in entry.expected_md5s.items():
            if fname not in present_files:
                per_file_status[fname] = "missing"
                continue
            any_present = True
            actual = _md5_of(present_files[fname])
            if actual == expected_md5:
                per_file_status[fname] = "ok"
            else:
                per_file_status[fname] = f"mismatch (got {actual[:8]}…)"
                all_match = False
        return {
            **base_template,
            "present": any_present,
            "cache_path": str(base),
            "verified": any_present and all_match,
            "note": f"directory with {len(contents)} entries; per-file md5: {per_file_status}",
        }
    return {
        **base_template,
        "present": bool(contents),
        "cache_path": str(base),
        "verified": False,
        "note": f"directory with {len(contents)} entries — manual sha256/md5 verification required",
    }


__all__ = [
    "dataset_status",
    "list_datasets",
    "list_scalar_references",
    "list_site_references",
    "load_dataset",
    "load_scalar_reference",
    "load_site_reference",
    "load_threshold_set",
    "load_validation_thresholds",
]
