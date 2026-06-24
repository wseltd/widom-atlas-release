"""SHA256 hashing + provenance.json for cached benchmark CIFs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from widom_atlas.core.benchmark_models import BenchmarkMaterial

CHUNK_SIZE_BYTES: Final[int] = 1 << 20  # 1 MiB
_LOGGER = logging.getLogger(__name__)
_HEX = frozenset("0123456789abcdef")


class ProvenanceMismatch(RuntimeError):
    """Raised when a recorded SHA256 disagrees with the live file digest."""


class ProvenanceRecord(BaseModel):
    """Pydantic v2 record persisted alongside each cached CIF."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    material_id: str
    source: str
    sha256: str
    file_size_bytes: int = Field(ge=0)
    license_tag: str
    citation_doi: str
    dataset_version: str
    fetched_at: datetime
    cif_path: str
    extra: dict[str, str] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if len(v) != 64 or not all(c in _HEX for c in v):
            raise ValueError(f"sha256 must be 64 lowercase hex chars; got len={len(v)}")
        return v


def sha256_file(path: Path, *, chunk_size: int = CHUNK_SIZE_BYTES) -> str:
    """Stream a file in ``chunk_size`` byte chunks and return its lowercase hex SHA256."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance_path(cache_dir: Path, material: BenchmarkMaterial) -> Path:
    return Path(cache_dir) / material.source / f"{material.material_id}.provenance.json"


def _meta_path(cache_dir: Path, material: BenchmarkMaterial) -> Path:
    return Path(cache_dir) / material.source / f"{material.material_id}.meta.json"


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise


def record_provenance(
    material: BenchmarkMaterial,
    cif_path: Path,
    cache_dir: Path,
    *,
    dataset_version: str | None = None,
) -> ProvenanceRecord:
    """Compute provenance fields and persist ``<material>.provenance.json`` atomically."""
    cif = Path(cif_path)
    if not cif.exists():
        raise FileNotFoundError(f"cif not found: {cif}")
    digest = sha256_file(cif)
    file_size = int(cif.stat().st_size)
    prov_path = _provenance_path(cache_dir, material)
    prov_path.parent.mkdir(parents=True, exist_ok=True)

    if prov_path.exists():
        existing_data = json.loads(prov_path.read_text(encoding="utf-8"))
        existing = ProvenanceRecord.model_validate(existing_data)
        if existing.sha256 != digest:
            raise ProvenanceMismatch(
                f"file digest changed since record: stored={existing.sha256} live={digest}"
            )
        fetched_at = existing.fetched_at
    else:
        fetched_at = datetime.now(UTC)

    if dataset_version is None:
        meta_p = _meta_path(cache_dir, material)
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                dataset_version = meta.get("dataset_version") or meta.get("version")
            except (OSError, ValueError):
                dataset_version = None
    if dataset_version is None:
        _LOGGER.warning("dataset_version unknown for material_id=%s", material.material_id)
        dataset_version = "unknown"

    record = ProvenanceRecord(
        material_id=material.material_id,
        source=material.source,
        sha256=digest,
        file_size_bytes=file_size,
        license_tag=material.license,
        citation_doi=material.citation,
        dataset_version=dataset_version,
        fetched_at=fetched_at,
        cif_path=str(cif),
    )
    payload = json.dumps(record.model_dump(mode="json"), sort_keys=True, indent=2)
    _atomic_write(prov_path, (payload + "\n").encode("utf-8"))
    return record


__all__ = ["CHUNK_SIZE_BYTES", "ProvenanceMismatch", "ProvenanceRecord", "record_provenance", "sha256_file"]
