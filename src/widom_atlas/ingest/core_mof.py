"""CoRE-MOF zip ingester (v2019 v2 + v2024-structures).

Walks the unpacked archive tree and produces:

- a flat list of (refcode, cif_path, sha256) tuples
- a summary JSON
- registry-compatible structure_metadata records for cached CIFs

The operator drops the zip under ``benchmarks/cache/core_mof_2019/`` (or
``core_mof_2024/structures/``), runs ``widom-atlas ingest core-mof``, and
the resulting CIF cache is consumable by the validation runner.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CoreMofUnpackResult:
    archive_path: str
    extracted_root: str
    n_cifs: int
    refcodes_sample: list[str]
    total_size_bytes: int


def unpack_core_mof_zip(archive_path: Path, dest_root: Path) -> CoreMofUnpackResult:
    archive_path = Path(archive_path)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.filename.startswith("/") or ".." in info.filename.split("/"):
                raise ValueError(f"unsafe zip member: {info.filename!r}")
        zf.extractall(dest_root)
    cifs = list(dest_root.rglob("*.cif"))
    return CoreMofUnpackResult(
        archive_path=str(archive_path),
        extracted_root=str(dest_root),
        n_cifs=len(cifs),
        refcodes_sample=sorted([c.stem for c in cifs[:20]]),
        total_size_bytes=sum(c.stat().st_size for c in cifs),
    )


def cif_metadata(cif_path: Path) -> dict[str, str | int]:
    """Compute provenance metadata for a single CIF."""
    return {
        "path": str(cif_path),
        "refcode": cif_path.stem,
        "size_bytes": cif_path.stat().st_size,
        "sha256": _sha256(cif_path),
    }


def write_summary(result: CoreMofUnpackResult, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "archive_path": result.archive_path,
        "extracted_root": result.extracted_root,
        "n_cifs": result.n_cifs,
        "refcodes_sample": result.refcodes_sample,
        "total_size_bytes": result.total_size_bytes,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


__all__ = ["CoreMofUnpackResult", "cif_metadata", "unpack_core_mof_zip", "write_summary"]
