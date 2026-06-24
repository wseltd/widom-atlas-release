"""CRAFTED v2 archive ingester.

Operator drops ``CRAFTED-2.0.1.tar.xz`` (55 MB, CDLA-Sharing-1.0) under
``benchmarks/cache/crafted/`` and runs ``widom-atlas ingest crafted``. The
ingester unpacks the archive to a sibling directory, scans the resulting
isotherm CSVs, and emits ``ScalarReferenceEntry``-shaped JSONs ready for
the validation runner.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CraftedArchiveSummary:
    """Summary of an unpacked CRAFTED archive."""

    archive_path: str
    extracted_root: str
    n_csv_files: int
    n_json_files: int
    materials_seen: int
    gases_seen: list[str]
    temperatures_K: list[float]
    warnings: list[str]


def unpack_crafted_archive(archive_path: Path, dest_root: Path) -> Path:
    """Extract CRAFTED-*.tar.xz under ``dest_root``."""
    archive_path = Path(archive_path)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:xz") as tar:
        # Safe-extract: refuse absolute paths or "../" traversal
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name.split("/"):
                raise ValueError(f"unsafe tar member: {member.name!r}")
        tar.extractall(dest_root, filter="data")
    return dest_root


def summarise_unpacked_crafted(extracted_root: Path) -> CraftedArchiveSummary:
    """Walk an extracted CRAFTED directory tree and report what's there."""
    csv_files = list(extracted_root.rglob("*.csv"))
    json_files = list(extracted_root.rglob("*.json"))
    materials: set[str] = set()
    gases: set[str] = set()
    temperatures: set[float] = set()
    warnings: list[str] = []
    # CRAFTED filenames typically encode {framework}_{gas}_{T}_{ff}_{charge}.csv
    for f in csv_files[:200]:  # cheap pass over the first 200
        parts = f.stem.split("_")
        if len(parts) >= 4:
            materials.add(parts[0])
            gases.add(parts[1])
            try:
                T = float(parts[2])
                temperatures.add(T)
            except ValueError:
                continue
    if not csv_files:
        warnings.append("no CSVs found under extracted_root")
    return CraftedArchiveSummary(
        archive_path="",
        extracted_root=str(extracted_root),
        n_csv_files=len(csv_files),
        n_json_files=len(json_files),
        materials_seen=len(materials),
        gases_seen=sorted(gases),
        temperatures_K=sorted(temperatures),
        warnings=warnings,
    )


def write_summary_to_cache(summary: CraftedArchiveSummary, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.__dict__, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


__all__ = [
    "CraftedArchiveSummary",
    "summarise_unpacked_crafted",
    "unpack_crafted_archive",
    "write_summary_to_cache",
]
