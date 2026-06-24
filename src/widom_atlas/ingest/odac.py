"""ODAC23 / ODAC25 ingester — OPTIONAL EXTRA.

This module is behind ``widom-atlas[odac]`` per reviewer decision 4. ODAC25
HuggingFace dataset is gated; ODAC23 is direct-download from
``dl.fbaipublicfiles.com`` and is the v0.4 fallback. v0.4 scope is
restricted to S2EF and DDEC subsets (IS2RE / IS2RS tasks deferred).

This module imports lmdb / fastavro lazily so the core install stays small.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OdacArchiveStatus:
    archive_path: str
    expected_md5: str | None
    actual_md5: str
    matches: bool
    size_bytes: int
    notes: str


def _md5(p: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_odac23_archive(archive_path: Path, expected_md5: str | None) -> OdacArchiveStatus:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        return OdacArchiveStatus(
            archive_path=str(archive_path),
            expected_md5=expected_md5,
            actual_md5="",
            matches=False,
            size_bytes=0,
            notes="missing — operator must download from dl.fbaipublicfiles.com",
        )
    actual = _md5(archive_path)
    matches = (expected_md5 is not None and actual == expected_md5)
    return OdacArchiveStatus(
        archive_path=str(archive_path),
        expected_md5=expected_md5,
        actual_md5=actual,
        matches=matches,
        size_bytes=archive_path.stat().st_size,
        notes="md5 verified" if matches else ("no expected md5" if expected_md5 is None else "MD5 MISMATCH"),
    )


def is_lmdb_available() -> bool:
    return shutil.which("python") is not None and _has_module("lmdb")


def _has_module(name: str) -> bool:
    import importlib

    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def parse_odac23_lmdb_summary(lmdb_dir: Path) -> dict[str, Any]:
    """Open an LMDB folder lazily and report n_entries + first-key sample.

    Requires ``widom-atlas[odac]`` extra (lmdb + fastavro). Without those
    installed, returns a structured error.
    """
    if not _has_module("lmdb"):
        return {
            "ok": False,
            "error": "lmdb module not installed; install via 'pip install widom-atlas[odac]'",
        }
    import lmdb

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        n = txn.stat().get("entries", 0)
        cursor = txn.cursor()
        first = next(iter(cursor), None)
        first_key = first[0].hex() if first is not None else None
    return {"ok": True, "n_entries": n, "first_key_hex": first_key}


__all__ = [
    "OdacArchiveStatus",
    "is_lmdb_available",
    "parse_odac23_lmdb_summary",
    "verify_odac23_archive",
]
