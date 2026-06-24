"""QMOF (Rosen) ingester — figshare 13147324, CC-BY-4.0.

The QMOF release ships ``qmof_database.zip`` (392 MB) and
``qmof_thermo_database.zip`` (153 MB). Each MOF entry has a CIF, a JSON
metadata blob with PACMAN charges, and DFT-derived properties.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QmofUnpackResult:
    archive_path: str
    extracted_root: str
    n_cifs: int
    n_json: int
    sample_refcodes: list[str]


def unpack_qmof_zip(archive_path: Path, dest_root: Path) -> QmofUnpackResult:
    archive_path = Path(archive_path)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.filename.startswith("/") or ".." in info.filename.split("/"):
                raise ValueError(f"unsafe zip member: {info.filename!r}")
        zf.extractall(dest_root)
    cifs = list(dest_root.rglob("*.cif"))
    jsons = list(dest_root.rglob("*.json"))
    return QmofUnpackResult(
        archive_path=str(archive_path),
        extracted_root=str(dest_root),
        n_cifs=len(cifs),
        n_json=len(jsons),
        sample_refcodes=sorted([c.stem for c in cifs[:10]]),
    )


def parse_qmof_metadata(json_path: Path) -> dict[str, object]:
    """Return the QMOF per-MOF metadata dict (just JSON load + light validation)."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return payload


__all__ = ["QmofUnpackResult", "parse_qmof_metadata", "unpack_qmof_zip"]
