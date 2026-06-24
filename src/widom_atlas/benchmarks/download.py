"""Resolve :class:`BenchmarkMaterial` entries to local CIF paths with offline caching.

Network access is opt-in via ``allow_network=True``; tests run with the
default (False) and require pre-cached fixtures or skip the network path.
Excludes CSD-derived material licenses up front.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from widom_atlas.core.benchmark_models import BenchmarkMaterial


class BenchmarkDataUnavailable(RuntimeError):
    """Raised when a benchmark material cannot be resolved from the cache and network is disabled."""


_FORBIDDEN_LICENSE_PREFIXES = ("csd", "cambridge structural database")


def _cache_paths(cache_dir: Path, material: BenchmarkMaterial) -> tuple[Path, Path]:
    base = Path(cache_dir) / material.source
    base.mkdir(parents=True, exist_ok=True)
    cif = base / f"{material.material_id}.cif"
    meta = base / f"{material.material_id}.meta.json"
    return cif, meta


def _atomic_write_bytes(path: Path, data: bytes) -> None:
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


def _fetch_core_mof(material: BenchmarkMaterial, cif_path: Path) -> str:
    """Resolve the CIF for a CoRE-MOF entry into the cache.

    Uses the bundled ``CoRE_MOF`` Python package — no network IO required
    after install. ``material.source_identifier`` must be a CSD-derived
    refcode present in ``CoRE_MOF.list_structures(material.core_mof_dataset)``.

    Side-effect: writes a sibling ``<material_id>.meta.json`` carrying the
    resolved ``dataset_version`` so :func:`record_provenance` can stamp the
    real version in the per-material provenance file (REPAIR-3).
    """
    import importlib.util
    from importlib import metadata as _metadata

    if importlib.util.find_spec("CoRE_MOF") is None:
        raise BenchmarkDataUnavailable(
            f"CoRE-MOF package not installed; cannot fetch {material.material_id}"
        )
    if not material.source_identifier:
        raise BenchmarkDataUnavailable(
            f"core_mof material {material.material_id} missing source_identifier (CSD refcode)"
        )
    dataset = material.core_mof_dataset or "2019-ASR"
    import CoRE_MOF as _core_mof

    data = _core_mof.get_CIF_structure_data(dataset, material.source_identifier)
    _atomic_write_bytes(cif_path, data.encode("utf-8") if isinstance(data, str) else data)

    try:
        package_version = _metadata.version("CoRE-MOF")
    except _metadata.PackageNotFoundError:
        package_version = "unavailable: importlib.metadata.version('CoRE-MOF') not found"
    dataset_version = f"CoRE-MOF=={package_version}; dataset={dataset}"

    meta_path = cif_path.with_suffix("").with_suffix(".meta.json")
    meta = {
        "material_id": material.material_id,
        "source": material.source,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "source_identifier": material.source_identifier,
        "core_mof_package_version": package_version,
        "fetched_at": datetime.now(UTC).isoformat(),
        "license": material.license,
    }
    _atomic_write_bytes(meta_path, json.dumps(meta, sort_keys=True, indent=2).encode("utf-8"))
    return f"core_mof://{dataset}/{material.source_identifier}"


def _fetch_qmof(material: BenchmarkMaterial, cif_path: Path) -> str:
    raise BenchmarkDataUnavailable(
        f"QMOF Figshare downloader not implemented in v1 (pin offline cache for {material.material_id})"
    )


def fetch_benchmark_material(
    material: BenchmarkMaterial,
    cache_dir: Path,
    *,
    allow_network: bool = False,
    fixtures_dir: Path | None = None,
) -> Path:
    """Resolve a :class:`BenchmarkMaterial` to a local CIF path.

    Returns the cached file when present; otherwise dispatches to the
    source-specific fetcher (or to ``fixtures_dir`` for ``manual`` entries).
    Raises :class:`BenchmarkDataUnavailable` when offline and uncached.
    """
    if any(material.license.lower().startswith(p) for p in _FORBIDDEN_LICENSE_PREFIXES):
        raise BenchmarkDataUnavailable(
            f"license {material.license!r} is excluded from widom-atlas v1 redistribution"
        )

    cif_path, meta_path = _cache_paths(cache_dir, material)
    if cif_path.exists():
        return cif_path

    if material.source == "manual":
        if fixtures_dir is None:
            raise BenchmarkDataUnavailable(
                f"manual benchmark material {material.material_id} requires fixtures_dir"
            )
        src = Path(fixtures_dir) / f"{material.material_id}.cif"
        if not src.exists():
            raise BenchmarkDataUnavailable(
                f"fixture not found at {src}; commit a CC-BY-4.0 CIF or skip {material.material_id}"
            )
        shutil.copy2(src, cif_path)
        meta = {
            "material_id": material.material_id,
            "source": "manual_fixture",
            "fetched_at": datetime.now(UTC).isoformat(),
            "license": material.license,
        }
        _atomic_write_bytes(meta_path, json.dumps(meta, sort_keys=True, indent=2).encode("utf-8"))
        return cif_path

    # CoRE-MOF data is bundled with the `CoRE-MOF` Python package — local, no network.
    if material.source == "core_mof":
        url = _fetch_core_mof(material, cif_path)
    elif material.source == "qmof":
        if not allow_network:
            raise BenchmarkDataUnavailable(
                f"{material.material_id} not cached at {cif_path} and allow_network=False"
            )
        url = _fetch_qmof(material, cif_path)
    else:
        raise BenchmarkDataUnavailable(f"unknown source: {material.source!r}")

    # Merge with any meta.json the source-specific fetcher already wrote
    # (e.g. ``_fetch_core_mof`` records ``dataset_version`` per REPAIR-3).
    existing_meta: dict = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing_meta = {}
    merged = {
        "material_id": material.material_id,
        "source": material.source,
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(),
        "license": material.license,
        **{k: v for k, v in existing_meta.items() if k not in {"url", "fetched_at"}},
    }
    _atomic_write_bytes(meta_path, json.dumps(merged, sort_keys=True, indent=2).encode("utf-8"))
    return cif_path


__all__ = ["BenchmarkDataUnavailable", "_fetch_core_mof", "_fetch_qmof", "fetch_benchmark_material"]
