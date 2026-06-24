"""Run manifest construction + serialisation."""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import sys
from datetime import UTC, datetime
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any

from widom_atlas.core.models import RunManifest

_DEPENDENCIES_TO_RECORD = (
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
    "ase",
    "pymatgen",
    "spglib",
    "scikit-learn",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dep_version(name: str) -> str:
    try:
        return str(_metadata.version(name))
    except _metadata.PackageNotFoundError:
        return "unknown"


def build_manifest(
    *,
    structure_id: str,
    gas: str,
    temperature_K: float,
    sample_path: Path,
    structure_path: Path,
    parameters: dict[str, Any],
    source_dataset: dict[str, Any] | None = None,
    license_metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> RunManifest:
    """Capture environment + input hashes into a :class:`RunManifest`."""
    package_version = _dep_version("widom-atlas")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    deps = {name: _dep_version(name) for name in _DEPENDENCIES_TO_RECORD}
    structure_sha = _sha256_file(Path(structure_path))
    samples_sha = _sha256_file(Path(sample_path))
    rid = run_id or f"{structure_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    dataset_source = None
    dataset_license = None
    if source_dataset is not None:
        dataset_source = str(source_dataset.get("name") or source_dataset.get("source") or "")
        dataset_license = (
            str(source_dataset.get("license") or "")
            if "license" in source_dataset
            else None
        )
    if license_metadata is not None and dataset_license is None:
        dataset_license = str(license_metadata.get("license") or "")

    return RunManifest(
        run_id=rid,
        package_version=package_version,
        python_version=python_version,
        platform=_platform.platform(),
        dependency_versions=deps,
        structure_id=structure_id,
        structure_source=str(Path(structure_path)),
        structure_sha256=structure_sha,
        input_samples_sha256=samples_sha,
        gas=gas,  # type: ignore[arg-type]
        temperature_K=float(temperature_K),
        parameters=dict(parameters),
        dataset_source=dataset_source,
        dataset_license=dataset_license,
        output_paths={},
    )


def write_manifest(manifest: RunManifest, out_path: Path) -> None:
    """Serialise a :class:`RunManifest` to ``out_path`` as deterministic JSON."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    text = json.dumps(payload, sort_keys=True, indent=2, separators=(",", ": "))
    p.write_text(text + "\n", encoding="utf-8")


__all__ = ["_sha256_file", "build_manifest", "write_manifest"]
