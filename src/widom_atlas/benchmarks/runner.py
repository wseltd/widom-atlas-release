"""Run the widom-atlas pipeline over a benchmark set with caching + provenance.

Synthetic Lennard-Jones samples are generated on the fly when no real
Widom samples are supplied — these are explicitly tagged ``synthetic_toy_lj``
in the run metadata so downstream readers cannot mistake them for real
adsorption data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any

import numpy as np

from widom_atlas.backends import AtlasBackend, BackendName, get_backend
from widom_atlas.benchmarks.download import (
    BenchmarkDataUnavailable,
    fetch_benchmark_material,
)
from widom_atlas.benchmarks.hashing import record_provenance
from widom_atlas.benchmarks.registry import get_benchmark_set
from widom_atlas.core.benchmark_models import BenchmarkMaterial
from widom_atlas.core.pipeline import PipelineParams, run_atlas
from widom_atlas.io.from_arrays import from_arrays

_LOGGER = logging.getLogger(__name__)


def _dep_version(name: str) -> str:
    try:
        return str(_metadata.version(name))
    except _metadata.PackageNotFoundError:
        return "unknown"


def _pipeline_params_hash(params: PipelineParams) -> str:
    payload = json.dumps(params.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_benchmark_set(set_name: str) -> tuple[BenchmarkMaterial, ...]:
    return get_benchmark_set(set_name)


def _toy_lj_samples(
    cell: np.ndarray,
    atoms_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    n_samples: int,
    seed: int,
    epsilon_eV: float = 0.005,
    sigma_A: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Toy Lennard-Jones insertion samples — NOT chemically meaningful (verdict §13.J)."""
    rng = np.random.default_rng(seed)
    frac = rng.random((n_samples, 3))
    cart = frac @ cell
    inv_cell = np.linalg.inv(cell)
    energies = np.empty(n_samples, dtype=np.float64)
    accessible = np.ones(n_samples, dtype=bool)
    for i in range(n_samples):
        r_cart = cart[i]
        delta = atoms_positions - r_cart
        delta_frac = delta @ inv_cell.T
        delta_frac -= np.round(delta_frac)
        delta_cart = delta_frac @ cell
        r2 = np.sum(delta_cart * delta_cart, axis=-1)
        r2 = np.maximum(r2, (0.5 * sigma_A) ** 2)  # cap to avoid singularity
        sr6 = (sigma_A * sigma_A / r2) ** 3
        sr12 = sr6 * sr6
        e_per_pair = 4.0 * epsilon_eV * (sr12 - sr6)
        energies[i] = float(e_per_pair.sum())
        if energies[i] > 5.0:
            accessible[i] = False
    return frac, energies, accessible


def _load_atoms_from_cif(cif_path: Path) -> Any:
    from ase.io import read

    atoms = read(str(cif_path))
    if hasattr(atoms, "set_pbc"):
        atoms.set_pbc(True)
    return atoms


def _run_one_material(
    material: BenchmarkMaterial,
    gas: str,
    structures_dir: Path,
    samples_dir: Path | None,
    output_dir: Path,
    cache_dir: Path,
    download: bool,
    temperature_K: float,
    n_samples: int,
    seed: int,
    params: PipelineParams,
    samples_kind: str = "widom",
    backend: AtlasBackend | None = None,
) -> dict[str, Any]:
    out = output_dir / material.material_id
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "material_id": material.material_id,
        "source": material.source,
        "license": material.license,
        "gas": gas,
        "temperature_K": float(temperature_K),
        "status": "pending",
        "error_class": None,
        "error_message": None,
    }
    try:
        cif_path = fetch_benchmark_material(
            material,
            cache_dir=cache_dir,
            allow_network=download,
            fixtures_dir=structures_dir,
        )
    except BenchmarkDataUnavailable as exc:
        record["status"] = "skipped"
        record["error_class"] = type(exc).__name__
        record["error_message"] = str(exc)
        return record

    try:
        provenance = record_provenance(material, cif_path, cache_dir)
        record["cif_sha256"] = provenance.sha256
        record["cif_path"] = str(cif_path)
        record["dataset_version"] = provenance.dataset_version
        record["citation_doi"] = provenance.citation_doi
        record["license"] = provenance.license_tag

        atoms = _load_atoms_from_cif(cif_path)
        atlas_input = _generate_atlas_input(
            material=material,
            atoms=atoms,
            gas=gas,
            temperature_K=temperature_K,
            n_samples=n_samples,
            seed=seed,
            samples_kind=samples_kind,
            backend=backend,
        )
        record["samples_origin"] = atlas_input.metadata.get("samples_origin", "unknown")
        record["backend"] = atlas_input.metadata.get("backend", "unknown")
        record["calculator"] = atlas_input.metadata.get("calculator", "unknown")
        scalars = atlas_input.metadata.get("widom_scalars")
        if isinstance(scalars, dict):
            record["henry_coefficient"] = scalars.get("henry_coefficient")
            record["heat_of_adsorption_kJmol"] = scalars.get("heat_of_adsorption_kJmol")

        result = run_atlas(atlas_input, params, out, structure=atoms)
        record["status"] = "ok"
        record["basins_count"] = len(result.basins)
        record["manifest_path"] = str((out / "manifest.json").resolve())
    except Exception as exc:
        record["status"] = "failed"
        record["error_class"] = type(exc).__name__
        record["error_message"] = str(exc)
    return record


def _generate_atlas_input(
    *,
    material: BenchmarkMaterial,
    atoms: Any,
    gas: str,
    temperature_K: float,
    n_samples: int,
    seed: int,
    samples_kind: str = "widom",
    backend: AtlasBackend | None = None,
) -> Any:
    """Generate insertion samples for one benchmark material.

    ``samples_kind="widom"`` (default) routes through the chosen
    :class:`~widom_atlas.backends.AtlasBackend` (defaults to the
    parameterised UFF+TraPPE backend). When the ``widom`` extra is not
    installed, a fast inline toy-LJ approximation is used so unit tests
    can run offline.
    """
    import importlib.util

    use_widom = samples_kind == "widom" and importlib.util.find_spec("widom") is not None
    if use_widom:
        if backend is None:
            backend = get_backend("parameterised_lj")
        try:
            backend_output = backend.generate(
                structure=atoms,
                gas=gas,
                temperature_K=float(temperature_K),
                n_samples=int(n_samples),
                seed=int(seed),
                material_id=material.material_id,
                material_source=material.source,
            )
            return backend_output.atlas_input
        except Exception as exc:
            _LOGGER.warning(
                "backend %r failed for %s (%s); falling back to synthetic_toy_lj",
                getattr(backend, "name", "?"),
                material.material_id,
                exc,
            )

    cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    frac, energies, accessible = _toy_lj_samples(cell, positions, numbers, n_samples=n_samples, seed=seed)
    return from_arrays(
        structure=atoms,
        positions_frac=frac,
        energies_eV=energies,
        accessible=accessible,
        temperature_K=float(temperature_K),
        gas=gas,
        metadata={
            "samples_origin": "synthetic_toy_lj",
            "backend": "synthetic_toy_lj",
            "calculator": "inline_toy_lj_offline_fallback",
            "warning": "toy LJ output — not chemically meaningful per implementation-verdict.txt §13.J",
            "benchmark_material_id": material.material_id,
            "benchmark_source": material.source,
        },
    )


@dataclass
class _BenchmarkSummary:
    """Lightweight container returned by :func:`run_benchmark_set` for inspection."""

    run_id: str
    set_name: str
    gas: str
    temperature_K: float
    materials: list[dict[str, Any]]
    benchmark_run_path: Path


def run_benchmark_set(
    set_name: str,
    gas: str,
    structures_dir: Path,
    samples_dir: Path | None,
    output_dir: Path,
    cache_dir: Path,
    download: bool = False,
    temperature_K: float = 298.15,
    n_samples: int = 500,
    seed: int = 0,
    params: PipelineParams | None = None,
    samples_kind: str = "widom",
    backend_name: BackendName = "parameterised_lj",
    external_samples_path: Path | None = None,
    external_manifest_path: Path | None = None,
    user_parameter_file: Path | None = None,
    allow_neutral_fallback: bool = False,
) -> _BenchmarkSummary:
    """Run the atlas pipeline over every material in ``set_name``.

    A failed material is recorded with ``status='failed'`` plus its error
    class and message; the rest of the run continues. The aggregate
    ``benchmark_run.json`` written under ``output_dir`` always reflects
    every attempted material (succeeded, skipped, or failed).
    """
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    structures_dir = Path(structures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    materials = _load_benchmark_set(set_name)
    if params is None:
        params = PipelineParams(n_grid=(16, 16, 16), dbscan_eps_A=2.0, min_samples=8)

    started = datetime.now(UTC)
    backend = get_backend(
        backend_name,
        external_samples_path=external_samples_path,
        external_manifest_path=external_manifest_path,
        user_parameter_file=user_parameter_file,
        allow_neutral_fallback=allow_neutral_fallback,
    )
    records: list[dict[str, Any]] = []
    for material in materials:
        records.append(
            _run_one_material(
                material,
                gas=gas,
                structures_dir=structures_dir,
                samples_dir=samples_dir,
                output_dir=output_dir,
                cache_dir=cache_dir,
                download=download,
                temperature_K=temperature_K,
                n_samples=n_samples,
                seed=seed,
                params=params,
                backend=backend,
            )
        )
    finished = datetime.now(UTC)

    run_id = f"{set_name}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    aggregate = {
        "run_id": run_id,
        "set_name": set_name,
        "gas": gas,
        "temperature_K": float(temperature_K),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "package_version": _dep_version("widom-atlas"),
        "dependency_versions": {
            name: _dep_version(name)
            for name in ("numpy", "scipy", "pandas", "pydantic", "ase", "pymatgen", "spglib", "scikit-learn", "widom", "CoRE-MOF")
        },
        "params_hash": _pipeline_params_hash(params),
        "params": params.model_dump(mode="json"),
        "samples_kind": samples_kind,
        "backend_name": backend_name,
        "backend_category": backend_name,
        "external_samples_path": str(external_samples_path) if external_samples_path else None,
        "external_manifest_path": str(external_manifest_path) if external_manifest_path else None,
        "user_parameter_file": str(user_parameter_file) if user_parameter_file else None,
        "allow_neutral_fallback": bool(allow_neutral_fallback),
        "random_seed": int(seed),
        "n_samples": int(n_samples),
        "materials": records,
    }
    benchmark_run_path = output_dir / "benchmark_run.json"
    benchmark_run_path.write_text(
        json.dumps(aggregate, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return _BenchmarkSummary(
        run_id=run_id,
        set_name=set_name,
        gas=gas,
        temperature_K=float(temperature_K),
        materials=records,
        benchmark_run_path=benchmark_run_path,
    )


__all__ = [
    "_load_benchmark_set",
    "_pipeline_params_hash",
    "_run_one_material",
    "run_benchmark_set",
]
