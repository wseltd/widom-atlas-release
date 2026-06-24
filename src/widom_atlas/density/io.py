"""``.npz`` IO for :class:`DensityGrid` with explicit schema versioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np

from widom_atlas.core.models import DensityGrid

DENSITY_NPZ_SCHEMA_VERSION: Final[int] = 1


def save_density_npz(grid: DensityGrid, path: Path) -> None:
    """Persist a :class:`DensityGrid` to ``path`` as a single compressed ``.npz``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        grid=grid.grid,
        shape=np.asarray(grid.shape, dtype=np.int64),
        cell_A=grid.cell_A,
        spacing_A=np.asarray(grid.spacing_A, dtype=np.float64),
        temperature_K=np.asarray(grid.temperature_K, dtype=np.float64),
        gas=np.asarray(grid.gas),
        normalisation=np.asarray(grid.normalisation),
        smoothing_sigma_A=np.asarray(grid.smoothing_sigma_A, dtype=np.float64),
        n_source_samples=np.asarray(grid.n_source_samples, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(grid.metadata, sort_keys=True, separators=(",", ":"))),
        schema_version=np.asarray(DENSITY_NPZ_SCHEMA_VERSION, dtype=np.int64),
    )


def load_density_npz(path: Path) -> DensityGrid:
    """Load a :class:`DensityGrid` written by :func:`save_density_npz`."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as f:
        version = int(f["schema_version"])
        if version != DENSITY_NPZ_SCHEMA_VERSION:
            raise ValueError(
                f"density .npz schema version {version} does not match "
                f"current {DENSITY_NPZ_SCHEMA_VERSION}: {p}"
            )
        shape_arr = np.asarray(f["shape"]).tolist()
        spacing_arr = np.asarray(f["spacing_A"]).tolist()
        metadata = json.loads(str(f["metadata_json"]))
        return DensityGrid(
            grid=f["grid"],
            shape=(int(shape_arr[0]), int(shape_arr[1]), int(shape_arr[2])),
            cell_A=f["cell_A"],
            spacing_A=(float(spacing_arr[0]), float(spacing_arr[1]), float(spacing_arr[2])),
            temperature_K=float(f["temperature_K"]),
            gas=str(f["gas"]),
            normalisation=str(f["normalisation"]),  # type: ignore[arg-type]
            smoothing_sigma_A=float(f["smoothing_sigma_A"]),
            n_source_samples=int(f["n_source_samples"]),
            metadata=metadata,
        )
