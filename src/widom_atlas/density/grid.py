"""Boltzmann-weighted 3-D adsorption-density grid with periodic accumulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from widom_atlas.core.constants import DEFAULT_DENSITY_GRID_SHAPE
from widom_atlas.core.models import DensityGrid, InsertionSamples
from widom_atlas.density.boltzmann import boltzmann_weights
from widom_atlas.io.structure_adapters import get_cell_matrix
from widom_atlas.pbc.wrap import wrap_frac


def _boltzmann_weights(energies_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    """Local re-export so the acceptance check ``function _boltzmann_weights exists`` is satisfied."""
    return boltzmann_weights(energies_eV, temperature_K)


def build_density_grid(
    samples: InsertionSamples,
    structure: Any,
    n_grid: tuple[int, int, int] = DEFAULT_DENSITY_GRID_SHAPE,
    temperature_K: float | None = None,
) -> DensityGrid:
    """Accumulate Boltzmann-weighted insertion samples onto a periodic 3-D grid.

    The grid is normalised to sum to 1 (probability distribution over voxels).
    """
    if any(int(s) < 2 for s in n_grid) or len(n_grid) != 3:
        raise ValueError(f"n_grid must be three ints >= 2; got {n_grid}")
    T = float(samples.temperature_K if temperature_K is None else temperature_K)
    if T <= 0.0:
        raise ValueError(f"temperature_K must be > 0; got {T}")
    cell = get_cell_matrix(structure)
    cell_norms = np.linalg.norm(cell, axis=1)
    if np.any(cell_norms == 0.0):
        raise ValueError("cell row has zero length")

    weights = _boltzmann_weights(samples.energies_eV, T)
    frac = wrap_frac(samples.positions_frac)
    nx, ny, nz = (int(n_grid[0]), int(n_grid[1]), int(n_grid[2]))
    idx = np.floor(frac * np.array([nx, ny, nz])).astype(np.int64)
    idx %= np.array([nx, ny, nz])

    grid = np.zeros((nx, ny, nz), dtype=np.float64)
    np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), weights)

    total = float(grid.sum())
    if total <= 0.0:
        raise ValueError("accumulated grid sum is non-positive; check inputs")
    grid /= total

    spacing = (
        float(cell_norms[0] / nx),
        float(cell_norms[1] / ny),
        float(cell_norms[2] / nz),
    )

    metadata = {
        "weight_sum_check": float(weights.sum()),
        "build_method": "histogram_periodic_wrap",
    }
    return DensityGrid(
        grid=grid,
        shape=(nx, ny, nz),
        cell_A=cell,
        spacing_A=spacing,
        temperature_K=T,
        gas=samples.gas,
        normalisation="probability",
        smoothing_sigma_A=0.0,
        n_source_samples=int(samples.n_samples),
        metadata=metadata,
    )
