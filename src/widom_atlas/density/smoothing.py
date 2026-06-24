"""Periodic Gaussian smoothing for density grids (mode='wrap' enforced)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from widom_atlas.core.models import DensityGrid


def _sigma_angstrom_to_voxels(
    sigma_A: float, cell_A: np.ndarray, shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Convert an isotropic spatial sigma (Angstrom) to per-axis voxel widths."""
    norms = np.linalg.norm(cell_A, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("cell row has zero length")
    return (
        float(sigma_A * shape[0] / norms[0]),
        float(sigma_A * shape[1] / norms[1]),
        float(sigma_A * shape[2] / norms[2]),
    )


def smooth_density(grid: DensityGrid, sigma_A: float, mode: str = "wrap") -> DensityGrid:
    """Return a new :class:`DensityGrid` with a periodic Gaussian smoothing applied."""
    if mode != "wrap":
        raise ValueError("smooth_density only supports mode='wrap' to preserve PBC")
    if sigma_A < 0.0:
        raise ValueError(f"sigma_A must be >= 0; got {sigma_A}")
    sigma_voxels = _sigma_angstrom_to_voxels(sigma_A, grid.cell_A, grid.shape)
    smoothed = gaussian_filter(grid.grid, sigma=sigma_voxels, mode="wrap")
    smoothed = np.clip(smoothed, 0.0, None)
    total = float(smoothed.sum())
    if total <= 0.0:
        raise ValueError("smoothed grid sum is non-positive")
    smoothed = smoothed / total

    metadata = dict(grid.metadata)
    metadata["smoothing_sigma_A"] = float(sigma_A)
    return DensityGrid(
        grid=smoothed,
        shape=grid.shape,
        cell_A=grid.cell_A,
        spacing_A=grid.spacing_A,
        temperature_K=grid.temperature_K,
        gas=grid.gas,
        normalisation=grid.normalisation,
        smoothing_sigma_A=float(sigma_A),
        n_source_samples=grid.n_source_samples,
        metadata=metadata,
    )
