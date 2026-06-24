"""Density-map module: Boltzmann weighting, periodic accumulation, smoothing, npz IO."""

from widom_atlas.density.boltzmann import boltzmann_weights, log_boltzmann_weights
from widom_atlas.density.grid import build_density_grid
from widom_atlas.density.io import (
    DENSITY_NPZ_SCHEMA_VERSION,
    load_density_npz,
    save_density_npz,
)
from widom_atlas.density.smoothing import smooth_density

__all__ = [
    "DENSITY_NPZ_SCHEMA_VERSION",
    "boltzmann_weights",
    "build_density_grid",
    "load_density_npz",
    "log_boltzmann_weights",
    "save_density_npz",
    "smooth_density",
]
