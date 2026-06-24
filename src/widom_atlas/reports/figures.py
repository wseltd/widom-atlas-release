"""Matplotlib figure rendering (Agg backend, headless-safe)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from widom_atlas.core.models import Basin, DensityGrid, RobustnessMetrics


def _ensure_parent(path: Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_density_slices(
    density_grid: DensityGrid,
    out_path: Path,
    axis: str = "c",
    n_slices: int = 3,
) -> Path:
    """Render orthogonal slices of the normalised density grid."""
    axis_idx = {"a": 0, "b": 1, "c": 2}[axis]
    p = _ensure_parent(out_path)
    grid = np.asarray(density_grid.grid, dtype=np.float64)
    n_along = grid.shape[axis_idx]
    indices = np.linspace(0, n_along - 1, n_slices, dtype=int)
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    for k, slice_idx in enumerate(indices):
        if axis_idx == 0:
            sl = grid[slice_idx, :, :]
            xlabel, ylabel = "frac b", "frac c"
        elif axis_idx == 1:
            sl = grid[:, slice_idx, :]
            xlabel, ylabel = "frac a", "frac c"
        else:
            sl = grid[:, :, slice_idx]
            xlabel, ylabel = "frac a", "frac b"
        im = axes[k].imshow(
            sl.T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="auto",
            cmap="viridis",
        )
        axes[k].set_xlabel(xlabel)
        axes[k].set_ylabel(ylabel)
        axes[k].set_title(f"slice {axis}={slice_idx}/{n_along}")
        fig.colorbar(im, ax=axes[k], shrink=0.8)
    fig.suptitle(f"Density slices along {axis}-axis (gas={density_grid.gas}, T={density_grid.temperature_K} K)")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_basin_centroids(
    basins: list[Basin],
    structure: Any,
    out_path: Path,
    projection: str = "ab",
) -> Path:
    """Scatter plot of basin centroids projected onto a fractional plane."""
    p = _ensure_parent(out_path)
    axes_lookup = {"ab": (0, 1), "ac": (0, 2), "bc": (1, 2)}
    if projection not in axes_lookup:
        raise ValueError(f"projection must be one of {sorted(axes_lookup)}; got {projection!r}")
    ix, iy = axes_lookup[projection]
    fig, ax = plt.subplots(figsize=(6, 6))
    if basins:
        x = np.array([b.centroid_frac[ix] for b in basins])
        y = np.array([b.centroid_frac[iy] for b in basins])
        sizes = np.array([200.0 * float(b.weight) + 40.0 for b in basins])
        ax.scatter(x, y, s=sizes, alpha=0.7, edgecolors="k")
        for b, xi, yi in zip(basins, x, y, strict=False):
            ax.annotate(str(b.basin_id), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"frac {projection[0]}")
    ax.set_ylabel(f"frac {projection[1]}")
    ax.set_title("Adsorption basin centroids (size ∝ weight)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_robustness_bar(metrics: RobustnessMetrics | dict, out_path: Path) -> Path:
    """Bar chart of selected robustness scalars."""
    p = _ensure_parent(out_path)
    m = metrics.model_dump() if isinstance(metrics, RobustnessMetrics) else dict(metrics)
    labels: list[str] = []
    values: list[float] = []
    for key in ("delta_ln_KH", "delta_Qads_kJmol", "basin_persistence_fraction", "mean_basin_displacement_A", "accessibility_change"):
        v = m.get(key)
        if v is None:
            continue
        labels.append(key)
        values.append(float(v))
    fig, ax = plt.subplots(figsize=(6, 4))
    if labels:
        ax.bar(labels, values, color="steelblue")
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Robustness metrics")
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


__all__ = [
    "plot_basin_centroids",
    "plot_density_slices",
    "plot_robustness_bar",
]
