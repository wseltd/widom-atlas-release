"""Extract :class:`Basin` records from PBC-aware DBSCAN cluster labels."""

from __future__ import annotations

from typing import Any

import numpy as np

from widom_atlas.clustering.pbc_dbscan import pbc_dbscan
from widom_atlas.core.models import Basin, InsertionSamples
from widom_atlas.density.boltzmann import boltzmann_weights
from widom_atlas.io.structure_adapters import get_cell_matrix
from widom_atlas.pbc.minimum_image import min_image_distance
from widom_atlas.pbc.wrap import wrap_frac


def _circular_mean_frac(frac_vals: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean on the unit circle for one fractional axis."""
    theta = 2.0 * np.pi * frac_vals
    w = np.asarray(weights, dtype=np.float64)
    sx = float(np.sum(w * np.sin(theta)))
    cx = float(np.sum(w * np.cos(theta)))
    if sx == 0.0 and cx == 0.0:
        return 0.0
    mean_theta = np.arctan2(sx, cx)
    out = (mean_theta / (2.0 * np.pi)) % 1.0
    if out >= 1.0:
        out = 0.0
    return float(out)


def extract_basins(
    samples: InsertionSamples,
    structure: Any,
    eps_A: float,
    min_samples: int,
    temperature_K: float | None = None,
) -> list[Basin]:
    """Return a list of :class:`Basin` records, one per non-noise DBSCAN cluster."""
    cell = get_cell_matrix(structure)
    T = float(samples.temperature_K if temperature_K is None else temperature_K)
    frac = wrap_frac(samples.positions_frac)
    energies = np.asarray(samples.energies_eV, dtype=np.float64)
    accessible = np.asarray(samples.accessible, dtype=bool)
    n = frac.shape[0]
    if n == 0:
        return []

    weights = boltzmann_weights(energies, T)
    labels = pbc_dbscan(frac, cell, eps_A=eps_A, min_samples=min_samples, weights=weights)
    unique_labels = sorted(set(int(x) for x in labels) - {-1})

    basins: list[Basin] = []
    for new_id, lab in enumerate(unique_labels):
        mask = labels == lab
        member_frac = frac[mask]
        member_w = weights[mask]
        member_e = energies[mask]
        member_acc = accessible[mask]

        w_sum = float(member_w.sum())
        if w_sum <= 0.0:
            continue
        w_norm = member_w / w_sum

        centroid_frac = (
            _circular_mean_frac(member_frac[:, 0], member_w),
            _circular_mean_frac(member_frac[:, 1], member_w),
            _circular_mean_frac(member_frac[:, 2], member_w),
        )
        cf_arr = np.asarray(centroid_frac, dtype=np.float64)
        centroid_cart = (cf_arr @ cell).tolist()

        mean_e = float(np.sum(w_norm * member_e))
        var_e = float(np.sum(w_norm * (member_e - mean_e) ** 2))
        std_e = float(np.sqrt(max(0.0, var_e)))

        # Spread = RMS minimum-image distance from centroid (Cartesian, weighted)
        cf_broadcast = np.broadcast_to(cf_arr, member_frac.shape)
        d = min_image_distance(member_frac, cf_broadcast, cell)
        rms = float(np.sqrt(np.sum(w_norm * d * d))) if d.size else 0.0

        accessible_fraction = float(np.sum(w_norm * member_acc.astype(np.float64)))
        # Clamp to [0, 1]; weighted sums can drift by O(eps) when w_norm ≈ 1.
        accessible_fraction = max(0.0, min(1.0, accessible_fraction))

        basins.append(
            Basin(
                basin_id=int(new_id),
                count=int(mask.sum()),
                weight=float(min(1.0, max(0.0, w_sum))),  # clamped to [0,1] against fp drift
                centroid_frac=centroid_frac,
                centroid_cart_A=(float(centroid_cart[0]), float(centroid_cart[1]), float(centroid_cart[2])),
                mean_energy_eV=mean_e,
                std_energy_eV=std_e,
                accessible_fraction=accessible_fraction,
                spread_A=rms,
            )
        )
    return basins


__all__ = ["_circular_mean_frac", "extract_basins"]
