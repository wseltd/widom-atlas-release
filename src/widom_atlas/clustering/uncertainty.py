"""Annotate :class:`Basin` records with uncertainty fields.

Computes:

- ``accessible_fraction`` (already populated by :func:`extract_basins`, kept consistent here);
- ``energy_stderr_eV`` via Kish-effective-sample-size correction;
- ``centroid_stderr_A`` via fixed-seed weighted bootstrap;
- ``weight_stderr`` via Poisson approximation;
- ``low_count_flag`` (True when sample count < 10).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from widom_atlas.core.models import Basin, InsertionSamples
from widom_atlas.density.boltzmann import boltzmann_weights
from widom_atlas.io.structure_adapters import get_cell_matrix
from widom_atlas.pbc.minimum_image import min_image_distance
from widom_atlas.pbc.wrap import wrap_frac

_BOOTSTRAP_N = 200
_BOOTSTRAP_SEED = 12345
_LOW_COUNT_THRESHOLD = 10


def _effective_sample_size_kish(weights: np.ndarray) -> float:
    """Kish effective sample size: ``(sum w)^2 / sum(w^2)``."""
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return s * s / s2


def _bootstrap_centroid_stderr(
    member_frac: np.ndarray, member_w: np.ndarray, cell: np.ndarray, seed: int = _BOOTSTRAP_SEED
) -> float:
    n = member_frac.shape[0]
    if n < 2:
        return 0.0
    w = member_w / float(member_w.sum())
    rng = np.random.default_rng(seed)
    centroids = np.empty((_BOOTSTRAP_N, 3), dtype=np.float64)
    for k in range(_BOOTSTRAP_N):
        idx = rng.choice(n, size=n, replace=True, p=w)
        sample = member_frac[idx]
        sample_w = member_w[idx]
        sw = float(sample_w.sum())
        if sw <= 0.0:
            centroids[k] = np.nan
            continue
        theta = 2.0 * np.pi * sample
        cs = np.sum(sample_w[:, None] * np.cos(theta), axis=0)
        ss = np.sum(sample_w[:, None] * np.sin(theta), axis=0)
        mean_theta = np.arctan2(ss, cs)
        centroids[k] = (mean_theta / (2.0 * np.pi)) % 1.0
    valid = centroids[~np.any(np.isnan(centroids), axis=1)]
    if valid.shape[0] < 2:
        return 0.0
    base = np.mean(valid, axis=0)
    base_b = np.broadcast_to(base, valid.shape)
    d = min_image_distance(valid, base_b, cell)
    return float(np.std(d))


def annotate_basin_uncertainty(
    basins: list[Basin], samples: InsertionSamples, structure: Any
) -> list[Basin]:
    """Return a new list of :class:`Basin` with uncertainty fields populated."""
    if not basins:
        return []
    cell = get_cell_matrix(structure)
    frac_all = wrap_frac(samples.positions_frac)
    energies = np.asarray(samples.energies_eV, dtype=np.float64)
    accessible = np.asarray(samples.accessible, dtype=bool)
    weights = boltzmann_weights(energies, samples.temperature_K)
    n_total = frac_all.shape[0]

    out: list[Basin] = []
    for b in basins:
        cf = np.asarray(b.centroid_frac, dtype=np.float64)
        d = min_image_distance(frac_all, np.broadcast_to(cf, frac_all.shape), cell)
        # Recover cluster membership by proximity (within 2x basin spread or 1 A, whichever larger)
        radius = max(2.0 * b.spread_A, 1.0)
        member_mask = d <= radius
        if int(member_mask.sum()) == 0:
            out.append(b)
            continue
        member_frac = frac_all[member_mask]
        member_e = energies[member_mask]
        member_w = weights[member_mask]
        member_acc = accessible[member_mask]

        ess = _effective_sample_size_kish(member_w)
        w_norm = member_w / float(member_w.sum())
        var_e = float(np.sum(w_norm * (member_e - b.mean_energy_eV) ** 2))
        std_e = float(np.sqrt(max(0.0, var_e)))
        energy_stderr = std_e / np.sqrt(max(1.0, ess))

        centroid_se = _bootstrap_centroid_stderr(member_frac, member_w, cell)

        weight = float(b.weight)
        weight_se = float(np.sqrt(max(0.0, weight * (1.0 - weight)) / max(1, n_total)))

        accessible_fraction = float(np.sum(w_norm * member_acc.astype(np.float64)))

        out.append(
            b.model_copy(
                update={
                    "accessible_fraction": accessible_fraction,
                    "energy_stderr_eV": float(energy_stderr),
                    "centroid_stderr_A": float(centroid_se),
                    "weight_stderr": weight_se,
                    "low_count_flag": bool(b.count < _LOW_COUNT_THRESHOLD),
                }
            )
        )
    return out


__all__ = [
    "_bootstrap_centroid_stderr",
    "_effective_sample_size_kish",
    "annotate_basin_uncertainty",
]
