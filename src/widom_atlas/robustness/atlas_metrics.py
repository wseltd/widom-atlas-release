"""Atlas-level robustness metrics: basin persistence, splitting, displacement."""

from __future__ import annotations

from typing import Any

import numpy as np

from widom_atlas.core.constants import DEFAULT_BASIN_MATCH_TOL_A
from widom_atlas.core.models import Basin
from widom_atlas.pbc.minimum_image import min_image_distance


def _match_basins_pbc(
    pristine: list[Basin],
    perturbed: list[Basin],
    cell: np.ndarray,
    match_tol_A: float,
) -> tuple[dict[int, int], list[float], list[list[int]]]:
    """Greedy nearest-centroid matching of perturbed basins to pristine basins.

    Returns:
        match_map: perturbed_basin_id -> pristine_basin_id (or -1 if unmatched)
        displacements_A: list of min-image distances for matched pairs
        per_pristine_matches: list-of-list of perturbed_basin_ids matched to each pristine basin
    """
    if not pristine or not perturbed:
        return ({pb.basin_id: -1 for pb in perturbed}, [], [[] for _ in pristine])

    pristine_centroids = np.array([p.centroid_frac for p in pristine], dtype=np.float64)
    match_map: dict[int, int] = {}
    displacements: list[float] = []
    per_pristine: list[list[int]] = [[] for _ in pristine]

    for pb in perturbed:
        cf = np.asarray(pb.centroid_frac, dtype=np.float64)
        dists = np.array(
            [float(min_image_distance(cf, pc, cell)) for pc in pristine_centroids],
            dtype=np.float64,
        )
        j = int(np.argmin(dists))
        if dists[j] <= match_tol_A:
            match_map[pb.basin_id] = pristine[j].basin_id
            displacements.append(float(dists[j]))
            per_pristine[j].append(pb.basin_id)
        else:
            match_map[pb.basin_id] = -1
    return match_map, displacements, per_pristine


def compute_atlas_metrics(
    pristine_basins: list[Basin],
    perturbed_basins: list[Basin],
    cell: np.ndarray,
    match_tol_A: float = DEFAULT_BASIN_MATCH_TOL_A,
) -> dict[str, Any]:
    """Atlas-level robustness metrics for one pristine vs one perturbed basin set."""
    cell_arr = np.asarray(cell, dtype=np.float64)
    match_map, displacements, per_pristine = _match_basins_pbc(
        pristine_basins, perturbed_basins, cell_arr, match_tol_A
    )

    n_pri = len(pristine_basins)
    n_per = len(perturbed_basins)
    matched_pristine = {pid for pid in match_map.values() if pid >= 0}
    persistence = float(len(matched_pristine) / n_pri) if n_pri else 0.0
    splitting = sum(max(0, len(group) - 1) for group in per_pristine)

    mean_disp = float(np.mean(displacements)) if displacements else 0.0
    pri_acc = float(np.mean([b.accessible_fraction for b in pristine_basins])) if pristine_basins else 0.0
    per_acc = float(np.mean([b.accessible_fraction for b in perturbed_basins])) if perturbed_basins else 0.0
    accessibility_change = per_acc - pri_acc

    ambiguity_flags: list[str] = []
    for j, group in enumerate(per_pristine):
        if len(group) > 1:
            ambiguity_flags.append(f"multiple_perturbed_within_tol:basin_{pristine_basins[j].basin_id}")

    missing_data_flags: list[str] = []
    if not pristine_basins:
        missing_data_flags.append("pristine_basins_empty")
    if not perturbed_basins:
        missing_data_flags.append("perturbed_basins_empty")

    return {
        "basin_count_pristine": n_pri,
        "basin_count_perturbed": n_per,
        "basin_count_change": n_per - n_pri,
        "basin_persistence_fraction": persistence,
        "basin_splitting_count": int(splitting),
        "mean_basin_displacement_A": mean_disp,
        "accessibility_change": accessibility_change,
        "ambiguity_flags": ambiguity_flags,
        "missing_data_flags": missing_data_flags,
    }


__all__ = ["_match_basins_pbc", "compute_atlas_metrics"]
