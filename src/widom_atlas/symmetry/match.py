"""Match symmetry-equivalent basin centroids and group via union-find."""

from __future__ import annotations

import numpy as np

from widom_atlas.core.constants import (
    DEFAULT_BASIN_MATCH_TOL_A,
    DEFAULT_ENERGY_MATCH_TOL_KJMOL,
    EV_TO_KJMOL,
)
from widom_atlas.core.models import Basin
from widom_atlas.pbc.minimum_image import min_image_distance
from widom_atlas.pbc.wrap import wrap_frac
from widom_atlas.symmetry.types import FrameworkSymmetry


def _apply_symmetry_op_to_frac(
    frac: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply ``frac' = R @ frac + t``, returning a wrapped fractional vector."""
    out = rotation @ np.asarray(frac, dtype=np.float64) + np.asarray(translation, dtype=np.float64)
    return wrap_frac(out)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _union_find_merge(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Return groups of indices in ``[0, n)`` merged by ``pairs``."""
    uf = _UnionFind(n)
    for a, b in pairs:
        uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)
    return [sorted(g) for g in groups.values()]


def group_equivalent_basins(
    basins: list[Basin],
    symmetry_group: FrameworkSymmetry,
    cell: np.ndarray,
    basin_match_tol_A: float = DEFAULT_BASIN_MATCH_TOL_A,
    energy_match_tol_kJmol: float = DEFAULT_ENERGY_MATCH_TOL_KJMOL,
) -> list[list[int]]:
    """Return ``list[list[basin_id]]`` of basins related by host symmetry."""
    n = len(basins)
    if n == 0:
        return []
    cell_arr = np.asarray(cell, dtype=np.float64)
    centroids = np.array([b.centroid_frac for b in basins], dtype=np.float64)
    energies_eV = np.array([b.mean_energy_eV for b in basins], dtype=np.float64)

    pairs: list[tuple[int, int]] = []
    for op_idx in range(symmetry_group.n_operations):
        R = np.asarray(symmetry_group.rotations[op_idx], dtype=np.float64)
        t = np.asarray(symmetry_group.translations[op_idx], dtype=np.float64)
        transformed = wrap_frac((centroids @ R.T) + t)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = float(min_image_distance(transformed[i], centroids[j], cell_arr))
                if d > basin_match_tol_A:
                    continue
                de_kjmol = abs(energies_eV[i] - energies_eV[j]) * EV_TO_KJMOL
                if de_kjmol > energy_match_tol_kJmol:
                    continue
                a, b = (basins[i].basin_id, basins[j].basin_id)
                pairs.append((min(a, b), max(a, b)))

    id_to_index = {b.basin_id: i for i, b in enumerate(basins)}
    index_pairs = [(id_to_index[a], id_to_index[b]) for (a, b) in pairs if a in id_to_index and b in id_to_index]
    groups_by_index = _union_find_merge(n, index_pairs)
    return [sorted(basins[i].basin_id for i in grp) for grp in groups_by_index]
