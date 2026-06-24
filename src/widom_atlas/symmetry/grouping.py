"""High-level basin grouping: detect host symmetry then match basin centroids."""

from __future__ import annotations

from typing import Any

import numpy as np

from widom_atlas.core.constants import (
    DEFAULT_ANGLE_TOLERANCE_DEG,
    DEFAULT_BASIN_MATCH_TOL_A,
    DEFAULT_ENERGY_MATCH_TOL_KJMOL,
    DEFAULT_SYMPREC,
    EV_TO_KJMOL,
)
from widom_atlas.core.models import Basin, SymmetryGroup
from widom_atlas.io.structure_adapters import get_cell_matrix
from widom_atlas.pbc.minimum_image import min_image_distance
from widom_atlas.pbc.wrap import wrap_frac
from widom_atlas.symmetry.match import (
    _union_find_merge,
)
from widom_atlas.symmetry.spglib_ops import detect_symmetry
from widom_atlas.symmetry.types import FrameworkSymmetry


def _apply_operation_to_centroid(
    centroid_frac: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return wrap_frac(rotation @ np.asarray(centroid_frac, dtype=np.float64) + translation)


def _match_transformed_centroid(
    transformed_frac: np.ndarray,
    centroids: np.ndarray,
    energies_eV: np.ndarray,
    energy_i_eV: float,
    cell: np.ndarray,
    basin_match_tol_A: float,
    energy_match_tol_kJmol: float,
) -> list[int]:
    """Return indices of basin centroids that match ``transformed_frac`` within tolerance."""
    matches: list[int] = []
    for j in range(centroids.shape[0]):
        d = float(min_image_distance(transformed_frac, centroids[j], cell))
        if d > basin_match_tol_A:
            continue
        if abs(energies_eV[j] - energy_i_eV) * EV_TO_KJMOL > energy_match_tol_kJmol:
            continue
        matches.append(j)
    return matches


def _compute_group_confidence(
    framework: FrameworkSymmetry,
    pair_distances: list[float],
    basin_match_tol_A: float,
    multi_match_count: int,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    base = {"high": 1.0, "medium": 0.7, "low": 0.4, "uncertain": 0.2}.get(framework.confidence, 0.5)
    if not pair_distances:
        return base, reasons
    max_d = max(pair_distances)
    fraction = float(max_d / max(basin_match_tol_A, 1e-12))
    confidence = base * (1.0 - 0.5 * fraction)
    if framework.confidence in {"low", "uncertain"}:
        reasons.append("low_symmetry_host")
    if max_d > 0.8 * basin_match_tol_A:
        reasons.append("tolerance_ambiguous")
    if multi_match_count > 0:
        reasons.append("partial_match")
    return float(max(0.0, min(1.0, confidence))), reasons


def group_basins(
    structure: Any,
    basins: list[Basin],
    symprec: float = DEFAULT_SYMPREC,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE_DEG,
    basin_match_tol_A: float = DEFAULT_BASIN_MATCH_TOL_A,
    energy_match_tol_kJmol: float = DEFAULT_ENERGY_MATCH_TOL_KJMOL,
) -> list[SymmetryGroup]:
    """Detect host symmetry and return :class:`SymmetryGroup` records for the basins."""
    if not basins:
        return []
    framework = detect_symmetry(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    cell = get_cell_matrix(structure)

    n = len(basins)
    centroids = np.array([b.centroid_frac for b in basins], dtype=np.float64)
    energies_eV = np.array([b.mean_energy_eV for b in basins], dtype=np.float64)

    pair_distances_per_pair: list[float] = []
    multi_match_count = 0
    pairs: list[tuple[int, int]] = []
    for op_idx in range(framework.n_operations):
        R = np.asarray(framework.rotations[op_idx], dtype=np.float64)
        t = np.asarray(framework.translations[op_idx], dtype=np.float64)
        transformed = wrap_frac((centroids @ R.T) + t)
        for i in range(n):
            matches = _match_transformed_centroid(
                transformed[i], centroids, energies_eV, energies_eV[i], cell,
                basin_match_tol_A, energy_match_tol_kJmol,
            )
            if not matches:
                continue
            if len(matches) > 1:
                multi_match_count += 1
            for j in matches:
                if i == j:
                    continue
                pairs.append((min(i, j), max(i, j)))
                pair_distances_per_pair.append(
                    float(min_image_distance(transformed[i], centroids[j], cell))
                )

    groups_by_index = _union_find_merge(n, pairs)
    confidence, base_reasons = _compute_group_confidence(
        framework, pair_distances_per_pair, basin_match_tol_A, multi_match_count
    )

    out: list[SymmetryGroup] = []
    tolerances = {
        "symprec": float(symprec),
        "angle_tolerance_deg": float(angle_tolerance),
        "basin_match_tol_A": float(basin_match_tol_A),
        "energy_match_tol_kJmol": float(energy_match_tol_kJmol),
    }
    if framework.is_low_symmetry:
        confidence = min(confidence, 0.45)
    for gi, indices in enumerate(groups_by_index):
        member_ids = tuple(sorted(int(basins[k].basin_id) for k in indices))
        flags = list(base_reasons)
        if framework.is_low_symmetry and "low_symmetry_host" not in flags:
            flags.append("low_symmetry_host")
        if framework.confidence in {"low", "uncertain"} and "low_symmetry_host" not in flags:
            flags.append("low_symmetry_host")
        flags_tuple = tuple(flags)
        if confidence < 0.5 and not flags_tuple:
            flags_tuple = ("tolerance_ambiguous",)
        out.append(
            SymmetryGroup(
                group_id=gi,
                member_basin_ids=member_ids,
                space_group_symbol=framework.international_symbol,
                space_group_number=framework.space_group_number,
                n_operations_used=framework.n_operations,
                tolerances=tolerances,
                grouping_confidence=confidence,
                uncertainty_flags=flags_tuple,
                notes="grouped via spglib + minimum-image centroid matching",
            )
        )
    return out


__all__ = [
    "_apply_operation_to_centroid",
    "_compute_group_confidence",
    "_match_transformed_centroid",
    "group_basins",
]
