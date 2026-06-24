"""Minimum-image displacement and distance with a 27-image safe fallback for skewed cells."""

from __future__ import annotations

import numpy as np

from widom_atlas.pbc.wrap import _validate_cell

_SKEW_THRESHOLD = 0.5  # max(|off-diag|/min(|diag|)) above which we use 27-image fallback


def _is_skewed(cell: np.ndarray) -> bool:
    diag = np.abs(np.diag(cell))
    if np.any(diag < 1e-12):
        return True
    off = np.abs(cell - np.diag(np.diag(cell)))
    return bool(np.max(off) / np.min(diag) > _SKEW_THRESHOLD)


def _round_image_displacement(
    frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    delta_frac = frac_a - frac_b
    delta_frac -= np.round(delta_frac)
    return delta_frac @ cell


def _safe_image_displacement(
    frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    delta_frac = frac_a - frac_b
    delta_frac -= np.round(delta_frac)
    offsets = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=np.float64,
    )
    candidate_frac = delta_frac[..., None, :] + offsets  # (..., 27, 3)
    candidate_cart = candidate_frac @ cell  # (..., 27, 3)
    sq = np.sum(candidate_cart * candidate_cart, axis=-1)  # (..., 27)
    best = np.argmin(sq, axis=-1)  # (...)
    return np.take_along_axis(candidate_cart, best[..., None, None], axis=-2).squeeze(-2)


def min_image_displacement(
    frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Cartesian displacement (a - b) under the minimum-image convention."""
    cell_arr = _validate_cell(cell)
    a = np.asarray(frac_a, dtype=np.float64)
    b = np.asarray(frac_b, dtype=np.float64)
    if a.shape != b.shape:
        try:
            a, b = np.broadcast_arrays(a, b)
        except ValueError as exc:
            raise ValueError(f"frac_a {a.shape} and frac_b {b.shape} not broadcast-compatible") from exc
    if _is_skewed(cell_arr):
        return _safe_image_displacement(a, b, cell_arr)
    return _round_image_displacement(a, b, cell_arr)


def min_image_distance(
    frac_a: np.ndarray, frac_b: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Minimum-image distance (Angstrom)."""
    disp = min_image_displacement(frac_a, frac_b, cell)
    return np.linalg.norm(disp, axis=-1)
