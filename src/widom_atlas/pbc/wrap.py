"""Fractional wrapping and Cartesian↔fractional conversion (cell rows = a, b, c)."""

from __future__ import annotations

import numpy as np

_DET_TOL_A3 = 1e-9


def _validate_cell(cell: np.ndarray) -> np.ndarray:
    arr = np.asarray(cell, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"cell must have shape (3,3); got {arr.shape}")
    det = float(np.linalg.det(arr))
    if abs(det) < _DET_TOL_A3:
        raise ValueError(f"cell is singular or near-degenerate: |det|={abs(det):g}")
    return arr


def wrap_frac(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into ``[0, 1)``.

    Idempotent under repeated application; handles negative inputs and
    floating-point residues that would otherwise produce ``1.0`` exactly.
    """
    arr = np.asarray(frac, dtype=np.float64)
    out = np.mod(arr, 1.0)
    out = np.where(out >= 1.0, 0.0, out)
    out = np.where(out < 0.0, out + 1.0, out)
    return np.where(out >= 1.0, 0.0, out)


def cart_to_frac(cart: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert Cartesian → fractional. Stable for triclinic cells via ``np.linalg.solve``."""
    cell_arr = _validate_cell(cell)
    cart_arr = np.asarray(cart, dtype=np.float64)
    if cart_arr.ndim == 1:
        if cart_arr.shape != (3,):
            raise ValueError(f"cart must have shape (3,) or (N,3); got {cart_arr.shape}")
        return np.linalg.solve(cell_arr.T, cart_arr)
    if cart_arr.ndim != 2 or cart_arr.shape[1] != 3:
        raise ValueError(f"cart must have shape (3,) or (N,3); got {cart_arr.shape}")
    return np.linalg.solve(cell_arr.T, cart_arr.T).T


def frac_to_cart(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert fractional → Cartesian: ``frac @ cell``."""
    cell_arr = _validate_cell(cell)
    frac_arr = np.asarray(frac, dtype=np.float64)
    if frac_arr.ndim == 1:
        if frac_arr.shape != (3,):
            raise ValueError(f"frac must have shape (3,) or (N,3); got {frac_arr.shape}")
        return frac_arr @ cell_arr
    if frac_arr.ndim != 2 or frac_arr.shape[1] != 3:
        raise ValueError(f"frac must have shape (3,) or (N,3); got {frac_arr.shape}")
    return frac_arr @ cell_arr
