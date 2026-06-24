"""``from_arrays`` — the stable adapter-first constructor for :class:`AtlasInput`.

The public widom-atlas API is intentionally array-based, not bound to any
particular Widom result-object schema. Callers pass numpy arrays + an ASE or
pymatgen structure; this module produces a validated :class:`AtlasInput`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from widom_atlas.io.models import ALLOWED_GASES, AtlasInput

_LOGGER = logging.getLogger(__name__)

# Tolerance for cart vs (frac @ cell) consistency check, in Angstrom.
_CART_FRAC_TOL_A = 1e-6


def _cell_from_structure(structure: Any) -> np.ndarray:
    if structure is None:
        raise TypeError("structure must not be None; pass an ASE Atoms or pymatgen Structure")
    cell_attr = getattr(structure, "cell", None)
    if cell_attr is not None:
        if hasattr(cell_attr, "array"):
            return np.asarray(cell_attr.array, dtype=np.float64)
        try:
            return np.asarray(np.asarray(cell_attr), dtype=np.float64)
        except (TypeError, ValueError):
            pass
    lattice = getattr(structure, "lattice", None)
    if lattice is not None and hasattr(lattice, "matrix"):
        return np.asarray(lattice.matrix, dtype=np.float64)
    raise TypeError(
        f"structure {type(structure).__name__} has no .cell.array (ASE) or .lattice.matrix (pymatgen)"
    )


def _structure_id(structure: Any) -> str:
    if hasattr(structure, "get_chemical_formula"):
        try:
            f = structure.get_chemical_formula()
            if f:
                return str(f)
        except Exception as exc:
            _LOGGER.warning("ASE get_chemical_formula failed; falling back: %s", exc)
    if hasattr(structure, "composition"):
        try:
            f = structure.composition.reduced_formula
            if f:
                return str(f)
        except Exception as exc:
            _LOGGER.warning("pymatgen composition.reduced_formula failed; falling back: %s", exc)
    return "unknown_structure"


def _structure_metadata(structure: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if hasattr(structure, "get_chemical_formula"):
        meta["source"] = "ase"
        try:
            meta["formula"] = structure.get_chemical_formula()
        except Exception as exc:
            _LOGGER.warning("ASE get_chemical_formula failed during metadata extraction: %s", exc)
        try:
            meta["n_atoms"] = len(structure)
        except TypeError as exc:
            _LOGGER.warning("len(structure) failed during metadata extraction: %s", exc)
    elif hasattr(structure, "lattice"):
        meta["source"] = "pymatgen"
        try:
            meta["formula"] = str(structure.composition.reduced_formula)
        except Exception as exc:
            _LOGGER.warning("pymatgen reduced_formula failed during metadata extraction: %s", exc)
        try:
            meta["n_atoms"] = len(structure)
        except TypeError as exc:
            _LOGGER.warning("len(structure) failed during metadata extraction: %s", exc)
    return meta


def _require_ndarray(name: str, arr: Any) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray; got {type(arr).__name__}. "
            "from_arrays() does not accept Python lists — convert with np.asarray() first."
        )
    return arr


def from_arrays(
    *,
    structure: Any,
    positions_cart: np.ndarray | None = None,
    positions_frac: np.ndarray | None = None,
    energies_eV: np.ndarray,
    accessible: np.ndarray | None = None,
    temperature_K: float,
    gas: str,
    metadata: dict[str, Any] | None = None,
) -> AtlasInput:
    """Construct a validated :class:`AtlasInput` from numpy arrays + a host structure.

    Exactly one of ``positions_cart`` / ``positions_frac`` must be supplied,
    or both with consistency check. Fractional coordinates are wrapped to
    ``[0, 1)`` before storage. ``accessible`` defaults to all-True.
    """
    if gas == "H2O":
        raise ValueError("gas='H2O' is excluded from widom-atlas v1; see implementation-verdict.txt §F")
    if gas not in ALLOWED_GASES:
        raise ValueError(f"gas={gas!r} not in v1 allowed set {sorted(ALLOWED_GASES)}")
    cell = _cell_from_structure(structure)
    if cell.shape != (3, 3):
        raise ValueError(f"cell extracted from structure has shape {cell.shape}; expected (3,3)")
    if abs(float(np.linalg.det(cell))) < 1e-6:
        raise ValueError("cell extracted from structure is singular or near-degenerate")

    energies_arr = _require_ndarray("energies_eV", energies_eV)
    if energies_arr.ndim != 1:
        raise ValueError(f"energies_eV must be 1-D; got shape {energies_arr.shape}")
    n = energies_arr.shape[0]

    if positions_cart is None and positions_frac is None:
        raise ValueError("must supply at least one of positions_cart or positions_frac")
    cart_arr: np.ndarray
    frac_arr: np.ndarray
    if positions_frac is not None:
        frac_in = _require_ndarray("positions_frac", positions_frac).astype(np.float64, copy=False)
        if frac_in.shape != (n, 3):
            raise ValueError(f"positions_frac shape {frac_in.shape} does not match (N={n},3)")
        if positions_cart is not None:
            cart_in = _require_ndarray("positions_cart", positions_cart).astype(np.float64, copy=False)
            if cart_in.shape != (n, 3):
                raise ValueError(f"positions_cart shape {cart_in.shape} does not match (N={n},3)")
            recomputed = frac_in @ cell
            max_diff = float(np.max(np.abs(recomputed - cart_in)))
            if max_diff > _CART_FRAC_TOL_A:
                raise ValueError(
                    "positions_cart and positions_frac inconsistent: "
                    f"max |frac@cell - cart| = {max_diff:g} A > {_CART_FRAC_TOL_A:g} A"
                )
            frac_arr = np.mod(frac_in, 1.0)
            cart_arr = frac_arr @ cell
        else:
            frac_arr = np.mod(frac_in, 1.0)
            cart_arr = frac_arr @ cell
    else:
        cart_in = _require_ndarray("positions_cart", positions_cart).astype(np.float64, copy=False)
        if cart_in.shape != (n, 3):
            raise ValueError(f"positions_cart shape {cart_in.shape} does not match (N={n},3)")
        cart_arr = cart_in
        frac_unwrapped = np.linalg.solve(cell.T, cart_in.T).T
        frac_arr = np.mod(frac_unwrapped, 1.0)

    if accessible is None:
        accessible_arr = np.ones(n, dtype=bool)
    else:
        accessible_arr = _require_ndarray("accessible", accessible).astype(bool, copy=False)
        if accessible_arr.shape != (n,):
            raise ValueError(f"accessible shape {accessible_arr.shape} does not match (N={n},)")

    merged_metadata: dict[str, Any] = _structure_metadata(structure)
    if metadata:
        merged_metadata.update(metadata)

    return AtlasInput(
        structure_id=_structure_id(structure),
        gas=gas,
        temperature_K=float(temperature_K),
        cell_matrix=cell.tolist(),
        positions_cart_A=cart_arr.tolist(),
        positions_frac=frac_arr.tolist(),
        energies_eV=energies_arr.astype(np.float64).tolist(),
        accessible=[bool(x) for x in accessible_arr.tolist()],
        metadata=merged_metadata,
    )
