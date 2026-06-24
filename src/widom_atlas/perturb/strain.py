"""Apply affine / isotropic / uniaxial / volume-preserving strain to a host structure.

Fractional positions are preserved (atoms move with the cell). The input
structure is never mutated; a new structure is returned.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

_LOGGER = logging.getLogger(__name__)

StrainMode = Literal["affine", "isotropic", "uniaxial", "volume_preserving"]
_AXIS_TO_INDEX = {"a": 0, "b": 1, "c": 2}


def _build_isotropic_strain(value: float) -> np.ndarray:
    return float(value) * np.eye(3, dtype=np.float64)


def _build_uniaxial_strain(value: float, axis: str | int) -> np.ndarray:
    if isinstance(axis, str):
        if axis not in _AXIS_TO_INDEX:
            raise ValueError(f"axis must be one of 'a','b','c' or 0,1,2; got {axis!r}")
        idx = _AXIS_TO_INDEX[axis]
    else:
        idx = int(axis)
        if idx not in (0, 1, 2):
            raise ValueError(f"axis index must be 0, 1, or 2; got {idx}")
    out = np.zeros((3, 3), dtype=np.float64)
    out[idx, idx] = float(value)
    return out


def _build_volume_preserving_strain(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"volume_preserving strain expects (3,3) input; got {arr.shape}")
    deviatoric = arr - (np.trace(arr) / 3.0) * np.eye(3)
    return deviatoric


def _structure_kind(structure: Any) -> str:
    try:
        from ase import Atoms
    except ImportError as exc:
        _LOGGER.warning("ASE unavailable while detecting structure kind: %s", exc)
    else:
        if isinstance(structure, Atoms):
            return "ase"
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        _LOGGER.warning("pymatgen unavailable while detecting structure kind: %s", exc)
    else:
        if isinstance(structure, Structure):
            return "pymatgen"
    raise TypeError(f"unsupported structure type: {type(structure).__name__}")


def apply_strain(
    structure: Any,
    mode: StrainMode,
    value: Any,
    axis: str | int | None = None,
) -> Any:
    """Return a new structure with the specified strain applied.

    The cell becomes ``(I + strain_matrix) @ old_cell``. Fractional positions
    are preserved; Cartesian positions are recomputed via ``frac @ new_cell``.
    """
    if mode == "affine":
        strain = np.asarray(value, dtype=np.float64)
        if strain.shape != (3, 3):
            raise ValueError(f"affine strain matrix must be (3,3); got {strain.shape}")
    elif mode == "isotropic":
        strain = _build_isotropic_strain(float(value))
    elif mode == "uniaxial":
        if axis is None:
            raise ValueError("uniaxial strain requires explicit axis")
        strain = _build_uniaxial_strain(float(value), axis)
    elif mode == "volume_preserving":
        strain = _build_volume_preserving_strain(value)
    else:
        raise ValueError(f"unknown strain mode: {mode!r}")

    kind = _structure_kind(structure)
    if kind == "ase":
        from ase import Atoms

        atoms: Atoms = structure
        old_cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
        new_cell = (np.eye(3) + strain) @ old_cell
        new_atoms = atoms.copy()
        scaled = atoms.get_scaled_positions()
        new_atoms.set_cell(new_cell, scale_atoms=False)
        new_atoms.set_scaled_positions(scaled)
        return new_atoms
    # pymatgen
    from pymatgen.core import Lattice, Structure

    s: Structure = structure
    old_cell = np.asarray(s.lattice.matrix, dtype=np.float64)
    new_cell = (np.eye(3) + strain) @ old_cell
    new_lattice = Lattice(new_cell)
    return Structure(
        new_lattice,
        s.species,
        s.frac_coords,
        coords_are_cartesian=False,
        site_properties=s.site_properties,
    )


__all__ = [
    "_build_isotropic_strain",
    "_build_uniaxial_strain",
    "_build_volume_preserving_strain",
    "apply_strain",
]
