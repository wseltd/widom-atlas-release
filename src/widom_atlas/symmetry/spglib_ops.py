"""Detect host-framework symmetry via spglib."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import spglib

from widom_atlas.core.constants import DEFAULT_ANGLE_TOLERANCE_DEG, DEFAULT_SYMPREC
from widom_atlas.symmetry.types import Confidence, FrameworkSymmetry

_LOGGER = logging.getLogger(__name__)


def _structure_to_spglib_cell(structure: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lattice, scaled_positions, atomic_numbers)`` accepted by spglib."""
    try:
        from ase import Atoms
    except ImportError as exc:
        _LOGGER.warning("ASE unavailable in _structure_to_spglib_cell: %s", exc)
    else:
        if isinstance(structure, Atoms):
            lattice = np.asarray(structure.get_cell().array, dtype=np.float64)
            positions = np.asarray(structure.get_scaled_positions(), dtype=np.float64)
            numbers = np.asarray(structure.get_atomic_numbers(), dtype=np.int64)
            return lattice, positions, numbers
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        _LOGGER.warning("pymatgen unavailable in _structure_to_spglib_cell: %s", exc)
    else:
        if isinstance(structure, Structure):
            lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
            positions = np.asarray(structure.frac_coords, dtype=np.float64)
            numbers = np.asarray([s.Z for s in structure.species], dtype=np.int64)
            return lattice, positions, numbers
    raise TypeError(
        f"structure {type(structure).__name__} is not an ASE Atoms or pymatgen Structure"
    )


def _attr(dataset: Any, name: str, default: Any = None) -> Any:
    """Tolerate both dict-style (older spglib) and attribute-style (newer spglib) datasets."""
    if dataset is None:
        return default
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if hasattr(dataset, "__getitem__"):
        try:
            return dataset[name]
        except (KeyError, TypeError):
            return default
    return default


def _confidence_from_displacement(
    original_frac: np.ndarray | None,
    refined_frac: np.ndarray | None,
    cell: np.ndarray,
    symprec: float,
) -> Confidence:
    if original_frac is None or refined_frac is None:
        return "medium"
    if original_frac.shape != refined_frac.shape:
        return "medium"
    delta = original_frac - refined_frac
    delta -= np.round(delta)
    delta_cart = delta @ cell
    rms = float(np.sqrt(np.mean(np.sum(delta_cart * delta_cart, axis=-1))))
    if rms < 0.5 * symprec:
        return "high"
    if rms < float(symprec):
        return "medium"
    if rms < 5.0 * symprec:
        return "low"
    return "uncertain"


def detect_symmetry(
    structure: Any,
    symprec: float = DEFAULT_SYMPREC,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE_DEG,
) -> FrameworkSymmetry:
    """Run :func:`spglib.get_symmetry_dataset` and return a :class:`FrameworkSymmetry`."""
    cell_tuple = _structure_to_spglib_cell(structure)
    dataset = spglib.get_symmetry_dataset(
        cell_tuple,  # type: ignore[arg-type]
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if dataset is None:
        return FrameworkSymmetry(
            space_group_number=1,
            international_symbol="P1",
            hall_number=1,
            rotations=np.eye(3, dtype=np.int64)[None, ...],
            translations=np.zeros((1, 3), dtype=np.float64),
            n_operations=1,
            confidence="uncertain",
            is_low_symmetry=True,
            is_triclinic=True,
            symprec=symprec,
            angle_tolerance_deg=angle_tolerance,
            notes="spglib returned no symmetry dataset; defaulting to P1",
        )

    rotations = np.asarray(_attr(dataset, "rotations"), dtype=np.int64)
    translations = np.asarray(_attr(dataset, "translations"), dtype=np.float64)
    sg_number = int(_attr(dataset, "number", 1))
    international = str(_attr(dataset, "international", "P1"))
    hall_number = int(_attr(dataset, "hall_number", 1))
    n_ops = int(rotations.shape[0])

    refined_positions = _attr(dataset, "std_positions")
    if refined_positions is None:
        refined_positions = _attr(dataset, "standardized_positions")
    refined_arr: np.ndarray | None = None
    if refined_positions is not None:
        try:
            refined_arr = np.asarray(refined_positions, dtype=np.float64)
        except (TypeError, ValueError):
            refined_arr = None
    confidence = _confidence_from_displacement(
        original_frac=cell_tuple[1],
        refined_frac=refined_arr,
        cell=cell_tuple[0],
        symprec=symprec,
    )

    return FrameworkSymmetry(
        space_group_number=sg_number,
        international_symbol=international,
        hall_number=hall_number,
        rotations=rotations,
        translations=translations,
        n_operations=n_ops,
        confidence=confidence,
        is_low_symmetry=bool(n_ops <= 4),
        is_triclinic=bool(sg_number <= 2),
        symprec=symprec,
        angle_tolerance_deg=angle_tolerance,
    )
