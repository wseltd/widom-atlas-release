"""Curated manual atom-removal for benchmark fixtures only.

No automatic defect-chemistry inference. Every removed atom is traceable via
the returned :class:`DefectRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DefectRecord:
    """Provenance record for an atom-removal perturbation."""

    removed_indices: tuple[int, ...]
    original_species: tuple[str, ...]
    provenance: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def _validate_indices(indices: list[int], n_sites: int) -> tuple[int, ...]:
    if not indices:
        raise ValueError("indices must be non-empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"indices contain duplicates: {indices}")
    for i in indices:
        if int(i) < 0:
            raise ValueError(f"index {i} is negative")
        if int(i) >= n_sites:
            raise ValueError(f"index {i} is out of range for structure of size {n_sites}")
    return tuple(int(i) for i in indices)


def remove_atoms(
    structure: Any,
    indices: list[int],
    provenance: str | None = None,
) -> tuple[Any, DefectRecord]:
    """Return ``(new_structure, defect_record)`` with the specified sites removed."""
    try:
        from ase import Atoms
        ase_atoms_cls: Any = Atoms
    except ImportError:
        ase_atoms_cls = None
    try:
        from pymatgen.core import Structure
        pymatgen_struct_cls: Any = Structure
    except ImportError:
        pymatgen_struct_cls = None

    if ase_atoms_cls is not None and isinstance(structure, ase_atoms_cls):
        n = len(structure)
        validated = _validate_indices(list(indices), n)
        symbols = structure.get_chemical_symbols()
        removed_species = tuple(symbols[i] for i in validated)
        new_atoms = structure.copy()
        del new_atoms[list(validated)]
        return new_atoms, DefectRecord(
            removed_indices=validated,
            original_species=removed_species,
            provenance=provenance or "manual_atom_removal",
        )
    if pymatgen_struct_cls is not None and isinstance(structure, pymatgen_struct_cls):
        n = len(structure)
        validated = _validate_indices(list(indices), n)
        removed_species = tuple(str(structure.species[i]) for i in validated)
        new_struct = structure.copy()
        new_struct.remove_sites(list(validated))
        return new_struct, DefectRecord(
            removed_indices=validated,
            original_species=removed_species,
            provenance=provenance or "manual_atom_removal",
        )
    raise TypeError(f"unsupported structure type: {type(structure).__name__}")


__all__ = ["DefectRecord", "_validate_indices", "remove_atoms"]
