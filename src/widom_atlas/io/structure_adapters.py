"""ASE ↔ pymatgen structure adapters.

Wraps :class:`pymatgen.io.ase.AseAtomsAdaptor` with widom-atlas v1 invariants:

- both representations must be fully periodic (PBC ``True`` on all axes);
- the cell must be a non-singular 3×3 matrix in Angstrom;
- 2-D periodic and non-periodic structures are rejected explicitly.

A row-vector convention is used for the cell matrix (lattice vectors as rows),
matching both ASE and pymatgen conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_DET_TOL_A3 = 1e-6


def _ensure_pbc_3d(atoms: Any) -> None:
    pbc = getattr(atoms, "pbc", None)
    if pbc is None:
        raise NotImplementedError("ase.Atoms object has no .pbc attribute; cannot determine periodicity")
    pbc_arr = np.asarray(pbc, dtype=bool).reshape(-1)
    if pbc_arr.size == 1:
        pbc_arr = np.full(3, bool(pbc_arr[0]))
    if pbc_arr.size != 3:
        raise NotImplementedError(f"PBC array must have length 3; got {pbc_arr.size}")
    if not bool(pbc_arr.all()):
        if int(pbc_arr.sum()) == 2:
            raise NotImplementedError(
                "2-D periodic structures are not supported in widom-atlas v1; got PBC=%s" % pbc_arr.tolist()
            )
        raise ValueError(
            "widom-atlas v1 only supports fully-periodic 3D crystals; got PBC=%s" % pbc_arr.tolist()
        )


def _ensure_nonsingular(cell: np.ndarray) -> None:
    arr = np.asarray(cell, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"cell must be shape (3,3); got {arr.shape}")
    det = float(np.linalg.det(arr))
    if abs(det) < _DET_TOL_A3:
        raise ValueError(
            "cell is singular or near-degenerate: |det| = %g < %g" % (abs(det), _DET_TOL_A3)
        )


def get_cell_matrix(obj: Any) -> np.ndarray:
    """Return the lattice matrix (rows = a, b, c) as a ``(3,3)`` float64 array."""
    cell_attr = getattr(obj, "cell", None)
    if cell_attr is not None and hasattr(cell_attr, "array"):
        return np.asarray(cell_attr.array, dtype=np.float64)
    lattice = getattr(obj, "lattice", None)
    if lattice is not None and hasattr(lattice, "matrix"):
        return np.asarray(lattice.matrix, dtype=np.float64)
    raise TypeError(
        "object has neither .cell.array (ASE) nor .lattice.matrix (pymatgen): %r" % type(obj).__name__
    )


def ase_to_pymatgen(atoms: Any) -> Any:
    """Convert an ASE ``Atoms`` to a pymatgen ``Structure``."""
    import ase
    from pymatgen.io.ase import AseAtomsAdaptor

    if not isinstance(atoms, ase.Atoms):
        raise TypeError(f"ase_to_pymatgen expects ase.Atoms; got {type(atoms).__name__}")
    _ensure_pbc_3d(atoms)
    cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
    _ensure_nonsingular(cell)
    return AseAtomsAdaptor.get_structure(atoms)


def pymatgen_to_ase(structure: Any) -> Any:
    """Convert a pymatgen ``Structure`` to an ASE ``Atoms`` (PBC=True on all axes)."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    if not isinstance(structure, Structure):
        raise TypeError(f"pymatgen_to_ase expects pymatgen.core.Structure; got {type(structure).__name__}")
    cell = np.asarray(structure.lattice.matrix, dtype=np.float64)
    _ensure_nonsingular(cell)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_pbc([True, True, True])
    return atoms
