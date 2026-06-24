"""High-level :func:`apply_perturbation` operating on an :class:`AtlasInput`.

This dispatches a :class:`PerturbationSpec` (or a list of specs for composite
perturbations) to the strain / atom-removal building blocks in ``perturb.strain``
and ``perturb.defects``. Insertion samples are cleared because the original
samples no longer apply to the perturbed cell — downstream callers must
recompute insertion samples on the new structure.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from widom_atlas.core.models import PerturbationSpec
from widom_atlas.io.models import AtlasInput

PerturbationLike = PerturbationSpec | Sequence[PerturbationSpec]


def _strain_matrix_from_spec(spec: PerturbationSpec) -> np.ndarray:
    if spec.kind == "affine":
        if spec.strain_matrix is None:
            raise ValueError("affine PerturbationSpec missing strain_matrix")
        return np.asarray(spec.strain_matrix, dtype=np.float64)
    if spec.kind == "isotropic":
        if spec.magnitude is None:
            raise ValueError("isotropic PerturbationSpec missing magnitude")
        return float(spec.magnitude) * np.eye(3, dtype=np.float64)
    if spec.kind == "uniaxial":
        if spec.magnitude is None or spec.axis is None:
            raise ValueError("uniaxial PerturbationSpec missing magnitude or axis")
        idx = {"a": 0, "b": 1, "c": 2}[spec.axis]
        out = np.zeros((3, 3), dtype=np.float64)
        out[idx, idx] = float(spec.magnitude)
        return out
    raise ValueError(f"non-strain spec kind {spec.kind!r}")


def _spec_to_history_entry(spec: PerturbationSpec) -> dict[str, Any]:
    return {
        "kind": spec.kind,
        "label": spec.label,
        "magnitude": spec.magnitude,
        "axis": spec.axis,
        "strain_matrix": spec.strain_matrix,
        "removed_atom_indices": spec.removed_atom_indices,
        "notes": spec.notes,
    }


def _apply_one(atlas_input: AtlasInput, spec: PerturbationSpec) -> AtlasInput:
    cell = np.asarray(atlas_input.cell_matrix, dtype=np.float64)
    metadata = copy.deepcopy(atlas_input.metadata)
    history = list(metadata.get("perturbation_history", []))
    history.append(_spec_to_history_entry(spec))
    metadata["perturbation_history"] = history
    metadata["samples_cleared_due_to_perturbation"] = True

    if spec.kind in {"affine", "isotropic", "uniaxial"}:
        strain = _strain_matrix_from_spec(spec)
        new_cell = (np.eye(3) + strain) @ cell
    elif spec.kind == "atom_removal":
        new_cell = cell
        metadata["removed_atom_indices"] = list(spec.removed_atom_indices or [])
    else:
        raise ValueError(f"unknown perturbation kind: {spec.kind!r}")

    return AtlasInput(
        structure_id=atlas_input.structure_id,
        gas=atlas_input.gas,
        temperature_K=atlas_input.temperature_K,
        cell_matrix=new_cell.tolist(),
        positions_cart_A=[],
        positions_frac=[],
        energies_eV=[],
        accessible=[],
        metadata=metadata,
    )


def apply_perturbation(
    atlas_input: AtlasInput, spec: PerturbationLike
) -> AtlasInput:
    """Return a new :class:`AtlasInput` with ``spec`` applied.

    ``spec`` may be a single :class:`PerturbationSpec` (most common) or a
    sequence applied in order. Insertion samples are cleared on the returned
    object because they no longer apply to the perturbed cell.
    """
    if isinstance(spec, PerturbationSpec):
        return _apply_one(atlas_input, spec)
    specs = list(spec)
    if not specs:
        raise ValueError("spec sequence must be non-empty")
    out = atlas_input
    for s in specs:
        if not isinstance(s, PerturbationSpec):
            raise TypeError(f"spec sequence contains non-PerturbationSpec: {type(s).__name__}")
        out = _apply_one(out, s)
    return out


__all__ = ["apply_perturbation"]
