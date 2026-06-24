"""Convenience adapter: CuspAI ``WidomInsertionResults`` → :class:`AtlasInput`.

This is the **convenience** path described in the brief. The stable foundation
remains :func:`widom_atlas.io.from_arrays` and :func:`widom_atlas.io.from_npz`;
this adapter is intentionally thin and does not let CuspAI's schema leak
into the rest of widom-atlas.

What it does:

1. Parses ``WidomInsertionResults.optimized_structure_cif`` into an ASE Atoms.
2. Reduces multi-atom guest positions to one Cartesian coordinate per
   insertion, defaulting to the geometric centroid (atom index ``-1``) so
   the choice is unambiguous for CO2/N2/CH4 and any other future gas.
3. Filters by ``is_valid`` (drops insertions Widom flagged as numerically
   bad) and preserves ``is_accessible`` as the accessibility mask passed
   into :class:`AtlasInput`.
4. Hands off to :func:`from_arrays` so all the array-shape, gas-allowlist,
   and frac-wrapping invariants of the foundation API are enforced.
5. Stashes the Widom scalars (Henry coefficient, heat of adsorption,
   averaged interaction energy + uncertainties) in
   ``metadata['widom_scalars']`` so the benchmark / robustness layers can
   compare against MOFX-DB / NIST without re-running Widom.

The CuspAI Widom result schema is real (verified at install time), but the
package contract still does not bind to it — :class:`AtlasInput` only sees
plain numpy arrays + an ASE structure.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.io.models import AtlasInput


def _parse_cif_to_atoms(cif_text: str) -> Any:
    """Parse a CIF string to an ASE ``Atoms`` with PBC=True on all axes."""
    from ase.io import read

    fd, path = tempfile.mkstemp(suffix=".cif", prefix="widom_atlas_optstruct_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(cif_text)
        atoms = read(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if hasattr(atoms, "set_pbc"):
        atoms.set_pbc(True)
    return atoms


def _gas_positions_to_cart(
    gas_positions: list[list[list[float]]] | np.ndarray,
    atom_index: int | None,
) -> np.ndarray:
    """Reduce ``(N, n_atoms, 3)`` insertion positions to ``(N, 3)`` Cartesian.

    ``atom_index=None`` (default) → geometric centroid; ``atom_index=k`` →
    pick the ``k``-th atom of each guest.
    """
    arr = np.asarray(gas_positions, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"gas_positions must have shape (N, n_atoms, 3); got {arr.shape}"
        )
    if atom_index is None:
        return arr.mean(axis=1)
    n_atoms = arr.shape[1]
    if not (-n_atoms <= atom_index < n_atoms):
        raise IndexError(
            f"atom_index={atom_index} out of range for guest with {n_atoms} atoms"
        )
    return arr[:, atom_index, :]


def from_widom_result(
    result: Any,
    *,
    gas: str,
    temperature_K: float,
    structure: Any | None = None,
    metadata: dict[str, Any] | None = None,
    atom_index: int | None = None,
    drop_invalid: bool = True,
) -> AtlasInput:
    """Convert a :class:`widom.WidomInsertionResults` to a validated :class:`AtlasInput`.

    Args:
        result: A ``widom.analyze.WidomInsertionResults`` instance (or any
            object exposing the same fields — duck-typed).
        gas: Adsorbate identifier. Must be one of ``{'CO2','N2','CH4'}``;
            ``H2O`` is excluded in v1.
        temperature_K: Temperature in Kelvin (the Widom result itself does
            not carry T; the caller must supply it).
        structure: Optional ASE Atoms / pymatgen Structure. When ``None``,
            the host is reconstructed from
            ``result.optimized_structure_cif``.
        metadata: Free-form metadata to merge into the
            :class:`AtlasInput.metadata` dict alongside ``widom_scalars``.
        atom_index: Which atom of the multi-atom guest to use for the
            insertion-site Cartesian coordinate. ``None`` → geometric
            centroid (recommended).
        drop_invalid: If ``True`` (default), insertions with
            ``is_valid=False`` are dropped before construction.
    """
    interaction_energies = np.asarray(result.interaction_energies, dtype=np.float64)
    is_accessible = np.asarray(result.is_accessible, dtype=bool)
    is_valid = np.asarray(result.is_valid, dtype=bool)
    gas_positions = result.gas_positions

    if structure is None:
        if not getattr(result, "optimized_structure_cif", ""):
            raise ValueError(
                "result.optimized_structure_cif is empty; supply `structure=` explicitly"
            )
        structure = _parse_cif_to_atoms(result.optimized_structure_cif)

    cart = _gas_positions_to_cart(gas_positions, atom_index=atom_index)

    if drop_invalid and not is_valid.all():
        keep = is_valid
        cart = cart[keep]
        interaction_energies = interaction_energies[keep]
        is_accessible = is_accessible[keep]

    if not np.all(np.isfinite(interaction_energies)):
        finite = np.isfinite(interaction_energies)
        cart = cart[finite]
        interaction_energies = interaction_energies[finite]
        is_accessible = is_accessible[finite]

    merged_metadata: dict[str, Any] = {
        "widom_scalars": {
            "henry_coefficient": float(getattr(result, "henry_coefficient", float("nan"))),
            "henry_coefficient_std": float(getattr(result, "henry_coefficient_std", float("nan"))),
            "heat_of_adsorption_kJmol": float(getattr(result, "heat_of_adsorption", float("nan"))),
            "heat_of_adsorption_std_kJmol": float(getattr(result, "heat_of_adsorption_std", float("nan"))),
            "averaged_interaction_energy_eV": float(getattr(result, "averaged_interaction_energy", float("nan"))),
            "averaged_interaction_energy_std_eV": float(getattr(result, "averaged_interaction_energy_std", float("nan"))),
            "atomic_density": float(getattr(result, "atomic_density", float("nan"))),
            "energy_gas_eV": float(getattr(result, "energy_gas", float("nan"))),
            "energy_structure_eV": float(getattr(result, "energy_structure", float("nan"))),
        },
        "samples_origin": "cuspai_widom",
        "n_insertions_total": len(result.interaction_energies),
        "n_insertions_kept": len(interaction_energies),
    }
    if metadata:
        merged_metadata.update(metadata)

    return from_arrays(
        structure=structure,
        positions_cart=cart,
        positions_frac=None,
        energies_eV=interaction_energies,
        accessible=is_accessible,
        temperature_K=float(temperature_K),
        gas=gas,
        metadata=merged_metadata,
    )


__all__ = ["from_widom_result"]
