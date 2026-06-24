"""``.npz`` round-trip for :class:`AtlasInput`.

Round-trip rules:

- positions and energies are bit-exact under the underlying :class:`np.savez_compressed`;
- metadata is stored as canonical JSON (``sort_keys=True``);
- pickling is never used (``allow_pickle=False`` on load);
- the parent directory of the destination must already exist;
- on load, the validated :func:`from_arrays` path is used so any future schema
  tightening is enforced uniformly on disk-loaded data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.io.models import AtlasInput


def save_samples_npz(atlas_input: AtlasInput, path: Path) -> None:
    """Persist an :class:`AtlasInput` to ``path`` as a single compressed ``.npz`` archive."""
    p = Path(path)
    if not p.parent.exists():
        raise FileNotFoundError(f"parent directory does not exist: {p.parent}")
    metadata_blob = json.dumps(
        {
            "gas": atlas_input.gas,
            "structure_id": atlas_input.structure_id,
            "metadata": atlas_input.metadata,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    np.savez_compressed(
        p,
        positions_cart=np.asarray(atlas_input.positions_cart_A, dtype=np.float64),
        positions_frac=np.asarray(atlas_input.positions_frac, dtype=np.float64),
        energies_eV=np.asarray(atlas_input.energies_eV, dtype=np.float64),
        accessible=np.asarray(atlas_input.accessible, dtype=bool),
        cell_matrix=atlas_input.cell_matrix_A,
        temperature_K=np.asarray(atlas_input.temperature_K, dtype=np.float64),
        metadata_json=np.asarray(metadata_blob),
    )


class _LooseStructure:
    """Lightweight cell carrier used when no original structure is supplied to :func:`from_npz`."""

    def __init__(self, cell: np.ndarray, formula: str = "unknown_structure") -> None:
        self._cell = np.asarray(cell, dtype=np.float64)
        self._formula = formula

    class _Cell:
        def __init__(self, arr: np.ndarray) -> None:
            self.array = arr

    @property
    def cell(self) -> _LooseStructure._Cell:
        return _LooseStructure._Cell(self._cell)

    def __len__(self) -> int:
        return 1

    def get_chemical_formula(self) -> str:
        return self._formula


def from_npz(path: Path, structure: Any | None = None) -> AtlasInput:
    """Load an :class:`AtlasInput` written by :func:`save_samples_npz`.

    If ``structure`` is not given, a minimal cell carrier is reconstructed from
    the stored ``cell_matrix`` so the validated :func:`from_arrays` constructor
    can be re-run without numpy-side surprises.
    """
    p = Path(path)
    with np.load(p, allow_pickle=False) as f:
        positions_cart = np.asarray(f["positions_cart"], dtype=np.float64)
        positions_frac = np.asarray(f["positions_frac"], dtype=np.float64)
        energies_eV = np.asarray(f["energies_eV"], dtype=np.float64)
        accessible = np.asarray(f["accessible"], dtype=bool)
        cell_matrix = np.asarray(f["cell_matrix"], dtype=np.float64)
        temperature_K = float(f["temperature_K"])
        metadata_blob = json.loads(str(f["metadata_json"]))

    gas = str(metadata_blob["gas"])
    structure_id_stored = str(metadata_blob.get("structure_id", "unknown_structure"))
    metadata: dict[str, Any] = dict(metadata_blob.get("metadata", {}))

    carrier = structure if structure is not None else _LooseStructure(cell_matrix, structure_id_stored)
    return from_arrays(
        structure=carrier,
        positions_cart=positions_cart,
        positions_frac=positions_frac,
        energies_eV=energies_eV,
        accessible=accessible,
        temperature_K=temperature_K,
        gas=gas,
        metadata=metadata,
    )
