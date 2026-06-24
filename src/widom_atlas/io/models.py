"""Pydantic v2 input contract: AtlasInput.

Frozen, JSON-serialisable representation of one Widom-style insertion campaign
expressed as plain Python lists. Numpy-array-backed validated forms are
exposed via the :pyattr:`AtlasInput.samples` property as
``widom_atlas.core.models.InsertionSamples``; the underlying storage is
list-based so the model round-trips through JSON without numpy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from widom_atlas.core.models import InsertionSamples

_LOGGER = logging.getLogger(__name__)

ALLOWED_GASES: frozenset[str] = frozenset({"CO2", "N2", "CH4"})
_FRAC_TOLERANCE = 0.5  # accept fractional coords in roughly [-0.5, 1.5]; wrapping happens elsewhere


class AtlasInput(BaseModel):
    """Frozen input contract for the widom-atlas pipeline.

    Use ``widom_atlas.io.from_arrays`` to construct from numpy arrays + an ASE
    or pymatgen structure. This model itself accepts only plain-Python list
    payloads so it can be serialised to JSON without numpy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=False, extra="forbid", frozen=True)

    structure_id: str = Field(min_length=1)
    gas: str
    temperature_K: float = Field(gt=0.0, le=2000.0)
    cell_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    positions_cart_A: list[list[float]]
    positions_frac: list[list[float]]
    energies_eV: list[float]
    accessible: list[bool]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("gas")
    @classmethod
    def _validate_gas(cls, v: str) -> str:
        if v == "H2O":
            raise ValueError(
                "gas='H2O' is excluded from widom-atlas v1 (force-field + charge handling not explicit yet); "
                "see implementation-verdict.txt §F"
            )
        if v not in ALLOWED_GASES:
            raise ValueError(
                f"gas={v!r} not in v1 allowed set {sorted(ALLOWED_GASES)}"
            )
        return v

    @field_validator("cell_matrix", mode="before")
    @classmethod
    def _coerce_cell_matrix(
        cls, v: Any
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        arr = np.asarray(v, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(f"cell_matrix must be shape (3,3); got {arr.shape}")
        rows: list[tuple[float, float, float]] = [
            (float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])) for i in range(3)
        ]
        return (rows[0], rows[1], rows[2])

    @field_validator("positions_cart_A", "positions_frac")
    @classmethod
    def _validate_positions_shape(cls, v: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(v):
            if len(row) != 3:
                raise ValueError(f"position row {i} has {len(row)} components; expected 3")
        return v

    @field_validator("positions_frac")
    @classmethod
    def _validate_frac_range(cls, v: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(v):
            for j, x in enumerate(row):
                if not (-_FRAC_TOLERANCE <= x <= 1.0 + _FRAC_TOLERANCE):
                    raise ValueError(
                        "positions_frac[%d][%d]=%g outside accepted pre-wrap range [-%g, %g]; "
                        "wrap before constructing AtlasInput"
                        % (i, j, x, _FRAC_TOLERANCE, 1.0 + _FRAC_TOLERANCE)
                    )
        return v

    @field_validator("energies_eV")
    @classmethod
    def _validate_energies_finite(cls, v: list[float]) -> list[float]:
        for i, e in enumerate(v):
            if not np.isfinite(e):
                raise ValueError("energies_eV[%d]=%r is not finite" % (i, e))
        return v

    @model_validator(mode="after")
    def _validate_lengths_and_cell(self) -> AtlasInput:
        n = len(self.positions_cart_A)
        for name, arr in (
            ("positions_frac", self.positions_frac),
            ("energies_eV", self.energies_eV),
            ("accessible", self.accessible),
        ):
            if len(arr) != n:
                raise ValueError(
                    "length mismatch: positions_cart_A has %d rows but %s has %d"
                    % (n, name, len(arr))
                )
        cell = np.asarray(self.cell_matrix, dtype=float)
        det = float(np.linalg.det(cell))
        if abs(det) < 1e-6:
            raise ValueError("cell_matrix is singular or near-degenerate (|det|=%g < 1e-6)" % abs(det))
        if not self.metadata:
            _LOGGER.warning("AtlasInput constructed with empty metadata for structure_id=%s", self.structure_id)
        return self

    @property
    def samples(self) -> InsertionSamples:
        """Return a numpy-array-backed :class:`InsertionSamples` view of the stored arrays."""
        from widom_atlas.core.models import InsertionSamples

        return InsertionSamples(
            positions_cart=np.asarray(self.positions_cart_A, dtype=np.float64),
            positions_frac=np.asarray(self.positions_frac, dtype=np.float64),
            energies_eV=np.asarray(self.energies_eV, dtype=np.float64),
            accessible=np.asarray(self.accessible, dtype=bool),
            temperature_K=self.temperature_K,
            gas=self.gas,
            metadata=dict(self.metadata),
        )

    @property
    def n_samples(self) -> int:
        """Number of insertion samples."""
        return len(self.energies_eV)

    @property
    def cell_matrix_A(self) -> np.ndarray:
        """Cell matrix as a numpy ``(3,3)`` float64 array (Angstrom)."""
        return np.asarray(self.cell_matrix, dtype=np.float64)

    @property
    def input_hash(self) -> str:
        """Deterministic SHA256 over canonicalised positions, energies, gas, cell, structure_id, metadata."""
        h = hashlib.sha256()
        h.update(np.round(np.asarray(self.positions_frac, dtype=np.float64), 9).tobytes())
        h.update(np.round(np.asarray(self.energies_eV, dtype=np.float64), 9).tobytes())
        h.update(np.round(self.cell_matrix_A, 9).tobytes())
        h.update(np.asarray(self.accessible, dtype=bool).tobytes())
        h.update(self.gas.encode("utf-8"))
        h.update(self.structure_id.encode("utf-8"))
        h.update(json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(np.float64(self.temperature_K).tobytes())
        return h.hexdigest()
