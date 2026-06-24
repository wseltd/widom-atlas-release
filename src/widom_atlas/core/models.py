"""Core Pydantic v2 schemas for widom-atlas.

All models are frozen and validate aggressively at construction time. Numpy
arrays are accepted where shape and dtype matter (``InsertionSamples``,
``DensityGrid``); plain-Python types are used where JSON round-tripping is
required (``RobustnessReport``, ``RunManifest``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

ALLOWED_GASES_V1: frozenset[str] = frozenset({"CO2", "N2", "CH4"})

# ---------------------------------------------------------------------------
# T005 — InsertionSamples
# ---------------------------------------------------------------------------


class InsertionSamples(BaseModel):
    """One Widom-style insertion campaign expressed as numpy arrays.

    Validators enforce shape consistency, fractional wrapping into [0, 1),
    finite energies, and gas allow-listing. ``to_npz`` / ``from_npz`` provide
    round-trip persistence used by ``widom_atlas.io``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    positions_cart: np.ndarray
    positions_frac: np.ndarray
    energies_eV: np.ndarray
    accessible: np.ndarray
    temperature_K: float = Field(gt=0.0)
    gas: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("gas")
    @classmethod
    def _validate_gas(cls, v: str) -> str:
        if v not in ALLOWED_GASES_V1:
            raise ValueError(
                f"gas={v!r} not in v1 allowed set {sorted(ALLOWED_GASES_V1)}"
            )
        return v

    @field_validator("temperature_K")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"temperature_K={v!r} is not finite")
        return v

    @field_validator("positions_cart", "positions_frac")
    @classmethod
    def _validate_position_array(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"position array must have shape (N,3); got {arr.shape}")
        return arr

    @field_validator("energies_eV")
    @classmethod
    def _validate_energies(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"energies_eV must be 1-D; got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            bad = int(np.flatnonzero(~np.isfinite(arr))[0])
            raise ValueError(f"energies_eV[{bad}]={arr[bad]!r} is not finite")
        return arr

    @field_validator("accessible")
    @classmethod
    def _validate_accessible(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=bool)
        if arr.ndim != 1:
            raise ValueError(f"accessible must be 1-D bool array; got shape {arr.shape}")
        return arr

    @model_validator(mode="after")
    def _validate_consistency(self) -> InsertionSamples:
        n = self.positions_cart.shape[0]
        for name, arr in (
            ("positions_frac", self.positions_frac),
            ("energies_eV", self.energies_eV),
            ("accessible", self.accessible),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    "length mismatch: positions_cart has %d rows but %s has %d"
                    % (n, name, arr.shape[0])
                )
        tol = 1e-9
        bad = np.where(
            (self.positions_frac < -tol) | (self.positions_frac >= 1.0 + tol)
        )
        if bad[0].size:
            i = int(bad[0][0])
            j = int(bad[1][0])
            raise ValueError(
                "positions_frac[%d][%d]=%g is outside wrapped range [0,1)"
                % (i, j, float(self.positions_frac[i, j]))
            )
        return self

    @property
    def n_samples(self) -> int:
        """Number of insertion samples (rows in the position arrays)."""
        return int(self.positions_cart.shape[0])

    def to_npz(self, path: Path) -> None:
        """Persist the samples to ``path`` as a single ``.npz`` archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            positions_cart=self.positions_cart,
            positions_frac=self.positions_frac,
            energies_eV=self.energies_eV,
            accessible=self.accessible,
            temperature_K=np.asarray(self.temperature_K, dtype=np.float64),
            gas=np.asarray(self.gas),
            metadata_json=np.asarray(json.dumps(self.metadata)),
        )

    @classmethod
    def from_npz(cls, path: Path) -> InsertionSamples:
        """Load samples written by :meth:`to_npz`."""
        with np.load(Path(path), allow_pickle=False) as f:
            metadata: dict[str, Any] = json.loads(str(f["metadata_json"]))
            return cls(
                positions_cart=f["positions_cart"],
                positions_frac=f["positions_frac"],
                energies_eV=f["energies_eV"],
                accessible=f["accessible"],
                temperature_K=float(f["temperature_K"]),
                gas=str(f["gas"]),
                metadata=metadata,
            )


# ---------------------------------------------------------------------------
# T006 — Basin
# ---------------------------------------------------------------------------


class Basin(BaseModel):
    """One adsorption basin extracted from clustered insertion samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    basin_id: int = Field(ge=0)
    count: int = Field(ge=1)
    weight: float = Field(ge=0.0, le=1.0)
    centroid_frac: tuple[float, float, float]
    centroid_cart_A: tuple[float, float, float]
    mean_energy_eV: float
    std_energy_eV: float = Field(ge=0.0)
    accessible_fraction: float = Field(ge=0.0, le=1.0)
    spread_A: float = Field(ge=0.0)
    uncertainty: dict[str, float] = Field(default_factory=dict)
    energy_stderr_eV: float | None = None
    centroid_stderr_A: float | None = None
    weight_stderr: float | None = None
    low_count_flag: bool | None = None

    @field_validator("centroid_frac")
    @classmethod
    def _validate_centroid_wrapped(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        for i, x in enumerate(v):
            if not (0.0 <= x < 1.0):
                raise ValueError(
                    "centroid_frac[%d]=%g must be wrapped to [0,1)" % (i, x)
                )
        return v

    @field_validator("accessible_fraction", "weight", mode="before")
    @classmethod
    def _clamp_unit_interval(cls, v: object) -> object:
        """Clamp weighted-sum drift back into [0, 1] before le=1.0 validation."""
        if v is None:
            return None
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return v
        if 1.0 < f <= 1.0 + 1e-9:
            return 1.0
        if -1e-9 <= f < 0.0:
            return 0.0
        return f

    @field_validator("mean_energy_eV", "std_energy_eV", "spread_A")
    @classmethod
    def _validate_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"value {v!r} is not finite")
        return v

    def as_row(self) -> dict[str, Any]:
        """Flat dict of scalar fields for CSV/JSON export."""
        return {
            "basin_id": self.basin_id,
            "count": self.count,
            "weight": self.weight,
            "centroid_frac_a": self.centroid_frac[0],
            "centroid_frac_b": self.centroid_frac[1],
            "centroid_frac_c": self.centroid_frac[2],
            "centroid_cart_x_A": self.centroid_cart_A[0],
            "centroid_cart_y_A": self.centroid_cart_A[1],
            "centroid_cart_z_A": self.centroid_cart_A[2],
            "mean_energy_eV": self.mean_energy_eV,
            "std_energy_eV": self.std_energy_eV,
            "accessible_fraction": self.accessible_fraction,
            "spread_A": self.spread_A,
        }


def as_row(basin: Basin) -> dict[str, Any]:
    """Module-level helper mirroring :meth:`Basin.as_row` for convenience."""
    return basin.as_row()


# ---------------------------------------------------------------------------
# T007 — DensityGrid
# ---------------------------------------------------------------------------


class DensityGrid(BaseModel):
    """A Boltzmann-weighted 3D adsorption-density map on a fractional grid."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    grid: np.ndarray
    shape: tuple[int, int, int]
    cell_A: np.ndarray
    spacing_A: tuple[float, float, float]
    temperature_K: float = Field(gt=0.0)
    gas: str
    normalisation: Literal["probability", "none"] = "probability"
    smoothing_sigma_A: float = Field(default=0.0, ge=0.0)
    n_source_samples: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("gas")
    @classmethod
    def _validate_gas(cls, v: str) -> str:
        if v not in ALLOWED_GASES_V1:
            raise ValueError(
                f"gas={v!r} not in v1 allowed set {sorted(ALLOWED_GASES_V1)}"
            )
        return v

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(v) != 3 or any(int(s) < 2 for s in v):
            raise ValueError(f"shape must be three ints >= 2; got {v}")
        return (int(v[0]), int(v[1]), int(v[2]))

    @field_validator("grid")
    @classmethod
    def _validate_grid_array(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"grid must be 3-D; got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("grid contains non-finite values")
        if np.any(arr < 0):
            raise ValueError("grid contains negative values")
        return arr

    @field_validator("cell_A")
    @classmethod
    def _validate_cell(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"cell_A must be shape (3,3); got {arr.shape}")
        if abs(float(np.linalg.det(arr))) < 1e-9:
            raise ValueError("cell_A is singular or near-degenerate")
        return arr

    @model_validator(mode="after")
    def _validate_consistency(self) -> DensityGrid:
        if tuple(self.grid.shape) != self.shape:
            raise ValueError(
                f"grid.shape={self.grid.shape} does not match declared shape={self.shape}"
            )
        if self.normalisation == "probability":
            total = float(self.grid.sum())
            if not np.isclose(total, 1.0, atol=1e-9):
                raise ValueError(
                    f"normalisation='probability' but grid.sum()={total!r} (tol 1e-9)"
                )
        cell_norms = np.linalg.norm(self.cell_A, axis=1)
        expected = tuple(float(cell_norms[i] / self.shape[i]) for i in range(3))
        for i, (got, exp) in enumerate(zip(self.spacing_A, expected, strict=False)):
            if exp == 0.0:
                raise ValueError(f"cell_A row {i} has zero norm")
            rel = abs(got - exp) / exp
            if rel > 1e-6:
                raise ValueError(
                    "spacing_A[%d]=%g inconsistent with cell row norm / shape (%g, rel %g)"
                    % (i, got, exp, rel)
                )
        return self

    def to_npz(self, path: Path) -> None:
        """Persist the density grid to ``path`` as a single ``.npz`` archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            grid=self.grid,
            shape=np.asarray(self.shape, dtype=np.int64),
            cell_A=self.cell_A,
            spacing_A=np.asarray(self.spacing_A, dtype=np.float64),
            temperature_K=np.asarray(self.temperature_K, dtype=np.float64),
            gas=np.asarray(self.gas),
            normalisation=np.asarray(self.normalisation),
            smoothing_sigma_A=np.asarray(self.smoothing_sigma_A, dtype=np.float64),
            n_source_samples=np.asarray(self.n_source_samples, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(self.metadata)),
        )

    @classmethod
    def from_npz(cls, path: Path) -> DensityGrid:
        """Load a density grid written by :meth:`to_npz`."""
        with np.load(Path(path), allow_pickle=False) as f:
            metadata: dict[str, Any] = json.loads(str(f["metadata_json"]))
            shape_arr = np.asarray(f["shape"]).tolist()
            spacing_arr = np.asarray(f["spacing_A"]).tolist()
            return cls(
                grid=f["grid"],
                shape=(int(shape_arr[0]), int(shape_arr[1]), int(shape_arr[2])),
                cell_A=f["cell_A"],
                spacing_A=(float(spacing_arr[0]), float(spacing_arr[1]), float(spacing_arr[2])),
                temperature_K=float(f["temperature_K"]),
                gas=str(f["gas"]),
                normalisation=str(f["normalisation"]),  # type: ignore[arg-type]
                smoothing_sigma_A=float(f["smoothing_sigma_A"]),
                n_source_samples=int(f["n_source_samples"]),
                metadata=metadata,
            )


# ---------------------------------------------------------------------------
# T008 — SymmetryGroup
# ---------------------------------------------------------------------------


_ALLOWED_UNCERTAINTY_FLAGS: frozenset[str] = frozenset(
    {
        "tolerance_ambiguous",
        "low_symmetry_host",
        "defective_structure",
        "strained_structure",
        "partial_match",
        "energy_mismatch",
    }
)
_REQUIRED_TOL_KEYS: frozenset[str] = frozenset(
    {"symprec", "angle_tolerance_deg", "basin_match_tol_A", "energy_match_tol_kJmol"}
)


class SymmetryGroup(BaseModel):
    """A set of symmetry-equivalent adsorption basins under a host space group."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: int = Field(ge=0)
    member_basin_ids: tuple[int, ...]
    space_group_symbol: str
    space_group_number: int = Field(ge=1, le=230)
    n_operations_used: int = Field(ge=1)
    tolerances: dict[str, float]
    grouping_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_flags: tuple[str, ...] = ()
    notes: str = ""

    @field_validator("member_basin_ids")
    @classmethod
    def _validate_members(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("member_basin_ids must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError(f"member_basin_ids contains duplicates: {v}")
        if list(v) != sorted(v):
            raise ValueError(f"member_basin_ids must be sorted ascending: {v}")
        for i, b in enumerate(v):
            if int(b) < 0:
                raise ValueError(f"member_basin_ids[{i}]={b} must be >= 0")
        return v

    @field_validator("tolerances")
    @classmethod
    def _validate_tolerances(cls, v: dict[str, float]) -> dict[str, float]:
        missing = _REQUIRED_TOL_KEYS - set(v)
        if missing:
            raise ValueError(f"tolerances missing required keys: {sorted(missing)}")
        for k in _REQUIRED_TOL_KEYS:
            val = float(v[k])
            if val <= 0.0:
                raise ValueError(f"tolerances[{k!r}]={val} must be > 0")
        return v

    @field_validator("uncertainty_flags")
    @classmethod
    def _validate_flags(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for f in v:
            if f not in _ALLOWED_UNCERTAINTY_FLAGS:
                raise ValueError(
                    f"uncertainty_flag {f!r} not in {sorted(_ALLOWED_UNCERTAINTY_FLAGS)}"
                )
        return v

    @model_validator(mode="after")
    def _validate_low_confidence_explained(self) -> SymmetryGroup:
        if self.grouping_confidence < 0.5 and not self.uncertainty_flags:
            raise ValueError(
                "grouping_confidence=%g < 0.5 requires non-empty uncertainty_flags"
                % self.grouping_confidence
            )
        return self


# ---------------------------------------------------------------------------
# T009 — PerturbationSpec
# ---------------------------------------------------------------------------


PerturbationKind = Literal["affine", "isotropic", "uniaxial", "atom_removal"]


class PerturbationSpec(BaseModel):
    """Schema for affine/isotropic/uniaxial strain or curated atom-removal defects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: PerturbationKind
    strain_matrix: list[list[float]] | None = None
    magnitude: float | None = None
    axis: Literal["a", "b", "c"] | None = None
    removed_atom_indices: list[int] | None = None
    label: str
    notes: str | None = None

    @field_validator("strain_matrix")
    @classmethod
    def _validate_strain_shape(
        cls, v: list[list[float]] | None
    ) -> list[list[float]] | None:
        if v is None:
            return None
        if len(v) != 3 or any(len(r) != 3 for r in v):
            raise ValueError(f"strain_matrix must be 3x3; got {len(v)} rows")
        return [[float(x) for x in row] for row in v]

    @field_validator("removed_atom_indices")
    @classmethod
    def _validate_indices(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return None
        if not v:
            raise ValueError("removed_atom_indices must be non-empty when present")
        for i, idx in enumerate(v):
            if int(idx) < 0:
                raise ValueError(f"removed_atom_indices[{i}]={idx} must be >= 0")
        if len(set(v)) != len(v):
            raise ValueError(f"removed_atom_indices contains duplicates: {v}")
        return [int(x) for x in v]

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> PerturbationSpec:
        kind = self.kind
        if kind == "affine":
            if self.strain_matrix is None:
                raise ValueError("kind='affine' requires strain_matrix")
            if self.magnitude is not None or self.axis is not None or self.removed_atom_indices is not None:
                raise ValueError("kind='affine' must not set magnitude/axis/removed_atom_indices")
        elif kind == "isotropic":
            if self.magnitude is None:
                raise ValueError("kind='isotropic' requires magnitude")
            if self.strain_matrix is not None or self.axis is not None or self.removed_atom_indices is not None:
                raise ValueError("kind='isotropic' must not set strain_matrix/axis/removed_atom_indices")
        elif kind == "uniaxial":
            if self.magnitude is None or self.axis is None:
                raise ValueError("kind='uniaxial' requires magnitude and axis")
            if self.strain_matrix is not None or self.removed_atom_indices is not None:
                raise ValueError("kind='uniaxial' must not set strain_matrix/removed_atom_indices")
        elif kind == "atom_removal":
            if self.removed_atom_indices is None:
                raise ValueError("kind='atom_removal' requires removed_atom_indices")
            if self.strain_matrix is not None or self.magnitude is not None or self.axis is not None:
                raise ValueError(
                    "kind='atom_removal' must not set strain_matrix/magnitude/axis"
                )
        return self


# ---------------------------------------------------------------------------
# T010 — RobustnessMetrics
# ---------------------------------------------------------------------------


class RobustnessMetrics(BaseModel):
    """Pristine-vs-perturbed comparison metrics with graceful KH/Qads fallback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta_ln_KH: float | None = None
    delta_Qads_kJmol: float | None = None
    basin_count_pristine: int = Field(ge=0)
    basin_count_perturbed: int = Field(ge=0)
    basin_count_change: int
    basin_persistence_fraction: float = Field(ge=0.0, le=1.0)
    basin_splitting_count: int = Field(ge=0)
    mean_basin_displacement_A: float = Field(ge=0.0)
    accessibility_change: float = Field(ge=-1.0, le=1.0)
    ambiguity_flags: list[str] = Field(default_factory=list)
    missing_data_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_consistency(self) -> RobustnessMetrics:
        expected = self.basin_count_perturbed - self.basin_count_pristine
        if self.basin_count_change != expected:
            raise ValueError(
                "basin_count_change=%d does not equal perturbed - pristine (%d - %d = %d)"
                % (
                    self.basin_count_change,
                    self.basin_count_perturbed,
                    self.basin_count_pristine,
                    expected,
                )
            )
        flags = list(self.missing_data_flags)
        if self.delta_ln_KH is None and "KH_unavailable" not in flags:
            flags.append("KH_unavailable")
        if self.delta_Qads_kJmol is None and "Qads_unavailable" not in flags:
            flags.append("Qads_unavailable")
        # frozen: mutate via __dict__ (Pydantic v2 lets model_validator mode='after' rewrite fields)
        object.__setattr__(self, "missing_data_flags", flags)
        return self


# ---------------------------------------------------------------------------
# T011 — RobustnessReport
# ---------------------------------------------------------------------------


class RobustnessReport(BaseModel):
    """Aggregates one pristine atlas vs one or more perturbed atlas comparisons."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str
    structure_id: str
    gas: str
    temperature_K: float = Field(gt=0.0)
    pristine_run_id: str
    perturbations: list[PerturbationSpec] = Field(min_length=1)
    metrics_per_perturbation: list[RobustnessMetrics]
    summary: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: Literal["1"] = "1"

    @field_validator("gas")
    @classmethod
    def _validate_gas(cls, v: str) -> str:
        if v not in ALLOWED_GASES_V1:
            raise ValueError(
                f"gas={v!r} not in v1 allowed set {sorted(ALLOWED_GASES_V1)}"
            )
        return v

    @model_validator(mode="after")
    def _validate_alignment(self) -> RobustnessReport:
        if len(self.perturbations) != len(self.metrics_per_perturbation):
            raise ValueError(
                "perturbations (n=%d) and metrics_per_perturbation (n=%d) must be equal length"
                % (len(self.perturbations), len(self.metrics_per_perturbation))
            )
        return self


# ---------------------------------------------------------------------------
# T012 — RunManifest
# ---------------------------------------------------------------------------


_HEX_CHARS = frozenset("0123456789abcdef")


def _is_lowercase_hex_sha256(s: str) -> bool:
    return len(s) == 64 and all(c in _HEX_CHARS for c in s)


class RunManifest(BaseModel):
    """Provenance record for one widom-atlas run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: Literal["1"] = "1"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    package_version: str
    python_version: str
    platform: str
    dependency_versions: dict[str, str]
    structure_id: str
    structure_source: str | None = None
    structure_sha256: str
    input_samples_sha256: str
    gas: Literal["CO2", "N2", "CH4"]
    temperature_K: float = Field(gt=0.0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    dataset_source: str | None = None
    dataset_license: str | None = None
    output_paths: dict[str, str] = Field(default_factory=dict)

    @field_validator("structure_sha256", "input_samples_sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if not _is_lowercase_hex_sha256(v):
            raise ValueError(
                f"sha256 must be 64 lowercase hex chars; got len={len(v)} value={v!r}"
            )
        return v

    @field_validator("parameters", "dependency_versions")
    @classmethod
    def _validate_json_serialisable(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"value is not JSON-serialisable: {exc}") from exc
        return v
