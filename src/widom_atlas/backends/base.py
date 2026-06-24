"""Backend protocol + factory.

A backend's only job is to produce an :class:`AtlasInput` from a structure +
gas + temperature + insertion-count specification, plus metadata describing
which calculator / engine produced the samples. This abstraction lets
downstream pipeline code (basin extraction, density grid, robustness,
launch reports) stay calculator-agnostic — the same atlas analysis runs on
toy-LJ outputs, TraPPE+UFF outputs, RASPA3 outputs, or kUPS outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

BackendName = Literal[
    "toy_lj",
    "parameterised_lj",
    "external_samples",
    "user_parameterised_coulomb_lj",
]


@dataclass(frozen=True)
class BackendOutput:
    """Result of one backend invocation.

    Attributes:
        atlas_input: A fully populated :class:`AtlasInput` ready for the
            atlas pipeline.
        backend_label: Short human-readable label
            (e.g. ``"parameterised_lj (TraPPE+UFF, Lorentz-Berthelot, cutoff=12A)"``).
        calculator_label: ASE calculator description
            (e.g. ``"widom_atlas.backends.parameterised_lj.ParameterisedLJCalculator"``).
        samples_origin: One of ``"cuspai_widom"``, ``"external_samples"``,
            ``"synthetic_toy_lj"``.
        provenance: Free-form dict (parameter-pack DOIs, mixing rule,
            cutoff, version metadata).
    """

    atlas_input: Any
    backend_label: str
    calculator_label: str
    samples_origin: str
    provenance: dict[str, Any]


@runtime_checkable
class AtlasBackend(Protocol):
    """Protocol every backend implements.

    The backend takes a structure, gas, temperature, and number of
    insertions, and returns an :class:`AtlasInput` containing the samples
    plus metadata describing what produced them.

    Backends are invoked by the benchmark runner, the convergence
    workflow, and the analyse-samples CLI. They are pure: same inputs +
    same seed → same output.
    """

    @property
    def name(self) -> str: ...

    def generate(
        self,
        *,
        structure: Any,
        gas: str,
        temperature_K: float,
        n_samples: int,
        seed: int,
        material_id: str,
        material_source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> BackendOutput: ...


def available_backends() -> tuple[BackendName, ...]:
    """Return the registry of backend names usable from the CLI."""
    return (
        "toy_lj",
        "parameterised_lj",
        "external_samples",
        "user_parameterised_coulomb_lj",
    )


def get_backend(
    name: BackendName,
    *,
    external_samples_path: Path | None = None,
    cutoff_A: float = 12.0,
    user_parameter_file: Path | None = None,
    allow_neutral_fallback: bool = False,
    wolf_alpha_inv_A: float = 0.20,
    external_manifest_path: Path | None = None,
) -> AtlasBackend:
    """Return a fresh instance of the named backend.

    Construction kwargs not relevant to the chosen backend are ignored.
    """
    if name == "toy_lj":
        from .toy_lj import ToyLennardJonesBackend
        return ToyLennardJonesBackend()
    if name == "parameterised_lj":
        from .parameterised_lj import ParameterisedLJBackend
        return ParameterisedLJBackend(cutoff_A=cutoff_A)
    if name == "external_samples":
        from .external import ExternalSamplesBackend
        if external_samples_path is None:
            raise ValueError(
                "external_samples backend requires --external-samples-path "
                "pointing at a .npz of pre-computed Widom samples"
            )
        return ExternalSamplesBackend(
            samples_path=external_samples_path,
            manifest_path=external_manifest_path,
        )
    if name == "user_parameterised_coulomb_lj":
        from .user_parameterised import UserChargeAwareBackend
        if user_parameter_file is None:
            raise ValueError(
                "user_parameterised_coulomb_lj backend requires --params PATH "
                "pointing at a JSON parameter file (see widom_atlas.backends.user_parameterised.UserParameterFile)"
            )
        return UserChargeAwareBackend(
            parameter_file=user_parameter_file,
            cutoff_A=cutoff_A,
            wolf_alpha_inv_A=wolf_alpha_inv_A,
            allow_neutral_fallback=allow_neutral_fallback,
        )
    raise ValueError(
        f"unknown backend {name!r}; choose from {available_backends()}"
    )


__all__ = [
    "AtlasBackend",
    "BackendName",
    "BackendOutput",
    "available_backends",
    "get_backend",
]
