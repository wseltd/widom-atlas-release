"""Verified unit conversions between energy / temperature / length units that
turn up in different external engines (RASPA3 in K, TraPPE in K, OpenMM /
DFT in kJ/mol or eV, classical FF papers in kcal/mol).

Constants are the CODATA 2018 values, taken from
``scipy.constants`` and verified against the conversion table in
``implementation-verdict-continuation.txt §10``:

- ``E_kJ_mol  = E_K * R / 1000``  with R = 8.314462618 J/(mol·K)
- ``E_eV      = E_kJ_mol / 96.48533212``
- ``E_kJ_mol  = E_kcal_mol * 4.184``

This module is the **only** place the package converts energies. Every
external-samples ingest path goes through it; if a sample arrives without
declared units, the call must :func:`raise_missing_units` rather than
guess.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np

# CODATA 2018, kept as Final scalars so they can be cross-checked.
GAS_CONSTANT_R_J_PER_MOL_K: Final[float] = 8.314462618  # exact
KJ_PER_MOL_PER_EV: Final[float] = 96.48533212  # exact
KJ_PER_MOL_PER_KCAL_MOL: Final[float] = 4.184  # exact (thermochemical kcal)
KELVIN_TO_EV: Final[float] = 8.617333262e-5  # k_B in eV/K (CODATA 2018)

EnergyUnit = Literal["K", "eV", "kJ_mol", "kcal_mol"]
ALLOWED_ENERGY_UNITS: Final[tuple[EnergyUnit, ...]] = ("K", "eV", "kJ_mol", "kcal_mol")


def to_eV(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert an array of energies in ``unit`` to eV (the package's internal unit).

    Args:
        values: array-like of energies in the given unit.
        unit: one of ``{"K", "eV", "kJ_mol", "kcal_mol"}``.

    Raises:
        ValueError: if ``unit`` is not in :data:`ALLOWED_ENERGY_UNITS`.
    """
    if unit not in ALLOWED_ENERGY_UNITS:
        raise ValueError(
            f"unknown energy unit {unit!r}; supported: {ALLOWED_ENERGY_UNITS}. "
            "External samples must declare units explicitly; see widom_atlas.backends.schema."
        )
    arr = np.asarray(values, dtype=np.float64)
    if unit == "eV":
        return arr
    if unit == "K":
        return arr * KELVIN_TO_EV
    if unit == "kJ_mol":
        return arr / KJ_PER_MOL_PER_EV
    if unit == "kcal_mol":
        return arr * KJ_PER_MOL_PER_KCAL_MOL / KJ_PER_MOL_PER_EV
    raise AssertionError(f"unreachable: unit={unit!r}")  # pragma: no cover


def kelvin_to_kJ_per_mol(values: np.ndarray) -> np.ndarray:
    """Convert RASPA-style energies in K (i.e. ε/k_B) to kJ/mol.

    ``E_kJ_mol = E_K × R / 1000``. Useful for human-readable Q_ads tables.
    """
    arr = np.asarray(values, dtype=np.float64)
    return arr * GAS_CONSTANT_R_J_PER_MOL_K / 1000.0


def raise_missing_units(field_name: str) -> None:
    """Fail fast with a clear message when an external sample omits units.

    Per ``implementation-verdict-continuation.txt`` §"Implement external sample
    schema": *"Do not silently assume units. If units are missing, fail."*
    """
    raise ValueError(
        f"external sample missing required unit declaration for {field_name!r}; "
        "supply 'energy_unit' (one of 'K', 'eV', 'kJ_mol', 'kcal_mol') in the manifest. "
        "widom-atlas refuses to guess — see widom_atlas.backends.units."
    )


__all__ = [
    "ALLOWED_ENERGY_UNITS",
    "GAS_CONSTANT_R_J_PER_MOL_K",
    "KELVIN_TO_EV",
    "KJ_PER_MOL_PER_EV",
    "KJ_PER_MOL_PER_KCAL_MOL",
    "EnergyUnit",
    "kelvin_to_kJ_per_mol",
    "raise_missing_units",
    "to_eV",
]
