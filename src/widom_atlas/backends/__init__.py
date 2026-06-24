"""widom-atlas backend layer — pluggable calculators / sample sources for Widom insertion.

The package's v1 ships three backends:

- ``toy_lj``: ASE LennardJones with ε=0.01 eV, σ=3.0 Å (the previous default; smoke-only,
  not parameterised for any chemistry; left in place because the verdict §G calls
  for ASE LJ as the smoke-test calculator).
- ``parameterised_lj``: multi-element Lennard-Jones with **published, citable
  force-field parameters** — TraPPE for CO2/N2/CH4 adsorbates and UFF for the
  full periodic table of framework atoms — combined via Lorentz-Berthelot
  mixing rules. This is the v0.2 path that escapes toy-LJ behaviour without
  pulling in a heavyweight engine like LAMMPS, OpenMM, or RASPA.
- ``external_samples``: ingest externally generated Widom samples from any
  engine (RASPA3, LAMMPS, kUPS, custom), letting power users bring their own
  serious calculator without modifying widom-atlas.

The backend is selected via ``--backend`` on the CLI and recorded in the
manifest of every run.
"""

from __future__ import annotations

from .base import (
    AtlasBackend,
    BackendName,
    BackendOutput,
    available_backends,
    get_backend,
)
from .schema import (
    SAMPLE_FORMAT_VERSION,
    BackendCategory,
    CitationEntry,
    ExternalSampleManifest,
    ForceFieldDescriptor,
)
from .units import (
    ALLOWED_ENERGY_UNITS,
    KELVIN_TO_EV,
    KJ_PER_MOL_PER_EV,
    KJ_PER_MOL_PER_KCAL_MOL,
    to_eV,
)

__all__ = [
    "ALLOWED_ENERGY_UNITS",
    "KELVIN_TO_EV",
    "KJ_PER_MOL_PER_EV",
    "KJ_PER_MOL_PER_KCAL_MOL",
    "SAMPLE_FORMAT_VERSION",
    "AtlasBackend",
    "BackendCategory",
    "BackendName",
    "BackendOutput",
    "CitationEntry",
    "ExternalSampleManifest",
    "ForceFieldDescriptor",
    "available_backends",
    "get_backend",
    "to_eV",
]
