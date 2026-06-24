"""widom-atlas data registry — provenance-tracked pointers to public datasets,
literature reference values, crystallographic site positions, and v0.4 / v0.5
validation thresholds.

The registry is a *provenance index*, not a data archive. Per the v0.3 backend
strategy verdict, widom-atlas does not bundle force-field parameters, charge
tables, or large MOF datasets whose redistribution rights have not been
verified. Instead, every entry here records:

- the source URL and DOI,
- the licence and redistribution status,
- the file format and an expected sha256 (placeholder until the operator
  verifies their download),
- the citation,
- a structured ``content_summary``,
- a ``last_verified`` timestamp (null until set by the operator).

When the operator drops a downloaded file into ``benchmarks/cache/`` (or wherever
the registry entry says) and runs ``widom-atlas data status``, the registry
checks the on-disk file against the recorded sha256 (if any) and confirms the
dataset is "present, verified" or "missing"; the registry never auto-fetches
from the internet.

Module map
==========

- :mod:`~widom_atlas.data_registry.schema` — Pydantic v2 schemas
  (``extra="forbid"``, ``frozen=True``).
- :mod:`~widom_atlas.data_registry.registry` — concrete registry entries loaded
  from ``data/{datasets,scalars,sites,thresholds}.yaml``.
- ``data/datasets.yaml`` — public dataset registry (CRAFTED, CoRE-MOF, QMOF,
  PACMOF service, MEPO-ML/ARC-MOF, MOFSimBench).
- ``data/scalars.yaml`` — literature reference KH / Q_ads tables, per
  ``(material, gas, T_K)``.
- ``data/sites.yaml`` — crystallographic gas-loaded site positions, per
  ``(material, gas, label)``.
- ``data/thresholds.yaml`` — v0.4 minimum + v0.5 broader validation thresholds.
"""

from __future__ import annotations

from .registry import (
    list_datasets,
    list_scalar_references,
    list_site_references,
    load_dataset,
    load_scalar_reference,
    load_site_reference,
    load_validation_thresholds,
)
from .schema import (
    MOFXDB_FORCE_FIELD_ID_TO_NAME,
    DatasetRegistryEntry,
    MofxdbProvenanceKind,
    ScalarReferenceEntry,
    SiteReferenceEntry,
    ValidationThresholds,
    mofxdb_provenance_kind_from_database,
)

__all__ = [
    "MOFXDB_FORCE_FIELD_ID_TO_NAME",
    "DatasetRegistryEntry",
    "MofxdbProvenanceKind",
    "ScalarReferenceEntry",
    "SiteReferenceEntry",
    "ValidationThresholds",
    "list_datasets",
    "list_scalar_references",
    "list_site_references",
    "load_dataset",
    "load_scalar_reference",
    "load_site_reference",
    "load_validation_thresholds",
    "mofxdb_provenance_kind_from_database",
]
