"""widom-atlas v0.4 ingest layer — format-specific ingesters that turn external
public-dataset payloads into provenance-rich widom-atlas types.

Every ingester:

- accepts a path or URL;
- validates the payload against a Pydantic schema (extra="forbid");
- records full provenance (source DOI, licence, redistribution status);
- records units explicitly;
- refuses to silently produce ambiguous data.

None of the ingesters in v0.4 fetch from the network during package import.
They are operator-driven: the operator either drops a downloaded archive
into ``benchmarks/cache/<name>/`` and runs the corresponding ``widom-atlas
ingest <kind> ...`` CLI, or — for live REST sources (NIST ISODB, MOFX-DB,
GitHub raw RASPA examples) — passes the URL on the CLI and the ingester
fetches with a clear timeout + verifies the redistribution policy.

The eleven v0.4 ingesters
=========================

- :mod:`~widom_atlas.ingest.raspa3_ff` — RASPA3 ``force_field.json`` +
  ``simulation.json`` + Component JSONs → :class:`~widom_atlas.backends.user_parameterised.UserParameterFile`.
- :mod:`~widom_atlas.ingest.mofxdb` — MOFX-DB JSON records → scalar refs +
  embedded ``simin`` extraction; classifies provenance per sub-database.
- :mod:`~widom_atlas.ingest.nist_isodb` — NIST ISODB REST records →
  experimental scalar refs.
- :mod:`~widom_atlas.ingest.crafted` — CRAFTED tar.xz → scalar refs.
- :mod:`~widom_atlas.ingest.core_mof` — CoRE-MOF zip → structure cache.
- :mod:`~widom_atlas.ingest.core_mof_ddec6` — CoRE-MOF-DFT-2014 DDEC6 →
  charges into UserParameterFile.framework_atom_types.
- :mod:`~widom_atlas.ingest.qmof` — QMOF zip → structures + PACMAN charges.
- :mod:`~widom_atlas.ingest.pacmof2` — PACMOF2 CLI shell-out → charges.
- :mod:`~widom_atlas.ingest.eqeq` — EQeq CLI shell-out → charges.
- :mod:`~widom_atlas.ingest.odac` — ODAC23 LMDB / extxyz (behind
  ``widom-atlas[odac]`` extra; not core).
- :mod:`~widom_atlas.ingest.ccdc_cif` — operator-supplied gas-loaded CIF →
  :class:`~widom_atlas.data_registry.SiteReferenceEntry`.
"""

from __future__ import annotations

__all__: list[str] = []
