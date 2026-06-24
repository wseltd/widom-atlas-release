"""Cached real-material structures for Layer 2 integration tests.

Where a license-clean reference CIF is already on disk (committed by the
operator), we load it. Where it is not, we synthesise a deterministic
stand-in using :func:`ase.build.bulk` for the diamond cubic Si structure;
this is not the real MOF/zeolite topology, only a periodic crystal that
exercises the full widom-atlas pipeline path.

NB: using a stand-in is **explicitly tagged** in the metadata of the
returned :class:`AtlasInput` so reports never claim chemical fidelity from
this fixture path.
"""

from widom_atlas.tests.fixtures_loader_passthrough import (
    REAL_BENCHMARK_IDS,
    load_real_material,
)

__all__ = ["REAL_BENCHMARK_IDS", "load_real_material"]
