"""RASPA2 v2.0.50 backend for v04.

Used for branch 1a (Mg-MOF-74 Lin/Mercado Buckingham). RASPA3 v3.0.29 JSON
force-field format silently drops `"type": "buckingham"` entries; RASPA2's
classic `.def` format has native `BUCKINGHAM` support, which is the exact
Lin/Mercado functional form `p_0·exp(-p_1·r) - p_2/r^6`.

This module is parallel to `widom_atlas.v04.raspa3.*` — same role,
different backend. The branches/executor dispatches to whichever backend
the YAML calls for.

Erratum trail: introduced 2026-05-17 to unblock 1a per V04_BLOCKER_REPORT.md.
The RASPA2 binary is sha256-pinned via `raspa2_binary.py` so the audit cannot
silently switch versions.
"""
from __future__ import annotations
