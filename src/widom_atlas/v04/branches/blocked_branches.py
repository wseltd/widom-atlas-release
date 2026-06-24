"""Branches that are scientifically BLOCKED on the RASPA3 v3.0.29 backend.

Per operator directive: 1a Lin/Mercado Mg-MOF-74 and 1b Dzubak Mg-MOF-74 use
pair-potential functional forms that RASPA3 v3.0.29 JSON force_field.json
cannot represent. The native RASPA3 JSON parser accepts only
`lennard-jones`, `morse`, and `none` types; entries with `"type": "buckingham"`
are silently dropped by the parser (verified by direct probe of
~/miniconda3/envs/raspa3/bin/raspa3 with a Buckingham
BinaryInteractions entry — the cross-pair was missing from the printed
"Force field status" table without any error).

We therefore mark these branches BLOCKED rather than silently substitute LJ.
The blocker is a backend capability issue, not a force-field error.
"""
from __future__ import annotations

BLOCKED_BRANCHES: dict[str, dict[str, str]] = {
    "1a": {
        "reason": (
            "RASPA3 v3.0.29 JSON force_field.json does not support the Buckingham "
            "potential V(r) = A·exp(-B·r) - C/r^6 prescribed by Lin/Mercado Model 4. "
            "BinaryInteractions entries with type=buckingham are silently dropped by "
            "the parser; SelfInteractions entries with type=buckingham yield zero "
            "LJ parameters. No tabulated/custom pair-potential JSON path exists in "
            "v3.0.29. Verified by direct probe of the installed binary."
        ),
        "prescribed_form": "BUCKINGHAM_A_EXP_C6 (Lin 2014 / Mercado Model 4)",
        "required_action": (
            "Resolve at RASPA3 layer: upgrade RASPA3 to a release that exposes "
            "Buckingham via JSON, OR add a tabulated/spline pair-potential JSON "
            "path, OR switch the strict tier backend to RASPA2/4 with the original "
            "force_field.def file `docs/research/dataset-research-for-v0.4/9/"
            "raspa_force_field.def`."
        ),
    },
    "1b": {
        "reason": (
            "RASPA3 v3.0.29 has no native representation for the Dzubak "
            "two-attraction form V(r) = A·exp(-B·r) - C/r^5 - D/r^6. Buckingham "
            "(C/r^6 only) is the closest available form but cannot recover the "
            "C/r^5 mid-range attraction term that Dzubak 2012 specifically "
            "introduced for Mg-MOF-74 + CO2."
        ),
        "prescribed_form": "DZUBAK_A_EXP_C5_D6 (Dzubak 2012)",
        "required_action": (
            "Same as 1a — backend-level fix required."
        ),
    },
}


def blocked_reason(branch_id: str) -> dict[str, str] | None:
    """Return a blocked-branch reason dict, or None if not blocked."""
    return BLOCKED_BRANCHES.get(branch_id)
