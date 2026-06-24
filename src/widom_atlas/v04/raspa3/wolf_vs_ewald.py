"""T027: Wolf-vs-Ewald deviation reporter (per locked_strict branch).

Reports the dimensionless deviation between widom-atlas internal Wolf
Henry-excess estimate and RASPA3's Ewald-strict K_H. This is the
"sanity check" line on the audit — large deviations indicate either
- Wolf parameter mismatch
- A different gas-model wiring between the two backends
- A bug in widom-atlas's internal energy fn
The deviation is informational; it does NOT affect the verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WolfEwaldDeviation:
    branch_id: str
    wolf_K_H_mol_per_kg_per_Pa: float | None
    ewald_K_H_mol_per_kg_per_Pa: float | None
    delta_log10: float | None
    note: str


def report_wolf_vs_ewald(
    branch_id: str,
    wolf_K_H: float | None,
    ewald_K_H: float | None,
) -> WolfEwaldDeviation:
    if wolf_K_H is None or ewald_K_H is None or wolf_K_H <= 0 or ewald_K_H <= 0:
        return WolfEwaldDeviation(
            branch_id=branch_id,
            wolf_K_H_mol_per_kg_per_Pa=wolf_K_H,
            ewald_K_H_mol_per_kg_per_Pa=ewald_K_H,
            delta_log10=None,
            note="Wolf/Ewald comparison N/A (at least one value missing or non-positive)",
        )
    delta = math.log10(wolf_K_H / ewald_K_H)
    return WolfEwaldDeviation(
        branch_id=branch_id,
        wolf_K_H_mol_per_kg_per_Pa=wolf_K_H,
        ewald_K_H_mol_per_kg_per_Pa=ewald_K_H,
        delta_log10=delta,
        note=f"Δlog10(K_H_Wolf/K_H_Ewald) = {delta:+.3f}",
    )
