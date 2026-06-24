"""T018: per-branch electrostatics configurator.

Derives a typed ElectrostaticsConfig from the locked YAML. Reads
`electrostatics_per_branch` fields (cutoff, lj_treatment, lj_tail_correction,
ewald_via_raspa3) and combines with global defaults (mixing_rule).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElectrostaticsConfig:
    branch_id: str
    ewald_via_raspa3: bool
    direct_cutoff_angstrom: float
    lj_treatment: str  # "shifted_truncated" or "truncated_only"
    lj_tail_correction: bool
    mixing_rule: str = "lorentz_berthelot"

    def __post_init__(self) -> None:
        if self.lj_treatment not in ("shifted_truncated", "truncated_only"):
            raise ValueError(f"unsupported lj_treatment: {self.lj_treatment}")


def derive_electrostatics_config(branch: dict, mixing_rule: str) -> ElectrostaticsConfig:
    eb = branch.get("electrostatics_per_branch") or {}
    return ElectrostaticsConfig(
        branch_id=branch["branch_id"],
        ewald_via_raspa3=bool(eb.get("ewald_via_raspa3", False)),
        direct_cutoff_angstrom=float(eb.get("direct_cutoff_angstrom", 12.0)),
        lj_treatment=str(eb.get("lj_treatment", "shifted_truncated")),
        lj_tail_correction=bool(eb.get("lj_tail_correction", False)),
        mixing_rule=mixing_rule,
    )
