"""T026: RASPA3 output parser.

Extracts K_H (mol/kg/Pa), Q_st (kJ/mol), per-component energies, and any
diagnostic counters from the standard RASPA3 v3.0.29 output file
(output/output_<T>_0.s0.txt). All values are reported with absolute paths
to the line in the text where they were parsed, so the evidence record
preserves traceability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RaspaParsedScalars:
    K_H_mol_per_kg_per_Pa: float | None
    K_H_uncertainty: float | None
    K_H_molec_per_uc_per_Pa: float | None
    Q_st_kJ_per_mol: float | None
    Q_st_uncertainty: float | None
    widom_insertions_total: int | None
    widom_runtime_s: float | None
    raw_lines: dict[str, str]


_KH_RX = re.compile(
    r"Average\s+Henry\s+coefficient:\s+([0-9eE.+-]+)\s+\+/-\s+([0-9eE.+-]+)\s+\[mol/kg/Pa\]"
)
_KH_MOLEC_RX = re.compile(
    r"Average\s+Henry\s+coefficient:\s+([0-9eE.+-]+)\s+\+/-\s+[0-9eE.+-]+\s+\[molec\./uc/Pa\]"
)
_QST_RX = re.compile(
    r"(?:Q_?st|Enthalpy)[^\n]*?([0-9eE.+\-]+)\s*\+/-\s*([0-9eE.+\-]+)\s*\[K\]"
)
# RASPA3 v3.0.29 prints "Excess chemical potential: -8.564e+02 +/- ... [K]"
_MU_EX_RX = re.compile(
    r"Excess\s+chemical\s+potential:\s+([0-9eE.+\-]+)\s+\+/-\s+([0-9eE.+\-]+)\s+\[K\]"
)
_WIDOM_TOTAL_RX = re.compile(r"Widom\s+total:\s+([0-9eE.+-]+)")
_WIDOM_TIME_RX = re.compile(r"Widom\s+([0-9.]+)\s+\[s\]")


def parse_raspa3_output(output_txt: Path) -> RaspaParsedScalars:
    text = output_txt.read_text()
    raw: dict[str, str] = {}
    K_H = None
    K_H_unc = None
    K_H_molec = None
    Q_st = None
    Q_st_unc = None
    widom_total = None
    widom_time = None

    m = _KH_RX.search(text)
    if m:
        K_H = float(m.group(1))
        K_H_unc = float(m.group(2))
        raw["K_H"] = m.group(0)
    m2 = _KH_MOLEC_RX.search(text)
    if m2:
        K_H_molec = float(m2.group(1))
        raw["K_H_molec_per_uc"] = m2.group(0)
    m3 = _QST_RX.search(text)
    if m3:
        Q_K = float(m3.group(1))
        Q_unc_K = float(m3.group(2))
        # Convert K → kJ/mol
        from ..units import energy_K_to_kjmol
        Q_st = abs(energy_K_to_kjmol(Q_K))
        Q_st_unc = abs(energy_K_to_kjmol(Q_unc_K))
        raw["Q_st"] = m3.group(0)
    m4 = _WIDOM_TOTAL_RX.search(text)
    if m4:
        widom_total = int(float(m4.group(1)))
        raw["widom_total"] = m4.group(0)
    m5 = _WIDOM_TIME_RX.search(text)
    if m5:
        widom_time = float(m5.group(1))
        raw["widom_time"] = m5.group(0)
    m6 = _MU_EX_RX.search(text)
    if m6:
        raw["excess_mu_K_value"] = m6.group(1)
        raw["excess_mu_K_uncertainty"] = m6.group(2)
        raw["excess_mu_K_line"] = m6.group(0)

    return RaspaParsedScalars(
        K_H_mol_per_kg_per_Pa=K_H,
        K_H_uncertainty=K_H_unc,
        K_H_molec_per_uc_per_Pa=K_H_molec,
        Q_st_kJ_per_mol=Q_st,
        Q_st_uncertainty=Q_st_unc,
        widom_insertions_total=widom_total,
        widom_runtime_s=widom_time,
        raw_lines=raw,
    )
