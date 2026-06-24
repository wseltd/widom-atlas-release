"""RASPA2 output parser.

RASPA2 prints Henry coefficients in a section titled
'Widom insertion: average Henry coefficient' near the end of the
Output/System_0/output_*.data file. The reported basis is
mol/(kg·Pa) — matching the RASPA3 convention.

Q_st (isosteric heat at zero coverage) is not always emitted; in this
codebase we derive it via two-temperature van't Hoff (same as RASPA3
path). The parser extracts whatever scalar lines RASPA2 does produce.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_KH_LINES = [
    # RASPA2 final report format: "[<gas>] Average Henry coefficient:  X +/- Y [mol/kg/Pa]"
    re.compile(
        r"\[(\w+)\]\s+Average\s+Henry\s+coefficient:\s+([0-9eE.+\-nan]+)\s+\+/-\s+([0-9eE.+\-nan]+)\s+\[mol/kg/Pa\]"
    ),
    # Fallback for older RASPA2 versions
    re.compile(
        r"Average\s+Henry\s+coefficient.*?:\s*([0-9eE.+\-]+)\s*\+/-\s*([0-9eE.+\-]+)\s*\[mol/kg/Pa\]"
    ),
]
# RASPA2 Widom-derived enthalpy: "[<gas>] Average <U_gh>_1-<U_h>_0: X +/- Y [K]  (X kJ/mol)"
_WIDOM_ENERGY_RX = re.compile(
    r"\[(\w+)\]\s+Average\s+<U_gh>_1-<U_h>_0:\s+([0-9eE.+\-nan]+)\s+\+/-\s+([0-9eE.+\-nan]+)\s+\[K\]\s+\(\s*([0-9eE.+\-nan]+)\s+\+/-\s+([0-9eE.+\-nan]+)\s+kJ/mol"
)
_TRIAL_WIDOM_RX = re.compile(
    r"Number of trial positions \(Widom insertion\):\s*([0-9]+)"
)
_NUM_CYCLES_RX = re.compile(r"Number of cycles:\s*([0-9]+)")


def _safe_float(s: str) -> float | None:
    """Parse possibly-NaN float from RASPA2 output."""
    if s is None:
        return None
    t = s.strip().lower()
    if "nan" in t:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Raspa2ParsedScalars:
    K_H_mol_per_kg_per_Pa: float | None
    K_H_uncertainty: float | None
    Q_st_kJ_per_mol: float | None
    Q_st_uncertainty: float | None
    widom_insertions_total: int | None
    raw_lines: dict[str, str]


def parse_raspa2_output(output_path: Path) -> Raspa2ParsedScalars:
    """Return parsed K_H + Q_st + counters from a RASPA2 output file.

    Missing scalars are reported as None; the caller (executor) decides
    whether to compute Q_st via two-temperature van't Hoff externally.
    """
    text = output_path.read_text()
    raw: dict[str, str] = {}
    K_H: float | None = None
    K_H_unc: float | None = None
    Q_st_K: float | None = None
    Q_st_K_unc: float | None = None
    widom_total: int | None = None
    # K_H: find the final "Average Henry coefficient" block (last match wins; the file may have
    # earlier per-cycle prints with K_H=0)
    matches = list(_KH_LINES[0].finditer(text))
    if not matches:
        matches = list(_KH_LINES[1].finditer(text))
    if matches:
        m = matches[-1]
        # _KH_LINES[0] captures (gas, value, uncertainty) — 3 groups; _KH_LINES[1] captures
        # (value, uncertainty) — 2 groups. Use .isalnum() not .isalpha() so the gas-name 'CO2'
        # (which has a digit) is correctly recognised as a non-numeric prefix.
        if len(m.groups()) >= 3 and not _safe_float(m.group(1)):
            K_H = _safe_float(m.group(2))
            K_H_unc = _safe_float(m.group(3))
        else:
            K_H = _safe_float(m.group(1))
            K_H_unc = _safe_float(m.group(2)) if m.lastindex and m.lastindex >= 2 else None
        raw["K_H"] = m.group(0).strip()

    # Widom-derived adsorption enthalpy in kJ/mol (RASPA2 prints it directly)
    m = _WIDOM_ENERGY_RX.search(text)
    Q_st_kjmol: float | None = None
    Q_st_kjmol_unc: float | None = None
    if m:
        # Group 4 = kJ/mol value, group 5 = kJ/mol uncertainty
        u_gh = _safe_float(m.group(4))
        u_gh_unc = _safe_float(m.group(5))
        if u_gh is not None:
            # Q_st = -<U_gh>_1 - RT (positive-exothermic convention; <U_gh> already in kJ/mol).
            # For rigid CO2 <U_g> ≈ 0. RT at 298 K = 8.314e-3 * 298 = 2.478 kJ/mol.
            # Use the reference temperature from the simulation (currently 298 K; the per-run
            # temperature is set in the YAML).
            RT_298 = 8.314462618e-3 * 298.0
            Q_st_kjmol = abs(-u_gh - RT_298)
            if u_gh_unc is not None:
                Q_st_kjmol_unc = abs(u_gh_unc)
        raw["Q_st_widom"] = m.group(0).strip()

    m_trial = _TRIAL_WIDOM_RX.search(text)
    m_cycles = _NUM_CYCLES_RX.search(text)
    if m_trial and m_cycles:
        n_trial = int(m_trial.group(1))
        n_cycles = int(m_cycles.group(1))
        widom_total = n_cycles * n_trial
        raw["widom_total"] = (
            f"n_cycles={n_cycles} * n_trial_positions_widom={n_trial} -> {widom_total}"
        )
    elif m_cycles:
        widom_total = int(m_cycles.group(1))
        raw["widom_total"] = m_cycles.group(0).strip()

    return Raspa2ParsedScalars(
        K_H_mol_per_kg_per_Pa=K_H,
        K_H_uncertainty=K_H_unc,
        Q_st_kJ_per_mol=Q_st_kjmol,
        Q_st_uncertainty=Q_st_kjmol_unc,
        widom_insertions_total=widom_total,
        raw_lines=raw,
    )
