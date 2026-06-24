"""Cross-validation tests: official TraPPE database records vs in-repo transcriptions.

The operator pulled TraPPE-CO2 (record 116), TraPPE-UA methane (record 1),
and TraPPE-EH methane (record 164) from http://trappe.oit.umn.edu/ on
2026-06-01 and archived them under
docs/research/dataset-research-for-v0.4/trappe_database_official/.

These tests guard against transcription drift: every in-repo TraPPE-CO2
or TraPPE-UA CH4 parameter must match the official record.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from widom_atlas.v04.native.maia_2023_loader import (
    TRAPPE_CO2_BOND_LENGTH_A,
    TRAPPE_CO2_CHARGES_E,
    TRAPPE_CO2_SELF_LJ,
)

REPO = Path(__file__).resolve().parents[2]
TRAPPE_DIR = REPO / "docs/research/dataset-research-for-v0.4/trappe_database_official"


def _read_trappe_parameters(path: Path) -> list[dict]:
    """Read a trappe_parameters_N.csv and return the (pseudo)atom rows."""
    rows = []
    in_atom_block = False
    with path.open() as fp:
        for line in fp:
            line = line.rstrip("\n")
            if line.startswith("#,(pseudo)atom"):
                in_atom_block = True
                continue
            if line.startswith("#,stretch") or line.startswith("#,bend"):
                in_atom_block = False
                continue
            if in_atom_block and line and not line.startswith("#"):
                fields = next(iter(csv.reader([line])))
                if len(fields) >= 6 and fields[0] and fields[0] != "":
                    try:
                        rows.append({
                            "idx": int(fields[0]),
                            "atom": fields[1].strip(),
                            "type": fields[2].strip(),
                            "epsilon_K": float(fields[3]),
                            "sigma_A": float(fields[4]),
                            "charge_e": float(fields[5]),
                        })
                    except (ValueError, IndexError):
                        continue
    return rows


def _read_trappe_bond_length(path: Path) -> float | None:
    """Read the first stretch length from a trappe_parameters_N.csv."""
    in_stretch = False
    with path.open() as fp:
        for line in fp:
            line = line.rstrip("\n")
            if line.startswith("#,stretch"):
                in_stretch = True
                continue
            if line.startswith("#,bend"):
                in_stretch = False
                continue
            if in_stretch and line and not line.startswith("#"):
                fields = next(iter(csv.reader([line])))
                if len(fields) >= 4 and fields[0] and fields[3]:
                    try:
                        return float(fields[3])
                    except ValueError:
                        continue
    return None


def test_trappe_co2_record_116_matches_maia_2023_table_1_transcription():
    """TraPPE-CO2 official ε, σ, q + bond length must match Maia 2023 Table 1
    transcription exactly."""
    path = TRAPPE_DIR / "trappe_parameters_116.csv"
    assert path.exists(), "TraPPE-CO2 record 116 not archived"
    rows = _read_trappe_parameters(path)
    by_atom: dict[str, dict] = {r["atom"]: r for r in rows}
    assert "C" in by_atom
    assert "O" in by_atom

    # Cross-check vs Maia 2023 Table 1 (used by 3b execution).
    assert by_atom["C"]["epsilon_K"] == pytest.approx(TRAPPE_CO2_SELF_LJ["C_co2"][0])
    assert by_atom["C"]["sigma_A"] == pytest.approx(TRAPPE_CO2_SELF_LJ["C_co2"][1])
    assert by_atom["C"]["charge_e"] == pytest.approx(TRAPPE_CO2_CHARGES_E["C_co2"])
    assert by_atom["O"]["epsilon_K"] == pytest.approx(TRAPPE_CO2_SELF_LJ["O_co2"][0])
    assert by_atom["O"]["sigma_A"] == pytest.approx(TRAPPE_CO2_SELF_LJ["O_co2"][1])
    assert by_atom["O"]["charge_e"] == pytest.approx(TRAPPE_CO2_CHARGES_E["O_co2"])

    bond = _read_trappe_bond_length(path)
    assert bond == pytest.approx(TRAPPE_CO2_BOND_LENGTH_A)


def test_trappe_ua_ch4_record_1_matches_148K_3p73A_neutral():
    path = TRAPPE_DIR / "trappe_parameters_1.csv"
    assert path.exists()
    rows = _read_trappe_parameters(path)
    by_atom = {r["atom"]: r for r in rows}
    assert "CH4" in by_atom
    assert by_atom["CH4"]["epsilon_K"] == pytest.approx(148.0)
    assert by_atom["CH4"]["sigma_A"] == pytest.approx(3.73)
    assert by_atom["CH4"]["charge_e"] == pytest.approx(0.0)


def test_trappe_zeo_records_NOT_yet_in_repo_4c_6e_blocker_stands():
    """No TraPPE-zeo zeolite framework record is present in the
    trappe_database_official/ archive. 4c and 6e remain blocked
    pending the framework parameters (Si LJ ε/σ + framework charges)."""
    files = list(TRAPPE_DIR.glob("trappe_parameters_*.csv"))
    record_ids = [int(p.stem.split("_")[-1]) for p in files]
    # The 3 records pulled are 1, 116, 164 (CH4 UA, CO2, CH4 EH). All gas-side.
    assert set(record_ids) == {1, 116, 164}
