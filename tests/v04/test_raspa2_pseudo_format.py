"""Regression tests for the RASPA2 pseudo_atoms.def column-format upgrade.

Discovered 2026-05-17 during the forensic 1a audit: the operator-supplied
Lin/Mercado raspa_pseudo_atoms.def uses a legacy 9-column schema
(type / print / as / scat / mass / charge / B-factor / radii / connectivity)
but RASPA2 v2.0.50 expects a 14-column schema with an `oxidation` column
inserted at position 4, pushing the `mass` column to position 5.

When the legacy file is fed to RASPA2 v2.0.50 directly, the reported
Framework Mass is ~317 g/mol per supercell instead of the correct
~11,645 g/mol — a 36.7× under-count. Since K_H = β · <exp(-βU)> /
(M_framework · N_A), the K_H gets OVER-stated by 36.7×.

This test pins the rewriter behaviour:

  1. The rewritten file contains exactly the v2.0.50 header schema.
  2. Every original (mass, charge) pair is preserved verbatim.
  3. The new `oxidation` column has value 0 for every framework atom.
  4. Total atomic mass sums to the Mg-MOF-74 primitive cell mass (728 amu).

A separate end-to-end test confirms RASPA2 then reports the correct
Framework Mass = 11,645 g/mol for the 4×2×2 supercell.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from widom_atlas.v04.raspa2.input_writer import (
    _rewrite_lin_mercado_pseudo_atoms_to_v2_50_format,
)


@pytest.fixture(scope="module")
def lin_mercado_src() -> Path:
    p = Path("docs/research/dataset-research-for-v0.4/9/raspa_pseudo_atoms.def")
    if not p.exists():
        pytest.skip(f"Lin/Mercado pseudo_atoms.def not present: {p}")
    return p


@pytest.fixture()
def rewritten(lin_mercado_src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "pseudo_atoms.def"
    _rewrite_lin_mercado_pseudo_atoms_to_v2_50_format(lin_mercado_src, dst)
    return dst


def test_rewriter_emits_v2_50_header(rewritten: Path):
    """The rewriter must emit the 14-column header that RASPA2 v2.0.50 expects."""
    text = rewritten.read_text()
    header = next(
        line for line in text.splitlines() if line.strip().startswith("#type")
    )
    expected_columns = (
        "type", "print", "as", "chem", "oxidation",
        "mass", "charge", "polarization", "B-factor",
        "radii", "connectivity", "anisotropic",
        "anisotropic-type", "tinker-type",
    )
    for col in expected_columns:
        assert col in header, f"missing column {col!r} in rewritten header: {header}"


def test_rewriter_preserves_mass_and_charge(lin_mercado_src: Path, rewritten: Path):
    """For every framework atom type, the new mass / charge match the source."""
    src_text = lin_mercado_src.read_text()
    dst_text = rewritten.read_text()

    src_records: dict[str, tuple[float, float]] = {}
    for line in src_text.splitlines():
        fields = re.split(r"\s+", line.strip())
        if len(fields) < 9 or not fields[0].startswith("Mof_"):
            continue
        src_records[fields[0]] = (float(fields[4]), float(fields[5]))

    dst_records: dict[str, tuple[float, float]] = {}
    for line in dst_text.splitlines():
        fields = re.split(r"\s+", line.strip())
        if len(fields) < 14 or not fields[0].startswith("Mof_"):
            continue
        dst_records[fields[0]] = (float(fields[5]), float(fields[6]))

    assert set(src_records) == set(dst_records)
    for atype, (m_src, q_src) in src_records.items():
        m_dst, q_dst = dst_records[atype]
        assert math.isclose(m_src, m_dst, abs_tol=1e-9), (
            f"{atype}: mass mismatch src={m_src} dst={m_dst}"
        )
        assert math.isclose(q_src, q_dst, abs_tol=1e-9), (
            f"{atype}: charge mismatch src={q_src} dst={q_dst}"
        )


def test_rewriter_inserts_zero_oxidation_column(rewritten: Path):
    """Every per-atom row's oxidation column (position 4) must be 0."""
    for line in rewritten.read_text().splitlines():
        fields = re.split(r"\s+", line.strip())
        if len(fields) < 14 or not fields[0].startswith("Mof_"):
            continue
        assert fields[4] == "0", f"{fields[0]}: oxidation column = {fields[4]} (must be 0)"


def test_rewriter_framework_atoms_sum_to_vogtiv_primitive_mass(rewritten: Path):
    """Per-atom-type masses × Mg-MOF-74 primitive stoichiometry must give 727.8 g/mol.

    Catches a bug where the rewriter silently zeroes or shifts the mass column.
    Stoichiometry: 6 Mg + 24 C + 18 O + 6 H per primitive.
    """
    masses: dict[str, float] = {}
    for line in rewritten.read_text().splitlines():
        fields = re.split(r"\s+", line.strip())
        if len(fields) < 14 or not fields[0].startswith("Mof_"):
            continue
        masses[fields[0]] = float(fields[5])

    # 1 Mg, 4 C (Ca/Cb/Cc/Cd), 3 O (Oa/Ob/Oc), 1 H per asym unit; primitive has 6×
    total = 6 * masses["Mof_Mg"]
    for label in ("Mof_Ca", "Mof_Cb", "Mof_Cc", "Mof_Cd"):
        total += 6 * masses[label]
    for label in ("Mof_Oa", "Mof_Ob", "Mof_Oc"):
        total += 6 * masses[label]
    total += 6 * masses["Mof_H"]
    assert 720.0 < total < 735.0, (
        f"VOGTIV primitive mass from rewritten pseudo_atoms = {total:.2f}, "
        f"expected ~727.82 g/mol (catches the 36.7× under-count bug)"
    )
