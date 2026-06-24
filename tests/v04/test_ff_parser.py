"""T011: FF parser + units + electroneutrality tests."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from widom_atlas.v04 import electroneutrality as en
from widom_atlas.v04 import units as u
from widom_atlas.v04.ff.dzubak import decode_dzubak_row, dzubak_energy
from widom_atlas.v04.ff.lin_mercado import decode_lin_mercado_row
from widom_atlas.v04.ff.mixing import lorentz_berthelot
from widom_atlas.v04.ff.parser import (
    PairTable,
    parse_cross_pair_table,
    parse_lj126_self_terms,
    serialize_pair_table_to_dict,
)
from widom_atlas.v04.ff.terms import (
    LJ126,
    BuckinghamLinMercado,
    DzubakAExpC5D6,
    FunctionalForm,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_lin_mercado_column_ordering_correct() -> None:
    """Properly ordered (A,B,C) row decodes cleanly."""
    term = decode_lin_mercado_row({"A": 4.088e5, "B": 3.965, "C": 2.473e7})
    assert isinstance(term, BuckinghamLinMercado)
    assert term.A_K == 4.088e5
    assert term.B_per_angstrom == 3.965
    assert term.C_K_angstrom6 == 2.473e7


def test_lin_mercado_column_swap_rejected() -> None:
    """Swapped columns (B and A swapped) must raise — magnitude heuristic catches it."""
    with pytest.raises(ValueError, match=r"swapped|> 0"):
        decode_lin_mercado_row({"A": 3.965, "B": 4.088e5, "C": 2.473e7})


def test_lin_mercado_C_already_scaled_flag_respected() -> None:
    """If C_already_scaled=True (default), S_g is NOT re-applied at energy eval."""
    term_scaled = decode_lin_mercado_row(
        {"A": 1e6, "B": 4.0, "C": 1e6}, S_g=2.0, C_already_scaled=True
    )
    term_unscaled = decode_lin_mercado_row(
        {"A": 1e6, "B": 4.0, "C": 1e6}, S_g=2.0, C_already_scaled=False
    )
    e_scaled = term_scaled.energy(3.5)
    e_unscaled = term_unscaled.energy(3.5)
    # Unscaled version should have stronger attraction (more negative) because S_g multiplies C
    assert e_unscaled < e_scaled


def test_dzubak_two_attraction_form() -> None:
    """Dzubak Mg-MOF-74-style decode keeps the two-attraction structure."""
    row = {"A": 4.067e7, "B": 4.152, "C5": 0.0, "D6": 4.062e5}
    term = decode_dzubak_row(row)
    assert isinstance(term, DzubakAExpC5D6)
    assert term.A_K == 4.067e7
    e = dzubak_energy(3.5, term)
    # Should be repulsive at short r? No, at 3.5 Å the attraction wins for Mg-O. Just check finite.
    assert math.isfinite(e)


def test_lj126_round_trip() -> None:
    a = LJ126(epsilon_K=166.4, sigma_angstrom=3.636)
    b = LJ126(epsilon_K=119.8, sigma_angstrom=3.405)
    mix = lorentz_berthelot(a, b)
    assert mix.sigma_angstrom == pytest.approx(0.5 * (a.sigma_angstrom + b.sigma_angstrom))
    assert mix.epsilon_K == pytest.approx(math.sqrt(a.epsilon_K * b.epsilon_K))


def test_electroneutrality_DDEC_cif() -> None:
    cif = REPO_ROOT / "fixtures/v04/RUBTAK01_SL_DDEC.cif"
    passes, total, n_atoms = en.check_electroneutrality(cif)
    assert n_atoms > 0
    assert passes, f"DDEC CIF charges sum {total} not neutral within tolerance"


def test_electroneutrality_passes_on_neutral_iza_cif() -> None:
    cif = REPO_ROOT / "docs/research/dataset-research-for-v0.4/7/MFI_iza.cif"
    passes, total, n_atoms = en.check_electroneutrality(cif)
    # IZA CIF has no charges column → returns (True, 0, 0)
    assert passes
    assert n_atoms == 0


def test_unit_conversion_round_trip() -> None:
    x = 12.345
    y = u.energy_kjmol_to_K(x)
    z = u.energy_K_to_kjmol(y)
    assert z == pytest.approx(x, rel=1e-9)
    p = 0.643
    pa = u.KH_mol_per_kg_per_bar_to_mol_per_kg_per_Pa(p)
    back = u.KH_mol_per_kg_per_Pa_to_mol_per_kg_per_bar(pa)
    assert back == pytest.approx(p, rel=1e-12)


def test_positive_exothermic_convention() -> None:
    assert u.positive_exothermic_Qads(-21.0) == 21.0
    assert u.positive_exothermic_Qads(21.0) == 21.0


def test_pair_table_serialization() -> None:
    table = PairTable()
    table.set("Si", "O_zeo",
              decode_lin_mercado_row({"A": 1e6, "B": 4.0, "C": 1e6}))
    out = serialize_pair_table_to_dict(table)
    assert "Si|O_zeo" not in out["terms"] and "O_zeo|Si" in out["terms"]
    entry = out["terms"]["O_zeo|Si"]
    assert entry["form"] == FunctionalForm.BUCKINGHAM_A_EXP_C6.value


def test_parse_lj126_self_terms() -> None:
    terms = parse_lj126_self_terms(
        {"Kr": {"epsilon_K": 166.4, "sigma_angstrom": 3.636},
         "Ar": {"epsilon_K": 119.8, "sigma_angstrom": 3.405}}
    )
    assert terms["Kr"].epsilon_K == 166.4
    assert terms["Ar"].sigma_angstrom == 3.405


def test_parse_cross_pair_table_LJ() -> None:
    table = parse_cross_pair_table(
        {("Kr", "O_zeo"): {"epsilon_K": 109.6, "sigma_angstrom": 3.450}},
        kind=FunctionalForm.LJ_12_6,
    )
    t = table.get("Kr", "O_zeo")
    assert isinstance(t, LJ126)
    assert t.epsilon_K == 109.6
