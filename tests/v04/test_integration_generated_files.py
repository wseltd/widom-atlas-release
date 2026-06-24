"""Integration tests that read the actually-generated RASPA3 input files.

These tests inspect the JSON output of `write_raspa_inputs` (not the parser
internals) and assert the operator's bug-class checklist:
- non-zero framework charges where required
- correct pair-potential form
- correct gas model
- correct temperature
- BinaryInteractions present for branches that need cross-LJ overrides
- 1a / 1b are BLOCKED at the executor level (not silently substituted)
- van't Hoff Q_st estimator is consistent
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from widom_atlas.v04.branches.blocked_branches import BLOCKED_BRANCHES, blocked_reason
from widom_atlas.v04.locked_inputs import load_locked_case_matrix
from widom_atlas.v04.raspa3.electroneutrality import derive_charge_neutrality
from widom_atlas.v04.raspa3.input_writer import write_raspa_inputs
from widom_atlas.v04.widom.vant_hoff import vant_hoff_two_point

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _matrix() -> Any:
    return load_locked_case_matrix(REPO_ROOT / "v04_case_matrix.yaml")


def _branch_by_id(branch_id: str) -> dict:
    for case in _matrix().cases:
        for b in case.get("branches", []):
            if b["branch_id"] == branch_id:
                return b
    raise KeyError(branch_id)


def _write_for(branch_id: str, tmp_path: Path, *, T: float | None = None) -> Path:
    """Helper to write inputs for a branch and return the work dir."""
    branch = _branch_by_id(branch_id)
    cif = REPO_ROOT / branch["framework"]["source_cif_path"]
    if T is None:
        T = (
            branch.get("temperature_K")
            or (branch.get("references") or {}).get("K_H", {}).get("temperature_K", 298.0)
        )
    bundle = write_raspa_inputs(
        work_dir=tmp_path / branch_id,
        branch=branch,
        cif_abs_path=cif,
        temperature_K=float(T),
        n_cycles=100,
        repo_root=REPO_ROOT,
    )
    return bundle.work_dir


def _ff(work_dir: Path) -> dict:
    return json.loads((work_dir / "force_field.json").read_text())


def _sim(work_dir: Path) -> dict:
    return json.loads((work_dir / "simulation.json").read_text())


# ---------- Bug A: framework charges ----------

@pytest.mark.parametrize("branch_id", ["2a", "3a"])
def test_charged_MOF_branch_has_nonzero_framework_charges(branch_id: str, tmp_path: Path) -> None:
    """2a (HKUST-1 Nazarian DDEC) and 3a (UiO-66 PACMOF2 DDEC6) must emit non-zero
    framework charges from their respective DDEC CIFs."""
    ff = _ff(_write_for(branch_id, tmp_path))
    framework_atoms = [a for a in ff["PseudoAtoms"] if a.get("framework")]
    assert framework_atoms, f"{branch_id} emitted zero framework PseudoAtoms"
    charges = [a["charge"] for a in framework_atoms]
    assert any(abs(q) > 1e-6 for q in charges), (
        f"{branch_id} framework charges all zero — DDEC-CIF charge column not read"
    )


def test_5b_Na_Rho_Al_charge_from_electroneutrality(tmp_path: Path) -> None:
    """5b Al charge must be derived from electroneutrality of
    Na9.2(Al9.8Si38.2O96), NOT the UFF default of +1.75."""
    ff = _ff(_write_for("5b", tmp_path))
    al_atoms = [a for a in ff["PseudoAtoms"] if a.get("element") == "Al"]
    assert al_atoms, "5b: no Al PseudoAtom"
    al_charge = al_atoms[0]["charge"]
    # Derivation: 9.2(+1.0) + 9.8*q_Al + 38.2(+2.05) + 96(-1.025) = 0 -> q_Al = +1.111e
    derived = derive_charge_neutrality(
        target_element="Al",
        composition={"Na": 9.2, "Al": 9.8, "Si": 38.2, "O": 96.0},
        explicit_charges={"Na": 1.0, "Si": 2.05, "O": -1.025},
    )
    assert math.isclose(al_charge, derived.charge, abs_tol=1e-3), (
        f"5b Al charge {al_charge} does not match electroneutrality-derived {derived.charge}"
    )
    assert abs(al_charge - 1.75) > 0.1, "5b Al charge collapsed to UFF default"


def test_5b_framework_total_charge_neutral(tmp_path: Path) -> None:
    """Sum of all framework PseudoAtom charges, weighted by composition counts,
    must be near zero for the prescribed Na-Rho stoichiometry."""
    ff = _ff(_write_for("5b", tmp_path))
    pa = {a["name"]: a["charge"] for a in ff["PseudoAtoms"] if a.get("framework")}
    comp = {"Na": 9.2, "Al": 9.8, "Si": 38.2, "O": 96.0}
    total = sum(comp[el] * pa.get(el, 0.0) for el in comp)
    assert abs(total) < 0.05, f"5b framework not neutral: net charge = {total:+.4f}"


# ---------- Bug B: pair-potential form + 1a/1b BLOCKED ----------

@pytest.mark.parametrize("branch_id", ["1a", "1b"])
def test_1a_1b_are_BLOCKED_with_explicit_reason(branch_id: str) -> None:
    """Per operator directive: 1a (Lin/Mercado Buckingham) and 1b (Dzubak
    A·exp - C/r^5 - C/r^6) MUST be marked BLOCKED on the RASPA3 v3.0.29
    backend. They MUST NOT be silently substituted with LJ."""
    info = blocked_reason(branch_id)
    assert info is not None, f"{branch_id} should be in BLOCKED_BRANCHES"
    assert "Buckingham" in info["reason"] or "Dzubak" in info["reason"]
    assert "RASPA3" in info["reason"]
    assert info["prescribed_form"]
    assert info["required_action"]


def test_blocked_branches_dict_contains_both_case1_branches() -> None:
    assert "1a" in BLOCKED_BRANCHES
    assert "1b" in BLOCKED_BRANCHES


# ---------- Bug C: temperature override ----------

def test_6d_temperature_is_87K_not_298K(tmp_path: Path) -> None:
    """6d MFI + Ar @ 87 K must actually run at 87 K — not the previous 298 K bug."""
    work_6d = _write_for("6d", tmp_path)
    sim = _sim(work_6d)
    assert sim["Systems"][0]["ExternalTemperature"] == pytest.approx(87.0)


def test_6c_vs_6d_temperature_differs(tmp_path: Path) -> None:
    """6c and 6d differ ONLY in temperature; their generated simulation.json
    must reflect different ExternalTemperature."""
    sim_6c = _sim(_write_for("6c", tmp_path))
    sim_6d = _sim(_write_for("6d", tmp_path))
    T_c = sim_6c["Systems"][0]["ExternalTemperature"]
    T_d = sim_6d["Systems"][0]["ExternalTemperature"]
    assert T_c != T_d, f"6c and 6d both at {T_c} K — temperature override broken"
    assert abs(T_c - 298.15) < 0.5
    assert abs(T_d - 87.0) < 0.5


# ---------- Bug — gas selection + cross-LJ ----------

def test_6a_writes_methane_no_CO2(tmp_path: Path) -> None:
    """6a is MFI + CH4. Generated component.json must be methane.json (TraPPE-UA).
    No CO2 component parameters should appear."""
    work = _write_for("6a", tmp_path)
    assert (work / "methane.json").exists()
    assert not (work / "CO2.json").exists()
    ff = _ff(work)
    pa_names = {a["name"] for a in ff["PseudoAtoms"]}
    assert "CH4" in pa_names
    assert "C_co2" not in pa_names
    assert "O_co2" not in pa_names


def test_6b_writes_Kr_with_talu_myers_cross(tmp_path: Path) -> None:
    """6b is MFI + Kr. Must use Talu-Myers Kr self-LJ AND emit a
    BinaryInteractions Kr-O override (109.6 K, 3.450 Å)."""
    work = _write_for("6b", tmp_path)
    assert (work / "krypton.json").exists()
    assert not (work / "CO2.json").exists()
    ff = _ff(work)
    kr_self = [si for si in ff["SelfInteractions"] if si["name"] == "Kr"]
    assert kr_self, "6b: no Kr self-interaction"
    assert kr_self[0]["parameters"][0] == pytest.approx(166.4)
    assert kr_self[0]["parameters"][1] == pytest.approx(3.636)
    bi = ff.get("BinaryInteractions", [])
    kr_o_cross = [
        b for b in bi
        if (set(b["names"]) == {"Kr", "O"}) and b["type"] == "lennard-jones"
    ]
    assert kr_o_cross, "6b: Kr-O cross LJ BinaryInteraction missing"
    assert kr_o_cross[0]["parameters"][0] == pytest.approx(109.6)
    assert kr_o_cross[0]["parameters"][1] == pytest.approx(3.450)


def test_6c_writes_Ar_with_talu_myers_cross(tmp_path: Path) -> None:
    """6c is MFI + Ar. Self-LJ + BinaryInteractions Ar-O cross required."""
    work = _write_for("6c", tmp_path)
    ff = _ff(work)
    ar_self = [si for si in ff["SelfInteractions"] if si["name"] == "Ar"]
    assert ar_self
    assert ar_self[0]["parameters"][0] == pytest.approx(119.8)
    bi = ff.get("BinaryInteractions", [])
    ar_o_cross = [
        b for b in bi
        if set(b["names"]) == {"Ar", "O"}
    ]
    assert ar_o_cross
    assert ar_o_cross[0]["parameters"][0] == pytest.approx(93.0)
    assert ar_o_cross[0]["parameters"][1] == pytest.approx(3.335)


# ---------- van't Hoff Q_st sanity ----------

def test_vant_hoff_two_point_positive_exothermic() -> None:
    """Q_st via van't Hoff with K_H(T_low) > K_H(T_high) gives positive Q_st (exothermic)."""
    vh = vant_hoff_two_point(
        K_H_low=10.0, T_low_K=273.15,
        K_H_high=1.0, T_high_K=323.15,
    )
    # Q_st = R * ln(10/1) / (1/273.15 - 1/323.15)
    expected = 8.314462618e-3 * math.log(10.0) / (1 / 273.15 - 1 / 323.15)
    assert vh.Q_st_kJ_per_mol == pytest.approx(expected, rel=1e-4)
    assert vh.Q_st_kJ_per_mol > 0


# ---------- Hash-difference invariants from operator brief ----------

def test_1a_and_1b_blocked_paths_dont_share_runtime_state() -> None:
    """1a and 1b must report different BLOCKED reasons (Buckingham vs Dzubak)."""
    info_1a = blocked_reason("1a")
    info_1b = blocked_reason("1b")
    assert info_1a is not None and info_1b is not None
    assert info_1a["prescribed_form"] != info_1b["prescribed_form"]
    assert "Lin" in info_1a["prescribed_form"] or "Mercado" in info_1a["prescribed_form"]
    assert "Dzubak" in info_1b["prescribed_form"]


# ---------- All-silica MFI sanity (6a) ----------

def test_6a_uses_charge_method_none_for_all_silica_MFI(tmp_path: Path) -> None:
    """All-silica MFI is electrostatically neutral; ChargeMethod must be None
    to avoid spending Ewald on zero charges."""
    sim = _sim(_write_for("6a", tmp_path))
    assert sim["Systems"][0]["ChargeMethod"] == "None"


def test_5b_uses_charge_method_ewald(tmp_path: Path) -> None:
    """5b Na-Rho carries Na+/Al+/Si+/O- charges; Ewald must be on."""
    sim = _sim(_write_for("5b", tmp_path))
    assert sim["Systems"][0]["ChargeMethod"] == "Ewald"


# ---------- Reproducibility: identical seeds in one dir = identical input ----------

def test_input_writer_deterministic_for_same_T(tmp_path: Path) -> None:
    branch = _branch_by_id("6a")
    cif = REPO_ROOT / branch["framework"]["source_cif_path"]
    a = write_raspa_inputs(
        tmp_path / "a", branch, cif, 298.0, 200, REPO_ROOT,
    )
    b = write_raspa_inputs(
        tmp_path / "b", branch, cif, 298.0, 200, REPO_ROOT,
    )
    assert a.sha256["force_field.json"] == b.sha256["force_field.json"]


# ---------- 5b derivation note is recorded for evidence ----------

def test_5b_derivation_note_recorded(tmp_path: Path) -> None:
    """The Al-electroneutrality derivation text must appear in the bundle's
    derivation_notes so the audit evidence carries it."""
    branch = _branch_by_id("5b")
    cif = REPO_ROOT / branch["framework"]["source_cif_path"]
    bundle = write_raspa_inputs(tmp_path / "5b", branch, cif, 298.0, 200, REPO_ROOT)
    assert bundle.derivation_notes is not None
    joined = " ".join(bundle.derivation_notes)
    assert "Al" in joined and "electroneutrality" in joined.lower()


# ---------- 5b BinaryInteractions present ----------

def test_5b_emits_garcia_sanchez_cross_pairs(tmp_path: Path) -> None:
    work = _write_for("5b", tmp_path)
    ff = _ff(work)
    bi = ff.get("BinaryInteractions") or []
    pair_names = {tuple(sorted(b["names"])) for b in bi}
    # García-Sánchez 2009 specifies at least Na-CO2_C, Na-CO2_O, O-CO2_C, O-CO2_O
    assert ("C_co2", "Na") in pair_names or ("Na", "C_co2") in pair_names \
        or any({"Na", "C_co2"} <= set(p) for p in pair_names), \
        f"5b missing Na-C_co2 cross: {pair_names}"


# ---------- 5b Na-Rho CIF normalisation: strip crystallographic CO2, keep Na ----------

def test_5b_crystallographic_CO2_stripped_from_framework_Na_retained(tmp_path: Path) -> None:
    """Per Lozinska 2012 SI Table S11.2 (0.1 bar), the Na-Rho CO2-open CIF
    contains both the relocated Na cations (Na2 at (0.4408,0,0), Na3 at
    (0.3084,0.3084,0.3084)) AND the crystallographically resolved adsorbed
    CO2 atoms (labels OC1, OC2, OC3, OC4, CO1, CO2_atom).

    The Widom-insertion framework MUST keep Na (it gates the CO2-open state)
    but MUST drop the crystallographic CO2 — RASPA3 inserts its own CO2 probes.
    Verifies the CIF normaliser produces this state without operator
    intervention.
    """
    import re
    branch = _branch_by_id("5b")
    src_cif_path = REPO_ROOT / branch["framework"]["source_cif_path"]
    src_text = src_cif_path.read_text()
    # Sanity: the source CIF DOES contain the crystallographic CO2 labels AND Na
    assert re.search(r"^\s*OC1\s+O\s+", src_text, re.MULTILINE), (
        "Source CIF must carry crystallographic OC1 label (it does in Lozinska 2012 SI)"
    )
    assert re.search(r"^\s*CO1\s+C\s+", src_text, re.MULTILINE), \
        "Source CIF must carry crystallographic CO1 label"
    assert re.search(r"^\s*Na2\s+Na\s+", src_text, re.MULTILINE), \
        "Source CIF must carry Na2 cation"
    assert re.search(r"^\s*Na3\s+Na\s+", src_text, re.MULTILINE), \
        "Source CIF must carry Na3 cation"

    work = _write_for("5b", tmp_path)
    out_cif_path = work / "Na-Rho_CO2_open_0p1bar.cif"
    out_text = out_cif_path.read_text()

    # Crystallographic CO2 adsorbates MUST be stripped
    for stripped_label in ("OC1", "OC2", "OC3", "OC4", "CO1", "CO2_atom"):
        assert not re.search(rf"^\s*{stripped_label}\s+", out_text, re.MULTILINE), (
            f"Normalised 5b CIF still contains framework-row {stripped_label!r}; "
            f"RASPA3 should insert its own CO2 probes."
        )

    # Na cations MUST be retained (relocated Na positions are part of the framework)
    assert re.search(r"^\s*Na\b", out_text, re.MULTILINE), \
        "Normalised 5b CIF must retain Na cation rows"

    # Force-field is consistent: framework charges include Na and Al, no C_co2 framework PA
    ff = _ff(work)
    framework_atoms = {a["name"] for a in ff["PseudoAtoms"] if a.get("framework")}
    assert "Na" in framework_atoms and "Al" in framework_atoms
    assert "C_co2" not in framework_atoms and "O_co2" not in framework_atoms


# ---------- van't Hoff uncertainty propagation ----------

def test_vant_hoff_uncertainty_propagates_when_K_H_uncertainties_provided() -> None:
    """When per-temperature K_H uncertainties are passed, the van't Hoff
    estimator reports a Q_st uncertainty in kJ/mol."""
    vh_no_unc = vant_hoff_two_point(
        K_H_low=10.0, T_low_K=273.15, K_H_high=1.0, T_high_K=323.15,
    )
    assert vh_no_unc.Q_st_uncertainty_kJ_per_mol is None

    vh_with_unc = vant_hoff_two_point(
        K_H_low=10.0, T_low_K=273.15, K_H_high=1.0, T_high_K=323.15,
        K_H_low_unc=0.5, K_H_high_unc=0.05,
    )
    assert vh_with_unc.Q_st_uncertainty_kJ_per_mol is not None
    assert vh_with_unc.Q_st_uncertainty_kJ_per_mol > 0


# ---------- 2a HKUST-1 Q_st reference window restored ----------

def test_2a_Q_st_reference_window_present(tmp_path: Path) -> None:
    """2a YAML uses Q_st.low_loading_value + Q_st.value_acceptance_range
    (not the .value / .acceptance_window_* schema). The reference extractor
    must read either schema."""
    from widom_atlas.v04.branches.executor import _extract_reference_block
    branch = _branch_by_id("2a")
    ref = _extract_reference_block(branch, T_ref=298.0)
    assert ref["Q_st_value_kj_per_mol"] is not None, "2a Q_st ref dropped"
    assert ref["Q_st_window_min"] is not None and ref["Q_st_window_max"] is not None
    assert 24.0 <= ref["Q_st_value_kj_per_mol"] <= 32.0


# ---------- Architectural vs scientific validation distinction ----------

def test_scientific_validation_pass_distinct_from_architectural() -> None:
    """A v0.4 scientific PASS requires every strict branch to PASS (or BROAD).
    BLOCKED + FAIL both disqualify. Architecture (spec §8) can still pass."""
    from widom_atlas.v04.audit.pass_criteria import scientific_validation_pass
    # Synthetic all-PASS → scientific PASS
    all_pass = {bid: {"verdict": "PASS"} for bid in
                ("1a", "1b", "2a", "3a", "4a", "5b", "6a", "6b", "6c")}
    ok, _ = scientific_validation_pass(all_pass)
    assert ok
    # One BLOCKED → NOT PASSED
    one_blocked = {**all_pass, "1a": {"verdict": "BLOCKED"}}
    ok, detail = scientific_validation_pass(one_blocked)
    assert not ok
    assert "BLOCKED" in detail and "1a" in detail
    # One FAIL → NOT PASSED
    one_fail = {**all_pass, "6b": {"verdict": "FAIL"}}
    ok, detail = scientific_validation_pass(one_fail)
    assert not ok
    assert "6b" in detail
