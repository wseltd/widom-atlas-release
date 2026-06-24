"""T028: RASPA3 input-generation determinism + output-parse correctness tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from widom_atlas.v04.locked_inputs import load_locked_case_matrix
from widom_atlas.v04.raspa3.input_writer import write_raspa_inputs
from widom_atlas.v04.raspa3.output_parser import parse_raspa3_output

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _branch_by_id(branch_id: str) -> dict:
    matrix = load_locked_case_matrix(REPO_ROOT / "v04_case_matrix.yaml")
    for case in matrix.cases:
        for b in case.get("branches", []):
            if b["branch_id"] == branch_id:
                return b
    raise KeyError(branch_id)


def test_raspa_input_writer_produces_4_files(tmp_path: Path) -> None:
    branch = _branch_by_id("6a")  # MFI + CH4
    cif = REPO_ROOT / "docs/research/dataset-research-for-v0.4/7/MFI_iza.cif"
    bundle = write_raspa_inputs(
        work_dir=tmp_path,
        branch=branch,
        cif_abs_path=cif,
        temperature_K=298.0,
        n_cycles=100,
        repo_root=REPO_ROOT,
    )
    assert bundle.simulation_json.exists()
    assert bundle.force_field_json.exists()
    assert bundle.component_json.exists()
    assert bundle.framework_cif.exists()
    sim = json.loads(bundle.simulation_json.read_text())
    assert sim["SimulationType"] == "MonteCarlo"
    assert sim["NumberOfCycles"] == 100


def test_raspa_input_determinism(tmp_path: Path) -> None:
    """Identical branch + cif + cycles → identical input sha256s."""
    branch = _branch_by_id("6a")
    cif = REPO_ROOT / "docs/research/dataset-research-for-v0.4/7/MFI_iza.cif"
    a = write_raspa_inputs(tmp_path / "a", branch, cif, 298.0, 100, REPO_ROOT)
    b = write_raspa_inputs(tmp_path / "b", branch, cif, 298.0, 100, REPO_ROOT)
    assert a.sha256["simulation.json"] == b.sha256["simulation.json"]
    assert a.sha256["force_field.json"] == b.sha256["force_field.json"]


def test_raspa_charge_method_uses_ewald_only_when_BOTH_framework_AND_gas_are_charged(
    tmp_path: Path,
) -> None:
    """ChargeMethod=Ewald iff framework has charges AND gas has charges.

    CO2-bearing branches (2a, 5b) → Ewald (framework charged + EPM2/Harris-Yung CO2 carries
    +/- charges).
    Noble-gas / CH4 branches (6a, 6b, 6c, 6d) → None (gas is neutral, so framework charges
    are irrelevant for the Widom insertion energy).
    """
    branch_2a = _branch_by_id("2a")
    bundle_2a = write_raspa_inputs(
        tmp_path / "2a", branch_2a, REPO_ROOT / branch_2a["framework"]["source_cif_path"],
        298.0, 100, REPO_ROOT,
    )
    assert json.loads(bundle_2a.simulation_json.read_text())["Systems"][0]["ChargeMethod"] == "Ewald"
    for branch_id in ("6a", "6b", "6c"):
        branch = _branch_by_id(branch_id)
        bundle = write_raspa_inputs(
            tmp_path / branch_id, branch, REPO_ROOT / branch["framework"]["source_cif_path"],
            298.15, 100, REPO_ROOT,
        )
        sim = json.loads(bundle.simulation_json.read_text())
        assert sim["Systems"][0]["ChargeMethod"] == "None", (
            f"branch {branch_id} should use None (gas is neutral)"
        )


def test_parse_raspa3_output_extracts_K_H(tmp_path: Path) -> None:
    """Parser handles the canonical RASPA3 output format."""
    sample = """[Input reader]: simulation.json
    Widom insertion Rosenbluth weight statistics:
    Henry coefficient based on Rosenbluth weight:
    Average Henry coefficient:    6.431925e-06 +/-  9.810509e-08 [mol/kg/Pa]
    Average Henry coefficient:    3.709994e-05 +/-  5.658793e-07 [molec./uc/Pa]
    Widom total:             1e+05
    Widom                                10.749004 [s]
"""
    out_path = tmp_path / "output.txt"
    out_path.write_text(sample)
    p = parse_raspa3_output(out_path)
    assert p.K_H_mol_per_kg_per_Pa == pytest.approx(6.431925e-06)
    assert p.K_H_uncertainty == pytest.approx(9.810509e-08)
    assert p.widom_insertions_total in (100000, int(1e5))


def test_parse_raspa3_output_handles_missing_K_H(tmp_path: Path) -> None:
    out_path = tmp_path / "output.txt"
    out_path.write_text("No K_H here.\n")
    p = parse_raspa3_output(out_path)
    assert p.K_H_mol_per_kg_per_Pa is None


def test_raspa_input_force_field_lists_TraPPE_sources(tmp_path: Path) -> None:
    branch = _branch_by_id("6b")  # MFI + Kr
    cif = REPO_ROOT / "docs/research/dataset-research-for-v0.4/7/MFI_iza.cif"
    bundle = write_raspa_inputs(tmp_path, branch, cif, 298.0, 100, REPO_ROOT)
    ff = json.loads(bundle.force_field_json.read_text())
    pa_sources = " ".join(a["source"] for a in ff["PseudoAtoms"])
    si_sources = " ".join(a["source"] for a in ff["SelfInteractions"])
    assert "Jaramillo-Auerbach" in pa_sources or "García-Pérez" in pa_sources
    assert "TraPPE-zeo" in si_sources
    assert "Talu-Myers 2001" in (pa_sources + si_sources)
