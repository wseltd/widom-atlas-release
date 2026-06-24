"""Smoke tests for the v0.4 validation layer (case-runner + tables + audit)."""

from __future__ import annotations

import json
from pathlib import Path

from widom_atlas.evaluator.parity import ParityRow
from widom_atlas.validation.audit import (
    derive_verdict,
    render_audit_markdown,
    write_audit,
)
from widom_atlas.validation.case_runner import (
    THRESHOLDS_BY_TIER,
    CaseResult,
    CaseSpec,
    run_case,
)
from widom_atlas.validation.flagship_specs import FLAGSHIP_DEFS, build_flagship_spec_list
from widom_atlas.validation.v04_tables import write_all_tables


def _toy_parity_pass_rows() -> list[ParityRow]:
    return [
        ParityRow(
            case_id="raspa3-mfi",
            kind="raspa3_reference",
            framework_name="MFI",
            component_name="CO2",
            temperature_K=298.0,
            n_insertions=64,
            seed=0,
            log10_KH_internal=-5.0,
            log10_KH_reference=-5.05,
            delta_log10_KH=0.05,
            Qads_internal_kJ_per_mol=22.0,
            Qads_reference_kJ_per_mol=22.5,
            delta_Qads_kJ_per_mol=0.5,
            threshold_log10_KH=0.10,
            threshold_Qads_kJ_per_mol=2.0,
            pass_log10_KH=True,
            pass_Qads=True,
            pass_overall=True,
            reference_provenance_sha256="sha",
            notes="ok",
        ),
        *[
            ParityRow(
                case_id=f"mofx-{i}",
                kind="mofxdb_simin",
                framework_name=f"MOF-{i}",
                component_name="CO2",
                temperature_K=298.0,
                n_insertions=64,
                seed=i,
                log10_KH_internal=-5.0,
                log10_KH_reference=-5.0,
                delta_log10_KH=0.0,
                Qads_internal_kJ_per_mol=22.0,
                Qads_reference_kJ_per_mol=22.0,
                delta_Qads_kJ_per_mol=0.0,
                threshold_log10_KH=0.10,
                threshold_Qads_kJ_per_mol=2.0,
                pass_log10_KH=True,
                pass_Qads=True,
                pass_overall=True,
                reference_provenance_sha256="sha",
                notes="ok",
            )
            for i in range(5)
        ],
    ]


def _flagship_passing_results() -> list[CaseResult]:
    out: list[CaseResult] = []
    for d in FLAGSHIP_DEFS:
        out.append(
            CaseResult(
                case_id=d.case_id,
                tier="flagship",
                status="ok",
                framework_name=d.framework_name,
                gas=d.gas,
                temperature_K=d.temperature_K,
                n_insertions_used=d.n_insertions,
                KH_internal_mol_per_kg_per_Pa=d.reference_KH,
                log10_KH_internal=-3.0,
                log10_KH_reference=-3.05,
                delta_log10_KH=0.05,
                Qads_internal_kJ_per_mol=d.reference_Qads_kJ_per_mol or 22.0,
                Qads_reference_kJ_per_mol=(d.reference_Qads_kJ_per_mol or 22.0) + 0.5,
                delta_Qads_kJ_per_mol=0.5,
                threshold_log10_KH=THRESHOLDS_BY_TIER["flagship"][0],
                threshold_Qads_kJ_per_mol=THRESHOLDS_BY_TIER["flagship"][1],
                pass_log10_KH=True,
                pass_Qads=True,
                pass_overall=True,
                framework_sha256="abc123",
                upf_sha256="def456",
                reference_doi=d.reference_doi,
                notes=d.notes,
            )
        )
    return out


def test_build_flagship_spec_list_returns_six(tmp_path: Path) -> None:
    specs = build_flagship_spec_list(structures_dir=tmp_path / "s", upf_dir=tmp_path / "u")
    assert len(specs) == 6
    case_ids = [s.case_id for s in specs]
    assert "flag-01-mg-mof-74-CO2-298" in case_ids
    assert "flag-06-mfi-CH4-Kr-298" in case_ids


def test_run_case_structure_missing(tmp_path: Path) -> None:
    spec = CaseSpec(
        case_id="x",
        framework_name="X",
        structure_path=tmp_path / "no.cif",
        gas="CO2",
        temperature_K=298.0,
        user_parameter_file_path=tmp_path / "no.json",
        n_insertions=10,
        seed=0,
        r_cut_A=8.0,
        grid_mode="stochastic_uniform",
        tier="flagship",
    )
    res = run_case(spec)
    assert res.status == "structure_missing"
    assert res.pass_overall is False


def test_derive_verdict_pass() -> None:
    cases = _flagship_passing_results()
    rows = _toy_parity_pass_rows()
    verdict, summary = derive_verdict(cases=cases, parity_rows=rows)
    assert verdict == "PASS"
    assert summary["flagship_passed_overall"] == 6


def test_derive_verdict_evaluator_parity_failed() -> None:
    cases = _flagship_passing_results()
    rows = []
    rows.append(ParityRow(
        case_id="raspa3", kind="raspa3_reference", framework_name="MFI",
        component_name="CO2", temperature_K=298.0, n_insertions=64, seed=0,
        log10_KH_internal=-5.0, log10_KH_reference=-7.0, delta_log10_KH=2.0,
        Qads_internal_kJ_per_mol=22.0, Qads_reference_kJ_per_mol=2.0,
        delta_Qads_kJ_per_mol=20.0,
        threshold_log10_KH=0.10, threshold_Qads_kJ_per_mol=2.0,
        pass_log10_KH=False, pass_Qads=False, pass_overall=False,
        reference_provenance_sha256="", notes="",
    ))
    verdict, _ = derive_verdict(cases=cases, parity_rows=rows)
    assert verdict == "EVALUATOR PARITY FAILED"


def test_derive_verdict_implemented_but_coverage_incomplete() -> None:
    cases: list[CaseResult] = []
    for i, d in enumerate(FLAGSHIP_DEFS):
        cases.append(CaseResult(
            case_id=d.case_id, tier="flagship",
            status="structure_missing" if i < 3 else "ok",
            framework_name=d.framework_name, gas=d.gas, temperature_K=d.temperature_K,
            n_insertions_used=0 if i < 3 else d.n_insertions,
            KH_internal_mol_per_kg_per_Pa=None if i < 3 else d.reference_KH,
            log10_KH_internal=None if i < 3 else -3.0,
            log10_KH_reference=None if i < 3 else -3.0,
            delta_log10_KH=None if i < 3 else 0.0,
            Qads_internal_kJ_per_mol=None if i < 3 else 22.0,
            Qads_reference_kJ_per_mol=None if i < 3 else 22.0,
            delta_Qads_kJ_per_mol=None if i < 3 else 0.0,
            threshold_log10_KH=THRESHOLDS_BY_TIER["flagship"][0],
            threshold_Qads_kJ_per_mol=THRESHOLDS_BY_TIER["flagship"][1],
            pass_log10_KH=i >= 3, pass_Qads=i >= 3, pass_overall=i >= 3,
            framework_sha256="", upf_sha256="", reference_doi=d.reference_doi,
            notes=d.notes,
        ))
    rows = _toy_parity_pass_rows()
    verdict, _ = derive_verdict(cases=cases, parity_rows=rows)
    assert verdict == "IMPLEMENTED BUT CASE COVERAGE INCOMPLETE"


def test_write_all_tables_produces_nine_files(tmp_path: Path) -> None:
    paths = write_all_tables(
        tmp_path / "tables",
        cases=_flagship_passing_results(),
        parity_rows=_toy_parity_pass_rows(),
        convergence_rows=[{"case_id": "flag-01", "n_insertions": 1024, "log10_KH": -3.0}],
        charge_sensitivity_rows=[{"mof": "Mg-MOF-74", "scheme": "DDEC6", "log10_KH": -3.0}],
        site_match_rows=[{"site": "OMS_Mg_CO2", "delta_A": 0.10}],
        provenance_rows=[{"dataset_name": "CRAFTED", "primary_doi": "10.5281/zenodo.x"}],
        registry_status_rows=[{"dataset_name": "CRAFTED", "present": True}],
    )
    assert sorted(paths.keys()) == ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
    for tid, p in paths.items():
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert payload["table_id"] == tid
        assert payload["schema_version"] == "0.4"


def test_render_audit_markdown_pass(tmp_path: Path) -> None:
    cases = _flagship_passing_results()
    rows = _toy_parity_pass_rows()
    paths = write_all_tables(
        tmp_path / "tables",
        cases=cases,
        parity_rows=rows,
        convergence_rows=[],
        charge_sensitivity_rows=[],
        site_match_rows=[],
        provenance_rows=[],
        registry_status_rows=[],
    )
    md = render_audit_markdown(cases=cases, parity_rows=rows, table_paths=paths)
    assert "Verdict: PASS" in md
    assert "MOFX-DB simin parity passed: **5/4**" in md
    out = write_audit(tmp_path / "FINAL_V04_VALIDATION_AUDIT.md", md)
    assert out.exists()
    assert "FINAL_V04_VALIDATION_AUDIT" in out.read_text(encoding="utf-8")
