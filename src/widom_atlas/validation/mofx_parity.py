"""MOFX-DB simin parity gate (real evaluator runs).

Each MOFX-DB record carries an inline CIF (`record["cif"]`) plus per-isotherm
`simin` strings that fully specify the RASPA simulation parameters
(framework name, force-field, gas component, temperature, supercell tile).
This module:

1. Caches the inline CIF as a real CIF file.
2. Parses the simin string for ``Forcefield <name>``, ``ExternalTemperature``,
   ``UnitCells``, and ``Component MoleculeName`` tokens.
3. Resolves the named force-field to the locally-cached RASPA2
   ExampleMOFsForceField table.
4. Builds a UserParameterFile per record, runs the internal evaluator, and
   compares the resulting K_H to the MOFX reference K_H from the record's
   `isotherm_data[0]`.

If a record's simin force-field is not in our cached UFF/DREIDING table (e.g.
PCFF, GAFF, MOF-FF, etc.), the row goes onto the structured blocker table
with the full reason + needed-action breakdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from widom_atlas.evaluator.parity import ParityRow, parity_row_from_pair
from widom_atlas.evaluator.runner import run_widom_evaluator


@dataclass(frozen=True)
class SiminSpec:
    forcefield_name: str | None
    framework_name: str | None
    temperature_K: float | None
    unit_cells: tuple[int, int, int] | None
    component_names: list[str]
    helium_void_fraction: float | None


def parse_simin_spec(simin_text: str) -> SiminSpec:
    """Lightweight parser for the RASPA simulation.input fields we need."""
    ff = None
    fw = None
    T = None
    nx = None
    voidf = None
    components: list[str] = []
    for raw in simin_text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if not toks:
            continue
        head = toks[0]
        if head == "Forcefield" and len(toks) >= 2:
            ff = toks[1]
        elif head == "FrameworkName" and len(toks) >= 2:
            fw = toks[1]
        elif head == "ExternalTemperature" and len(toks) >= 2:
            try:
                T = float(toks[1])
            except ValueError:
                pass
        elif head == "UnitCells" and len(toks) >= 4:
            try:
                nx = (int(toks[1]), int(toks[2]), int(toks[3]))
            except ValueError:
                pass
        elif head == "HeliumVoidFraction" and len(toks) >= 2:
            try:
                voidf = float(toks[1])
            except ValueError:
                pass
        elif head == "Component":
            for j, t in enumerate(toks):
                if t == "MoleculeName" and j + 1 < len(toks):
                    components.append(toks[j + 1])
    return SiminSpec(
        forcefield_name=ff, framework_name=fw, temperature_K=T,
        unit_cells=nx, component_names=components, helium_void_fraction=voidf,
    )


def is_supported_forcefield(name: str | None) -> bool:
    """We support the UFF / DREIDING combo from RASPA2 ExampleMOFsForceField."""
    if name is None:
        return False
    n = name.strip().lower()
    return n in {"uff", "dreiding"}


def build_mofxdb_blocker_row(
    *, rec: Any, simin: SiminSpec, reason: str, needed_path: str,
) -> dict[str, Any]:
    return {
        "record_id": rec.mofx_record_id,
        "material": rec.framework_name,
        "gas": rec.gas,
        "temperature_K": rec.temperature_K,
        "simin_forcefield": simin.forcefield_name,
        "missing_item": reason,
        "needed_path": needed_path,
        "reason": (
            f"MOFX simin requested force-field {simin.forcefield_name!r}; "
            "the locally-cached RASPA2 ExampleMOFsForceField only resolves "
            "UFF + DREIDING + Garcia-Sanchez/TraPPE gas atoms. Other FFs "
            "(GAFF, OPLS-AA, PCFF, MOF-FF, ZIF-FF, etc.) need their own "
            "parameter tables, which are not bundled."
        ),
        "source_url": f"https://mof.tech.northwestern.edu/mofs/{rec.mofx_record_id}.json",
        "licence_status": "MOFX-DB licence: unclear (cite-only by inventory.json policy)",
    }


def run_mofxdb_simin_parity(
    *,
    distilled_records: list[dict[str, Any]],
    cif_dir: Path,
    upf_dir: Path,
    ff_table: dict[str, dict[str, float]],
    n_records_to_run: int = 5,
    seed: int = 17,
    n_insertions: int = 512,
    r_cut_A: float = 10.0,
) -> tuple[list[ParityRow], list[dict[str, Any]]]:
    """Run the MOFX simin parity gate and return (parity_rows, blocker_rows)."""
    from widom_atlas.backends.user_parameterised import UserParameterFile
    from widom_atlas.evaluator.component import (
        Component,
        ch4_trappe_ua,
        co2_garcia_sanchez_2009,
        n2_trappe,
    )
    from widom_atlas.evaluator.runner import load_atoms
    from widom_atlas.ingest.mofxdb import select_deterministic_simin_records
    from widom_atlas.validation.prepare_inputs import _gas_template

    picks = select_deterministic_simin_records(distilled_records, n=n_records_to_run, seed=seed)
    parity_rows: list[ParityRow] = []
    blockers: list[dict[str, Any]] = []
    cif_dir.mkdir(parents=True, exist_ok=True)
    upf_dir.mkdir(parents=True, exist_ok=True)

    comp_factories = {
        "CO2": co2_garcia_sanchez_2009,
        "N2": n2_trappe,
        "CH4": ch4_trappe_ua,
    }
    if "Ar" in ff_table:
        ar_comp = Component(
            name="Ar", site_labels=["Ar"], site_offsets=np.zeros((1, 3)),
            site_charges=np.zeros(1), site_masses=np.array([39.948]),
            rotational=False, notes="Single-LJ Ar",
        )
        def _make_ar(comp: Component = ar_comp) -> Component:
            return comp
        comp_factories["Ar"] = _make_ar

    for rec in picks:
        simin_spec = parse_simin_spec(rec.simin_text)
        cif_path = cif_dir / f"{rec.framework_name}.cif"

        if rec.gas not in comp_factories:
            blockers.append({
                "record_id": rec.mofx_record_id,
                "material": rec.framework_name,
                "gas": rec.gas,
                "temperature_K": rec.temperature_K,
                "simin_forcefield": simin_spec.forcefield_name,
                "missing_item": f"gas Component for {rec.gas!r}",
                "needed_path": "src/widom_atlas/evaluator/component.py",
                "reason": (
                    f"Gas {rec.gas!r} has no rigid-body Component definition in the "
                    "internal evaluator. v0.4 supports CO2/N2/CH4/Ar."
                ),
                "source_url": f"https://mof.tech.northwestern.edu/mofs/{rec.mofx_record_id}.json",
                "licence_status": "MOFX-DB licence: unclear (cite-only)",
            })
            parity_rows.append(_blocker_parity_row(rec, simin_spec, "gas not supported"))
            continue

        if not is_supported_forcefield(simin_spec.forcefield_name):
            blockers.append(build_mofxdb_blocker_row(
                rec=rec, simin=simin_spec,
                reason=f"force-field {simin_spec.forcefield_name!r} not in cached FF table",
                needed_path=f"benchmarks/cache/raspa2_ff/{simin_spec.forcefield_name}_mixing_rules.def",
            ))
            parity_rows.append(_blocker_parity_row(
                rec, simin_spec,
                f"FF {simin_spec.forcefield_name!r} not cached locally",
            ))
            continue

        if not cif_path.exists():
            blockers.append({
                "record_id": rec.mofx_record_id,
                "material": rec.framework_name,
                "gas": rec.gas,
                "temperature_K": rec.temperature_K,
                "simin_forcefield": simin_spec.forcefield_name,
                "missing_item": "framework CIF",
                "needed_path": str(cif_path),
                "reason": "MOFX record CIF was not split out; re-run prepare-validation-inputs.",
                "source_url": f"https://mof.tech.northwestern.edu/mofs/{rec.mofx_record_id}.json",
                "licence_status": "MOFX-DB CIF: unclear (record-level)",
            })
            parity_rows.append(_blocker_parity_row(rec, simin_spec, "CIF unavailable"))
            continue

        atoms = load_atoms(cif_path)
        elements = sorted(set(atoms.get_chemical_symbols()))
        fw_entries: list[dict[str, Any]] = []
        missing_elem: list[str] = []
        for elem in elements:
            ff = ff_table.get(f"{elem}_") or ff_table.get(elem)
            if ff is None:
                missing_elem.append(elem)
                continue
            fw_entries.append({
                "label": elem, "sigma_A": ff["sigma_A"], "epsilon_K": ff["epsilon_K"],
                "charge_e": 0.0,
                "source": f"RASPA2/ExampleMOFsForceField (UFF/DREIDING) for MOFX record {rec.mofx_record_id}",
                "doi": "https://github.com/iRASPA/RASPA2",
            })
        if missing_elem:
            blockers.append({
                "record_id": rec.mofx_record_id,
                "material": rec.framework_name,
                "gas": rec.gas,
                "temperature_K": rec.temperature_K,
                "simin_forcefield": simin_spec.forcefield_name,
                "missing_item": f"FF entries for elements {missing_elem}",
                "needed_path": "benchmarks/cache/raspa2_ff/ExampleMOFsForceField_mixing_rules.def",
                "reason": (
                    f"MOFX framework {rec.framework_name} contains elements {missing_elem} "
                    "not covered by the cached UFF/DREIDING table."
                ),
                "source_url": f"https://mof.tech.northwestern.edu/mofs/{rec.mofx_record_id}.json",
                "licence_status": "MIT (RASPA2 redistributable)",
            })
            parity_rows.append(_blocker_parity_row(
                rec, simin_spec,
                f"FF missing for elements {missing_elem}",
            ))
            continue

        try:
            gas_sites = _gas_template(rec.gas, ff_table)
        except (KeyError, ValueError) as exc:
            blockers.append({
                "record_id": rec.mofx_record_id,
                "material": rec.framework_name,
                "gas": rec.gas,
                "temperature_K": rec.temperature_K,
                "simin_forcefield": simin_spec.forcefield_name,
                "missing_item": f"gas template for {rec.gas!r}",
                "needed_path": "src/widom_atlas/validation/prepare_inputs.py:_gas_template",
                "reason": f"_gas_template raised: {exc}",
                "source_url": f"https://mof.tech.northwestern.edu/mofs/{rec.mofx_record_id}.json",
                "licence_status": "MIT (gas templates from public lit.)",
            })
            parity_rows.append(_blocker_parity_row(rec, simin_spec, f"gas template error: {exc}"))
            continue

        upf_path = upf_dir / f"{rec.framework_name}_{rec.gas}_mofxdb_{rec.mofx_record_id}.json"
        upf_payload = {
            "framework_atom_types": fw_entries,
            "gas_sites": gas_sites,
            "mixing_rules": "Lorentz-Berthelot",
            "electrostatics": "Wolf",
            "redistribution_status": "open_access_with_attribution",
            "hybrid_warning": (
                f"UPF derived from MOFX record {rec.mofx_record_id} simin "
                f"(FF={simin_spec.forcefield_name}, T={simin_spec.temperature_K}, "
                f"unit_cells={simin_spec.unit_cells}) projected onto cached "
                "RASPA2 ExampleMOFsForceField. Charges set to 0 (UFF baseline)."
            ),
        }
        upf_path.write_text(json.dumps(upf_payload, indent=2, sort_keys=True), encoding="utf-8")
        upf = UserParameterFile.model_validate_json(upf_path.read_text(encoding="utf-8"))
        component = comp_factories[rec.gas]()

        res = run_widom_evaluator(
            atoms=atoms, framework_name=rec.framework_name,
            user_parameter_file=upf, component=component,
            temperature_K=rec.temperature_K, n_insertions=n_insertions,
            seed=seed, r_cut_A=r_cut_A, grid_mode="stochastic_uniform",
            framework_source_path=cif_path,
        )

        if res.status == "ok" and rec.KH_value is not None and rec.KH_value > 0:
            parity_rows.append(parity_row_from_pair(
                case_id=f"mofxdb-{rec.mofx_record_id}",
                kind="mofxdb_simin",
                internal=res,
                KH_reference=rec.KH_value,
                Qads_reference_kJ_per_mol=rec.Qads_value,
                reference_provenance_sha256=rec.simin_sha256,
                threshold_log10_KH=0.20,
                threshold_Qads=4.0,
                notes=(
                    f"MOFX live-fetched record {rec.mofx_record_id}; "
                    f"FF: {simin_spec.forcefield_name}; KH units: {rec.KH_units}; "
                    f"reference K_H = isotherm_data[0].adsorption / pressure (low-P limit)"
                ),
            ))
        elif res.status == "ok":
            parity_rows.append(ParityRow(
                case_id=f"mofxdb-{rec.mofx_record_id}",
                kind="mofxdb_simin",
                framework_name=rec.framework_name,
                component_name=rec.gas,
                temperature_K=rec.temperature_K,
                n_insertions=res.n_insertions_used,
                seed=seed,
                log10_KH_internal=(
                    None if res.KH_mol_per_kg_per_Pa is None or res.KH_mol_per_kg_per_Pa <= 0
                    else float(np.log10(res.KH_mol_per_kg_per_Pa))
                ),
                log10_KH_reference=None,
                delta_log10_KH=None,
                Qads_internal_kJ_per_mol=res.Qads_kJ_per_mol,
                Qads_reference_kJ_per_mol=rec.Qads_value,
                delta_Qads_kJ_per_mol=None,
                threshold_log10_KH=0.20,
                threshold_Qads_kJ_per_mol=4.0,
                pass_log10_KH=False,
                pass_Qads=False,
                pass_overall=False,
                reference_provenance_sha256=rec.simin_sha256,
                notes="evaluator ran but MOFX reference K_H was None or non-positive",
                warnings=rec.warnings,
            ))
        else:
            parity_rows.append(_blocker_parity_row(rec, simin_spec, f"evaluator status={res.status}"))

    return parity_rows, blockers


def _blocker_parity_row(rec: Any, simin: SiminSpec, note: str) -> ParityRow:
    return ParityRow(
        case_id=f"mofxdb-{rec.mofx_record_id}",
        kind="mofxdb_simin",
        framework_name=rec.framework_name,
        component_name=rec.gas,
        temperature_K=rec.temperature_K,
        n_insertions=0,
        seed=17,
        log10_KH_internal=None,
        log10_KH_reference=(
            None if rec.KH_value is None or rec.KH_value <= 0 else float(np.log10(rec.KH_value))
        ),
        delta_log10_KH=None,
        Qads_internal_kJ_per_mol=None,
        Qads_reference_kJ_per_mol=rec.Qads_value,
        delta_Qads_kJ_per_mol=None,
        threshold_log10_KH=0.20,
        threshold_Qads_kJ_per_mol=4.0,
        pass_log10_KH=False,
        pass_Qads=False,
        pass_overall=False,
        reference_provenance_sha256=rec.simin_sha256,
        notes=note,
        warnings=rec.warnings,
    )


__all__ = [
    "SiminSpec",
    "build_mofxdb_blocker_row",
    "is_supported_forcefield",
    "parse_simin_spec",
    "run_mofxdb_simin_parity",
]
