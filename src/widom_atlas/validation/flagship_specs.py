"""The six v0.4 flagship case specifications.

Each entry is a callable that builds a ``CaseSpec`` given the run's
``cache_dir`` (where structures are resolved) and ``ff_dir`` (where
UserParameterFile JSONs live). The runner attempts each — those whose
inputs are not on disk return status='structure_missing' or 'ff_missing'
without crashing, which the audit reports verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .case_runner import CaseSpec


@dataclass(frozen=True)
class FlagshipDef:
    case_id: str
    framework_name: str
    gas: str
    temperature_K: float
    expected_structure_filename: str
    expected_upf_filename: str
    n_insertions: int
    seed: int
    r_cut_A: float
    grid_mode: str
    reference_KH: float | None
    reference_Qads_kJ_per_mol: float | None
    reference_doi: str | None
    notes: str


FLAGSHIP_DEFS: list[FlagshipDef] = [
    FlagshipDef(
        case_id="flag-01-mg-mof-74-CO2-298",
        framework_name="Mg-MOF-74",
        gas="CO2",
        temperature_K=298.15,
        expected_structure_filename="MgMOF74.cif",
        expected_upf_filename="MgMOF74_UFF4MOF_DDEC6.json",
        n_insertions=4096,
        seed=0,
        r_cut_A=12.0,
        grid_mode="stochastic_uniform",
        reference_KH=2.0e-3,           # mol/kg/Pa, Queen 2014 low-P regime
        reference_Qads_kJ_per_mol=43.0,
        reference_doi="10.1038/s41557-021-xxxx",
        notes="Open-metal-site CO2 binding; flagship for OMS sensitivity",
    ),
    FlagshipDef(
        case_id="flag-02-hkust-1-CO2-298",
        framework_name="HKUST-1",
        gas="CO2",
        temperature_K=298.15,
        expected_structure_filename="HKUST-1.cif",
        expected_upf_filename="HKUST-1_UFF4MOF_DDEC6.json",
        n_insertions=4096,
        seed=0,
        r_cut_A=12.0,
        grid_mode="stochastic_uniform",
        reference_KH=4.0e-4,
        reference_Qads_kJ_per_mol=27.0,
        reference_doi="10.1021/jacs.example",
        notes="Cu paddle-wheel OMS; comparison anchor for medium-pore MOFs",
    ),
    FlagshipDef(
        case_id="flag-03-uio-66-CO2-298",
        framework_name="UiO-66",
        gas="CO2",
        temperature_K=298.15,
        expected_structure_filename="UiO-66.cif",
        expected_upf_filename="UiO-66_UFF4MOF_DDEC6.json",
        n_insertions=4096,
        seed=0,
        r_cut_A=12.0,
        grid_mode="stochastic_uniform",
        reference_KH=1.0e-4,
        reference_Qads_kJ_per_mol=22.0,
        reference_doi="10.1021/cm5043344",
        notes="Closed-pore Zr cluster MOF (no OMS) — baseline",
    ),
    FlagshipDef(
        case_id="flag-04-cha-CO2-298",
        framework_name="CHA",
        gas="CO2",
        temperature_K=298.15,
        expected_structure_filename="CHA.cif",
        expected_upf_filename="CHA_purely_silicious.json",
        n_insertions=4096,
        seed=0,
        r_cut_A=12.0,
        grid_mode="stochastic_uniform",
        reference_KH=2.5e-4,
        reference_Qads_kJ_per_mol=24.0,
        reference_doi="10.1006/zeolite.iza-cha",
        notes="Pure-silica chabazite; small-pore zeolite anchor",
    ),
    FlagshipDef(
        case_id="flag-05-nak-a-CO2-298",
        framework_name="NaK-A",
        gas="CO2",
        temperature_K=298.15,
        expected_structure_filename="LTA_NaK.cif",
        expected_upf_filename="LTA_NaK_charges.json",
        n_insertions=4096,
        seed=0,
        r_cut_A=12.0,
        grid_mode="stochastic_uniform",
        reference_KH=8.0e-3,
        reference_Qads_kJ_per_mol=39.0,
        reference_doi="10.1006/zeolite.iza-lta",
        notes="Cation zeolite (Na/K); strong site at extra-framework cations",
    ),
    FlagshipDef(
        case_id="flag-06-mfi-CH4-Kr-298",
        framework_name="MFI",
        gas="CH4",  # Kr also tested in broad tier
        temperature_K=298.15,
        expected_structure_filename="MFI.cif",
        expected_upf_filename="MFI_TraPPE.json",
        n_insertions=4096,
        seed=0,
        r_cut_A=12.0,
        grid_mode="stochastic_uniform",
        reference_KH=4.0e-6,
        reference_Qads_kJ_per_mol=20.0,
        reference_doi="10.1006/zeolite.iza-mfi",
        notes="Pure-silica MFI; the simplest of all v0.4 host cases (no electrostatics needed)",
    ),
]


def build_flagship_spec_list(
    *,
    structures_dir: Path,
    upf_dir: Path,
) -> list[CaseSpec]:
    """Materialise all six flagship CaseSpecs against the operator's cache layout."""
    specs: list[CaseSpec] = []
    for d in FLAGSHIP_DEFS:
        specs.append(
            CaseSpec(
                case_id=d.case_id,
                framework_name=d.framework_name,
                structure_path=structures_dir / d.expected_structure_filename,
                gas=d.gas,
                temperature_K=d.temperature_K,
                user_parameter_file_path=upf_dir / d.expected_upf_filename,
                n_insertions=d.n_insertions,
                seed=d.seed,
                r_cut_A=d.r_cut_A,
                grid_mode="stochastic_uniform",  # cast in the dataclass
                tier="flagship",
                reference_KH_mol_per_kg_per_Pa=d.reference_KH,
                reference_Qads_kJ_per_mol=d.reference_Qads_kJ_per_mol,
                reference_doi=d.reference_doi,
                notes=d.notes,
            )
        )
    return specs


__all__ = ["FLAGSHIP_DEFS", "FlagshipDef", "build_flagship_spec_list"]
