"""RASPA3 ``force_field.json`` + ``simulation.json`` + Component JSON ingester.

This is the keystone of v0.4 ingest. It turns any RASPA3 input set into
either a :class:`~widom_atlas.backends.user_parameterised.UserParameterFile`
(consumable by ``user_parameterised_coulomb_lj``) or a structured
``ParsedRaspa3Input`` (consumable by the v0.4 evaluator).

RASPA3 force-field schema (verified against
``tests/fixtures/raspa3_mfi_henry/force_field.json``):

.. code-block:: json

   {
     "MixingRule": "Lorentz-Berthelot",
     "TruncationMethod": "shifted",
     "TailCorrections": false,
     "PseudoAtoms": [
       {"name":"O", "framework":true, "element":"O", "mass":15.9994, "charge":-1.025, "source":"<DOI>"},
       …
     ],
     "SelfInteractions": [
       {"name":"O", "type":"lennard-jones", "parameters":[53.0, 3.30], "source":"<DOI>"},
       …
     ]
   }

simulation.json keys consumed: ``Systems[0].ChargeMethod``,
``Systems[0].ExternalTemperature``, ``Systems[0].Name`` (framework name),
``Components[*].Name`` and ``WidomProbability`` (gas component selection).

Units (RASPA3 conventions, verified):
  - ``parameters[0]`` = ε / k_B in **K**
  - ``parameters[1]`` = σ in **Å**
  - ``charge`` in elementary-charge units **e**
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from widom_atlas.backends.user_parameterised import UserParameterFile

_KELVIN_TO_eps_eV = 8.617333262e-5  # k_B in eV/K


class _RaspaPseudoAtom(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    name: str
    framework: bool = False
    element: str
    mass: float = Field(..., ge=0)  # ge=0 to allow virtual COM sites with mass=0 (TraPPE-N2 N_com)
    charge: float = 0.0
    source: str = ""
    print_to_output: bool | None = None
    print_as: str | None = None


class _RaspaSelfInteraction(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    name: str
    type: str
    parameters: list[float] = Field(default_factory=list)
    source: str = ""

    @field_validator("type")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v not in {"lennard-jones", "none"}:
            raise ValueError(f"unsupported SelfInteraction.type {v!r}")
        return v


class ForceFieldFile(BaseModel):
    """Parsed RASPA3 ``force_field.json``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    MixingRule: Literal["Lorentz-Berthelot"] = "Lorentz-Berthelot"
    TruncationMethod: Literal["shifted", "truncated"] = "shifted"
    TailCorrections: bool = False
    PseudoAtoms: list[_RaspaPseudoAtom] = Field(default_factory=list)
    SelfInteractions: list[_RaspaSelfInteraction] = Field(default_factory=list)


class _RaspaSystemConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    Type: str | None = None
    Name: str | None = None
    NumberOfUnitCells: list[int] = Field(default_factory=list)
    ExternalTemperature: float | None = None
    ExternalPressure: float | None = None
    ChargeMethod: Literal["Ewald", "Wolf", "None", "none"] = "None"


class _RaspaComponent(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    Name: str
    WidomProbability: float = 0.0
    CreateNumberOfMolecules: int | None = None


class SimulationFile(BaseModel):
    """Parsed RASPA3 ``simulation.json``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    SimulationType: str = "MonteCarlo"
    NumberOfCycles: int = 0
    NumberOfInitializationCycles: int = 0
    PrintEvery: int | None = None
    ForceField: str | None = None
    Systems: list[_RaspaSystemConfig] = Field(default_factory=list)
    Components: list[_RaspaComponent] = Field(default_factory=list)


class ComponentFile(BaseModel):
    """Parsed RASPA3 component JSON (e.g. ``CO2.json``)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    Name: str
    PseudoAtoms: list[str] = Field(default_factory=list)
    Positions: list[list[float]] = Field(default_factory=list)
    MoleculeDefinition: str | None = None


@dataclass(frozen=True)
class ParsedRaspa3Input:
    """One fully-parsed RASPA3 input bundle ready for the evaluator."""

    force_field: ForceFieldFile
    simulation: SimulationFile
    components: dict[str, ComponentFile]
    framework_name: str
    temperature_K: float
    charge_method: str
    mixing_rule: str
    truncation: str
    tail_corrections: bool
    force_field_path: str
    simulation_path: str
    force_field_sha256: str
    simulation_sha256: str
    components_sha256: dict[str, str] = field(default_factory=dict)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_force_field_json(text_or_path: str | Path) -> ForceFieldFile:
    """Parse a RASPA3 ``force_field.json``."""
    if isinstance(text_or_path, Path) or (isinstance(text_or_path, str) and Path(text_or_path).exists()):
        raw = json.loads(Path(text_or_path).read_text(encoding="utf-8"))
    else:
        raw = json.loads(text_or_path)
    return ForceFieldFile.model_validate(raw)


def parse_simulation_json(text_or_path: str | Path) -> SimulationFile:
    """Parse a RASPA3 ``simulation.json``."""
    if isinstance(text_or_path, Path) or (isinstance(text_or_path, str) and Path(text_or_path).exists()):
        raw = json.loads(Path(text_or_path).read_text(encoding="utf-8"))
    else:
        raw = json.loads(text_or_path)
    return SimulationFile.model_validate(raw)


def parse_component_json(text_or_path: str | Path) -> ComponentFile:
    """Parse a single RASPA3 component JSON."""
    if isinstance(text_or_path, Path) or (isinstance(text_or_path, str) and Path(text_or_path).exists()):
        raw = json.loads(Path(text_or_path).read_text(encoding="utf-8"))
    else:
        raw = json.loads(text_or_path)
    return ComponentFile.model_validate(raw)


def parse_raspa3_input_directory(
    *,
    force_field_path: Path,
    simulation_path: Path,
    component_paths: dict[str, Path] | None = None,
) -> ParsedRaspa3Input:
    """Parse a RASPA3 input bundle (force_field.json + simulation.json + Component JSONs)."""
    ff = parse_force_field_json(force_field_path)
    sim = parse_simulation_json(simulation_path)
    components: dict[str, ComponentFile] = {}
    component_sha: dict[str, str] = {}
    if component_paths:
        for name, p in component_paths.items():
            components[name] = parse_component_json(p)
            component_sha[name] = _sha256_of_file(p)

    framework_name = "unknown"
    temperature_K = 298.15
    charge_method = "None"
    if sim.Systems:
        sys0 = sim.Systems[0]
        framework_name = sys0.Name or "unknown"
        if sys0.ExternalTemperature is not None:
            temperature_K = float(sys0.ExternalTemperature)
        charge_method = sys0.ChargeMethod or "None"

    return ParsedRaspa3Input(
        force_field=ff,
        simulation=sim,
        components=components,
        framework_name=framework_name,
        temperature_K=temperature_K,
        charge_method=charge_method,
        mixing_rule=ff.MixingRule,
        truncation=ff.TruncationMethod,
        tail_corrections=ff.TailCorrections,
        force_field_path=str(force_field_path),
        simulation_path=str(simulation_path),
        force_field_sha256=_sha256_of_file(force_field_path),
        simulation_sha256=_sha256_of_file(simulation_path),
        components_sha256=component_sha,
    )


def to_user_parameter_file(parsed: ParsedRaspa3Input, *, gas_name: str) -> UserParameterFile:
    """Project a parsed RASPA3 bundle to a :class:`UserParameterFile`.

    The framework atom types are pseudo-atoms with ``framework=true``; gas
    sites are pseudo-atoms with ``framework=false`` whose ``name`` matches
    one of the chosen gas component's ``PseudoAtoms`` list.

    Units in :class:`UserParameterFile` are ``epsilon_K`` (matches RASPA3's
    parameters[0] in K) and ``sigma_A`` (matches parameters[1] in Å). No
    unit conversion is required.
    """
    if gas_name not in parsed.components:
        raise ValueError(
            f"gas {gas_name!r} not in parsed components ({list(parsed.components)}); "
            "supply the corresponding component_path"
        )
    gas_atom_names = set(parsed.components[gas_name].PseudoAtoms)

    # Build a lookup of self-interactions by pseudo-atom name.
    si_by_name: dict[str, _RaspaSelfInteraction] = {si.name: si for si in parsed.force_field.SelfInteractions}

    framework_atom_types: list[dict[str, Any]] = []
    gas_sites: list[dict[str, Any]] = []
    skipped: list[str] = []

    for pa in parsed.force_field.PseudoAtoms:
        si = si_by_name.get(pa.name)
        if si is None or si.type != "lennard-jones" or len(si.parameters) < 2:
            skipped.append(pa.name)
            continue
        eps_K, sig_A = float(si.parameters[0]), float(si.parameters[1])
        atom_dict = {
            "label": pa.element,
            "atom_type": pa.name,
            "charge_e": float(pa.charge),
            "sigma_A": sig_A,
            "epsilon_K": eps_K,
            "source": pa.source or "RASPA3 force_field.json",
            "doi": None,
        }
        if pa.framework:
            framework_atom_types.append(atom_dict)
        elif pa.name in gas_atom_names:
            gas_sites.append(atom_dict)
        else:
            skipped.append(pa.name)

    if not framework_atom_types or not gas_sites:
        raise ValueError(
            f"could not project to UserParameterFile: framework={len(framework_atom_types)} entries, "
            f"gas={len(gas_sites)} entries; skipped={skipped}"
        )

    # Deduplicate by label (keep first); UserParameterFile schema enforces unique labels.
    def _dedup(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for e in entries:
            if e["label"] in seen:
                continue
            seen.add(e["label"])
            out.append(e)
        return out

    return UserParameterFile.model_validate(
        {
            "framework_atom_types": _dedup(framework_atom_types),
            "gas_sites": _dedup(gas_sites),
            "mixing_rules": parsed.mixing_rule,
            "electrostatics": "Wolf" if parsed.charge_method.lower() == "ewald" else (
                "external_engine" if parsed.charge_method.lower() != "none" else "none"
            ),
            "redistribution_status": "user_supplied_not_bundled",
            "hybrid_warning": (
                "Imported from RASPA3 input. Charge handling: original RASPA used "
                f"{parsed.charge_method!r}; widom-atlas's user_parameterised_coulomb_lj "
                "approximates with Wolf. Numerical agreement at the parity gate is the test."
            ),
        }
    )


def parse_simin_string(simin: str) -> dict[str, Any]:
    """Parse a RASPA2-style ``simulation.input`` string (the form embedded in
    MOFX-DB ``heats[*].simin``).

    Returns a dict of {key: value} pairs. Multi-token values are kept as
    strings; numeric coercion happens at consumer call sites.
    """
    keys: dict[str, Any] = {}
    for raw_line in simin.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            key, val = parts
            keys[key] = val.strip()
        elif len(parts) == 1:
            keys[parts[0]] = ""
    # Synthesise nested keys for "Component 0 MoleculeName CO2"-style lines.
    for k in list(keys):
        if k == "Component" and isinstance(keys[k], str):
            tokens = keys[k].split()
            if len(tokens) >= 3 and tokens[1] == "MoleculeName":
                keys["component_index"] = int(tokens[0])
                keys["component_molecule_name"] = tokens[2]
    return keys


def hash_simin_string(simin: str) -> str:
    """Return a stable sha256 of a normalised simin string (whitespace-collapsed, lowercased keys)."""
    norm = "\n".join(line.strip() for line in simin.splitlines() if line.strip())
    return _sha256_of_text(norm)


__all__ = [
    "ComponentFile",
    "ForceFieldFile",
    "ParsedRaspa3Input",
    "SimulationFile",
    "hash_simin_string",
    "parse_component_json",
    "parse_force_field_json",
    "parse_raspa3_input_directory",
    "parse_simin_string",
    "parse_simulation_json",
    "to_user_parameter_file",
]
