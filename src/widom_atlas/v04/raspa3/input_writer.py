"""T024: RASPA3 input-file generator.

Every parameter is derived programmatically from the v04_case_matrix.yaml
branch entry — no manual patching. Each invocation generates four files
into a fresh evidence directory:

- simulation.json     : MC config (Type=Framework, ExternalTemperature=298, etc.)
- force_field.json    : pseudo-atoms + LJ self-terms + charges
- {gas_species}.json  : adsorbate molecule definition
- {framework}.cif     : copy of the locked CIF, optionally with labels normalised
                        to match the pseudo-atom name space
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ...backends.parameters import KELVIN_TO_EV, UFF_TABLE
from ..cif.normalize import normalize_cif_to_workdir
from .binary_interactions import binary_interactions_for_branch
from .electroneutrality import derive_charge_neutrality


def _uff_lj_K(element: str) -> tuple[float, float]:
    """Genuine UFF (Rappe 1992) Lennard-Jones for a framework element, returned
    as ``(epsilon_K, sigma_A)``.

    Used for MOF frameworks so the emitted ``source="UFF ..."`` provenance is
    truthful. Raises ``KeyError`` on an unknown element rather than inventing a
    silent ``(50.0, 3.0)`` default (the v0.4.2 silent-substitution fix)."""
    entry = UFF_TABLE.get(element)
    if entry is None:
        raise KeyError(
            f"no UFF Lennard-Jones parameters for framework element {element!r}; "
            "refusing to invent a silent default"
        )
    return (entry.eps_eV / KELVIN_TO_EV, entry.sigma_A)


@dataclass(frozen=True)
class RaspaInputBundle:
    work_dir: Path
    simulation_json: Path
    force_field_json: Path
    component_json: Path
    framework_cif: Path
    sha256: dict[str, str]
    # Provenance for evidence + integration tests
    derived_charges: dict[str, float] | None = None
    derivation_notes: list[str] | None = None
    binary_interactions: list[dict] | None = None


_GAS_TEMPLATES: dict[str, dict] = {
    "CO2": {
        "CriticalTemperature": 304.1,
        "CriticalPressure": 7.38e6,
        "AcentricFactor": 0.225,
        "Type": "rigid",
        "pseudoAtoms": [
            ["C_co2", [0.0, 0.0, 0.0]],
            ["O_co2", [0.0, 0.0, 1.149]],
            ["O_co2", [0.0, 0.0, -1.149]],
        ],
    },
    "CH4": {
        "CriticalTemperature": 190.564,
        "CriticalPressure": 4.5992e6,
        "AcentricFactor": 0.01142,
        "Type": "rigid",
        "pseudoAtoms": [["CH4", [0.0, 0.0, 1.0]]],
    },
    "Kr": {
        "CriticalTemperature": 209.4,
        "CriticalPressure": 5.5e6,
        "AcentricFactor": 0.0,
        "Type": "rigid",
        "pseudoAtoms": [["Kr", [0.0, 0.0, 1.0]]],
    },
    "Ar": {
        "CriticalTemperature": 150.8,
        "CriticalPressure": 4.87e6,
        "AcentricFactor": 0.0,
        "Type": "rigid",
        "pseudoAtoms": [["Ar", [0.0, 0.0, 1.0]]],
    },
}


# Per-element LJ + nominal charge defaults (used when CIF labels are stripped
# down to element symbols). For MOF DDEC CIFs we read per-label charges from
# the CIF directly.
_ELEMENT_LJ: dict[str, tuple[float, float]] = {
    # element -> (epsilon_K, sigma_angstrom)
    "Si": (22.0, 2.30),   # TraPPE-zeo (Bai 2013)
    "O":  (53.0, 3.30),   # TraPPE-zeo (Bai 2013)
    "Al": (22.0, 2.30),   # treat Al like Si for LJ; charge differs in zeolite cases
    "Na": (15.0966, 2.6576),  # UFF Na+ fallback
    "K":  (17.6128, 2.6576),  # UFF K+ fallback
    "Cu": (2.5161, 3.1137),   # UFF Cu fallback
    "Zr": (34.7222, 2.78315), # UFF Zr fallback (approx)
    "Mg": (55.857, 2.6914),   # UFF Mg
    "C":  (28.129, 2.76),     # generic aromatic C (Harris-Yung default)
    "H":  (7.6493, 2.5711),   # UFF H
}


def _zeolite_charge(element: str) -> float:
    """Framework-only charges used in Si/O/Al zeolites (Jaramillo-Auerbach)."""
    return {"Si": 2.05, "O": -1.025, "Al": 1.75, "Na": 1.0, "K": 1.0,
            "Cs": 1.0, "Li": 1.0}.get(element, 0.0)


def _gas_pseudoatoms(gas_species: str) -> list[dict]:
    if gas_species == "CO2":
        return [
            {"name": "C_co2", "framework": False, "element": "C", "print_as": "C",
             "mass": 12.0107, "charge": 0.6512,
             "source": "Harris-Yung 1995 CO2"},
            {"name": "O_co2", "framework": False, "element": "O", "print_as": "O",
             "mass": 15.9994, "charge": -0.3256,
             "source": "Harris-Yung 1995 CO2"},
        ]
    if gas_species == "CH4":
        return [{"name": "CH4", "framework": False, "element": "C", "print_as": "C",
                 "mass": 16.04246, "charge": 0.0,
                 "source": "TraPPE-UA Martin-Siepmann 2001"}]
    if gas_species == "Kr":
        return [{"name": "Kr", "framework": False, "element": "Kr", "print_as": "Kr",
                 "mass": 83.798, "charge": 0.0,
                 "source": "Talu-Myers 2001"}]
    if gas_species == "Ar":
        return [{"name": "Ar", "framework": False, "element": "Ar", "print_as": "Ar",
                 "mass": 39.948, "charge": 0.0,
                 "source": "Talu-Myers 2001"}]
    raise ValueError(f"unknown gas: {gas_species}")


def _gas_self_LJ(gas_species: str) -> list[dict]:
    if gas_species == "CO2":
        return [
            {"name": "C_co2", "type": "lennard-jones",
             "parameters": [28.129, 2.76], "source": "Harris-Yung 1995"},
            {"name": "O_co2", "type": "lennard-jones",
             "parameters": [80.507, 3.033], "source": "Harris-Yung 1995"},
        ]
    if gas_species == "CH4":
        return [{"name": "CH4", "type": "lennard-jones",
                 "parameters": [158.5, 3.72], "source": "TraPPE-UA Martin-Siepmann 2001"}]
    if gas_species == "Kr":
        return [{"name": "Kr", "type": "lennard-jones",
                 "parameters": [166.4, 3.636], "source": "Talu-Myers 2001"}]
    if gas_species == "Ar":
        return [{"name": "Ar", "type": "lennard-jones",
                 "parameters": [119.8, 3.405], "source": "Talu-Myers 2001"}]
    raise ValueError(f"unknown gas: {gas_species}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_gas_component(gas_species: str) -> tuple[str, dict]:
    if gas_species == "CO2":
        return "CO2", _GAS_TEMPLATES["CO2"]
    if gas_species == "CH4":
        return "methane", _GAS_TEMPLATES["CH4"]
    if gas_species == "Kr":
        return "krypton", _GAS_TEMPLATES["Kr"]
    if gas_species == "Ar":
        return "argon", _GAS_TEMPLATES["Ar"]
    raise ValueError(f"unknown gas species: {gas_species}")


def _is_zeolite_branch(branch: dict) -> bool:
    """Returns True for branches where the CIF contains stripped-element labels.

    Zeolite CIFs (Si-CHA, MFI, Na-Rho) need Si*/O*/Al*/Na* label collapsing.
    MOF CIFs with DDEC charges (FIQCEN, RUBTAK01_SL_DDEC) carry per-label charges
    and must NOT be relabelled.
    """
    cif_path = (branch.get("framework") or {}).get("source_cif_path") or ""
    return any(token in cif_path for token in (
        "iza", "Na-Rho", "Na_Rho", "NaRho",
    ))


_ELEMENT_PATTERN = re.compile(r"^([A-Z][a-z]?)")


def _element_from_label(label: str) -> str:
    m = _ELEMENT_PATTERN.match(label)
    return m.group(1) if m else label


def _read_ddec_cif_pseudo_atoms(cif_text: str) -> list[dict]:
    """For MOF DDEC CIFs: extract unique (label, element, charge) triples.

    Handles CIFs that omit `_atom_site_type_symbol` by deriving element
    from the label prefix (Cu1 → Cu, H1 → H).
    """
    lines = cif_text.splitlines()
    columns: list[str] = []
    in_loop = False
    rows: list[list[str]] = []
    for line in lines:
        s = line.strip()
        if s == "loop_":
            columns = []
            in_loop = False
            continue
        if s.startswith("_atom_site_"):
            columns.append(s)
            if "_atom_site_label" in columns:
                in_loop = True
            continue
        if in_loop and s and not s.startswith("#") and not s.startswith("_"):
            if s.startswith("loop_"):
                break
            rows.append(re.split(r"\s+", s))
    if "_atom_site_label" not in columns:
        return []
    label_idx = columns.index("_atom_site_label")
    type_idx = columns.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in columns else None
    charge_idx = columns.index("_atom_site_charge") if "_atom_site_charge" in columns else None
    seen: dict[str, dict] = {}
    for r in rows:
        if len(r) <= label_idx:
            continue
        label = r[label_idx]
        element = (
            r[type_idx]
            if type_idx is not None and len(r) > type_idx
            else _element_from_label(label)
        )
        if label in seen:
            continue
        q = float(r[charge_idx]) if charge_idx is not None and len(r) > charge_idx else 0.0
        # MOF frameworks use genuine UFF per element (Rappe 1992). Fail loud on an
        # unknown element rather than silently substituting (50,3). The earlier
        # _ELEMENT_LJ default mislabelled guest/zeolite values as "UFF" (v0.4.2 fix).
        eps_sigma = _uff_lj_K(element)
        seen[label] = {
            "name": label,
            "element": element,
            "mass": _ELEMENT_MASS.get(element, 12.0),
            "charge": q,
            "LJ": eps_sigma,
        }
    return list(seen.values())


_ELEMENT_MASS: dict[str, float] = {
    "Si": 28.0855, "O": 15.9994, "Al": 26.9815, "Na": 22.9898,
    "K": 39.0983, "Cu": 63.546, "Zr": 91.224, "Mg": 24.305,
    "C": 12.0107, "H": 1.00794, "Cs": 132.9054, "Li": 6.941,
    "Kr": 83.798, "Ar": 39.948,
}


def _na_rho_composition_from_yaml(branch: dict) -> dict[str, float] | None:
    """Parse the Na-Rho composition string from YAML framework note.

    Returns None if no recognizable composition string is found.
    The Na-Rho fixture CIF has `_chemical_formula_structural`
    'Na9.2 (Al9.8 Si38.2 O96) (CO2)10.1' but we read from YAML directly to
    avoid depending on the CIF parser. The YAML key is `framework.note`.
    """
    framework = branch.get("framework") or {}
    note = (framework.get("note") or "") + " " + (framework.get("name") or "")
    import re
    m = re.search(r"Na(\d+(?:\.\d+)?)\s*\(?\s*Al(\d+(?:\.\d+)?)\s*Si(\d+(?:\.\d+)?)\s*O(\d+(?:\.\d+)?)", note)
    if not m:
        return None
    return {"Na": float(m.group(1)), "Al": float(m.group(2)),
            "Si": float(m.group(3)), "O": float(m.group(4))}


def _na_rho_composition_fallback() -> dict[str, float]:
    """Na-Rho composition from Lozinska 2012 SI: Na9.2(Al9.8Si38.2O96)."""
    return {"Na": 9.2, "Al": 9.8, "Si": 38.2, "O": 96.0}


def write_raspa_inputs(
    work_dir: Path,
    branch: dict,
    cif_abs_path: Path,
    temperature_K: float,
    n_cycles: int,
    repo_root: Path,
) -> RaspaInputBundle:
    work_dir.mkdir(parents=True, exist_ok=True)
    framework_stem = cif_abs_path.stem
    out_cif = work_dir / f"{framework_stem}.cif"

    derivation_notes: list[str] = []
    derived_charges: dict[str, float] = {}

    # CIF normalization
    is_zeolite = _is_zeolite_branch(branch)
    if is_zeolite:
        normalize_cif_to_workdir(cif_abs_path, out_cif, mode="zeolite")
    else:
        normalize_cif_to_workdir(cif_abs_path, out_cif, mode="preserve")

    # Build framework PseudoAtoms + SelfInteractions
    framework_pa: list[dict] = []
    framework_si: list[dict] = []
    present_framework_elements: set[str] = set()
    branch_id = str(branch.get("branch_id") or "")
    if is_zeolite:
        # Scan the normalized CIF for the framework elements actually present
        framework_charges = (branch.get("force_field") or {}).get("framework_charges") or {
            "Si": 2.05, "O_zeo": -1.025,
        }
        spec_charges_by_element: dict[str, float] = {}
        for key, q in framework_charges.items():
            element = key.replace("_zeo", "").rstrip("_")
            spec_charges_by_element[element] = float(q)
        # Add Na charge from YAML if present (separate key)
        na_charge_spec = (branch.get("force_field") or {}).get("Na_charge")
        if na_charge_spec is not None:
            spec_charges_by_element["Na"] = float(na_charge_spec)
        # Detect elements actually in the normalized CIF
        ddec_atoms = _read_ddec_cif_pseudo_atoms(out_cif.read_text())
        for a in ddec_atoms:
            present_framework_elements.add(a["element"])

        # Electroneutrality-derived Al for Na-Rho 5a/5b (operator directive: no UFF default).
        # 5a and 5b share the Na9.2(Al9.8Si38.2O96) framework — derive Al from composition.
        is_na_rho = branch_id in ("5a", "5b")
        if is_na_rho and "Al" in present_framework_elements:
            # Ensure Na charge is present (5a may not have Na_charge directly)
            if "Na" not in spec_charges_by_element:
                spec_charges_by_element["Na"] = 1.0  # Garcia-Perez 2007 Na+ convention
            comp = _na_rho_composition_from_yaml(branch) or _na_rho_composition_fallback()
            explicit = {el: spec_charges_by_element[el]
                        for el in ("Si", "O", "Na") if el in spec_charges_by_element}
            try:
                derived = derive_charge_neutrality("Al", comp, explicit)
                spec_charges_by_element["Al"] = derived.charge
                derived_charges["Al"] = derived.charge
                derivation_notes.append(
                    f"{branch_id} Al charge derived from electroneutrality: {derived.derivation_text}. "
                    f"Composition source: Lozinska_2012 SI Na9.2(Al9.8Si38.2O96)."
                )
            except Exception as e:
                derivation_notes.append(f"{branch_id} Al derivation FAILED: {e!r}")
                raise

        for element in sorted(present_framework_elements):
            if element in ("C", "H"):
                continue  # CO2 site labels were stripped; gas atoms handled below
            if element not in spec_charges_by_element:
                raise ValueError(
                    f"branch {branch_id}: zeolite framework contains element {element} "
                    f"with no charge in YAML force_field.framework_charges (Si/O/Al/Na). "
                    f"Operator directive: do not invent silently. Add it to YAML or block."
                )
            charge = spec_charges_by_element[element]
            if element not in _ELEMENT_LJ:
                continue
            eps, sig = _ELEMENT_LJ[element]
            source = (
                "Jaramillo-Auerbach 1999 via García-Pérez 2007" if element in ("Si", "O")
                else "Lozinska_2012_composition_electroneutrality" if (is_na_rho and element == "Al")
                else "UFF cation fallback"
            )
            framework_pa.append({
                "name": element, "framework": True, "element": element,
                "mass": _ELEMENT_MASS[element], "charge": float(charge),
                "source": source,
            })
            framework_si.append({
                "name": element, "type": "lennard-jones",
                "parameters": [eps, sig],
                "source": "TraPPE-zeo Bai 2013" if element in ("Si", "O") else "UFF cation fallback",
            })
        # Spec Na-LJ override (e.g., 5b uses UFF Na+ from YAML)
        na_lj = (branch.get("force_field") or {}).get("Na_LJ")
        if na_lj is not None:
            for si in framework_si:
                if si["name"] == "Na":
                    si["parameters"] = [float(na_lj["epsilon_K"]), float(na_lj["sigma_angstrom"])]
                    si["source"] = "UFF Na+ fallback (spec)"
    else:
        # MOF DDEC CIF: extract per-label charges
        ddec_atoms = _read_ddec_cif_pseudo_atoms(cif_abs_path.read_text())
        for atom in ddec_atoms:
            present_framework_elements.add(atom["element"])
            framework_pa.append({
                "name": atom["name"],
                "framework": True,
                "element": atom["element"],
                "mass": atom["mass"],
                "charge": atom["charge"],
                "source": "DDEC6 from PACMOF2 / Nazarian 2016",
            })
            eps, sig = atom["LJ"]
            framework_si.append({
                "name": atom["name"],
                "type": "lennard-jones",
                "parameters": [eps, sig],
                "source": "UFF (Rappe 1992), per-element, MOF framework",
            })

    # Gas pseudo-atoms + self-LJ
    gas_species = branch["gas"]["species"]
    component_name, component_def = _select_gas_component(gas_species)
    gas_pa = _gas_pseudoatoms(gas_species)
    gas_si = _gas_self_LJ(gas_species)

    component_json = work_dir / f"{component_name}.json"
    component_json.write_text(json.dumps(component_def, indent=2))

    # Force field
    ep = branch.get("electrostatics_per_branch") or {}
    cutoff = float(ep.get("direct_cutoff_angstrom", 12.0))
    binary_interactions = binary_interactions_for_branch(
        branch_id=branch_id,
        present_framework_elements=present_framework_elements,
        gas_species=gas_species,
    )
    ff: dict = {
        "PseudoAtoms": framework_pa + gas_pa,
        "SelfInteractions": framework_si + gas_si,
        "MixingRule": "Lorentz-Berthelot",
        "TruncationMethod": "shifted" if ep.get("lj_treatment") == "shifted_truncated" else "truncated",
        "TailCorrections": bool(ep.get("lj_tail_correction", False)),
        "CutOffVDW": cutoff,
        "CutOffCoulomb": cutoff,
    }
    if binary_interactions:
        ff["BinaryInteractions"] = binary_interactions
    ff_json = work_dir / "force_field.json"
    ff_json.write_text(json.dumps(ff, indent=2))

    # Decide ChargeMethod: Ewald only when BOTH framework AND adsorbate carry charges.
    # If the adsorbate is neutral (CH4, Kr, Ar), framework charges are irrelevant for
    # the Widom insertion energy and Ewald adds compute without changing the result.
    framework_has_charges = any(abs(p["charge"]) > 1e-6 for p in framework_pa)
    gas_has_charges = any(abs(p.get("charge", 0.0)) > 1e-6 for p in gas_pa)
    charge_method = (
        "Ewald"
        if (framework_has_charges and gas_has_charges and ep.get("ewald_via_raspa3"))
        else "None"
    )

    sim = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": int(n_cycles),
        "NumberOfInitializationCycles": 0,
        "PrintEvery": max(int(n_cycles) // 10, 1),
        "Systems": [
            {
                "Type": "Framework",
                "Name": framework_stem,
                "NumberOfUnitCells": [2, 2, 2],
                "ExternalTemperature": float(temperature_K),
                "ChargeMethod": charge_method,
            }
        ],
        "Components": [
            {
                "Name": component_name,
                "IdealGasRosenbluthWeight": 1.0,
                "WidomProbability": 1.0,
                "CreateNumberOfMolecules": 0,
            }
        ],
    }
    sim_json = work_dir / "simulation.json"
    sim_json.write_text(json.dumps(sim, indent=2))

    return RaspaInputBundle(
        work_dir=work_dir,
        simulation_json=sim_json,
        force_field_json=ff_json,
        component_json=component_json,
        framework_cif=out_cif,
        sha256={
            "simulation.json": _sha256(sim_json),
            "force_field.json": _sha256(ff_json),
            f"{component_name}.json": _sha256(component_json),
            f"{framework_stem}.cif": _sha256(out_cif),
        },
        derived_charges=derived_charges if derived_charges else None,
        derivation_notes=derivation_notes if derivation_notes else None,
        binary_interactions=binary_interactions if binary_interactions else None,
    )
