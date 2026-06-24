"""User-supplied charge-aware LJ + Coulomb backend.

Per ``implementation-verdict-continuation.txt`` §"Implement user-supplied
charge-aware backend": this backend reads a user-supplied JSON parameter
file, runs CuspAI Widom with a Coulomb-LJ ASE calculator built from it,
and stamps full provenance (DOI, redistribution status, hybrid-warning)
into the manifest.

**Refusal contract**: if the parameter file does not declare partial
charges for both the framework and the gas, the backend refuses to run
unless ``allow_neutral_fallback=True`` is set explicitly. The neutral
fallback is documented and warned about — it is not advertised as
scientifically equivalent.

The parameter file shape (validated by Pydantic):

.. code-block:: json

   {
     "framework_atom_types": [
       {"label": "Mg", "atom_type": "Mg_OMS", "charge_e": 1.2,
        "sigma_A": 2.69, "epsilon_K": 55.85, "source": "user-DDEC", "doi": "..."},
       …
     ],
     "gas_sites": [
       {"label": "C_CO2", "charge_e": 0.7,
        "sigma_A": 2.80, "epsilon_K": 27.0, "source": "TraPPE-CO2", "doi": "10.1002/aic.690470719"},
       {"label": "O_CO2", "charge_e": -0.35,
        "sigma_A": 3.05, "epsilon_K": 79.0, "source": "TraPPE-CO2", "doi": "10.1002/aic.690470719"}
     ],
     "mixing_rules": "Lorentz-Berthelot",
     "electrostatics": "Wolf",
     "redistribution_status": "user_supplied_not_bundled",
     "hybrid_warning": "TraPPE-CO2 + DDEC framework charges + UFF LJ — hybrid approximation, not a published validated FF"
   }

**Atom labels are matched against ASE chemical symbols** (the
``label`` field is the ASE element symbol). A future extension can add a
finer "atom_type" mapping via custom tags; v0.3 keeps it element-keyed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import BackendOutput
from .coulomb import WolfParameters, cross_only_wolf_energy
from .units import KELVIN_TO_EV

_LOGGER = logging.getLogger(__name__)

_FRAMEWORK_TAG = 1
_GAS_TAG = 0


class _AtomEntry(BaseModel):
    """One framework or gas-site parameter entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(..., min_length=1, description="ASE chemical symbol (e.g. 'Mg', 'C', 'O').")
    atom_type: str | None = Field(default=None, description="Optional finer type tag (e.g. 'Mg_OMS', 'C_CO2').")
    charge_e: float | None = Field(default=None, description="Partial charge in elementary charges.")
    sigma_A: float = Field(..., gt=0)
    epsilon_K: float = Field(..., ge=0)
    source: str = Field(..., min_length=1)
    doi: str | None = None


class UserParameterFile(BaseModel):
    """Schema for the user-supplied parameter file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    framework_atom_types: list[_AtomEntry] = Field(..., min_length=1)
    gas_sites: list[_AtomEntry] = Field(..., min_length=1)
    mixing_rules: Literal["Lorentz-Berthelot", "user_supplied"] = "Lorentz-Berthelot"
    electrostatics: Literal["Wolf", "Ewald", "external_engine", "none"] = "Wolf"
    redistribution_status: Literal[
        "bundled_safe",
        "user_supplied_not_bundled",
        "user_supplied_not_redistributed",
        "open_access_with_attribution",
        "unknown",
    ] = "user_supplied_not_bundled"
    hybrid_warning: str | None = Field(
        default=None,
        description="Free-form note when the file mixes parameters from different FF families.",
    )

    @field_validator("framework_atom_types", "gas_sites")
    @classmethod
    def _check_unique_labels(cls, v: list[_AtomEntry]) -> list[_AtomEntry]:
        labels = [e.label for e in v]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate labels in {labels}")
        return v


def _has_charges(file_: UserParameterFile) -> bool:
    """True iff at least one framework AND at least one gas atom carries a non-zero charge."""
    fw_any = any(e.charge_e is not None and e.charge_e != 0.0 for e in file_.framework_atom_types)
    gas_any = any(e.charge_e is not None and e.charge_e != 0.0 for e in file_.gas_sites)
    return fw_any and gas_any


def load_user_parameter_file(path: Path) -> UserParameterFile:
    """Load + validate a user-supplied parameter JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return UserParameterFile.model_validate(raw)


@dataclass(frozen=True)
class UserChargeAwareBackend:
    """Runs CuspAI Widom with a user-supplied Coulomb-LJ calculator.

    Refuses to start if charges are missing unless
    ``allow_neutral_fallback=True``. In neutral-fallback mode it emits a
    prominent warning into ``BackendOutput.provenance["warnings"]`` and the
    manifest's ``warnings`` list.
    """

    parameter_file: Path
    cutoff_A: float = 12.0
    wolf_alpha_inv_A: float = 0.20
    allow_neutral_fallback: bool = False
    name: str = "user_parameterised_coulomb_lj"

    def generate(
        self,
        *,
        structure: Any,
        gas: str,
        temperature_K: float,
        n_samples: int,
        seed: int,
        material_id: str,
        material_source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> BackendOutput:
        from widom import run_widom_insertion

        from widom_atlas.io.from_widom_result import from_widom_result

        params_file = load_user_parameter_file(self.parameter_file)
        warnings: list[str] = []
        charges_present = _has_charges(params_file)
        if not charges_present:
            if not self.allow_neutral_fallback:
                raise ValueError(
                    f"user_parameterised_coulomb_lj parameter file {self.parameter_file} "
                    "does not declare non-zero partial charges for both framework and gas. "
                    "Re-supply with charges or pass allow_neutral_fallback=True (NOT recommended). "
                    "Neutral fallback is NOT scientifically equivalent to a charge-aware run."
                )
            warnings.append(
                "neutral_fallback: charges absent or all zero; this run is LJ-only "
                "and is not a charge-aware result. Site-localising electrostatics absent."
            )
        if params_file.hybrid_warning:
            warnings.append(f"hybrid: {params_file.hybrid_warning}")

        atoms_for_widom = structure.copy()
        if hasattr(atoms_for_widom, "set_pbc"):
            atoms_for_widom.set_pbc(True)
        atoms_for_widom.set_tags(np.full(len(atoms_for_widom), _FRAMEWORK_TAG))

        calc = UserChargeAwareCalculator(
            params_file=params_file,
            cutoff_A=self.cutoff_A,
            wolf_alpha_inv_A=self.wolf_alpha_inv_A,
        )

        results = run_widom_insertion(
            calculator=calc,
            structure=atoms_for_widom,
            gas=gas,
            temperature=float(temperature_K),
            model_outputs_interaction_energy=True,
            num_insertions=int(n_samples),
            random_seed=int(seed),
        )

        provenance: dict[str, Any] = {
            "parameter_file": str(self.parameter_file),
            "mixing_rules": params_file.mixing_rules,
            "electrostatics": params_file.electrostatics,
            "redistribution_status": params_file.redistribution_status,
            "framework_atom_count": str(len(params_file.framework_atom_types)),
            "gas_site_count": str(len(params_file.gas_sites)),
            "cutoff_A": str(self.cutoff_A),
            "wolf_alpha_inv_A": str(self.wolf_alpha_inv_A),
            "charges_present": str(charges_present),
            "warnings": warnings,
            "framework_atom_types": [e.model_dump() for e in params_file.framework_atom_types],
            "gas_sites": [e.model_dump() for e in params_file.gas_sites],
        }

        calculator_label = (
            "widom_atlas.backends.user_parameterised.UserChargeAwareCalculator("
            f"params={self.parameter_file!s}, "
            f"mixing={params_file.mixing_rules}, electrostatics={params_file.electrostatics}, "
            f"cutoff_A={self.cutoff_A})"
        )
        metadata: dict[str, Any] = {
            "benchmark_material_id": material_id,
            "benchmark_source": material_source,
            "calculator": calculator_label,
            "backend": self.name,
            "backend_category": "user_parameterised_coulomb_lj",
            "parameter_pack_provenance": provenance,
            "warnings": warnings,
            "random_seed": int(seed),
            "n_insertions": int(n_samples),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        atlas_input = from_widom_result(
            results,
            gas=gas,
            temperature_K=float(temperature_K),
            structure=structure,
            metadata=metadata,
        )
        return BackendOutput(
            atlas_input=atlas_input,
            backend_label=(
                f"user_parameterised_coulomb_lj ({self.parameter_file.name}, "
                f"electrostatics={params_file.electrostatics}, cutoff={self.cutoff_A} Å)"
            ),
            calculator_label=calculator_label,
            samples_origin="cuspai_widom",
            provenance=provenance,
        )


class UserChargeAwareCalculator(Calculator):
    """ASE calculator: framework × gas LJ + Wolf-Coulomb interaction energy.

    Returns 0.0 for framework-alone and gas-alone calls (no inter-tag pairs)
    and the framework-gas pair sum (LJ + Coulomb) on the combined call.
    Used with ``run_widom_insertion(model_outputs_interaction_energy=True)``.
    """

    implemented_properties: ClassVar[list[str]] = ["energy"]  # type: ignore[misc]
    nolabel: ClassVar[bool] = True

    def __init__(
        self,
        *,
        params_file: UserParameterFile,
        cutoff_A: float = 12.0,
        wolf_alpha_inv_A: float = 0.20,
    ) -> None:
        super().__init__()
        self._fw_eps_eV: dict[str, float] = {}
        self._fw_sig_A: dict[str, float] = {}
        self._fw_q: dict[str, float] = {}
        for e in params_file.framework_atom_types:
            self._fw_eps_eV[e.label] = e.epsilon_K * KELVIN_TO_EV
            self._fw_sig_A[e.label] = e.sigma_A
            self._fw_q[e.label] = float(e.charge_e or 0.0)
        self._gas_eps_eV: dict[str, float] = {}
        self._gas_sig_A: dict[str, float] = {}
        self._gas_q: dict[str, float] = {}
        for e in params_file.gas_sites:
            sym = e.label.split("_")[0]  # "C_CO2" -> "C", or "O" -> "O"
            self._gas_eps_eV[sym] = e.epsilon_K * KELVIN_TO_EV
            self._gas_sig_A[sym] = e.sigma_A
            self._gas_q[sym] = float(e.charge_e or 0.0)
        self._cutoff_A = float(cutoff_A)
        self._wolf = WolfParameters(alpha_inv_A=wolf_alpha_inv_A, cutoff_A=cutoff_A)
        self._do_coulomb = params_file.electrostatics in ("Wolf",)
        self._unknown_warned: set[str] = set()

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("UserChargeAwareCalculator.calculate requires atoms")
        E_lj = self._lj_interaction_energy(atoms)
        E_coul = self._coulomb_interaction_energy(atoms) if self._do_coulomb else 0.0
        self.results = {"energy": float(E_lj + E_coul)}

    def _lj_interaction_energy(self, atoms: Atoms) -> float:
        n = len(atoms)
        if n < 2:
            return 0.0
        symbols = atoms.get_chemical_symbols()
        tags = np.asarray(atoms.get_tags(), dtype=np.int64)
        if tags.size != n:
            tags = np.zeros(n, dtype=np.int64)
        framework_mask = tags == _FRAMEWORK_TAG
        gas_mask = tags == _GAS_TAG
        if not (framework_mask.any() and gas_mask.any()):
            return 0.0

        eps = np.zeros(n, dtype=np.float64)
        sig = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(symbols):
            if framework_mask[i]:
                eps[i] = self._fw_eps_eV.get(s, 0.0)
                sig[i] = self._fw_sig_A.get(s, 0.0)
            else:
                eps[i] = self._gas_eps_eV.get(s, 0.0)
                sig[i] = self._gas_sig_A.get(s, 0.0)
            if eps[i] == 0.0 and sig[i] == 0.0 and s not in self._unknown_warned:
                _LOGGER.warning(
                    "UserChargeAware: no parameters for element %r; pair contribution = 0",
                    s,
                )
                self._unknown_warned.add(s)

        cutoff = self._cutoff_A
        radii = [cutoff * 0.5] * n
        nl = NeighborList(radii, self_interaction=False, bothways=False, skin=0.0)
        nl.update(atoms)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        is_gas_i = gas_mask
        is_fw_i = framework_mask

        eps_pairs: list[np.ndarray] = []
        sig_pairs: list[np.ndarray] = []
        r_pairs: list[np.ndarray] = []
        for i in range(n):
            if eps[i] == 0.0:
                continue
            indices, offsets = nl.get_neighbors(i)
            if indices.size == 0:
                continue
            same_tag = (is_gas_i[i] & gas_mask[indices]) | (is_fw_i[i] & framework_mask[indices])
            keep = ~same_tag
            indices = indices[keep]
            offsets = offsets[keep]
            if indices.size == 0:
                continue
            zero_eps = eps[indices] == 0.0
            indices = indices[~zero_eps]
            offsets = offsets[~zero_eps]
            if indices.size == 0:
                continue
            r_vec = positions[indices] + offsets @ cell - positions[i]
            r2 = np.einsum("ij,ij->i", r_vec, r_vec)
            mask_within = (r2 > 1e-12) & (r2 < cutoff * cutoff)
            if not mask_within.any():
                continue
            r2 = r2[mask_within]
            jdx = indices[mask_within]
            r_pairs.append(np.sqrt(r2))
            eps_pairs.append(np.sqrt(eps[i] * eps[jdx]))
            sig_pairs.append(0.5 * (sig[i] + sig[jdx]))

        if not r_pairs:
            return 0.0
        r_arr = np.concatenate(r_pairs)
        eps_arr = np.concatenate(eps_pairs)
        sig_arr = np.concatenate(sig_pairs)
        sr6 = (sig_arr / r_arr) ** 6
        sr12 = sr6 * sr6
        return float((4.0 * eps_arr * (sr12 - sr6)).sum())

    def _coulomb_interaction_energy(self, atoms: Atoms) -> float:
        n = len(atoms)
        if n < 2:
            return 0.0
        symbols = atoms.get_chemical_symbols()
        tags = np.asarray(atoms.get_tags(), dtype=np.int64)
        if tags.size != n:
            tags = np.zeros(n, dtype=np.int64)
        framework_mask = tags == _FRAMEWORK_TAG
        gas_mask = tags == _GAS_TAG
        if not (framework_mask.any() and gas_mask.any()):
            return 0.0

        charges = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(symbols):
            if framework_mask[i]:
                charges[i] = self._fw_q.get(s, 0.0)
            else:
                charges[i] = self._gas_q.get(s, 0.0)

        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        pbc = np.asarray(atoms.get_pbc())
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        return cross_only_wolf_energy(
            positions=positions,
            charges=charges,
            tags=tags,
            cell=cell,
            pbc=pbc,
            params=self._wolf,
            framework_tag=_FRAMEWORK_TAG,
            gas_tag=_GAS_TAG,
        )


__all__ = [
    "UserChargeAwareBackend",
    "UserChargeAwareCalculator",
    "UserParameterFile",
    "load_user_parameter_file",
]
