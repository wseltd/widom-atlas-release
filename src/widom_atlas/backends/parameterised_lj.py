"""Multi-element Lennard-Jones ASE calculator + Widom-driving backend.

This is the v0.2 path that escapes toy-LJ behaviour without pulling in a
heavyweight engine. The calculator computes the LJ pair sum for any ASE
:class:`~ase.atoms.Atoms` using:

1. **Per-element ε,σ** loaded from :mod:`widom_atlas.backends.parameters`.
   By default: TraPPE for the gas atoms (tagged ``0``) and UFF for the
   framework (tagged ``1``).
2. **Lorentz-Berthelot mixing** (``ε_ij = √(ε_i ε_j)``, ``σ_ij = (σ_i+σ_j)/2``)
   — the most common rule for cross-element LJ.
3. **Periodic minimum-image distances via ASE's NeighborList**.
4. **Cutoff** (default 12 Å) — pair contributions beyond cutoff drop to zero.

By construction the calculator is **interaction-only**: the Widom-relevant
energy is the framework-gas pair sum. Intra-framework and intra-gas pair
sums are returned as zero, matching ``run_widom_insertion`` with
``model_outputs_interaction_energy=True``. This keeps each insertion's
work down to ``n_gas × neighbours_per_gas_atom`` LJ evaluations — about
150 for CO2 in UiO-66 — so a 10 000-insertion run finishes in ~1 minute
on one CPU.

Tagging convention
==================

The calculator distinguishes framework from gas via ASE atom tags:
``tags == 1`` is framework (use :func:`~.parameters.framework_parameter_pack`),
``tags == 0`` is gas (use :func:`~.parameters.gas_parameter_pack`). The
backend below sets these tags on the structure before passing to
``run_widom_insertion``. The gas atoms (built internally by widom via
:func:`ase.build.molecule`) inherit ``tags == 0`` by default, so no
patching of widom is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList

from .base import BackendOutput
from .parameters import (
    LJEntry,
    framework_parameter_pack,
    gas_parameter_pack,
    parameter_pack_provenance,
)

_LOGGER = logging.getLogger(__name__)

_FRAMEWORK_TAG = 1
_GAS_TAG = 0


@dataclass(frozen=True)
class ParameterisedLJBackend:
    """Backend implementation for the multi-element Lennard-Jones path.

    Construction is cheap; per-call work happens inside :meth:`generate`.
    """

    cutoff_A: float = 12.0
    name: str = "parameterised_lj"

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
        """Run a Widom insertion campaign with the parameterised LJ calculator.

        Returns:
            A :class:`BackendOutput` whose ``atlas_input`` is a fully
            populated :class:`~widom_atlas.io.models.AtlasInput`.
        """
        from widom import run_widom_insertion

        from widom_atlas.io.from_widom_result import from_widom_result

        atoms_for_widom = structure.copy()
        if hasattr(atoms_for_widom, "set_pbc"):
            atoms_for_widom.set_pbc(True)
        atoms_for_widom.set_tags(np.full(len(atoms_for_widom), _FRAMEWORK_TAG))

        gas_pack = gas_parameter_pack(gas)
        framework_pack = framework_parameter_pack()
        calc = ParameterisedLJCalculator(
            gas_parameters=gas_pack,
            framework_parameters=framework_pack,
            cutoff_A=self.cutoff_A,
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

        provenance = parameter_pack_provenance(gas)
        provenance["cutoff_A"] = str(self.cutoff_A)
        provenance["interaction_only"] = "true"
        provenance["framework_tag"] = str(_FRAMEWORK_TAG)
        provenance["gas_tag"] = str(_GAS_TAG)

        calculator_label = (
            "widom_atlas.backends.parameterised_lj.ParameterisedLJCalculator("
            f"gas_pack={provenance['gas_pack']!r}, framework_pack={provenance['framework_pack']!r}, "
            f"mixing={provenance['mixing_rule']}, cutoff_A={self.cutoff_A})"
        )

        metadata: dict[str, Any] = {
            "benchmark_material_id": material_id,
            "benchmark_source": material_source,
            "calculator": calculator_label,
            "backend": self.name,
            "backend_category": "parameterised_lj",
            "parameter_pack_provenance": provenance,
            "random_seed": int(seed),
            "n_insertions": int(n_samples),
            "warnings": [
                "TraPPE-CO2/N2 LJ parameters are mixed with UFF framework LJ via "
                "Lorentz-Berthelot. This is a hybrid approximation, NOT a published "
                "validated force field. Charges are not modelled — see "
                "user_parameterised_coulomb_lj for charge-aware runs.",
            ],
            "suitable_for_quantitative_interpretation": False,
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
            backend_label=f"parameterised_lj (TraPPE+UFF, Lorentz-Berthelot, cutoff={self.cutoff_A}A)",
            calculator_label=calculator_label,
            samples_origin="cuspai_widom",
            provenance=provenance,
        )


class ParameterisedLJCalculator(Calculator):
    """ASE calculator that returns the framework-gas Lennard-Jones interaction energy.

    Designed to be used with :func:`widom.run_widom_insertion` running with
    ``model_outputs_interaction_energy=True``. Returns zero on the
    framework-alone and gas-alone calls (no inter-tag pairs) and the
    framework-gas pair sum on the combined-system call.
    """

    implemented_properties: ClassVar[list[str]] = ["energy"]  # type: ignore[misc]
    nolabel: ClassVar[bool] = True

    def __init__(
        self,
        *,
        gas_parameters: dict[str, LJEntry],
        framework_parameters: dict[str, LJEntry],
        cutoff_A: float = 12.0,
    ) -> None:
        super().__init__()
        self._gas_eps: dict[str, float] = {s: e.eps_eV for s, e in gas_parameters.items()}
        self._gas_sig: dict[str, float] = {s: e.sigma_A for s, e in gas_parameters.items()}
        self._fw_eps: dict[str, float] = {s: e.eps_eV for s, e in framework_parameters.items()}
        self._fw_sig: dict[str, float] = {s: e.sigma_A for s, e in framework_parameters.items()}
        self._cutoff_A = float(cutoff_A)
        self._unknown_warned: set[str] = set()

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("ParameterisedLJCalculator.calculate requires atoms")
        E = self._interaction_energy(atoms)
        self.results = {"energy": float(E)}

    def _interaction_energy(self, atoms: Atoms) -> float:
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
                e = self._fw_eps.get(s)
                g = self._fw_sig.get(s)
            else:
                e = self._gas_eps.get(s)
                g = self._gas_sig.get(s)
                if e is None:
                    e = self._fw_eps.get(s)
                    g = self._fw_sig.get(s)
            if e is None or g is None:
                if s not in self._unknown_warned:
                    _LOGGER.warning(
                        "ParameterisedLJ: no parameters for element %r; pair contribution = 0",
                        s,
                    )
                    self._unknown_warned.add(s)
                eps[i] = 0.0
                sig[i] = 0.0
            else:
                eps[i] = e
                sig[i] = g

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
            same_tag = ((is_gas_i[i] & gas_mask[indices]) | (is_fw_i[i] & framework_mask[indices]))
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


__all__ = ["ParameterisedLJBackend", "ParameterisedLJCalculator"]
