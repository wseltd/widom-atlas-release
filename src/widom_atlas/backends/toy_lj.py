"""Toy Lennard-Jones backend — the v1 smoke calculator (verdict §G).

This is the previous default: a single-element ASE :class:`LennardJones`
calculator with ``ε=0.01 eV, σ=3.0 Å``. Kept here behind the new backend
abstraction so it can still be selected with ``--backend toy_lj`` when
the operator wants the legacy behaviour for direct comparison against
the parameterised path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BackendOutput

_TOY_EPSILON_EV = 0.01
_TOY_SIGMA_A = 3.0


@dataclass(frozen=True)
class ToyLennardJonesBackend:
    """Backend wrapping the verdict §G ASE LJ smoke calculator."""

    name: str = "toy_lj"

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
        from ase.calculators.lj import LennardJones
        from widom import run_widom_insertion

        from widom_atlas.io.from_widom_result import from_widom_result

        atoms_for_widom = structure.copy()
        if hasattr(atoms_for_widom, "set_pbc"):
            atoms_for_widom.set_pbc(True)

        calc = LennardJones(epsilon=_TOY_EPSILON_EV, sigma=_TOY_SIGMA_A)
        results = run_widom_insertion(
            calculator=calc,
            structure=atoms_for_widom,
            gas=gas,
            temperature=float(temperature_K),
            model_outputs_interaction_energy=False,
            num_insertions=int(n_samples),
            random_seed=int(seed),
        )

        calculator_label = (
            f"ase.calculators.lj.LennardJones(epsilon={_TOY_EPSILON_EV}, sigma={_TOY_SIGMA_A})"
        )
        provenance: dict[str, Any] = {
            "calculator_kind": "toy_lj_smoke_only",
            "epsilon_eV": str(_TOY_EPSILON_EV),
            "sigma_A": str(_TOY_SIGMA_A),
            "verdict_reference": "implementation-verdict.txt §G — smoke calculator only",
        }
        metadata: dict[str, Any] = {
            "benchmark_material_id": material_id,
            "benchmark_source": material_source,
            "calculator": calculator_label,
            "backend": self.name,
            "backend_category": "toy_lj",
            "parameter_pack_provenance": provenance,
            "random_seed": int(seed),
            "n_insertions": int(n_samples),
            "warnings": [
                "toy_lj smoke calculator only — verdict §G. NOT chemically meaningful "
                "for adsorption interpretation."
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
            backend_label=f"toy_lj (epsilon={_TOY_EPSILON_EV} eV, sigma={_TOY_SIGMA_A} A — smoke only)",
            calculator_label=calculator_label,
            samples_origin="cuspai_widom",
            provenance=provenance,
        )


__all__ = ["ToyLennardJonesBackend"]
