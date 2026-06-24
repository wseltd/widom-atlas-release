"""ASE Calculator wrapper around the native widom-atlas evaluator.

The upstream cusp-ai-oss/widom library is a pure NumPy/ASE Widom
estimator that delegates ALL energy evaluation to any ASE Calculator
it receives. By wrapping the native evaluator (which carries LJ 12-6 +
Buckingham + Dzubak + Ongari + Ewald + the polarizable extension) as
an ASE Calculator, we get a free third independent cross-engine check:

  native_widom_v04  (our estimator) -> our FF       -> per-branch verdict
  RASPA3 v3.0.29    (their estimator) -> their FF   -> cross-check
  cusp-ai-oss/widom (their estimator) -> our FF via -> cross-check
                                          this wrapper

Bugs that show up in the first but not the third are FF physics bugs;
bugs that show up in the third but not the second are upstream-widom
estimator bugs; bugs that show up in only the first are our estimator
bugs. This isolation is the main payoff.

The Calculator API used (ase.calculators.calculator.Calculator):

  results['energy'] : float, total energy in eV (ASE convention)
  results['forces'] : (N, 3) array in eV/Angstrom (optional)

We compute energy in K (the native evaluator's internal unit) and
convert to eV via the conversion factor.

This module is import-safe even when ASE is not installed; the
Calculator subclass is only defined on demand.
"""
from __future__ import annotations

import typing

import numpy as np

from .potentials import PairTable
from .system import NativeSystem

K_to_eV = 8.617333262e-5
"""Convert energy from K (k_B T at T = 1 K) to eV."""


def make_native_ase_calculator(
    native_system: NativeSystem,
    treat_all_atoms_as_test_particle: bool = False,
):
    """Construct an ASE Calculator that evaluates the energy of an ASE Atoms
    object against the native_system's pair_table.

    The wrapper interprets the ASE Atoms object as either:

      * `treat_all_atoms_as_test_particle=False` (default): the Atoms
        object is the full framework + probe; we look at the LAST
        n_probe_atoms atoms as the inserted probe and the rest as
        framework. Used when the upstream `widom` library inserts a
        whole molecule and asks for the full-system energy minus the
        framework reference.

      * `treat_all_atoms_as_test_particle=True`: the Atoms object IS
        the probe at its insertion pose; framework atoms come from
        the native_system. (Faster: re-uses the pre-built pair table.)

    Returns a Calculator subclass bound to this native_system.
    """
    from ase.calculators.calculator import Calculator, all_changes

    class WidomAtlasNativeCalculator(Calculator):
        implemented_properties: typing.ClassVar = ["energy"]  # type: ignore[misc]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._native_system = native_system
            self._probe_types = list(native_system.probe.types)
            self._n_probe = native_system.probe.n_atoms()
            self._treat_all_as_probe = treat_all_atoms_as_test_particle

        def calculate(
            self,
            atoms=None,
            properties=("energy",),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            if atoms is None:
                self.results["energy"] = 0.0
                return

            positions_A = np.asarray(atoms.get_positions())

            if self._treat_all_as_probe:
                probe_carts = positions_A
                framework_types, framework_carts = (
                    self._native_system.supercell_positions()
                )
            else:
                if len(positions_A) < self._n_probe:
                    raise ValueError(
                        f"ASE atoms has {len(positions_A)} positions but probe "
                        f"requires {self._n_probe}"
                    )
                probe_carts = positions_A[-self._n_probe :]
                framework_types, framework_carts = (
                    self._native_system.supercell_positions()
                )

            U_total_K = _evaluate_framework_probe_lj_energy_K(
                framework_types=framework_types,
                framework_carts=framework_carts,
                probe_types=self._probe_types,
                probe_carts=probe_carts,
                cell_matrix=self._native_system.supercell_cell(),
                pair_table=self._native_system.pair_table,
                cutoff_angstrom=self._native_system.energy_cutoff_angstrom,
            )

            self.results["energy"] = U_total_K * K_to_eV

    return WidomAtlasNativeCalculator()


def _evaluate_framework_probe_lj_energy_K(
    framework_types: list[str],
    framework_carts: np.ndarray,
    probe_types: list[str],
    probe_carts: np.ndarray,
    cell_matrix: np.ndarray,
    pair_table: PairTable,
    cutoff_angstrom: float,
) -> float:
    """LJ-only framework-probe interaction energy in K with minimum-image PBC.

    Mirrors the inner loop of `run_native_widom` but without insertion
    sampling -- evaluates one probe pose. Electrostatics are NOT included;
    when called through ASE, the upstream `widom` library does not require
    them (the test-particle energy can be defined as LJ-only for cross-
    check purposes).
    """
    inv_cell = np.linalg.inv(cell_matrix)
    cutoff_sq = cutoff_angstrom ** 2
    U_total = 0.0
    for ptype, p_pos in zip(probe_types, probe_carts, strict=True):
        d = p_pos[None, :] - framework_carts
        frac_d = d @ inv_cell
        frac_d -= np.round(frac_d)
        d = frac_d @ cell_matrix
        r2 = np.einsum("ij,ij->i", d, d)
        within = r2 < cutoff_sq
        if not np.any(within):
            continue
        r = np.sqrt(r2[within])
        framework_types_within = [framework_types[k] for k in np.where(within)[0]]
        u_arr = np.zeros_like(r)
        for ftype in set(framework_types_within):
            mask = np.array(
                [ft == ftype for ft in framework_types_within], dtype=bool
            )
            if not np.any(mask):
                continue
            u_arr[mask] = pair_table.pair_energy(ftype, ptype, r[mask])
        U_total += float(np.sum(u_arr))
    return U_total
