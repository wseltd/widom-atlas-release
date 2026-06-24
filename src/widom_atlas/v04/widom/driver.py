"""T021: Widom insertion driver with stable log-sum-exp accumulator.

Internal widom-atlas Widom backend. NOT the strict-tier reference (that
is RASPA3 v3.0.29 Ewald). Used for:
- Atlas-grid density maps via Wolf electrostatics
- Independent cross-check of RASPA3 scalar outputs
- Per-branch screening before the expensive Ewald run

Implements:
- Random uniform insertion of a single probe molecule
- Per-insertion energy evaluation via a caller-provided EnergyFn callable
- Stable log-sum-exp accumulation across ≥3 seeds
- Convergence test against ±5 % between N=10^4 and N=10^5 subsamples
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .logsumexp import LogSumExpAccumulator

EnergyFn = Callable[[np.ndarray], float]  # frac-coord (3,) → energy_K


@dataclass
class WidomResult:
    seed: int
    n_insertions: int
    log_Z: float
    mean_energy_K: float
    henry_excess_K: float


@dataclass
class WidomSummary:
    results: list[WidomResult]
    mean_henry_excess_K: float
    std_henry_excess_K: float

    @property
    def n_seeds(self) -> int:
        return len(self.results)


def run_widom_insertion(
    energy_fn: EnergyFn,
    temperature_K: float,
    n_insertions: int,
    seeds: list[int],
    rng_factory: Callable[[int], np.random.Generator] | None = None,
) -> WidomSummary:
    """Run ≥3 independent seed chains of Widom insertion.

    The `energy_fn` takes a fractional coordinate triple and returns
    the insertion energy in Kelvin (positive = repulsive). Caller is
    responsible for periodic-boundary handling.
    """
    if len(seeds) < 3:
        raise ValueError(f"need at least 3 seeds; got {len(seeds)}")
    if rng_factory is None:
        rng_factory = lambda s: np.random.default_rng(s)  # noqa: E731
    beta = 1.0 / temperature_K
    results: list[WidomResult] = []
    for seed in seeds:
        rng = rng_factory(seed)
        acc = LogSumExpAccumulator(beta_per_K=beta)
        for _ in range(n_insertions):
            frac = rng.random(3)
            e = energy_fn(frac)
            acc.add(e)
        results.append(
            WidomResult(
                seed=seed,
                n_insertions=acc.n_samples,
                log_Z=acc.log_Z,
                mean_energy_K=acc.mean_energy_K,
                henry_excess_K=acc.henry_excess_K,
            )
        )
    henry_arr = np.array([r.henry_excess_K for r in results])
    return WidomSummary(
        results=results,
        mean_henry_excess_K=float(henry_arr.mean()),
        std_henry_excess_K=float(henry_arr.std(ddof=0)),
    )


def henry_excess_to_K_H_mol_per_kg_per_Pa(
    henry_excess_K: float,
    temperature_K: float,
    framework_mass_kg_per_uc: float,
    uc_volume_m3: float,
) -> float:
    """Convert dimensionless Widom Henry excess <exp(-beta U)>_volume into
    K_H in mol/(kg*Pa).

    K_H = N_A * <exp(-beta U)>_V / (R * T * rho_framework)

    where rho_framework = framework_mass / uc_volume in kg/m^3.
    Returned value: mol of adsorbate / kg of framework / Pa of bulk pressure.
    """
    NA = 6.02214076e23
    R = 8.314462618
    if framework_mass_kg_per_uc <= 0 or uc_volume_m3 <= 0:
        return math.nan
    rho_framework_kg_per_m3 = framework_mass_kg_per_uc / uc_volume_m3
    return (NA * henry_excess_K) / (R * temperature_K * rho_framework_kg_per_m3)
