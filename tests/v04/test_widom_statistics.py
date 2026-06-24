"""T023: Widom statistics tests (log-sum-exp, no clipping, ≥3 seeds)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from widom_atlas.v04.widom.driver import (
    henry_excess_to_K_H_mol_per_kg_per_Pa,
    run_widom_insertion,
)
from widom_atlas.v04.widom.estimators import estimate_KH_Qst_from_widom
from widom_atlas.v04.widom.logsumexp import LogSumExpAccumulator


def test_logsumexp_accumulator_zero_energy_matches_uniform() -> None:
    acc = LogSumExpAccumulator(beta_per_K=1.0 / 298.0)
    for _ in range(100):
        acc.add(0.0)
    # All weights = 1; Z = 100; <E> = 0
    assert acc.henry_excess_K == pytest.approx(1.0, rel=1e-12)
    assert acc.mean_energy_K == 0.0


def test_logsumexp_overflow_protection() -> None:
    """Large negative energies (deep wells) must not overflow."""
    acc = LogSumExpAccumulator(beta_per_K=1.0 / 298.0)
    for e in [-100000.0, -100001.0, -100002.0]:
        acc.add(e)
    # logZ is finite and large
    assert math.isfinite(acc.log_Z)
    assert acc.log_Z > 0


def test_logsumexp_skip_infinite_energy() -> None:
    """Overlap (inf energy) contributes weight 0 but increments N."""
    acc = LogSumExpAccumulator(beta_per_K=1.0 / 298.0)
    acc.add(math.inf)
    acc.add(0.0)
    assert acc.n_samples == 2
    # Z = 0 + 1 = 1; henry_excess = Z/N = 0.5
    assert acc.henry_excess_K == pytest.approx(0.5, rel=1e-12)


def test_widom_insertion_three_seeds_required() -> None:
    energy_fn = lambda f: 0.0  # noqa: E731
    with pytest.raises(ValueError, match="3 seeds"):
        run_widom_insertion(energy_fn, 298.0, 100, seeds=[1])


def test_widom_insertion_with_3_seeds_zero_energy() -> None:
    energy_fn = lambda f: 0.0  # noqa: E731
    summary = run_widom_insertion(energy_fn, 298.0, 100, seeds=[1, 2, 3])
    assert summary.n_seeds == 3
    # Zero-energy uniform → henry_excess = 1.0
    assert summary.mean_henry_excess_K == pytest.approx(1.0, abs=1e-12)


def test_widom_insertion_attractive_well() -> None:
    """A single deep well must produce henry_excess > 1.0."""
    rng = np.random.default_rng(42)
    def energy_fn(frac):
        # Gaussian well at (0.5, 0.5, 0.5) of depth -1000 K, width 0.1
        r2 = ((frac - 0.5) ** 2).sum()
        return -1000.0 * math.exp(-r2 / (2 * 0.01))
    summary = run_widom_insertion(energy_fn, 298.0, 5000, seeds=[1, 2, 3])
    assert summary.mean_henry_excess_K > 1.0


def test_henry_excess_conversion() -> None:
    K_H = henry_excess_to_K_H_mol_per_kg_per_Pa(
        henry_excess_K=2.5,
        temperature_K=298.0,
        framework_mass_kg_per_uc=3084.95e-3 / 6.02214076e23,  # one UC in kg
        uc_volume_m3=3130.23e-30,  # one UC in m^3
    )
    assert K_H > 0
    assert math.isfinite(K_H)


def test_estimate_KH_Qst_from_widom_simple() -> None:
    result = estimate_KH_Qst_from_widom(
        henry_excess_K=2.5,
        mean_energy_K=-2500.0,
        temperature_K=298.0,
        framework_mass_kg_per_uc=3084.95e-3 / 6.02214076e23,
        uc_volume_m3=3130.23e-30,
        n_seeds=3,
        n_insertions_total=15000,
    )
    assert result.K_H_mol_per_kg_per_Pa > 0
    assert result.Q_st_kJ_per_mol > 0  # positive-exothermic convention


def test_widom_seed_reproducibility() -> None:
    """Same seed → identical run."""
    energy_fn = lambda f: float(np.sin(f.sum()))  # noqa: E731
    s1 = run_widom_insertion(energy_fn, 298.0, 200, seeds=[42, 43, 44])
    s2 = run_widom_insertion(energy_fn, 298.0, 200, seeds=[42, 43, 44])
    assert s1.mean_henry_excess_K == pytest.approx(s2.mean_henry_excess_K)
    assert s1.results[0].henry_excess_K == pytest.approx(s2.results[0].henry_excess_K)


def test_no_clipped_boltzmann_weights_at_low_T() -> None:
    """At very low T, the partition function is dominated by the lowest energy.
    Naive sum-of-Boltzmann-weights would overflow / clip; log-sum-exp doesn't."""
    acc = LogSumExpAccumulator(beta_per_K=1.0 / 10.0)  # T=10 K
    for e in [-100.0, -50.0, 0.0, 50.0, 100.0]:
        acc.add(e)
    # log_Z must be finite and dominated by -100 K sample at beta=0.1
    assert math.isfinite(acc.log_Z)
    assert acc.log_Z >= -0.1 * (-100.0) - 1.0
