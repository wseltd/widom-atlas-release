"""Bayesian K_H comparator in log space.

Compares the simulated K_H (with bootstrap-derived uncertainty from the
native Widom accumulator) against the experimental K_H (with literature-
scatter-derived uncertainty) on the log scale, returning a probability
of agreement under a Gaussian-in-log-K_H model.

The math:

  Let X = log10(K_H_simulated) ~ Normal(mu_sim, sigma_sim^2)
  Let Y = log10(K_H_experimental) ~ Normal(mu_exp, sigma_exp^2)
  Their difference D = X - Y ~ Normal(mu_sim - mu_exp, sigma_sim^2 + sigma_exp^2)

  Probability of agreement = P(|D| < d_threshold) for some band d_threshold.
  Z-score |mu_sim - mu_exp| / sqrt(sigma_sim^2 + sigma_exp^2).

This is the closest production-grade Bayesian framework adapted from
McCready, Sladekova, Conroy, Gomes, Fletcher, Jorge 2024 (J Chem Theory
Comput 20, 4869, DOI 10.1021/acs.jctc.4c00287), which uses a consensus
isotherm with uncertainty bands from an FF ensemble. We adapt it to
single-FF comparison against an experimental scatter band by treating
the experimental scatter as a prior on the reference K_H.

Note: this is NOT a full Bayesian FF-parameter UQ framework (KLIFF
PTMCMC is the reference for that). It is a lightweight comparison
adequate for headline disposition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BayesianKHResult:
    """Output of the Bayesian K_H comparison."""

    K_H_sim_mol_per_kg_per_bar: float
    K_H_sim_log10_std: float
    K_H_exp_mol_per_kg_per_bar: float
    K_H_exp_log10_std: float
    delta_log10: float
    combined_log10_std: float
    z_score: float
    p_agreement_within_1_sigma: float
    p_agreement_within_2_sigma: float
    p_agreement_within_tier_b_band: float | None
    tier_b_delta_log10_threshold: float | None
    classification: str
    note: str


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_sided_prob_within(d_threshold: float, mu: float, sigma: float) -> float:
    """P(|X - mu| < d_threshold) when X ~ Normal(mu, sigma^2)."""
    if sigma <= 0:
        return 1.0 if abs(mu) < d_threshold else 0.0
    z_hi = (d_threshold - mu) / sigma
    z_lo = (-d_threshold - mu) / sigma
    return _normal_cdf(z_hi) - _normal_cdf(z_lo)


def compare_K_H_in_log_space(
    K_H_sim_mol_per_kg_per_bar: float,
    K_H_sim_seed_values_mol_per_kg_per_bar: list[float],
    K_H_exp_mol_per_kg_per_bar: float,
    K_H_exp_log10_std: float | None = None,
    tier_b_delta_log10_threshold: float | None = None,
) -> BayesianKHResult:
    """Compare two K_H values in log space with Gaussian-in-log Bayesian framework.

    Parameters
    ----------
    K_H_sim_mol_per_kg_per_bar : float
        Atlas simulation mean K_H.
    K_H_sim_seed_values_mol_per_kg_per_bar : list[float]
        Per-seed K_H values for bootstrap log-std computation.
    K_H_exp_mol_per_kg_per_bar : float
        Experimental K_H reference value (best-estimate point).
    K_H_exp_log10_std : float | None
        Experimental Δlog10 uncertainty (1-sigma). If None, fall back
        to a conservative default keyed by Park 2017 / McCready 2024:
        0.20 (factor of ~1.6 scatter, the median MOF reproducibility).
    tier_b_delta_log10_threshold : float | None
        Tier B physical-accuracy band; if supplied, reports the
        probability the |delta| is within that band.
    """
    if K_H_sim_mol_per_kg_per_bar <= 0 or K_H_exp_mol_per_kg_per_bar <= 0:
        raise ValueError("K_H values must be positive for log-space comparison")

    sim_log_seeds = [
        math.log10(k) for k in K_H_sim_seed_values_mol_per_kg_per_bar if k > 0
    ]
    sim_log10_mean = math.log10(K_H_sim_mol_per_kg_per_bar)
    sim_log10_std = (
        float(np.std(sim_log_seeds, ddof=1)) if len(sim_log_seeds) >= 2 else 0.02
    )  # below 2 seeds, std is not informative

    if K_H_exp_log10_std is None:
        K_H_exp_log10_std = 0.20  # conservative Park 2017 median scatter default
    exp_log10_mean = math.log10(K_H_exp_mol_per_kg_per_bar)

    delta_log = sim_log10_mean - exp_log10_mean
    combined_std = math.sqrt(sim_log10_std ** 2 + K_H_exp_log10_std ** 2)
    z_score = delta_log / combined_std if combined_std > 0 else float("inf")

    p_1sigma = _two_sided_prob_within(combined_std, delta_log, combined_std)
    p_2sigma = _two_sided_prob_within(2.0 * combined_std, delta_log, combined_std)
    p_band = (
        _two_sided_prob_within(tier_b_delta_log10_threshold, delta_log, combined_std)
        if tier_b_delta_log10_threshold is not None
        else None
    )

    if abs(z_score) < 1.0:
        classification = "AGREEMENT_WITHIN_1_SIGMA"
    elif abs(z_score) < 2.0:
        classification = "AGREEMENT_WITHIN_2_SIGMA"
    elif abs(z_score) < 3.0:
        classification = "TENSION_2_TO_3_SIGMA"
    else:
        classification = "STRONG_DISAGREEMENT_GT_3_SIGMA"

    note = (
        f"Δlog10 = {delta_log:.3f} ± {combined_std:.3f} (sim 1σ ≈ {sim_log10_std:.3f}, "
        f"exp scatter 1σ ≈ {K_H_exp_log10_std:.3f}). |Z| = {abs(z_score):.2f}."
    )

    return BayesianKHResult(
        K_H_sim_mol_per_kg_per_bar=K_H_sim_mol_per_kg_per_bar,
        K_H_sim_log10_std=sim_log10_std,
        K_H_exp_mol_per_kg_per_bar=K_H_exp_mol_per_kg_per_bar,
        K_H_exp_log10_std=K_H_exp_log10_std,
        delta_log10=delta_log,
        combined_log10_std=combined_std,
        z_score=z_score,
        p_agreement_within_1_sigma=p_1sigma,
        p_agreement_within_2_sigma=p_2sigma,
        p_agreement_within_tier_b_band=p_band,
        tier_b_delta_log10_threshold=tier_b_delta_log10_threshold,
        classification=classification,
        note=note,
    )


# Per-system experimental K_H Δlog10 1-sigma defaults from literature
# scatter. These can be overridden per branch in YAML.
PER_SYSTEM_EXPERIMENTAL_KH_LOG10_STD: dict[str, float] = {
    # Open-metal-site MOFs: substantial synthesis scatter.
    "case_1_mg_mof_74_co2": 0.20,  # K_H window 384-414 vs Mason 381 -> ~0.04 log; +syntheses 2x = ~0.2
    "case_2_hkust_1_co2": 0.18,    # triangulation 5.5-9.0 mol/(kg.bar) -> log 0.74-0.95 -> ~0.10; +synthesis ~0.18
    "case_3_uio66_co2": 0.30,      # documented 5.14 vs 1.99 -> Δlog ~ 0.41; half = 0.20 -> conservative 0.30
    # All-silica zeolites: narrow scatter.
    "case_4_si_cha_co2": 0.10,     # Maghsoudi 2013 Toth fit; well-characterized
    "case_5_na_rho_co2": float("nan"),  # ensemble mismatch, not defined
    "case_6_mfi_small_gas": 0.05,  # Hufton 1993 vs Talu 1989 vs Sun 1998 cluster tightly
}
