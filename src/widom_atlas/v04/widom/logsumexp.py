"""T020: numerically stable log-sum-exp partition-function accumulator.

A Widom-insertion-test sample contributes a Boltzmann weight
exp(-beta * E_test). Naive summation of these weights overflows or
catastrophically cancels at the well; log-sum-exp keeps the partition
function finite by factoring out the largest exponent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LogSumExpAccumulator:
    """Online log-sum-exp accumulator with running mean energy.

    State invariant: after N samples,
        Z = sum_{i=1..N} exp(-beta * E_i)
        <E> = Z^-1 * sum_{i=1..N} E_i * exp(-beta * E_i)

    Stored as (log_Z, <E_weighted>, N) for numerical stability.
    """

    beta_per_K: float
    _log_Z: float = -math.inf
    _energy_log_Z: float = -math.inf  # log( sum_i E_i * exp(-beta * E_i) )
    _energy_sign: float = 1.0
    _n: int = 0

    def add(self, energy_K: float) -> None:
        """Add a Widom-insertion sample with energy in Kelvin."""
        if not math.isfinite(energy_K):
            # Infinite (overlap) energy gets weight 0 — skip
            self._n += 1
            return
        log_w = -self.beta_per_K * energy_K  # log of Boltzmann weight
        # Update Z = exp(log_Z) + exp(log_w) → log_Z' = logaddexp
        if self._log_Z == -math.inf:
            self._log_Z = log_w
        else:
            self._log_Z = _logaddexp(self._log_Z, log_w)
        # Update sum of E_i * exp(-beta E_i): track log|sum| + sign
        log_term = math.log(abs(energy_K)) + log_w if energy_K != 0 else -math.inf
        term_sign = 1.0 if energy_K >= 0 else -1.0
        if self._energy_log_Z == -math.inf:
            self._energy_log_Z = log_term
            self._energy_sign = term_sign
        else:
            self._energy_log_Z, self._energy_sign = _logaddexp_signed(
                self._energy_log_Z, self._energy_sign, log_term, term_sign
            )
        self._n += 1

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def log_Z(self) -> float:
        return self._log_Z

    @property
    def mean_energy_K(self) -> float:
        """<E_K>_Boltzmann = (sum_i E_i exp(-beta E_i)) / (sum_i exp(-beta E_i))."""
        if self._n == 0 or self._log_Z == -math.inf or self._energy_log_Z == -math.inf:
            return 0.0
        return self._energy_sign * math.exp(self._energy_log_Z - self._log_Z)

    @property
    def henry_excess_K(self) -> float:
        """<exp(-beta E)>_uniform in K units, computed as (Z / N)."""
        if self._n == 0 or self._log_Z == -math.inf:
            return 0.0
        return math.exp(self._log_Z - math.log(self._n))


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    big, small = (a, b) if a >= b else (b, a)
    return big + math.log1p(math.exp(small - big))


def _logaddexp_signed(
    log_a: float, sign_a: float, log_b: float, sign_b: float
) -> tuple[float, float]:
    if sign_a == sign_b:
        return _logaddexp(log_a, log_b), sign_a
    # Different signs: difference; keep the dominant sign
    if log_a == log_b:
        return -math.inf, 1.0
    if log_a > log_b:
        diff = math.log1p(-math.exp(log_b - log_a))
        return log_a + diff, sign_a
    diff = math.log1p(-math.exp(log_a - log_b))
    return log_b + diff, sign_b
