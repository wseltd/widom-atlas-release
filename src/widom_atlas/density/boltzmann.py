"""Boltzmann weights from interaction energies (eV) at temperature (K)."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from widom_atlas.core.constants import KB_EV_PER_K


def _validate_inputs(energies_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be a positive finite number; got {temperature_K!r}")
    arr = np.asarray(energies_eV, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"energies_eV must be 1-D; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        bad = int(np.flatnonzero(~np.isfinite(arr))[0])
        raise ValueError(f"energies_eV[{bad}]={arr[bad]!r} is not finite")
    return arr


def log_boltzmann_weights(energies_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    """Return log-weights ``log w_i`` summing to log(1) within numerical precision."""
    e = _validate_inputs(energies_eV, temperature_K)
    if e.size == 0:
        return e.copy()
    beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
    logw = -beta * e
    return logw - logsumexp(logw)


def boltzmann_weights(energies_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    """Return Boltzmann weights ``w_i`` summing to 1 within numerical precision."""
    return np.exp(log_boltzmann_weights(energies_eV, temperature_K))
