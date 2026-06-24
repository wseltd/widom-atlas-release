"""Force-field runtime tables for the evaluator.

Lifts a UserParameterFile (or a freshly-parsed RASPA3 bundle) into NumPy
tables the energy kernel can index without any per-step dict lookups:

- ``framework_atom_table``  : structured (label, sigma_A, epsilon_K, charge_e)
- ``gas_atom_table``        : same shape, indexed by component site labels
- ``mixing_rule``           : Lorentz-Berthelot vs. user_supplied
- ``coulomb_method``        : Wolf | Ewald | none
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from widom_atlas.backends.user_parameterised import UserParameterFile


@dataclass(frozen=True)
class AtomTable:
    """Lookup table indexed by integer atom_id, with parallel NumPy arrays."""

    labels: tuple[str, ...]
    sigma_A: np.ndarray
    epsilon_K: np.ndarray
    charge_e: np.ndarray

    @property
    def n(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class FFTables:
    framework: AtomTable
    gas: AtomTable
    mixing_rule: str  # "Lorentz-Berthelot" or "user_supplied"
    coulomb_method: str  # "Wolf" | "Ewald" | "none"
    redistribution_status: str
    hybrid_warning: str | None


def _to_table(entries: list) -> AtomTable:
    labels = tuple(e.label for e in entries)
    return AtomTable(
        labels=labels,
        sigma_A=np.array([e.sigma_A for e in entries], dtype=float),
        epsilon_K=np.array([e.epsilon_K for e in entries], dtype=float),
        charge_e=np.array([float(e.charge_e or 0.0) for e in entries], dtype=float),
    )


def load_user_parameter_file(path: Path) -> UserParameterFile:
    """Read & validate a UserParameterFile JSON."""
    return UserParameterFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def lift_to_tables(upf: UserParameterFile) -> FFTables:
    """Project a validated UserParameterFile into runtime FF tables."""
    return FFTables(
        framework=_to_table(upf.framework_atom_types),
        gas=_to_table(upf.gas_sites),
        mixing_rule=upf.mixing_rules,
        coulomb_method=upf.electrostatics,
        redistribution_status=upf.redistribution_status,
        hybrid_warning=upf.hybrid_warning,
    )


def lj_pair_tables(framework: AtomTable, gas: AtomTable, mixing_rule: str) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute (n_fw, n_gas) σ_ij and ε_ij arrays.

    For ``Lorentz-Berthelot``:
      σ_ij = (σ_i + σ_j) / 2,    ε_ij = sqrt(ε_i ε_j)

    For ``user_supplied`` we still apply LB by default — the caller is
    expected to overwrite specific (i, j) cells from explicit pair entries
    before invoking the energy kernel. We log a placeholder warning the
    parity layer can pick up.
    """
    sig_i = framework.sigma_A[:, None]  # (n_fw, 1)
    sig_j = gas.sigma_A[None, :]        # (1, n_gas)
    eps_i = framework.epsilon_K[:, None]
    eps_j = gas.epsilon_K[None, :]
    sigma = 0.5 * (sig_i + sig_j)
    epsilon = np.sqrt(eps_i * eps_j)
    return sigma, epsilon


__all__ = [
    "AtomTable",
    "FFTables",
    "lift_to_tables",
    "lj_pair_tables",
    "load_user_parameter_file",
]
