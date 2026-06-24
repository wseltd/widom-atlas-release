"""Published, citable Lennard-Jones parameters for adsorbates and frameworks.

This module is the *parameter source* for :mod:`widom_atlas.backends.parameterised_lj`.
Every value is taken verbatim from the cited primary literature and stored
together with the DOI so it can be audited. We do **not** redistribute any
force-field package's source code or proprietary data — these are scalar
LJ parameters from peer-reviewed papers, used here under fair-use as cited
constants.

Two parameter families are provided:

UFF (Rappé et al. 1992, JACS 114, 10024 — DOI ``10.1021/ja00051a040``):
    Universal Force Field — tabulated x_i (equilibrium pair separation) +
    D_i (well depth in kcal/mol) for every element from H to Lr. We use the
    LJ-12-6 functional form ``E(r) = D[(R_eq/r)^12 - 2(R_eq/r)^6]`` which is
    equivalent to ``E(r) = 4 ε [(σ/r)^12 - (σ/r)^6]`` with
    ``σ = R_eq / 2^(1/6)`` and ``ε = D``. Mixing rule is Lorentz-Berthelot
    by default in this module.

TraPPE — Transferable Potentials for Phase Equilibria — adsorbate models:
    - CO2 (Potoff & Siepmann 2001, AIChE J. 47, 1676 — DOI ``10.1002/aic.690470719``):
      3-site rigid linear C=O=C with C: σ=2.80 Å, ε/k_B=27.0 K;
      O: σ=3.05 Å, ε/k_B=79.0 K; bond 1.160 Å.
    - N2 (Potoff & Siepmann 2001, same paper): N: σ=3.31 Å, ε/k_B=36.0 K.
      (TraPPE-N2 includes a virtual COM charge site that we omit because we
      run LJ-only — the LJ atoms are reproduced exactly.)
    - CH4 (Martin & Siepmann 1998, JPC B 102, 2569 — DOI ``10.1021/jp972543+``):
      United-atom single-site CH4: σ=3.73 Å, ε/k_B=148.0 K. (When ASE
      ``molecule('CH4')`` returns a 5-atom representation we fall back to
      UFF-on-CH4 — see :data:`CH4_UFF_FALLBACK`.)

Why these and not (e.g.) UFF4MOF or DREIDING:
    - UFF4MOF (Addicoat 2014; Coupry-Addicoat 2016) requires an atom-typing
      pass (each carbon needs C_3 / C_R / C_2 etc.) which v1 does not
      automate.
    - DREIDING is normally used inside RASPA-style external tools and
      requires charge equilibration for charged frameworks — we keep
      charge handling out of v1 per the verdict §8.

Unit convention used internally:
    ``(eps_eV, sigma_A)`` tuples — eV for energy, Å for length. All
    dataclass entries below export the converted values plus the source.

If the requested element is not in a parameter pack the calculator returns
``(0.0, 0.0)`` (pair contribution = 0); the run still completes and the
unrecognised element is logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

KCAL_PER_MOL_TO_EV: Final[float] = 0.0433641043  # 1 kcal/mol = 0.0433641 eV
KELVIN_TO_EV: Final[float] = 8.617333262e-5  # k_B in eV/K
TWO_POWER_ONE_SIXTH: Final[float] = 2.0 ** (1.0 / 6.0)


@dataclass(frozen=True)
class LJEntry:
    """One element-or-site parameter line with provenance."""

    symbol: str
    eps_eV: float
    sigma_A: float
    citation: str
    doi: str


def _uff(symbol: str, x_i: float, D_i_kcal: float) -> LJEntry:
    """Build a UFF-derived :class:`LJEntry`. ``x_i`` is the LJ R_eq in Å,
    ``D_i_kcal`` is the well depth in kcal/mol per Rappé 1992 Table 1."""
    sigma_A = x_i / TWO_POWER_ONE_SIXTH
    eps_eV = D_i_kcal * KCAL_PER_MOL_TO_EV
    return LJEntry(
        symbol=symbol,
        eps_eV=eps_eV,
        sigma_A=sigma_A,
        citation="Rappé, Casewit, Colwell, Goddard, Skiff. UFF: A full periodic table force field for molecular mechanics and molecular dynamics simulations. JACS 114, 10024 (1992).",
        doi="10.1021/ja00051a040",
    )


# ---------- UFF (covers full periodic table; relevant subset for v1 MOFs)
UFF_TABLE: Final[dict[str, LJEntry]] = {
    e.symbol: e
    for e in (
        _uff("H", x_i=2.886, D_i_kcal=0.044),
        _uff("He", x_i=2.362, D_i_kcal=0.056),
        _uff("Li", x_i=2.451, D_i_kcal=0.025),
        _uff("Be", x_i=2.745, D_i_kcal=0.085),
        _uff("B", x_i=4.083, D_i_kcal=0.180),
        _uff("C", x_i=3.851, D_i_kcal=0.105),
        _uff("N", x_i=3.660, D_i_kcal=0.069),
        _uff("O", x_i=3.500, D_i_kcal=0.060),
        _uff("F", x_i=3.364, D_i_kcal=0.050),
        _uff("Ne", x_i=3.243, D_i_kcal=0.042),
        _uff("Na", x_i=2.983, D_i_kcal=0.030),
        _uff("Mg", x_i=3.021, D_i_kcal=0.111),
        _uff("Al", x_i=4.499, D_i_kcal=0.505),
        _uff("Si", x_i=4.295, D_i_kcal=0.402),
        _uff("P", x_i=4.147, D_i_kcal=0.305),
        _uff("S", x_i=4.035, D_i_kcal=0.274),
        _uff("Cl", x_i=3.947, D_i_kcal=0.227),
        _uff("Ar", x_i=3.868, D_i_kcal=0.185),
        _uff("K", x_i=3.812, D_i_kcal=0.035),
        _uff("Ca", x_i=3.399, D_i_kcal=0.238),
        _uff("Sc", x_i=3.295, D_i_kcal=0.019),
        _uff("Ti", x_i=3.175, D_i_kcal=0.017),
        _uff("V", x_i=3.144, D_i_kcal=0.016),
        _uff("Cr", x_i=3.023, D_i_kcal=0.015),
        _uff("Mn", x_i=2.961, D_i_kcal=0.013),
        _uff("Fe", x_i=2.912, D_i_kcal=0.013),
        _uff("Co", x_i=2.872, D_i_kcal=0.014),
        _uff("Ni", x_i=2.834, D_i_kcal=0.015),
        _uff("Cu", x_i=3.495, D_i_kcal=0.005),
        _uff("Zn", x_i=2.763, D_i_kcal=0.124),
        _uff("Ga", x_i=4.383, D_i_kcal=0.415),
        _uff("Ge", x_i=4.280, D_i_kcal=0.379),
        _uff("As", x_i=4.230, D_i_kcal=0.309),
        _uff("Se", x_i=4.205, D_i_kcal=0.291),
        _uff("Br", x_i=4.189, D_i_kcal=0.251),
        _uff("Kr", x_i=4.141, D_i_kcal=0.220),
        _uff("Rb", x_i=4.114, D_i_kcal=0.040),
        _uff("Sr", x_i=3.641, D_i_kcal=0.235),
        _uff("Zr", x_i=3.124, D_i_kcal=0.069),
        _uff("Mo", x_i=3.052, D_i_kcal=0.056),
        _uff("Ru", x_i=2.963, D_i_kcal=0.056),
        _uff("Rh", x_i=2.929, D_i_kcal=0.053),
        _uff("Pd", x_i=2.899, D_i_kcal=0.048),
        _uff("Ag", x_i=3.148, D_i_kcal=0.036),
        _uff("Cd", x_i=2.848, D_i_kcal=0.228),
        _uff("In", x_i=4.463, D_i_kcal=0.599),
        _uff("Sn", x_i=4.392, D_i_kcal=0.567),
        _uff("Sb", x_i=4.420, D_i_kcal=0.449),
        _uff("Te", x_i=4.470, D_i_kcal=0.398),
        _uff("I", x_i=4.500, D_i_kcal=0.339),
        _uff("Xe", x_i=4.404, D_i_kcal=0.332),
        _uff("Cs", x_i=4.517, D_i_kcal=0.045),
        _uff("Ba", x_i=3.703, D_i_kcal=0.364),
        _uff("La", x_i=3.522, D_i_kcal=0.017),
        _uff("Ce", x_i=3.556, D_i_kcal=0.013),
        _uff("Hf", x_i=3.295, D_i_kcal=0.072),
        _uff("Ta", x_i=3.149, D_i_kcal=0.081),
        _uff("W", x_i=2.963, D_i_kcal=0.067),
        _uff("Re", x_i=2.954, D_i_kcal=0.066),
        _uff("Os", x_i=3.120, D_i_kcal=0.037),
        _uff("Ir", x_i=2.840, D_i_kcal=0.073),
        _uff("Pt", x_i=2.754, D_i_kcal=0.080),
        _uff("Au", x_i=3.293, D_i_kcal=0.039),
        _uff("Hg", x_i=2.705, D_i_kcal=0.385),
        _uff("Tl", x_i=4.347, D_i_kcal=0.680),
        _uff("Pb", x_i=4.297, D_i_kcal=0.663),
    )
}


# ---------- TraPPE adsorbate parameters
def _trappe(symbol: str, eps_K: float, sigma_A: float, citation: str, doi: str) -> LJEntry:
    return LJEntry(
        symbol=symbol,
        eps_eV=eps_K * KELVIN_TO_EV,
        sigma_A=sigma_A,
        citation=citation,
        doi=doi,
    )


TRAPPE_CO2: Final[dict[str, LJEntry]] = {
    "C": _trappe(
        symbol="C",
        eps_K=27.0,
        sigma_A=2.80,
        citation="Potoff & Siepmann. Vapor-liquid equilibria of mixtures containing alkanes, carbon dioxide, and nitrogen. AIChE J. 47, 1676 (2001).",
        doi="10.1002/aic.690470719",
    ),
    "O": _trappe(
        symbol="O",
        eps_K=79.0,
        sigma_A=3.05,
        citation="Potoff & Siepmann. Vapor-liquid equilibria of mixtures containing alkanes, carbon dioxide, and nitrogen. AIChE J. 47, 1676 (2001).",
        doi="10.1002/aic.690470719",
    ),
}

TRAPPE_N2: Final[dict[str, LJEntry]] = {
    "N": _trappe(
        symbol="N",
        eps_K=36.0,
        sigma_A=3.31,
        citation="Potoff & Siepmann. AIChE J. 47, 1676 (2001) — TraPPE-N2 LJ atoms (the COM partial-charge site is omitted in our LJ-only run).",
        doi="10.1002/aic.690470719",
    ),
}

# CH4 — TraPPE united-atom would model CH4 as a single LJ site, but ASE's
# ``molecule('CH4')`` returns 5 explicit atoms (C + 4 H). For an
# explicit-atom Widom run we therefore use UFF for CH4 atoms. This is
# documented in ``CH4_UFF_FALLBACK`` so the choice is auditable.
CH4_UFF_FALLBACK: Final[bool] = True


def gas_parameter_pack(gas: str) -> dict[str, LJEntry]:
    """Return the TraPPE-derived LJ pack for the named gas; falls back to
    UFF for CH4 (see module docstring)."""
    g = gas.upper()
    if g == "CO2":
        return dict(TRAPPE_CO2)
    if g == "N2":
        return dict(TRAPPE_N2)
    if g == "CH4":
        return {sym: UFF_TABLE[sym] for sym in ("C", "H") if sym in UFF_TABLE}
    raise ValueError(
        f"unsupported gas {gas!r}; v1 backends support CO2, N2, CH4"
    )


def framework_parameter_pack() -> dict[str, LJEntry]:
    """The framework parameter pack is UFF (covers all elements relevant to
    v1 MOFs). Returns a copy to avoid accidental mutation."""
    return dict(UFF_TABLE)


def parameter_pack_provenance(gas: str) -> dict[str, str]:
    """Provenance dict suitable for stamping into manifest.json."""
    g = gas.upper()
    if g == "CO2":
        gas_doi = TRAPPE_CO2["C"].doi
        gas_label = "TraPPE-CO2 (Potoff & Siepmann 2001)"
    elif g == "N2":
        gas_doi = TRAPPE_N2["N"].doi
        gas_label = "TraPPE-N2 (Potoff & Siepmann 2001)"
    elif g == "CH4":
        gas_doi = UFF_TABLE["C"].doi
        gas_label = "UFF (Rappé 1992) — CH4 explicit-atom fallback (see widom_atlas.backends.parameters.CH4_UFF_FALLBACK)"
    else:
        gas_doi = ""
        gas_label = ""
    return {
        "gas_pack": gas_label,
        "gas_pack_doi": gas_doi,
        "framework_pack": "UFF (Rappé 1992)",
        "framework_pack_doi": UFF_TABLE["C"].doi,
        "mixing_rule": "Lorentz-Berthelot",
        "energy_unit_conversion": (
            "kcal/mol -> eV via 0.0433641 (UFF); K -> eV via k_B = 8.617e-5 (TraPPE)"
        ),
    }


__all__ = [
    "CH4_UFF_FALLBACK",
    "TRAPPE_CO2",
    "TRAPPE_N2",
    "UFF_TABLE",
    "LJEntry",
    "framework_parameter_pack",
    "gas_parameter_pack",
    "parameter_pack_provenance",
]
