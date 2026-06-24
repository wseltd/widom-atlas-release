"""T005: typed pair-table force-field parser.

Produces a PairTable (keyed by ordered atom-label pairs) of FFTerm
records. No hand-transcribed numerical constants — all FF data must
come from either:

- The RASPA3 force_field.json (T024 produces these from the case matrix)
- Lin/Mercado SI rows (via decode_lin_mercado_row, T006)
- Dzubak SI rows (via decode_dzubak_row, T007)
- Explicit per-branch YAML in the locked case matrix
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .dzubak import decode_dzubak_row
from .lin_mercado import decode_lin_mercado_row
from .terms import LJ126, FFTerm, FunctionalForm, HardSphere


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass
class PairTable:
    terms: dict[tuple[str, str], FFTerm] = field(default_factory=dict)
    mixing_rule: str = "lorentz_berthelot"
    cutoff_angstrom: float = 12.0
    lj_treatment: str = "shifted_truncated"
    lj_tail_correction: bool = False

    def add(self, a: str, b: str, term: FFTerm) -> None:
        self.terms[_ordered_pair(a, b)] = term

    # Backward-compatible alias kept for callers using the original .set() name
    set = add

    def get(self, a: str, b: str) -> FFTerm | None:
        return self.terms.get(_ordered_pair(a, b))

    def keys(self) -> list[tuple[str, str]]:
        return sorted(self.terms.keys())

    def forms_present(self) -> frozenset[FunctionalForm]:
        return frozenset(t.form for t in self.terms.values())


def parse_lj126_self_terms(
    self_terms: Mapping[str, Mapping[str, float]],
) -> dict[str, LJ126]:
    """Convert {atom_label: {'epsilon_K': ..., 'sigma_angstrom': ...}} -> {label: LJ126}."""
    out: dict[str, LJ126] = {}
    for label, params in self_terms.items():
        out[label] = LJ126(
            epsilon_K=float(params["epsilon_K"]),
            sigma_angstrom=float(params["sigma_angstrom"]),
        )
    return out


def parse_cross_pair_table(
    cross_terms: Mapping[tuple[str, str] | str, Mapping[str, float]],
    *,
    kind: FunctionalForm,
    extra: Mapping[tuple[str, str], dict] | None = None,
) -> PairTable:
    """Build a PairTable from explicit cross-term entries.

    `cross_terms` keys must be 2-tuples (or 'a|b' strings); values are
    parameter mappings interpreted by `kind`.

    For LJ_12_6, the params are {epsilon_K, sigma_angstrom}.
    For BUCKINGHAM_A_EXP_C6, the params are {A, B, C, S_g?, C_already_scaled?}.
    For DZUBAK_A_EXP_C5_D6, the params are {A, B, C5, D6}.
    """
    table = PairTable()
    for key, params in cross_terms.items():
        if isinstance(key, str) and "|" in key:
            a, b = key.split("|", 1)
        else:
            a, b = key[0], key[1]
        if kind is FunctionalForm.LJ_12_6:
            term: FFTerm = LJ126(
                epsilon_K=float(params["epsilon_K"]),
                sigma_angstrom=float(params["sigma_angstrom"]),
            )
        elif kind is FunctionalForm.BUCKINGHAM_A_EXP_C6:
            term = decode_lin_mercado_row(
                {"A": float(params["A"]), "B": float(params["B"]), "C": float(params["C"])},
                S_g=float(params.get("S_g", 1.0)),
                C_already_scaled=bool(params.get("C_already_scaled", True)),
            )
        elif kind is FunctionalForm.DZUBAK_A_EXP_C5_D6:
            term = decode_dzubak_row(
                {
                    "A": float(params["A"]),
                    "B": float(params["B"]),
                    "C5": float(params["C5"]),
                    "D6": float(params["D6"]),
                }
            )
        elif kind is FunctionalForm.HARD_SPHERE:
            term = HardSphere(r_cut_angstrom=float(params["r_cut_angstrom"]))
        else:
            raise ValueError(f"Unsupported FunctionalForm: {kind}")
        table.add(a, b, term)
    return table


def serialize_pair_table_to_dict(table: PairTable) -> dict[str, object]:
    """Lossless dict view of PairTable used for evidence files + comparison tests."""
    out: dict[str, object] = {
        "mixing_rule": table.mixing_rule,
        "cutoff_angstrom": table.cutoff_angstrom,
        "lj_treatment": table.lj_treatment,
        "lj_tail_correction": table.lj_tail_correction,
        "terms": {},
    }
    terms_out = out["terms"]
    assert isinstance(terms_out, dict)
    for (a, b), term in sorted(table.terms.items()):
        terms_out[f"{a}|{b}"] = {"form": term.form.value, **vars(term)}
    return out
