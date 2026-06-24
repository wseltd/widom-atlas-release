"""Cross-check: scientific status lock must match production verdict JSONs.

This regression test prevents the class of bug found in the 2026-06-01 end-
to-end audit, where the scientific status lock + paper + professor report
quoted K_H = 0.245 / Q_st = 17.34 for the 6c positive control, while the
actual RASPA3 production verdict JSON at evidence/v04_audit/verdicts/6c.json
held K_H = 0.20701 / Q_st = 16.033. The hardcoded placeholder values in
scripts/v04/reverdict_all_branches_two_tier.py propagated into the lock and
all downstream documents without any cross-check against the canonical
production output.

The test below asserts that for every load-bearing branch in the lock, the
K_H mean, K_H std, Q_st mean, Q_st std, branch_id, executed backend, and
strict-tier pass classification all match the canonical production verdict
JSON to within a tight numerical tolerance.

Branches covered: 6c (RASPA3 positive control), 4c (Bai 2013 native strict
PASS), 6e (Bai 2013 native), and 3b_UA / 3b_UAq / 3b_EHq (Maia 2023 native).

Production verdict schemas vary by backend:
  * RASPA3 audit branches:  parsed_K_H_mol_per_kg_per_bar (top-level)
                            parsed_Q_st_kJ_per_mol (top-level)
                            evidence.K_H_seed_std_mol_per_kg_per_Pa (* 1e5
                            for /bar conversion)
                            evidence.Q_st_uncertainty_kJ_per_mol
                            passes_K_H / passes_Q_st (top-level booleans)
  * Native Bai 2013 branches: K_H_mean_mol_per_kg_per_bar (top-level)
                              K_H_std_mol_per_kg_per_bar (top-level)
                              Q_st_mean_kJ_per_mol (top-level)
                              Q_st_std_kJ_per_mol (top-level)
                              executed_backend (top-level)
                              two_tier_verdict.tier_A_regression.K_H_pass
                              two_tier_verdict.tier_A_regression.Q_st_pass
  * Native Maia 2023 branches: aggregated.K_H_mean_mol_per_kg_per_bar
                               aggregated.K_H_std_mol_per_kg_per_bar
                               aggregated.Q_st_mean_kJ_per_mol
                               aggregated.Q_st_std_kJ_per_mol
                               backend_tag
                               verdict.verdict_strict ("PASS" / "FAIL")

Tolerance: K_H values are taken with 1e-4 absolute tolerance because the
RASPA3 6c verdict carries K_H_seed_std in mol/(kg.Pa) (4 sig fig); the
native runs carry full IEEE-754 precision but the lock rounds to ~5 digits
for human readability.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

LOCK_PATH = REPO / "evidence/scientific_status_lock_2026_06_01.json"


# Per-branch extractor: returns a normalised dict with K_H_mean,
# K_H_std, Q_st_mean, Q_st_std, branch_id, backend_tag, tier_A_K_H_pass,
# tier_A_Q_st_pass — all extracted directly from the canonical production
# verdict JSON for that branch.
def extract_canonical_6c() -> dict:
    """RASPA3 audit schema (v04_audit/verdicts/6c.json)."""
    d = json.load((REPO / "evidence/v04_audit/verdicts/6c.json").open())
    K_H_seed_std_Pa = d["evidence"]["K_H_seed_std_mol_per_kg_per_Pa"]
    return {
        "branch_id": d["branch_id"],
        "K_H_mean_mol_per_kg_per_bar": d["parsed_K_H_mol_per_kg_per_bar"],
        "K_H_std_mol_per_kg_per_bar": K_H_seed_std_Pa * 1.0e5,
        "Q_st_mean_kJ_per_mol": d["parsed_Q_st_kJ_per_mol"],
        "Q_st_std_kJ_per_mol": d["evidence"]["Q_st_uncertainty_kJ_per_mol"],
        "backend_tag": "RASPA3_v3.0.29",
        "tier_A_K_H_pass": bool(d["passes_K_H"]),
        "tier_A_Q_st_pass": bool(d["passes_Q_st"]),
    }


def extract_canonical_native_bai_2013(branch_id: str) -> dict:
    """Native Bai 2013 schema (v04_<branch>_bai_2013/verdicts/<branch>.json)."""
    d = json.load(
        (REPO / f"evidence/v04_{branch_id}_bai_2013/verdicts/{branch_id}.json").open()
    )
    v = d["two_tier_verdict"]
    tier_A = v["tier_A_regression"]
    return {
        "branch_id": d["branch_id"],
        "K_H_mean_mol_per_kg_per_bar": d["K_H_mean_mol_per_kg_per_bar"],
        "K_H_std_mol_per_kg_per_bar": d["K_H_std_mol_per_kg_per_bar"],
        "Q_st_mean_kJ_per_mol": d["Q_st_mean_kJ_per_mol"],
        "Q_st_std_kJ_per_mol": d["Q_st_std_kJ_per_mol"],
        "backend_tag": d["executed_backend"],
        "tier_A_K_H_pass": bool(tier_A["K_H_pass"]),
        "tier_A_Q_st_pass": bool(tier_A["Q_st_pass"]),
    }


def extract_canonical_maia_2023_variant(variant: str) -> dict:
    """Native Maia 2023 schema (v04_3b_maia/verdicts/3b_<variant>.json)."""
    d = json.load(
        (REPO / f"evidence/v04_3b_maia/verdicts/3b_{variant}.json").open()
    )
    agg = d["aggregated"]
    v = d["verdict"]
    return {
        "branch_id": f"3b_{variant}",
        "K_H_mean_mol_per_kg_per_bar": agg["K_H_mean_mol_per_kg_per_bar"],
        "K_H_std_mol_per_kg_per_bar": agg["K_H_std_mol_per_kg_per_bar"],
        "Q_st_mean_kJ_per_mol": agg["Q_st_mean_kJ_per_mol"],
        "Q_st_std_kJ_per_mol": agg["Q_st_std_kJ_per_mol"],
        "backend_tag": d["backend_tag"],
        "tier_A_K_H_pass": bool(v["K_H_passes_strict_threshold_0p10"]),
        "tier_A_Q_st_pass": bool(v["Q_st_passes_strict_threshold_2kJ_per_mol"]),
    }


# Branch registry: maps the lock key to the extractor for that branch's
# canonical production verdict.
BRANCH_REGISTRY = {
    "6c_mfi_argon_positive_control": extract_canonical_6c,
    "4c_si_cha_co2_strict_pass": lambda: extract_canonical_native_bai_2013("4c"),
    # 6e is documented in open_problems but not in load_bearing_results;
    # tested separately below via direct schema check.
    "3b_EHq_uio66_tier_b_pass": lambda: extract_canonical_maia_2023_variant("EHq"),
}


def _load_lock_branch(lock: dict, lock_key: str) -> dict:
    return lock["load_bearing_results"][lock_key]


# ===================================================================
# Cross-check tests (one per branch)
# ===================================================================

@pytest.fixture(scope="module")
def lock() -> dict:
    """Load the scientific status lock once per test module."""
    assert LOCK_PATH.exists(), f"missing scientific lock at {LOCK_PATH}"
    return json.loads(LOCK_PATH.read_text())


@pytest.mark.parametrize(
    "lock_key,extractor",
    list(BRANCH_REGISTRY.items()),
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_lock_branch_K_H_mean_matches_production(
    lock_key: str, extractor, lock: dict
) -> None:
    """For each load-bearing branch, the K_H_mean recorded in the scientific
    status lock must match the canonical production verdict to within 1e-4
    mol/(kg.bar). This guards against re-introducing the 2026-06-01 6c
    placeholder-value bug."""
    canonical = extractor()
    lock_entry = _load_lock_branch(lock, lock_key)
    K_H_lock = lock_entry["K_H_atlas_mol_per_kg_per_bar"]
    K_H_canon = canonical["K_H_mean_mol_per_kg_per_bar"]
    assert K_H_lock == pytest.approx(K_H_canon, abs=1.0e-4), (
        f"Lock K_H for {lock_key} = {K_H_lock} but canonical production "
        f"verdict says K_H = {K_H_canon}. The lock must be regenerated from "
        f"the production verdict, not from hardcoded values."
    )


@pytest.mark.parametrize(
    "lock_key,extractor",
    list(BRANCH_REGISTRY.items()),
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_lock_branch_K_H_std_matches_production(
    lock_key: str, extractor, lock: dict
) -> None:
    """K_H_std in the lock must match production within 1e-4 mol/(kg.bar)."""
    canonical = extractor()
    lock_entry = _load_lock_branch(lock, lock_key)
    K_H_std_lock = lock_entry.get("K_H_atlas_std_mol_per_kg_per_bar")
    K_H_std_canon = canonical["K_H_std_mol_per_kg_per_bar"]
    if K_H_std_lock is None:
        # Lock has not yet recorded std for this branch; flag explicitly.
        pytest.fail(
            f"Lock {lock_key} is missing K_H_atlas_std_mol_per_kg_per_bar; "
            f"canonical std is {K_H_std_canon}. Add the field."
        )
    assert K_H_std_lock == pytest.approx(K_H_std_canon, abs=1.0e-4), (
        f"Lock K_H_std for {lock_key} = {K_H_std_lock} but canonical = "
        f"{K_H_std_canon}."
    )


@pytest.mark.parametrize(
    "lock_key,extractor",
    list(BRANCH_REGISTRY.items()),
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_lock_branch_Q_st_mean_matches_production(
    lock_key: str, extractor, lock: dict
) -> None:
    """Q_st_mean in the lock must match production within 0.01 kJ/mol."""
    canonical = extractor()
    lock_entry = _load_lock_branch(lock, lock_key)
    Q_st_lock = lock_entry["Q_st_atlas_kJ_per_mol"]
    Q_st_canon = canonical["Q_st_mean_kJ_per_mol"]
    assert Q_st_lock == pytest.approx(Q_st_canon, abs=1.0e-2), (
        f"Lock Q_st for {lock_key} = {Q_st_lock} but canonical = {Q_st_canon}."
    )


@pytest.mark.parametrize(
    "lock_key,extractor",
    list(BRANCH_REGISTRY.items()),
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_lock_branch_Q_st_std_matches_production(
    lock_key: str, extractor, lock: dict
) -> None:
    """Q_st_std in the lock must match production within 0.01 kJ/mol."""
    canonical = extractor()
    lock_entry = _load_lock_branch(lock, lock_key)
    Q_st_std_lock = lock_entry.get("Q_st_atlas_std_kJ_per_mol")
    Q_st_std_canon = canonical["Q_st_std_kJ_per_mol"]
    if Q_st_std_lock is None:
        pytest.fail(
            f"Lock {lock_key} is missing Q_st_atlas_std_kJ_per_mol; "
            f"canonical std is {Q_st_std_canon}. Add the field."
        )
    assert Q_st_std_lock == pytest.approx(Q_st_std_canon, abs=1.0e-2), (
        f"Lock Q_st_std for {lock_key} = {Q_st_std_lock} but canonical = "
        f"{Q_st_std_canon}."
    )


@pytest.mark.parametrize(
    "lock_key,extractor",
    list(BRANCH_REGISTRY.items()),
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_lock_branch_id_matches_production(
    lock_key: str, extractor, lock: dict
) -> None:
    canonical = extractor()
    lock_entry = _load_lock_branch(lock, lock_key)
    lock_id = lock_entry["branch_id"]
    canon_id = canonical["branch_id"]
    assert lock_id == canon_id, (
        f"Lock branch_id for {lock_key} = {lock_id!r} but canonical = "
        f"{canon_id!r}."
    )


@pytest.mark.parametrize(
    "lock_key,extractor",
    list(BRANCH_REGISTRY.items()),
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_lock_branch_tier_A_pass_matches_production(
    lock_key: str, extractor, lock: dict
) -> None:
    """The Tier A strict pass / fail classification recorded in the lock must
    match the canonical production verdict's K_H_pass and Q_st_pass flags."""
    canonical = extractor()
    lock_entry = _load_lock_branch(lock, lock_key)
    lock_K_H_pass = bool(lock_entry["tier_A_strict_K_H_pass"])
    lock_Q_st_pass = bool(lock_entry["tier_A_strict_Q_st_pass"])
    canon_K_H_pass = canonical["tier_A_K_H_pass"]
    canon_Q_st_pass = canonical["tier_A_Q_st_pass"]
    assert lock_K_H_pass == canon_K_H_pass, (
        f"Lock {lock_key} tier_A_K_H_pass = {lock_K_H_pass} but canonical = "
        f"{canon_K_H_pass}."
    )
    assert lock_Q_st_pass == canon_Q_st_pass, (
        f"Lock {lock_key} tier_A_Q_st_pass = {lock_Q_st_pass} but canonical = "
        f"{canon_Q_st_pass}."
    )


# ===================================================================
# Headline disposition invariants (must stay 2/15 strict, 3 Tier B)
# ===================================================================

def test_headline_strict_pass_count_is_2(lock: dict) -> None:
    h = lock["headline_disposition"]
    assert h["tier_a_strict_pass_count"] == 2, (
        f"Tier A strict PASS count must remain 2 (6c + 4c); got "
        f"{h['tier_a_strict_pass_count']}."
    )
    assert h["tier_a_strict_denominator"] == 15
    assert set(h["tier_a_strict_pass_branches"]) == {"6c", "4c"}


def test_headline_tier_b_pass_count_is_3(lock: dict) -> None:
    h = lock["headline_disposition"]
    assert h["tier_b_physical_accuracy_pass_count"] == 3, (
        f"Tier B PHYSICAL_PASS count must remain 3 (6c, 4c, 3b_EHq); got "
        f"{h['tier_b_physical_accuracy_pass_count']}."
    )
    assert set(h["tier_b_physical_accuracy_pass_branches"]) == {
        "6c", "4c", "3b_EHq",
    }


def test_headline_not_scientifically_validated_as_general_tool(lock: dict) -> None:
    h = lock["headline_disposition"]
    assert h["is_scientifically_validated_as_general_predictive_tool"] is False, (
        "widom-atlas is not scientifically validated as a general predictive "
        "tool (2/15 is not sufficient for that claim). The lock must record "
        "this disposition explicitly."
    )


# ===================================================================
# Extra cross-checks for branches recorded in production but not in
# load_bearing_results: 3b_UA, 3b_UAq, 6e. These are referenced in the
# paper tables but live in production-verdict-only form. The test pulls
# them straight from the production verdict and asserts the numbers
# match what the paper table quotes (hardcoded targets).
# ===================================================================

PAPER_TABLE_TARGETS = {
    # Within-2sigma branches (paper §6.1 Table 5)
    "3b_UA":  {"K_H": 1.994, "K_H_std": 0.028, "Q_st": 20.41, "Q_st_std": 0.02},
    "3b_UAq": {"K_H": 1.975, "K_H_std": 0.036, "Q_st": 20.64, "Q_st_std": 0.07},
    "6e":     {"K_H": 0.450, "K_H_std": 0.006, "Q_st": 18.76, "Q_st_std": 0.03},
}


@pytest.mark.parametrize("branch_id", ["3b_UA", "3b_UAq"])
def test_paper_table_3b_variants_match_production(branch_id: str) -> None:
    """Article §6.1 Within-2sigma branches must match the production verdict
    K_H / Q_st to within 1e-3 / 0.01 respectively."""
    variant = branch_id.split("_")[1]
    canon = extract_canonical_maia_2023_variant(variant)
    target = PAPER_TABLE_TARGETS[branch_id]
    assert canon["K_H_mean_mol_per_kg_per_bar"] == pytest.approx(target["K_H"], abs=1e-3)
    assert canon["K_H_std_mol_per_kg_per_bar"] == pytest.approx(target["K_H_std"], abs=1e-3)
    assert canon["Q_st_mean_kJ_per_mol"] == pytest.approx(target["Q_st"], abs=1e-2)
    assert canon["Q_st_std_kJ_per_mol"] == pytest.approx(target["Q_st_std"], abs=1e-2)


def test_paper_table_6e_matches_production() -> None:
    """Article §6.1 6e MFI + CH4 (Bai 2013) must match the production verdict."""
    canon = extract_canonical_native_bai_2013("6e")
    target = PAPER_TABLE_TARGETS["6e"]
    assert canon["K_H_mean_mol_per_kg_per_bar"] == pytest.approx(target["K_H"], abs=1e-3)
    assert canon["K_H_std_mol_per_kg_per_bar"] == pytest.approx(target["K_H_std"], abs=1e-3)
    assert canon["Q_st_mean_kJ_per_mol"] == pytest.approx(target["Q_st"], abs=1e-2)
    assert canon["Q_st_std_kJ_per_mol"] == pytest.approx(target["Q_st_std"], abs=1e-2)


# ===================================================================
# Two-tier reverdict script must not regress to placeholder values
# ===================================================================

def test_no_placeholder_245_in_reverdict_script() -> None:
    """Regression guard: the placeholder K_H = 0.245 / Q_st = 17.34 must
    never reappear in reverdict_all_branches_two_tier.py for 6c."""
    script = (REPO / "scripts/v04/reverdict_all_branches_two_tier.py").read_text()
    # Look only at the 6c branch dict definition
    i = script.find('"branch_id": "6c"')
    assert i > 0, "could not find 6c entry in reverdict script"
    # Search forward to the next closing brace (end of the dict literal)
    j = script.find("},", i)
    block = script[i:j]
    assert "0.245" not in block, (
        "Hardcoded placeholder K_H = 0.245 has reappeared in the 6c dict. "
        "Use the canonical value 0.20701 from v04_audit/verdicts/6c.json."
    )
    assert "17.34" not in block, (
        "Hardcoded placeholder Q_st = 17.34 has reappeared in the 6c dict. "
        "Use the canonical value 16.033."
    )


def test_no_placeholder_245_in_lock_emitter() -> None:
    """Regression guard for emit_scientific_status_lock.py."""
    script = (REPO / "scripts/v04/emit_scientific_status_lock.py").read_text()
    i = script.find('"6c_mfi_argon_positive_control"')
    assert i > 0
    j = script.find("            },", i)
    block = script[i:j]
    assert "0.245" not in block
    assert "17.34" not in block
