# FINAL_V04_VALIDATION_AUDIT

**Verdict: EVALUATOR PARITY FAILED**

_Generated: 2026-05-08T14:26:25Z_

## Parity gate (Phase C)

- RASPA3 reference parity passed: **False**
- RASPA3 reference skipped (no raspa3 binary): **True**
- MOFX-DB simin parity passed: **0/4** required → False
- Total parity rows recorded: 6
- Overall parity pass: **False**

## Flagship-case roll-up (Phase D, tier=flagship)

- Cases attempted: 6
- Cases the evaluator ran on: 6
- Cases passing tier threshold: 0
- Cases passing or with no reference (evaluator ran): 0
- Cases with missing structure or FF inputs: 0
- Required pass count for verdict PASS: ≥ 4

### Per-flagship-case detail

| case_id | framework | gas | T_K | status | log10_KH (int / ref / Δ) | Q_ads kJ/mol (int / ref / Δ) | pass |
|---------|-----------|-----|-----|--------|---------------------------|-------------------------------|------|
| flag-01-mg-mof-74-CO2-298 | Mg-MOF-74 | CO2 | 298.15 | ok | -4.282 / -2.699 / 1.583 | 17.722 / 43.000 / 25.278 | False |
| flag-02-hkust-1-CO2-298 | HKUST-1 | CO2 | 298.15 | ok | -4.652 / -3.398 / 1.254 | 17.147 / 27.000 / 9.853 | False |
| flag-03-uio-66-CO2-298 | UiO-66 | CO2 | 298.15 | ok | -4.222 / -4.000 / 0.222 | 18.546 / 22.000 / 3.454 | False |
| flag-04-cha-CO2-298 | CHA | CO2 | 298.15 | ok | -3.064 / -3.602 / 0.538 | 26.080 / 24.000 / 2.080 | False |
| flag-05-nak-a-CO2-298 | NaK-A | CO2 | 298.15 | ok | -2.468 / -2.097 / 0.371 | 34.767 / 39.000 / 4.233 | False |
| flag-06-mfi-CH4-Kr-298 | MFI | CH4 | 298.15 | ok | -2.969 / -5.398 / 2.429 | 27.770 / 20.000 / 7.770 | False |

## Tables (machine-readable, schema_version=0.4)

- T1: `benchmarks/results/v0.4-validation/tables/T1_flagship_case_results.json`
- T10: `benchmarks/results/v0.4-validation/tables/T10_blocker_rows.json`
- T2: `benchmarks/results/v0.4-validation/tables/T2_broad-tier_coverage_summary.json`
- T3: `benchmarks/results/v0.4-validation/tables/T3_exploratory-tier_coverage_summary.json`
- T4: `benchmarks/results/v0.4-validation/tables/T4_convergence_evidence_n_insertion_ladder.json`
- T5: `benchmarks/results/v0.4-validation/tables/T5_charge-scheme_sensitivity_ddec6___eqeq___pacman.json`
- T6: `benchmarks/results/v0.4-validation/tables/T6_gas-loaded_cif_site_match_summary.json`
- T7: `benchmarks/results/v0.4-validation/tables/T7_provenance_inventory_datasets_used.json`
- T8: `benchmarks/results/v0.4-validation/tables/T8_registry_status_at_run_time.json`
- T9: `benchmarks/results/v0.4-validation/tables/T9_mofx-db_simin_parity_deterministic_5.json`

## Verdict definitions

- **PASS**: parity gate green AND ≥4/6 flagship cases pass the tier threshold.
- **IMPLEMENTED BUT CASE COVERAGE INCOMPLETE**: parity green but some flagship inputs missing.
- **EVALUATOR PARITY FAILED**: parity gate not green (raspa3 not pass+not skipped, or <4/5 MOFX).
- **FAIL**: anything else.

## Notes

v0.4 actually-ran build: 6/6 atlas-eligible flagship cases completed the full pipeline (samples → density → basins → reports). Strict thresholds applied: flagship 0.10 / 2.0; broad 0.20 / 4.0; exploratory 0.40 / 7.0. 4 MOFX simin rows are blocked (see T10_blocker_rows.json + blockers.jsonl). raspa3 reference: NOT on PATH (skipped, audit verdict tolerates skip).
