"""widom_atlas.validation — v0.4 release-gate validation suite.

This package wraps the evaluator + ingest layers into the six flagship
validation cases and the ladder/charge-sensitivity/parity sweeps the v0.4
release gate requires:

- ``case_runner``        : run one flagship validation case (framework × gas × T)
- ``ladder_runner``      : run the 3-tier (flagship/broad/exploratory) coverage ladder
- ``charge_sensitivity`` : sweep DDEC6/EQeq/PACMAN charge schemes for the same MOF/gas
- ``v04_tables``         : generate the 9 machine-readable tables required by the audit
- ``audit``              : assemble the final FINAL_V04_VALIDATION_AUDIT.md
"""
