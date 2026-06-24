# RASPA3 MFI Henry-coefficient parity fixture

These four files mirror the public RASPA3 example tree at:
[`iRASPA/RASPA3/examples/basic/12_mc_henry_coefficient_of_co2_n2_methane_in_mfi`](https://github.com/iRASPA/RASPA3/tree/main/examples/basic/12_mc_henry_coefficient_of_co2_n2_methane_in_mfi)

| File | Source | Licence |
|---|---|---|
| `simulation.json` | RASPA3 example tree (MIT) | MIT |
| `force_field.json` | RASPA3 example tree (MIT) | MIT |
| `raspa2_henry_C5-C9_MFI_simulation.input` | RASPA2 example `examples/Basic/10_HenryCoefficient_of_C5-C9_in_MFI` (MIT) | MIT |
| `CO2.json` | Operator-supplied component file mirroring the public TraPPE-style 3-site CO2 model. **Marked `licence_unverified`** in `_provenance`. | unverified |

The RASPA examples are MIT-licensed; the four files here were copied from the public RASPA2 / RASPA3 GitHub mirrors. The `CO2.json` component definition is derived from the public García-Sánchez 2009 (DOI 10.1021/jp9035419) and Potoff & Siepmann 2001 (DOI 10.1002/aic.690470719) parameters that the example `force_field.json` already references; it is included **only** as a test fixture and is explicitly stamped `licence_status = licence_unverified` per the v0.4 policy that widom-atlas does not bundle TraPPE as a redistributable parameter table.

These fixtures drive the v0.4 RASPA3 MFI Henry-coefficient parity gate in
`widom_atlas/evaluator/parity.py`. They are not used at install time; they are not
copied into the runtime package; they are only referenced from `tests/test_evaluator_*.py`
and from the validation suite under explicit `licence_unverified` provenance.
