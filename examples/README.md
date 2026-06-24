# widom-atlas examples

These templates are starting points for the v0.3 backend interfaces. **None
of the numbers in these templates are validated parameters.** Replace
every value with one whose source and redistribution rights you have
verified yourself.

## `external_samples_manifest_template.json`

Pair with a `samples.npz` produced by an external engine (RASPA3, OpenMM,
LAMMPS, kUPS, ML-FF, …). Save next to the npz as
`samples.npz.manifest.json`. The strict schema is
`widom_atlas.backends.schema.ExternalSampleManifest`; required fields:

- `sample_format_version` — must be `"0.3"`
- `framework`, `gas`, `temperature_K`, `n_insertions`
- `backend` — one of `toy_lj | parameterised_lj | user_parameterised_coulomb_lj | external_samples | raspa3_external | ml_external`
- `energy_unit` — one of `K | eV | kJ_mol | kcal_mol` (mandatory; widom-atlas refuses to guess)
- `parameter_mode` — declares which provenance category this run is in
- `force_field` — descriptive {framework_lj, framework_charges, gas_model, mixing_rules, electrostatics}
- `citations` — list of (role, doi, source); roles are
  `gas_model | framework_lj | framework_charges | engine | training_data | validation | other`
- `redistribution_status`
- `warnings` — must list any hybrid approximations
- `suitable_for_quantitative_interpretation` — `true` only when the operator
  takes responsibility for the FF combination

Validate with: `widom-atlas external-samples validate samples.npz.manifest.json`.

## `user_parameter_file_template.json`

Drives the `user_parameterised_coulomb_lj` backend
(`widom_atlas.backends.user_parameterised.UserParameterFile`). Required:

- `framework_atom_types` and `gas_sites` — lists of `(label, charge_e,
  sigma_A, epsilon_K, source, doi)`. Values can be null in the template,
  but the backend **will refuse to run** unless at least one framework
  atom AND at least one gas site carries a non-zero charge — and you do
  not pass `--allow-neutral-fallback`. Neutral fallback is documented as
  a warning in the manifest, not advertised as scientifically equivalent.
- `mixing_rules`: `Lorentz-Berthelot` or `user_supplied`.
- `electrostatics`: `Wolf` (default), `Ewald` (delegated to external
  engines), `external_engine`, or `none`.
- `redistribution_status`: must be set so the manifest carries it.
- `hybrid_warning`: free-text note when you mix parameter sources from
  different FF families (TraPPE adsorbates + DDEC framework charges +
  UFF framework LJ is a hybrid approximation, NOT a published validated
  force field — and the run will record this).

## Running with the templates

After replacing the placeholders:

```bash
# user_parameterised_coulomb_lj on UiO-66 + CO2
widom-atlas benchmark --set small --gas CO2 --n-samples 1000 \
    --backend user_parameterised_coulomb_lj \
    --params my_uio66_co2_params.json \
    --out /tmp/run_charge_aware

# Convert a RASPA3 run directory into a scalar-only sidecar
widom-atlas external-samples convert-raspa3 \
    --input my_raspa_run/ \
    --structure tests/fixtures/real_structures/UiO-66.cif \
    --gas CO2 --temperature 298.15 \
    --out /tmp/run_raspa3.scalar.json

# Compare backends side-by-side
widom-atlas compare-backends \
    /tmp/run_charge_aware /tmp/run_raspa3.scalar.json \
    --out /tmp/comparison
```

The `compare-backends` command writes `backend_comparison.{json,md}`
that lists KH, Q_ads, basin centroids, log-ratio vs reference, and a
yes/no/partial `improved?` flag against the LJ-only baseline within
each (material, gas) group.
