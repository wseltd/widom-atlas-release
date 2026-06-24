# widom-atlas v0.4 / v0.5 validation plan

This is the operator-facing roadmap for the next two releases. It encodes
the v0.4 brief (research summary supplied with `implementation-verdict-continuation.txt`)
into a concrete plan that is wired to the registry under
`src/widom_atlas/data_registry/`.

The thresholds in §6 below match the **`v0_4_minimum`** and **`v0_5_broader`**
sets in `data_registry/data/thresholds.yaml`. Every dataset, scalar
reference, and site reference cited here is recorded in the registry with
full citation + DOI + license + redistribution status.

---

## 1. Goal of v0.4

Demonstrate that widom-atlas can produce **defensible adsorption results**
on a small set of priority MOFs with credible reference comparators. v0.4
is a *minimum* validation suite — not full chemical validation, but enough
to prove the v0.3 backend strategy delivers science.

## 2. v0.4 minimum scope

### 2.1 Materials (4–6 MOFs)

| material_id | refcode (CoRE-MOF) | rationale |
|---|---|---|
| Mg-MOF-74  | VOGTIV_clean_h | open-metal-site flagship case |
| Zn-MOF-74  | (operator-supplied via CoRE-MOF) | OMS analog; calibrates M-MOF-74 family |
| UiO-66     | RUBTAK04_clean | closed-shell physisorption case |
| ZIF-8      | OFERUN_clean | sodalite cage |

(Optional v0.4: HKUST-1 + MOF-5 if time permits.)

### 2.2 Gases

CO2 + CH4 (the v0.4 brief minimum set).

### 2.3 Backend pairings (per material × gas)

For every (material, gas) pair, run:

1. **`parameterised_lj`** baseline (LJ-only TraPPE+UFF — already in the
   package) at N=50,000 insertions.
2. **`user_parameterised_coulomb_lj`** with operator-supplied charges +
   TraPPE LJ at N=50,000 insertions. Sources for the charges (operator
   chooses one):
   - PACMOF-service output (per-MOF JSON, ~10 sec per material)
   - DDEC charges from CoRE-MOF-2019 (subset)
   - User's own DFT-derived DDEC6 charges
3. **`raspa3_external`** scalar comparator for at least one (material,
   gas) pair to anchor the comparison against an established engine.

### 2.4 Datasets used (registry-cited)

- `CRAFTED` (Zenodo 10.5281/zenodo.10120180, CDLA-Sharing-1.0) — KH / Q_ads
  reference for the 690-MOF screening backdrop. ~97k isotherms.
- `CoRE-MOF-DFT-2014-DDEC6` (Zenodo 10.5281/zenodo.3986569, CC-BY-4.0) —
  v0.4 fast-path: bypass operator-side EQeq / PACMOF runs by consuming
  pre-computed DDEC6 charges for ~3000 MOFs.
- `CoRE-MOF-2019` — structures (when v0.4 needs entries beyond the 2014
  set).
- `QMOF` (PACMAN charges) **or** `PACMOF-service` — operator's choice for
  v0.4 charges; CoRE-MOF-DFT-2014-DDEC6 is generally preferred for speed.
- `NIST-ISODB` (REST API, public-domain US gov) — experimental scalar
  references (Henry regime + Q_ads); cross-check against CRAFTED simulated.
- `MOFX-DB` (Snurr group, `mofdb_client`) — RASPA-simulated values for
  cross-engine sanity.
- `RASPA3-templates-MFI-henry` (iRASPA/RASPA3 examples, MIT) — canonical
  FF + gas templates that the v0.4 internal evaluator (§2.6 below) reads.
- `Dzubak-MgMOF74-CO2-FF` (paper SI, redistribution unverified) — when
  generic UFF undershoots Mg-MOF-74 OMS by ≥30 kJ/mol the operator
  extracts the QM-derived parameters into a UserParameterFile under
  their own provenance.
- `MACE-MP-0` / `ODAC25-MACE` / `UMA-FAIR-Chem` — ML-FF candidates for
  v0.5 (external_samples ingest path).
- Per-material literature scalars: see `data_registry/data/scalars.yaml`
  (Caskey 2008, Mancini 2016, Pandey 2025, Cmarik 2012, Park 2006,
  Walton 2008, McEwen 2013, Pérez-Pellitero 2010, Becker 2018).

### 2.6 Sample-generation engine choice (v0.4 follow-up brief §5)

**Critical change from v0.3.** The v0.4 follow-up brief confirmed that
RASPA / RASPA3 **do not export per-insertion XYZ positions or energies**:
``raspakit/mc_moves/component/widom.cpp`` overwrites the ``growData``
struct each step, and ``WriteMoviesEvery`` only tracks the accepted
trajectory of the combined run. Per-insertion atlas input therefore
cannot come from a vanilla RASPA3 run. Two paths remain:

1. **Internal evaluator** (recommended for v0.4): widom-atlas's own NumPy
   energy evaluator reads a force-field template from
   ``RASPA3-templates-MFI-henry`` (or any user-supplied template) and
   computes ``N_insertions`` random-pose energies in-process. The
   already-shipped ``user_parameterised_coulomb_lj`` backend is the
   first concrete instance of this path.
2. **Patched RASPA fork** (out of scope for v0.4 / v0.5): operator
   patches ``raspakit/mc_moves/component/widom.cpp`` to stream
   ``growData`` to disk and feeds the result into widom-atlas via
   ``write_canonical_external_samples``.

For scalar comparison only, the existing scalar-mode RASPA3 ingest
(``widom-atlas external-samples convert-raspa3 …``) is still the right
tool — RASPA3 KH / Q_ads scalars are the v0.4 strict-threshold reference
values.

### 2.5 Site validation

Use `widom_atlas.sites.match_basins_to_expected_sites` plus the new
`data_registry` entries:

- Mg-MOF-74 / OMS-A_endon (Queen 2014, DOI 10.1039/C4SC02064B)
- Mg-MOF-74 / OMS-B_secondary (Pandey 2025, DOI 10.1021/acs.langmuir.5c04277)
- UiO-66 / octahedral_cage_centre + tetrahedral_cage_centre
- ZIF-8 / sodalite_cage_centre
- (Optional) HKUST-1 / Cu_paddle_wheel
- (Optional) MOF-5 / alpha_pocket_near_Zn4O

## 3. v0.5 broader scope

| material_id | gases | rationale |
|---|---|---|
| v0.4 set + 6 more MOFs from CoRE-MOF | CO2, N2, CH4, H2, Xe, Kr, Ar | broader coverage |
| MFI, FAU, LTA, CHA zeolites | CO2, N2, CH4 | per the brief §8 v0.5 plan |

### 3.1 Backend pairings

- `user_parameterised_coulomb_lj` with **DDEC** charges (gold standard) +
  refined UFF for OMS metals + TraPPE adsorbates.
- ML potential option: fine-tuned NequIP or MACE per-MOF, ingested via
  `external_samples`.
- 8–10 documented gas-loaded structures for site validation.

### 3.2 Datasets added

- `ARC-MOF-MEPO-ML` (charges; verify license)
- `MOFSimBench` (reference scalars; verify license)
- Operator-trained ML force-field checkpoints (per MOF)

## 4. Convergence and runtime budget

| step | v0.4 | v0.5 |
|---|---|---|
| KH convergence | N ≥ 50,000 | N ≥ 100,000 |
| Q_ads convergence | N ≥ 100,000 | N ≥ 200,000 |
| Repeatability | 3 independent seeds, ±10% basin-weight agreement | 5 seeds, ±7.5% |
| Backend cross-check | RASPA3 vs widom-atlas total uptake within 5% | within 3% |

GPU acceleration via the brief's notes on gRASPA / MACE recommended for
v0.5 throughput.

## 5. Acceptance verdicts

A v0.4 row promotes to `within_range` when **all** of:

- |KH_log_ratio| ≤ 0.30 (relative ±20%)
- |ΔQ_ads| ≤ 3 kJ/mol vs the chosen reference
- basin centroid distance to crystallographic site < 1.5 Å (when site is
  registered in `data_registry/data/sites.yaml`)
- repeatability across 3 seeds within ±10% on basin weight

A `v0_4_strict` row (per the v0.4 follow-up brief §9) tightens this to:

- KH within ±5% of the **RASPA3 reference value** (not the experimental
  literature value, which has its own measurement uncertainty)
- |ΔQ_ads| ≤ 2 kJ/mol vs RASPA3 reference
- basin centroid distance < 1.0 Å from experimental neutron-diffraction
  centre-of-mass
- running mean of KH varies < 2% over the final 20% of insertions
- 50,000 — 100,000 insertions per unit cell minimum

A v0.5 row promotes to `within_range` when:

- |KH_log_ratio| ≤ 0.20 (relative ±15%)
- |ΔQ_ads| ≤ 2.5 kJ/mol
- basin centroid distance < 1.0 Å
- repeatability ±7.5%

The thresholds are encoded in `data_registry/data/thresholds.yaml` and
loadable via `widom_atlas.data_registry.load_threshold_set("v0_4_minimum")`,
`("v0_4_strict")`, `("v0_5_broader")`, `("flagship")`, or
`("broad_screening")`. The convergence study and scalar comparator
should read them from there rather than hard-code.

## 6. Risks and unresolved issues (from brief §8)

1. **ML force-fields show systematic PES softening at high energies.**
   Fine-tuning on single data points helps but is not a general cure.
   Mitigation: stick to physisorption regions for v0.4; reserve ML for v0.5.
2. **Water handling.** Many MOFs lose stability in humidity; Mg-MOF-74
   is particularly sensitive. v0.5 H2O entries gated on operator
   confirming material stability under simulation conditions.
3. **Framework flexibility.** Rigid-cell assumptions overestimate
   adsorption; flexible MLPs critical for accurate diffusion. v0.4
   stays rigid-cell; v0.5 adds optional flex via external_samples
   ingest of MD trajectories.
4. **Dispersion interactions.** Critical but unevenly represented in
   uMLIPs; D3 corrections recommended. Documented per ML run in the
   manifest's `force_field` descriptor.
5. **Transferability across ZIFs.** ML models trained on ZIF-8 fail on
   ZIF-3/4/6; explicit per-structure training is needed. Per-MOF NNPs
   are recorded as separate `external_samples` runs with their own
   provenance.
6. **License verification.** PACMOF, MEPO-ML, MOFSimBench redistribution
   status all marked `license_unverified` in the registry until the
   operator confirms with the upstream authors.

## 7. Operator workflow checklist

```bash
# 1. Inspect the registry — what's available and what's bundled
widom-atlas data list

# 2. Show one entry's full record (license, DOI, cache_path, citations)
widom-atlas data show CRAFTED
widom-atlas data show thresholds:v0_4_minimum

# 3. Validate operator-supplied YAML (e.g. their own scalar refs)
widom-atlas data validate my_scalars.yaml --kind scalar

# 4. After downloading datasets to the recommended cache_path, check
#    presence + sha256 verification status
widom-atlas data status

# 5. Run the v0.4 minimum suite
widom-atlas benchmark --set small --gas CO2 --n-samples 50000 \
    --backend user_parameterised_coulomb_lj \
    --params my_uio66_co2_params.json \
    --out /tmp/v04_uio66_co2

# 6. Compare backends side-by-side
widom-atlas compare-backends \
    /tmp/v04_uio66_co2 /tmp/v04_uio66_co2_lj_baseline /tmp/v04_uio66_co2_raspa.json \
    --out /tmp/v04_comparison
```

## 8. Final verdict criteria for v0.4 release

The package can be tagged `v0.4.0-validation-grade` when **all** of:

- ≥ 4 (material × gas) pairs each show at least one backend reaching
  v0_4_minimum thresholds.
- ≥ 1 OMS case (Mg-MOF-74 + CO2) reaches v0_4_minimum on Q_ads.
- ≥ 2 site references from `sites.yaml` reach `confidence ∈ {medium, high}`
  on at least one backend per pair.
- Backend cross-check (RASPA3 vs widom-atlas) within 5% on at least one
  pair.
- Every run in the report has full provenance: backend_category, charges
  source, force-field source, citations, redistribution_status.

If any of those fail, the verdict stays `engineering_preview` and the
audit report names the specific gap.
