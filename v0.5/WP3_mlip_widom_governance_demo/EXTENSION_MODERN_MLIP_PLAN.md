# WP3 extension plan — modern-MLIP discrimination (PREPARED, not executed)

**Status:** plan only. Post-root-cause extension. Do not execute until 1b is resolved and the
operator approves.

## Goal
Move WP3 from **refusal** ("this MLIP-Widom number is untrustworthy") to **discrimination**
("refuse the unsafe MLIP-Widom output, *certify* the safe one") on the *same* OOD diagnostic.
This pre-empts the obvious rebuttal — "nobody uses 2023 small-MACE for adsorption" — by showing
the governance layer distinguishes a weak model from a stronger one rather than blanket-refusing.

## Candidate MLIPs (run the current MACE-MP small as the negative control)
| Model | Why | Notes |
|---|---|---|
| MACE-MP **medium / large** (same family) | direct "bigger model, fewer OOD overlaps?" test | drop-in via `mace_mp(model="medium"/"large")` |
| **MACE-MPA-0** (2024 MPtrj+Alexandria) | a stronger, more recent foundation model | check checkpoint availability + license |
| **MACE-OMAT / MatterSim / SevenNet / ORB** | independent architectures | each needs its own ASE calculator + weights |

## Install / environment requirements
- Python 3.13, torch **2.12.0+cu130** (already installed, sm_120 verified), device 0 only.
- Per-model: ASE calculator package + weights; verify cp313 + CUDA-13 compatibility *before*
  committing (some MLIP packages lag Python/CUDA). cuequivariance optional (deferred, beta).
- Expected GPU footprint: small (single-point energies on ≤ few-hundred-atom cells), ≪ 96 GB.

## Test systems
- Si-CHA + Ar (the WP3 control; light binder, clean overlaps).
- Add a moderate binder (e.g. UiO-66 + CO₂) and an OMS case (Mg-MOF-74 + CO₂) to probe whether
  stronger models still hallucinate attractive overlaps at open metal sites.

## OOD diagnostics (unchanged, form-agnostic)
Same two flags as WP3: hard-overlap geometry (< 0.80 σ_min) and energetic anomaly (U below a
physical floor, or far below the classical baseline at the same point). Report per-model:
flagged fraction, flagged Boltzmann-weight fraction, K_H with/without flagged, verdict.

## Success criteria
- **Discrimination demonstrated** if a stronger model yields a materially lower flagged-weight
  fraction (e.g. < 0.5 → "governed pass with flags") while small-MACE stays REFUSE — on the same
  systems, seeds, and diagnostic.
- **Honest null result** is also publishable: if *all* tested MLIPs are dominated by OOD overlaps
  at open metal sites, that strengthens "ungoverned MLIP-Widom is unsafe across models."

## Risks
- Some MLIP packages may not yet support Python 3.13 / CUDA 13 → may need a separate env.
- Checkpoint licensing/availability varies; record SHA-256 of every checkpoint used.
- Do not over-claim: this is a governance demonstration, not an MLIP benchmark.
