# upstream_candidate/ — PREPARED, NOT SUBMITTED

These artifacts are drafted for possible contribution to kUPS. **Nothing here has been submitted,
filed, or sent.** No PR, no issue, no contact. Onur decides if/when/how any of it goes upstream.

Contents:
- `widom_validation_6c.yaml` — a ready-to-run kUPS `mcmc_widom` config for the Ar/all-silica MFI
  positive control, with the expected reference values inline. Proposed as a validation example for
  kUPS's (currently unvalidated, unreleased) Widom module.
- `UPSTREAM_NOTE.md` — the main note: independent validation on two controls (6c LJ, 4c Ewald), the
  +16 % K_H root-caused to an LJ shift convention (one-line doc suggestion + optional `shift:` flag),
  and the JAX-export faithfulness check (2.1e-7).
- `ISSUE_DRAFT_kH_normalization.md` — SUPERSEDED stub (the normalization hypothesis was falsified).
- `DOCS_NOTE_widom_omission.md` — a one-line README/docs note: the Monte-Carlo feature list omits the
  Widom module that exists in source.

All three are framed as *help from a first independent validator*, not criticism. The model is the
MACE team's, the engine is CuspAI's, and the tone matches that.
