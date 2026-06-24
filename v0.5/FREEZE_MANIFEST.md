# v0.5 FREEZE MANIFEST

**Frozen:** 2026-06-12
**Tag:** `v0.5-frozen` (annotated)
**Safety branch:** `freeze/v0.5`
**Freeze content commit:** `e1fbdcdf034e05b647b55a9661df26ef57fa1221`
**Tree hash:** `2ab8c7ac5f61eb8eddf7b9f54a36191f4871c54f`
**Work branch at freeze:** `article/v3-author-written-manuscript`

## Statement
This commit is the authoritative frozen baseline of all v0.5 work, exactly as v0.4 was
frozen before it. It includes:
- the full `v0.5/` evidence tree (WP1 source-paper parity, WP2 convergence + gRASPA GPU,
  WP3 MLIP-Widom OOD, the 1b realization-gap dossier, all reports, figures, CSV/JSON),
- the vendored source PDFs (Becker 2017 jp6b12052, Talu–Myers 2001),
- the **merged** `repair/v04-provenance` provenance corrections (6c.json Q_st reference
  17.0→15.7; becker_loader citation 2018→2017; REPAIR_LOG_v04_provenance.md).

**No v0.4 or v0.5 science file, verdict JSON, case matrix, report, figure, or vendored
artifact will be modified by v0.6 work.** All v0.6 work lives under a new `v0.6/` tree on
branch `v0.6/kups-retarget`. The merged repair/v04-provenance branch state is part of this
freeze. To restore this exact state: `git checkout v0.5-frozen` (or `freeze/v0.5`).
