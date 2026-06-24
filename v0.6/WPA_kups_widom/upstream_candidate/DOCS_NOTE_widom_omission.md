# DRAFT DOCS NOTE (NOT SUBMITTED) — Widom module missing from the feature list

> Status: draft for Onur's review. Not filed.

The README's Monte-Carlo feature list reads "NVT and GCMC ensembles" and does not mention the **Widom
test-particle insertion** module that exists in source (`kups.mcmc.widom` — `widom_test()`,
`GhostProbe`, `WidomStatistics`; `kups.application.simulations.mcmc_widom`; `examples/mcmc_widom.yaml`;
`analyze_mcmc` reducing to μ_ex / K_H / q_st; console script `kups_mcmc_widom`).

Suggested one-line addition to the feature list:

> - **Widom test-particle insertion** for Henry coefficients, excess chemical potential, and isosteric
>   heat of adsorption (`kups_mcmc_widom`).

Note for maintainers: the Widom code is present in source `main` but not in the released PyPI package
(`kups 1.0.1`) — so the README is accurate *for the release* and behind *for source*. A note that the
module is unreleased/experimental would also help external users (we installed editable from source to
reach it).
