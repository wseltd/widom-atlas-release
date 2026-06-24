# WPA-A3 — Blocking-sphere governance study (the v0.5 1b lesson as kUPS's own API)

**Date:** 2026-06-12. kUPS in `venv-kups`, device 0. Numbers regenerate from `blocking_sweep_6c.json`
and the kUPS runs on `configs/widom_6c_block_r{2.0,3.0}.yaml`.

## Why this matters
v0.5's headline failure class (1b): *"a published parameter table is not a complete recipe; the
load-bearing unpublished ingredient is the short-range protection convention."* kUPS makes that
convention an **explicit, configurable API** — `blocking_spheres` (`kups.potential.classical.blocking`):
per-host hard-sphere infinite-barrier exclusions (center + radius). So the v0.5 claim is no longer
ours to argue — it is **CuspAI's own engineering**: a recipe ported without the right blocking is, by
their API's existence, an incomplete recipe. A3 measures how load-bearing it is.

## Experiment
6c (MFI all-silica + Ar, full Ar-O + Ar-Si LB recipe — the WPA-A2 system). Blocking spheres placed on
64 framework-O positions (every 3rd O), radius swept; K_H and q_st vs blocking off. Block-average SEM
on kUPS K_H ≈ 3.4 %.

| blocking | spheres | radius (Å) | K_H (Å³/eV) | ΔK_H | q_st (kJ/mol) |
|---|---|---|---|---|---|
| **off** | 0 | — | 1.798×10⁷ | — | 16.25 |
| on | 64 | 2.0 | 1.856×10⁷ | +3.2 % | 16.34 |
| on | 64 | 3.0 | 1.812×10⁷ | +0.8 % | 16.28 |

## Finding — blocking is a *no-op* here, and that is the point
On a **physically-repulsive LJ recipe**, blocking changes K_H by less than the SEM and leaves q_st
unchanged. The reason is structural: the Ar-O/Ar-Si LJ already diverges at short range, so overlap
insertions carry ≈0 Boltzmann weight whether or not a hard sphere also excludes them. **Blocking is
redundant precisely when the potential is well-behaved.**

The corollary is the governance lesson: **blocking is load-bearing exactly when the recipe has an
*unphysical* short-range attraction the LJ does not repel** — the open-metal-site over-binding case,
and the v0.5 1b case (a Buckingham/exp-6 form whose `A·exp(−r/ρ)` core turns over to −∞ without a
short-range guard). There, removing the near-metal insertions is the difference between a physical and
a divergent K_H. So a recipe's blocking configuration is **part of the recipe and must be locked** — on
an all-silica LJ host it is invisible; on an OMS host it is decisive. Shipping a recipe without its
blocking spec is the same domain-of-validity failure as omitting the Ar-Si term (A2, F1) — silent where
the test is easy, fatal where it is not.

## Secondary observation (a small, separate insertion-count effect)
The small +0.8…+3.2 % K_H nudge from blocking is in the **insertion-count normalization**, not the
physics: excluding near-zero-weight overlap insertions changes the denominator of ⟨W⟩ = Σw/N, nudging
K_H up while q_st is untouched. Note this is a *distinct, much smaller* effect from the +16 % K_H of
WPA-A2 F2, which is now root-caused to the **shifted-vs-truncated LJ convention** (δ ≈ 0.43 kJ/mol; see
`F2_rootcause_shift.json`), not a normalization issue. Both are within the SEM here and neither affects
q_st.

## Scope (honest)
- **Negative control delivered** on a representable host (6c all-silica LJ) with real kUPS runs.
- **Positive (OMS) case scoped, not run.** The dramatic demonstration — blocking around an
  under-coordinated metal collapsing a divergent K_H to a physical value — needs a kUPS-representable
  OMS recipe with charges + Ewald (1c Becker reduced LJ+charges, or 2a UFF Cu + EPM2). That wiring is
  the documented next step (`../BLOCKERS.md` B3-adjacent). The exotic 1a/1b/1d Buckingham forms are
  **not representable in kUPS** (no Buckingham; `../BRIEF_ERRATA.md` E4), so the OMS-LJ recipes are the
  correct vehicle for the positive case.
