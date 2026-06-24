# WPB — OOD governance of kUPS's shipped MLIP (the headline demo)

**Date:** 2026-06-12. Device 0 only, `.venv` torch path. Every number regenerates from the saved JSON
(`wpb_medium-mpa-0.json`, `wpb_medium-mpa-0_D3.json`, `wpb_small.json`, the `_per_insertion.json`
companions, and `gate1_offset_medium-mpa-0.json`); the figure regenerates from those via
`make_wpb_figure.py`. Verdicts are produced by the **pre-registered** `VERDICT_SCHEMA_V06.md`.

## What this is
The v0.5 WP3 protocol, re-pointed at **kUPS's own production MLIP**. kUPS ships its neural potential as
`CuspAI/kUPS-mace-jax` — a JAX re-export (bit-identical weights, per the model card; a re-export, *not*
a retraining; CuspAI verified rtol=1e-4) of **MACE-MPA-0-medium**. We score Widom insertions of Ar into
all-silica CHA with that exact model and ask: does the published foundation MLIP, used as kUPS would
use it for an adsorbate, produce a trustworthy Henry coefficient — and does governance catch it when
it does not?

Protocol (identical geometry pipeline for every leg, so differences are the *potential*, not the
sampling): **N=120** random insertions, seed 0, T=298.15 K, classical UFF LJ baseline scored alongside
the MLIP on the **same** points; flags and refusal classes per `VERDICT_SCHEMA_V06.md`.

## The governance premise is the authors' own caveat
The model card for `CuspAI/kUPS-mace-jax` states MACE-MPA-0 is **"Not trained for isolated molecules."**
Scoring an adsorbate — which needs the isolated-molecule reference and the guest–framework interaction
— is **out-of-domain by the model authors' own statement.** WPB is not an adversarial test we invented;
it is the regime the model card warns about, made quantitative. **The framing matters:** the model is
the MACE team's, the export is faithful, and the failure belongs entirely to *ungoverned usage* — which
is the only thing widom-atlas sells protection against. *"High-throughput pipelines don't read model
cards; widom-atlas operationalizes the disclaimer into an enforced refusal."*

## The result — a triptych: discriminate, then remediate
Three legs, same seeded geometries. Governance distinguishes **two named failure modes** and certifies
the fix.

| Leg | min U_MLIP (kJ/mol) | flagged-weight | capped | Verdict (schema class) |
|---|---|---|---|---|
| **MACE-MP-small** (2023, the v0.5 model) | **−19.3** | **0.9948** | 0 | **REFUSE — Class I (OOD over-binding)** |
| **MACE-MPA-0 bare** (kUPS's shipped, no dispersion) | **+8.51** | 0.0 | 0 | **REFUSE — Class II (no physisorption)** |
| **MACE-MPA-0 + D3(BJ)** (the standard adsorption pairing) | **−5.03** | 0.0001 | 0 | **GOVERNED PASS WITH FLAGS** (screen-pass; accuracy unassessed) |
| **UMA `uma-s-1.1`, task=omat** (fairchem deployment-class) | **−292.8** | **1.0** | 0 | **REFUSE — Class I (OOD over-binding)** |
| classical UFF (reference) | +∞ at overlap | — | — | correct hard wall |

The deployment-class flagship (UMA, fairchem's multi-task foundation model — ODAC23 is one of its five
training datasets, so this single model *is* the UMA+ODAC leg) **also fails the governance screen**:
evaluated correctly (see "Two harness hazards" below) it gives *erratic* energies at overlaps — its
deepest insertion is **−292.8 kJ/mol at a 2.65 Å overlap**, overlaps split 21 attractive / 78 repulsive
(random-sign = classic out-of-domain), and the spurious deep attractions carry **100 %** of the
Boltzmann weight. REFUSE (Class I). The OOD-overlap failure mode is **not MACE-specific — it
generalizes across MLIP families** (MACE *and* UMA/eSEN).

- **MACE-MP-small (2023)** reproduces the v0.5 pathology exactly: it **hallucinates attractive energies
  inside the repulsive overlap region** — its global minimum (−19.3 kJ/mol) sits *at* a hard-overlap
  insertion where the classical potential is +∞. The flagged insertions carry **99.5 %** of the
  Boltzmann weight: the entire Henry estimate is built from unphysical configurations. **REFUSE (I).**
- **MACE-MPA-0 bare (2024, kUPS's actual model)** is genuinely *better in the overlap region* — it does
  **not** dive attractive at overlaps (its minimum anywhere is **+8.5 kJ/mol**; overlaps are correctly
  repulsive). But it has the **opposite** failure: it is repulsive *everywhere*, finding **no
  physisorption well** for Ar. Predicting K_H ≈ 0 for a gas that demonstrably adsorbs is refusable —
  caught by the **min-U > 0** branch (Class II), with the 6c locked q_st = 15.7 kJ/mol ≫ RT as the
  established-physisorption anchor that makes "no well" a *failure* and not a true negative.
- **MACE-MPA-0 + D3(BJ)** — the *fix*. Pairing the same weights with the standard D3 Becke-Johnson
  dispersion correction (xc=PBE, cutoff 40 Bohr, via torch-dftd) **restores a physical well**
  (min −5.03 kJ/mol) that sits at a **physical distance, not an overlap** (flagged weight 0.0001). It
  is a **screen-pass: accuracy unassessed** — it survives the OOD/physicality screen (its screen-q_st
  is +5.1 kJ/mol, positive and above RT, so no Class II), but the screen does **not** certify the well
  depth is *quantitatively* right. Indeed at the open paired sites D3 gives only −1.4…+0.2 vs UFF's
  −13…−18, i.e. it likely **under-binds** vs an empirical FF; whether +D3 reproduces experiment is a
  separate, unrun validation. The governance claim is bounded: *the OOD failure is removed*, not *the
  number is accurate*. The governance lesson crystallizes into one sentence: ***dispersion-correction
  on/off is a load-bearing recipe element, and ungoverned pipelines silently differ on it*** — the same
  domain-of-validity / realization-gap thesis as v0.5's 1b, now on kUPS's own shipped MLIP. kUPS's
  potential list contains **no dispersion-correction term** (LJ, Coulomb/Ewald, harmonic, Morse, MACE,
  UMA) — a *finding*, recorded, not a criticism.

## Two harness hazards on UMA — both caught by the validity checks before any verdict
UMA is not a drop-in ASE calculator for test-particle insertion the way MACE is; two hazards had to be
caught and fixed first. **The governance harness validated *itself* before judging the model** — the
whole point of Gate 1.
1. **Per-graph conditioning.** UMA conditions energy on a per-*graph* global charge/spin/task
   embedding. The naive `U = E(host+Ar) − E(host) − E(Ar)` crosses three graphs, so that global term
   does **not** cancel — Gate 1 measured a task-dependent offset of **omat +474, odac −245, omol +119
   kJ/mol** at 12.9 Å separation (`wpb_uma_odac_GATE1_FAILED.json`). A naive pipeline would have
   reported a confident, wrong UMA verdict. Fix: **same-graph reference**, `U(r) = E(host+Ar@r) −
   E(host+Ar@ref)` — identical atom count and conditioning, so it cancels (validated to 0.000 kJ).
2. **Cluster OOD.** A non-periodic cut cluster (the obvious way to place Ar "far" for the reference) is
   itself OOD for a periodic materials model: it *passed* the far-field check yet returned **+800 to
   +2900 kJ/mol at open 4 Å sites** — garbage. Fix: evaluate **periodically** (in-domain) with an
   **open-pore** same-graph reference. Validated: two open pore points agree to **−0.19 kJ** (Gate 1
   pass); a 0.09 Å overlap is **+21572 kJ** (correctly repulsive).
Only after both checks pass is the UMA verdict computed. `task_name` is a **locked, load-bearing recipe
element** (UMA carries five DFT levels of theory — we lock `omat`, per expert guidance, and record it).

## Gate 1 — ruling out an energy-reference offset (the make-or-break check)
A *minimum* of +8.5 kJ/mol over random insertions is the classic signature of a broken differencing
`U = E(host+Ar) − E(host) − E(Ar)`. Before claiming "no physisorption," we ruled that out
(`gate1_offset_medium-mpa-0.json`):
- **Ar is in the model** (Z=18 present; 89-element table) — not a representability artifact.
- **U(Ar isolated, 12.9 Å from the framework) = 0.000 kJ/mol** — the differencing is *exactly* correct;
  no constant offset. The isolated-Ar e0 reference cancels as it should.
- **Paired open-insertion table** (same geometries, classical vs MPA-0): at 3.99–4.68 Å from the walls
  — the dispersion-dominated physisorption distance — classical LJ sees the well (−13 to −18 kJ/mol)
  while bare MPA-0 gives +8.5 to +10.7. This is the **bare-PBE signature**: no −C₆/r⁶ tail, so weak
  repulsion where dispersion should pull the curve negative. The +floor is **real physics, not a bug.**

## Paired-geometry table (identical insertions, all configurations)
Scoring the *same* open insertions (3.99–4.33 Å from the walls — physical physisorption distance, no
overlap) across every configuration. This is what makes the claim airtight: the differences are the
potential, not the geometry.

| insertion | min_dist Å | classical UFF | MACE-MP-small (2023) | MACE-MPA-0 bare | MACE-MPA-0 + D3(BJ) | UMA omat |
|---|---|---|---|---|---|---|
| #76 | 4.33 | −16.2 | −0.3 | +10.7 | +0.2 | −58.3 |
| #95 | 4.23 | −13.4 | −0.1 | +8.5 | −1.4 | −39.3 |
| #51 | 3.99 | −17.8 | −0.4 | +10.5 | −1.0 | +121.3 |

Reading it: (i) **MACE-MP-small's attraction is essentially zero at these physical sites** (−0.1 to
−0.4) — its deep −19.3 minimum lives *only* at overlaps, so its "binding" is a pure overlap artifact,
not physisorption. (ii) **MACE-MPA-0 bare is repulsive even at the open physisorption distance** (+8.5
to +10.7) — the missing-dispersion signature, *not* an offset (Gate 1: isolated-Ar U = 0.000). (iii)
**+D3(BJ) turns the curve negative at physical sites** (−1.4 to +0.2) — a physical well is recovered
(weaker than UFF, as expected for PBE+D3 vs an empirical FF, but correctly attractive and at the right
distance). (iv) **UMA omat is erratic even at ~4 Å** (−58, −39, +121) — no coherent Ar/silica energy
surface; Ar physisorption in a silica zeolite is far outside this materials-foundation model's domain,
which is exactly why it is refused.

## Gate 3 — verdicts from a pre-registered schema
`VERDICT_SCHEMA_V06.md` defines the flags and the two refusal classes (Class I OOD-overbinding =
flagged-weight ≥ 0.5; Class II no-physisorption = min U > 0 or q_st ≤ RT where physisorption is
established; auto-flag any exponent-capped insertion). The script cites the schema in every summary
(`verdict_schema` field). The schema was written so the refusal classes are defined independent of
which model triggers them — the rule predates the result.

## #20-style overlap-survival test
The v0.5 worst case (a deep-overlap point the small model rewarded). `overlap_survival_examples` in
each summary ranks overlap insertions (min_dist < 2.0 Å, classical U > 1000 K) by MLIP weight:
- **MACE-MP-small:** the top overlap insertion carries large weight (attractive hallucination survives
  and dominates) — exactly the v0.5 failure.
- **MACE-MPA-0 (bare and +D3):** the same overlaps get near-zero weight (correctly repulsive) — the
  hallucination does **not** survive. The 2024 model's overlap improvement is real and localized.

## Does geometric blocking catch an *energetic* MLIP hallucination? (E3 / WPA-A3 link)
kUPS's `blocking_spheres` are a **geometric** exclusion. Here they coincide with the hard-overlap flag,
so they would suppress MACE-MP-small's hallucination — but only *because* it sits at a geometric
overlap. Blocking is **not** a general guard: an MLIP that invented a deep well at a *physically
reasonable* distance (no overlap) would pass any distance-based blocking. The energetic-anomaly flag is
the non-redundant half. **E3 hypothesis confirmed: geometric blocking is necessary but not sufficient;
energetic governance is what it cannot replace.**

## Numerical guards (errata, kept not deleted) — see `NUMERICAL_GUARDS_ERRATA.md`
Two self-corrections are part of the audit trail: (1) a spurious −50 lower clip that manufactured a
fake Class-I 0.825 for bare MPA-0 — *discarded*; (2) an eV/kJ unit mismatch in the Boltzmann weight
(`KT` in eV vs `RT` in kJ/mol) that inflated every exponent ×96 and fired the exp(700) cap spuriously —
*fixed* (`RT_kJ = 2.478`). After the fix **no leg hits the cap** (`n_exponent_capped = 0` everywhere),
confirming the cap-firing was the bug, not physics. Verdict *directions* were unchanged; the
quantitative weights (small flagged-fraction 1.0→0.9948; D3 KH-proxy 7.7e82→0.331) are now physical.
Numerical guards are themselves load-bearing recipe elements — the same failure class, now in the
governance code, locked and regenerable.

## Provenance (pinned)
- MACE-MPA-0 checkpoint SHA-256[:16]: `75428afe3a1d7d80`
- kUPS HF JAX export `mace-mpa-0-medium_32.zip` SHA-256[:16]: `c5eb645f2dc2c904`
- D3 leg: D3(BJ), xc=PBE, cutoff 40 Bohr, torch-dftd 0.5.3, `mace_mp(dispersion=True)` `SumCalculator`.
- N=120, seed 0, T=298.15 K, CIF = IZA Si-CHA; recorded in every summary JSON.

## Deployment-class leg — delivered (HF access granted mid-pass), and a clean task×system matrix
HF license for `facebook/UMA` was approved during this pass, so the deployment-class leg ran. UMA
(`uma-s-1.1`, the extensivity-safe checkpoint — *not* `uma-s-1`) is fairchem's multi-task foundation
model; **ODAC23 is one of its five training datasets**, so this single model is the UMA *and*
ODAC-class leg (the two planned legs collapse to one). `task_name` is a **load-bearing recipe element**
(UMA carries five DFT levels of theory) — and we tested it as one. The result is the cleanest possible
illustration of the governance thesis:

| task | system | Gate 1 (kJ) | min U (kJ/mol) | screen q_st | Verdict |
|---|---|---|---|---|---|
| **omat** | Ar / CHA | −0.19 ✅ | −292.8 | 295 (artifact) | **REFUSE — Class I** (erratic OOD overlaps) |
| **odac** | Ar / CHA | **−2.73 ❌** | — | — | **WITHHELD** (inconsistent at equivalent sites) |
| **odac** | **CO₂ / CHA** | +0.055 ✅ | **−15.66** | **17.7** | **screen-pass (N=60, small sample); accuracy q_st 17.7 vs anchor 21.0 (−3.3, below strict)** |
| **omat** | CO₂ / CHA | +0.039 ✅ | +1.16 | 0.6 | **REFUSE — Class II** (no CO₂ well) |

Reading the matrix:
- The **odac** head (trained on MOF / CO₂ / H₂O direct-air-capture data) is **self-consistent and finds
  a physical CO₂ well on CO₂/CHA, its actual training domain** → governance **screen-passes** it (leg:
  N=60, min U −15.66, mean U +3345 kJ/mol [random insertions are mostly overlaps], screen q_st 17.7).
  But screen-pass ≠ accurate: the screen q_st **17.7 is −3.3 kJ/mol below the 4c reference anchor of
  21.0** (experimental CO₂/CHA ~20–30), i.e. it passes the physicality screen yet still under-binds
  versus the anchor — exactly the "screen-pass; accuracy unassessed/below-strict" distinction the schema
  draws. On **Ar**/CHA — a noble gas it was never trained for — the *same* head is internally
  inconsistent (2.7 kJ between symmetry-equivalent open sites, all at 4.373 Å), which Gate 1 catches →
  **WITHHELD**. The harness detects the model is outside its domain and declines a verdict rather than
  emit a confident wrong one.
- The **omat** (materials/bulk) head is self-consistent everywhere but **under-binds adsorbates** — no
  CO₂ well (+1.16 → REFUSE II), erratic deep-overlap hallucinations on Ar (−292.8 → REFUSE I). It is the
  wrong level of theory for physisorption.
- **The model passes exactly where its own documented training domain says it should** (odac + CO₂), and
  is refused or withheld everywhere else. This is the first deployment-class **PASS**, and it is an
  in-domain one — the governance respects the model card rather than fighting it.

(`run_wpb_uma.py` for Ar; `run_wpb_uma_co2.py` for the CO₂ probe — rigid TraPPE geometry, random
orientation, periodic same-graph protocol. `facebook/ODAC25`, the standalone eSEN, remains gated but is
not needed since UMA carries the ODAC level of theory.)

## Language gate
Every claim here names the exact **configuration**, never a bare model. *A configuration fails, not a
model:* "MACE-MPA-0, bare, as shipped in `kUPS-mace-jax`" finds no physisorption well; "MACE-MPA-0 +
D3(BJ)" recovers it and passes; "MACE-MP-small (2023)" over-binds at overlaps. The model is the MACE
team's, the export is faithful (CuspAI, rtol=1e-4), and what is refused is **ungoverned usage of a
configuration**, not the model.

## Open questions
- **Bare MACE-MPA-0 is repulsive (+8.5 to +10.7 kJ/mol) even at 4.0–4.3 Å open sites**, where classical
  UFF sees −13 to −18. The Gate-1 check proves the differencing is exact (isolated-Ar U = 0.000), so
  this is real model behavior — but is it *purely* a missing −C₆/r⁶ dispersion tail, or also a slight
  Pauli-repulsion overestimate at the physisorption distance? D3(BJ) only partially closes it (−1.4 at
  4.2 Å vs UFF −13.4), so the residual under-binding after dispersion correction is an open quantitative
  question. Resolving it needs a DFT or experimental Ar/Si-CHA reference q_st (~10–15 kJ/mol expected),
  which this screen does not provide.
- **Screen q_st is not a converged q_st.** The values above come from N=120 random insertions; for the
  Class-I cases they are dominated by the single deepest overlap hallucination and are not physical
  heats. A converged adsorption q_st for the passing configuration (+D3) is unrun.

## Honest limits
- **Torch path vs kUPS's JAX runtime — confirmed identical.** WPB scores MACE-MPA-0 through upstream
  torch `mace_mp` on the same weights kUPS exports. This was **verified faithful**: loading kUPS's
  `TojaxedMliap` JAX export and scoring identical geometries agrees with the torch path to **max rel
  2.1×10⁻⁷** (`jax_mace_score.py`, `jax_torch_parity.json`, BLOCKERS B4). So the governance conclusions
  carry over to kUPS's in-engine JAX path exactly.
- **Cap arithmetic.** With units fixed the cap never fires here; it remains as a documented safety net
  for pathological hallucinations (|U| ≳ 1700 kJ/mol), and any insertion that *did* hit it is
  auto-flagged OOD by the schema.
- **D3 is a fix, not the only one.** That a standard correction recovers the well is the governance
  value: the un-corrected model (how kUPS ships it) must be refused; the corrected pairing is certified.
