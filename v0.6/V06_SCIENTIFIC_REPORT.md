# widom-atlas v0.6 — Scientific Report

**Retargeting a recipe-governance layer onto CuspAI's kUPS engine and its shipped MLIP**

**Onur Alexander William Aydogan** — Waynestark Enterprises Limited. Licensed Apache-2.0.

*Si-CHA / MFI + Ar and CO₂ test systems · device-0 GPU · 2026-06-12. **The evidence base is the
version-controlled repository state: every number in this document traces to a committed JSON/CSV and
regenerates from the figure scripts (see the repo README / `CLAIMS.md`); checkpoint and export hashes
are pinned inline.** This document makes no claim the committed evidence does not support.*

---

## 1. Thesis

A published force-field parameter table is **not** a complete simulation recipe. The load-bearing
ingredients are frequently unpublished: a short-range protection convention, a mixing rule applied to
the framework atoms, a dispersion correction, a blocking radius, an energy-reference convention. An
ungoverned pipeline that guesses any of them wrong returns a confident, wrong number with no warning.
This is the **domain-of-validity / realization-gap** failure class. v0.4–v0.5 established it on
classical force fields; **v0.6 reproduces it on CuspAI's own engine (kUPS) and on the machine-learning
interatomic potential (MLIP) kUPS ships** — and shows a governance layer that *characterizes,
discriminates, and where the evidence supports it, screen-passes a governed remediation.*

Three results carry the report:
1. **A2/F1** — the originally-locked positive control *double-counted silicon*; its strict-band agreement was a fortuitous four-error cancellation; rebuilt clean (single-source, Talu's calibration Olson geometry) it reproduces the corrected reference K_H = 0.200 cross-engine (native 0.200, RASPA3 0.199) (Finding F1, §2.4).
2. **A2-F2** — to our knowledge, the first published independent validation of kUPS's (six-week-old, unreleased) Widom estimator.
3. **WPB** — a four-configuration MLIP demonstration: discriminate two failure modes, screen-pass a dispersion-corrected route,
   and show the failure generalizes across MLIP families.

---

## 2. Cross-engine Widom parity (WPA-A2)

### 2.1 System and recipe
6c positive control: **MFI (all-silica) + Ar**, García-Pérez 2007 framework + Talu-Myers 2001
cross-pair. The complete locked recipe (recovered verbatim from the RASPA3 evidence `force_field.json`):

| element | self LJ (ε/K, σ/Å) | source |
|---|---|---|
| Ar | 119.8 / 3.405 | Talu-Myers 2001 |
| O | 53.0 / 3.3 | TraPPE-zeo, Bai 2013 |
| Si | 22.0 / 2.3 | TraPPE-zeo, Bai 2013 |

Ar–O is an **explicit** binary (93.0 K / 3.335 Å, Talu-Myers); **MixingRule Lorentz-Berthelot**,
**truncation shifted**, cutoff 12 Å, **TailCorrections false**, no electrostatics. Critically, the
Lorentz-Berthelot rule applied to the framework self-parameters generates a **second** interaction —
**Ar–Si (51.34 K / 2.8525 Å)** — in addition to the explicit Ar–O term. This is the *as-locked* recipe,
recorded verbatim; Finding F1 (§2.4) shows this Ar–Si term is a *silicon double-count* (Talu's Ar–O
already folds silicon's dispersion into oxygen) and gives the clean single-source rebuild that reproduces
the corrected reference.

### 2.2 The clean metric
Engines report different units (kUPS: K_H in Å³/eV; RASPA3/native: mol·kg⁻¹·bar⁻¹), so the comparison
is on the engine-agnostic **⟨exp(−βU)⟩**, with K_H and q_st reported in common units.

### 2.3 Result

| Engine (full recipe unless noted) | ⟨W⟩ | K_H (mol·kg⁻¹·bar⁻¹) | q_st (kJ/mol) |
|---|---|---|---|
| RASPA3 (locked reference) | — | **0.2070** | **16.03** |
| native independent (Ar–O **+ Ar–Si**) | 9.54 | **0.2093** (+1.1 %) | 15.91 (−0.12) |
| kUPS Widom (Ar–O **+ Ar–Si**) | 11.08 | 0.2432 (+16.2 %) | 16.25 (+0.22) |
| native (Ar–O **only**, naive) | 5.24 | 0.1149 (**1.8× low**) | 14.36 |
| kUPS (Ar–O only) | 5.98 | 0.1313 | — |

### 2.4 Finding F1 — a silicon double-count caught by audit (and a clean rebuild)
The originally-locked 6c recipe **double-counted silicon**, and agreed with its reference only as a fortuitous
cancellation of four compounding errors. Talu–Myers' Ar–O cross-parameter (93.0 K) is the
Lorentz–Berthelot combination of argon with an oxygen-only *effective* oxygen (72.2 K, Talu Table 1) that
already folds silicon's dispersion into oxygen — Talu's model is oxygen-only by construction (Talu §2:
"ignores partially concealed silicon atoms and lumps their dispersion energy with the oxygen atoms"). The
locked recipe nonetheless added an explicit Ar–Si from an active TraPPE-zeo framework silicon, counting
silicon twice (×1.82 on K_H). That spurious term compensated three further errors — an idealized (IZA)
rather than the calibration (Olson) geometry (~−25 %), an under-converged 12 Å tail-off cutoff (~−25 %
more), and a reference value mis-derived by ~12 % (0.224 vs the correct 0.200) — which multiplied back to
roughly the (wrong) reference. Removing all four and running the coherent **single-source** recipe (pure
Talu, oxygen-only, no Ar–Si) at a converged cutoff on Talu's own (Olson) geometry reproduces the
**corrected** reference (K_H = 0.200) exactly and **cross-engine**: native 0.200, RASPA3 0.199 (parity
0.5 %), q_st 14.97 (within 0.7 kJ/mol of calorimetry). In strict-band terms the as-locked hybrid lands
in-band against **both** the mis-derived 0.224 and the corrected 0.200 reference; the rebuild changes the
recipe's coherence, not the pass/fail verdict.

| 6c recipe / setup | K_H (mol/kg/bar) | note |
|---|---|---|
| locked hybrid (IZA, 12 Å tail-off) | 0.207 | in-band by a fortuitous cancellation (also passes 0.200) |
| − double-counted silicon | 0.115 | pure Talu, oxygen-only (IZA, 12 Å) |
| + converged 24 Å + tail | 0.154 | numerical defect fixed (IZA) |
| + Olson calibration geometry | **0.200** | native; RASPA3 0.199 (parity 0.5 %) |
| **corrected reference** (Dunne 1996, K_H = B/kT) | **0.200** | strict window [0.159, 0.252] |
| structure sensitivity (single-source) | 0.200 / 0.205 / 0.154 | Olson / van Koningsveld / IZA |

(The locked hybrid evidence is retained intact as the documented negative case.)

### 2.5 Finding F2 — independent cross-engine validation of kUPS's Widom estimator
kUPS ships a Widom test-particle module (`kups.mcmc.widom`; in source `main`, absent from the released
PyPI `kups 1.0.1`; no published external validation). v0.6 is, to our knowledge, the first *published*
independent cross-engine check of this specific module — distinct from standalone Widom packages (e.g. the
`widom` ASE add-on) and from machine-learned-potential adsorption benchmarks such as ODAC25
(Sriram et al. 2025, arXiv:2508.03162) — based on public-source searches (the kUPS source carries internal unit tests only;
the open tracker was inspected 2026-06-12, one unrelated feature issue).

- **The estimator is correct.** `widom_test()` returns lnα = −βΔU for a ghost insertion ⇒
  **W ≡ exp(−βU)** — identical to the textbook Widom weight; the μ_ex / K_H / q_st reductions match
  widom-atlas's definitions. Verified by reading the source *and* by the parity run.
- **q_st is validated to 0.22 kJ/mol** (16.25 vs 16.03 RASPA3; 15.91 native). The energies are right.
- **The +16 % K_H is a shifted-vs-truncated LJ convention — root-caused, not a normalization bug.**
  kUPS's `lennard_jones_energy` is **truncated, not energy-shifted** (`4ε(c6²−c6)` masked at cutoff, no
  `−U(rc)` term); our config (`tail_correction: false`, no `truncation_radius`) selects that plain
  variant. RASPA3's recipe is `TruncationMethod: shifted`, and the native estimator shifts too. The
  unshifted potential is more attractive by U(rc) per interaction. Toggling the shift off in native
  reproduces kUPS:

  | native (with-Si) | ⟨W⟩ | K_H | q_st (kJ/mol) |
  |---|---|---|---|
  | shifted (RASPA convention) | 9.54 | 0.2093 | 15.91 |
  | truncated / no-shift (kUPS convention) | 11.35 | 0.2490 | 16.34 |
  | kUPS Widom | 11.08 | 0.2432 | 16.25 |

  The implied constant shift δ = RT·ln(11.35/9.54) = **0.43 kJ/mol** (predicted ≈0.35) explains **both**
  the +16 % K_H **and** the +0.34 kJ/mol q_st: matching kUPS's convention, native-no-shift agrees with
  kUPS to **+2.4 % on K_H (within the 3.4 % SEM) and 0.09 kJ/mol on q_st**. Neither engine is wrong —
  but the shift convention is a load-bearing recipe element worth ~16 % on K_H, which must be locked and
  matched. The exponential sensitivity of K_H to a constant energy offset, and the resulting need to lock
  the tail/truncation convention, are established (Jablonka, Ongari & Smit 2019, *J. Chem. Theory Comput.*
  **15**, 5635; Dubbeldam et al. 2016, *Mol. Simul.* **42**, 81); what is new here is the cross-engine
  *quantification* of the effect (δ ≈ 0.43 kJ/mol). Does **not** affect q_st-as-physics or the OOD
  governance demo (energy path).

The recipe all three engines share for this parity is the as-locked hybrid now reframed in F1 (§2.4) as a
silicon double-count: F2 establishes that independent estimators agree on a *common* recipe (to 1.1 % on
K_H), not that the recipe reproduces experiment — that correction lives entirely in F1, and the two
findings are consistent.

---

## 3. Blocking-sphere governance (WPA-A3)

kUPS exposes the v0.5 short-range-protection lesson as an explicit API — `blocking_spheres`
(per-host hard-sphere exclusions). So the convention is no longer a claim of ours; it is CuspAI's own
engineering. A sweep on the 6c system (64 spheres on framework-O positions):

| blocking | radius (Å) | K_H (Å³/eV) | ΔK_H | q_st (kJ/mol) |
|---|---|---|---|---|
| off | — | 1.798×10⁷ | — | 16.25 |
| on | 2.0 | 1.856×10⁷ | +3.2 % | 16.34 |
| on | 3.0 | 1.812×10⁷ | +0.8 % | 16.28 |

**Finding.** On a physically-repulsive LJ recipe blocking is a **no-op** (ΔK_H < SEM, q_st unchanged):
the potential already excludes overlaps, so a hard sphere on top adds nothing. The corollary is the
governance lesson — **blocking is load-bearing exactly when the recipe has an *unphysical* short-range
attraction the potential does not repel** (open-metal-site over-binding; the v0.5 1b exp-6 case). A
recipe's blocking configuration is therefore part of the recipe and must be locked: invisible on an
all-silica LJ host, decisive on an OMS host. The small +0.8…+3.2 % K_H nudge from blocking is an
insertion-count effect (a distinct, much smaller thing than F2's +16 %, which is the LJ shift
convention), within the SEM and not affecting q_st.

---

## 4. MLIP out-of-distribution governance (WPB) — the headline

### 4.1 Setup
Widom insertions of **Ar into all-silica CHA**, N = 120, seed 0, T = 298.15 K. A classical UFF LJ
baseline scores the **same** insertion geometries as each MLIP, so all differences are the potential,
not the sampling. Two transparent OOD flags (hard-overlap at < 0.80 σ_min = 2.762 Å; energetic-anomaly
at U < −25 kJ/mol or U < U_classical − 50) and a **pre-registered verdict schema** with two refusal
classes: **Class I** (flagged Boltzmann-weight fraction ≥ 0.5 → OOD over-binding) and **Class II**
(min U > 0 or q_st ≤ RT where physisorption is established → under-binding). The schema was written
before the results so refusal is rule-based, not retrofit.

The governance premise is the model authors' own statement: CuspAI's `kUPS-mace-jax` model card declares
MACE-MPA-0 *"not trained for isolated molecules"* — scoring an adsorbate is out-of-domain by the
authors' own words. What is refused is **ungoverned usage of a configuration**, never a model.

### 4.2 Result — four configurations, discriminate then remediate

| Configuration | min U (kJ/mol) | flagged-weight | Verdict |
|---|---|---|---|
| MACE-MP-small (2023) | −19.3 | 0.995 | **REFUSE — Class I** (over-binds overlaps) |
| MACE-MPA-0 bare (kUPS's shipped export) | +8.5 | 0.0 | **REFUSE — Class II** (no physisorption) |
| MACE-MPA-0 **+ D3(BJ)** | −5.03 | 0.0001 | **screen-pass** (physical but shallow well; accuracy unassessed) |
| UMA `uma-s-1.1`, task = omat (fairchem flagship) | −292.8 | 1.0 | **REFUSE — Class I** (erratic OOD) |

Paired energies at identical *open* (~4 Å, no overlap) insertions make the mechanism airtight:

| insertion | min_dist Å | classical | small (2023) | MPA-0 bare | + D3(BJ) | UMA omat |
|---|---|---|---|---|---|---|
| #76 | 4.33 | −16.2 | −0.3 | +10.7 | +0.2 | −58.3 |
| #95 | 4.23 | −13.4 | −0.1 | +8.5 | −1.4 | −39.3 |
| #51 | 3.99 | −17.8 | −0.4 | +10.5 | −1.0 | +121.3 |

### 4.3 Interpretation
- **MACE-MP-small (2023)** hallucinates *attractive* energies inside the repulsive overlap region — its
  global minimum (−19.3) sits *at* a hard overlap, and the flagged insertions carry **99.5 %** of the
  partition weight. At physical sites its attraction is ≈ 0 (−0.1…−0.4): the "binding" is a pure
  overlap artifact, not physisorption. **Class I.**
- **MACE-MPA-0 bare (2024, kUPS's shipped weights)** *fixed* the overlap hallucination — it is correctly
  repulsive at overlaps — but is repulsive *everywhere* (min +8.5), finding no physisorption well. This
  is the bare-PBE signature: no −C₆/r⁶ dispersion tail. **Class II.**
- **MACE-MPA-0 + D3(BJ)** — pairing the *same* weights with the standard D3 Becke-Johnson dispersion
  correction (xc = PBE, cutoff 40 Bohr, torch-dftd 0.5.3) restores a physical well (min −5.03) at a
  non-overlap distance (3.37 Å; flagged-weight 0.0001). **Screen-passes; accuracy unassessed.** The governance lesson crystallizes:
  *dispersion-correction on/off is a load-bearing recipe element, and ungoverned pipelines silently
  differ on it.* kUPS's potential list contains no dispersion term — a finding, recorded, not a
  criticism.
- **UMA `uma-s-1.1` (deployment-class flagship)** — fairchem's multi-task foundation model (ODAC23 is
  one of its five training sets, so it is the UMA *and* ODAC leg). `task_name` is a load-bearing recipe
  element (five DFT levels of theory), and it produces a clean **task × system matrix**:

  | task | system | Gate 1 | min U | Verdict |
  |---|---|---|---|---|
  | omat | Ar/CHA | −0.19 ✅ | −292.8 | REFUSE — Class I (erratic OOD overlaps) |
  | odac | Ar/CHA | −2.73 ❌ | — | WITHHELD (inconsistent at equivalent sites) |
  | **odac** | **CO₂/CHA** | +0.06 ✅ | **−15.66** | **screen-pass** (in-domain, N=60 small sample; q_st 17.7 vs anchor 21.0, −3.3 below strict) |
  | omat | CO₂/CHA | +0.04 ✅ | +1.16 | REFUSE — Class II (no CO₂ well) |

  The **odac** head (trained on MOF/CO₂/H₂O direct-air-capture data) finds a **physical CO₂ well
  (−15.7 kJ/mol, screen q_st 17.7; experimental CO₂/CHA ≈ 20–30) on CO₂/CHA — its actual training
  domain → governance screen-passes it**. The screen q_st is 17.7 kJ/mol against a ~21 reference anchor (~3 kJ/mol low, −3.3 below strict): a screen-pass, not an accuracy certificate. On Ar/CHA (a noble gas it never trained on) the *same* head is
  internally inconsistent — 2.7 kJ between symmetry-equivalent open sites — which Gate 1 catches →
  **withheld**. The **omat** (materials) head under-binds adsorbates (no CO₂ well; erratic deep-overlap
  hallucinations on Ar). **The model passes exactly where its documented training domain says it
  should, and is refused/withheld everywhere else** — the governance respects the model card rather than
  fighting it. Separately, the OOD-overlap failure on Ar is not MACE-specific: it generalizes across
  MLIP families (MACE-MP-small and UMA-omat alike).

### 4.4 Validity gates (why the numbers can be trusted)
- **MACE Gate 1 (reference offset).** A +8.5 floor could be a broken differencing. Ruled out: Ar is in
  the model (Z = 18, 89-element table); **U(Ar isolated, 12.9 Å) = 0.000 kJ/mol** — differencing exact.
  The +floor is real missing-dispersion physics.
- **UMA — two hazards, both caught by Gate 1 before any verdict.** (1) UMA conditions on a **per-graph**
  global charge/spin/task embedding, so naive three-graph differencing does not cancel — a
  task-dependent far-field offset (omat +474, odac −245, omol +119 kJ at 12.9 Å). Fixed with a
  **same-graph reference** U(r) = E(host+Ar@r) − E(host+Ar@ref). (2) A non-periodic cut cluster is
  itself OOD for a periodic materials model (+800…+2900 kJ at open sites) despite passing the narrow
  far-field check; fixed by **periodic** in-domain evaluation with an open-pore reference (Gate 1 =
  −0.19 kJ; a 0.09 Å overlap → +21 572 kJ, correctly repulsive). **task_name is a locked, load-bearing
  recipe element** (UMA carries five DFT levels of theory; we lock `omat`). The governance harness thus
  validates *itself* for an engine before it is permitted to judge that engine.

### 4.5 Provenance (pinned)
MACE-MPA-0 checkpoint SHA-256[:16] `75428afe3a1d7d80`; kUPS export `mace-mpa-0-medium_32.zip` SHA-256[:16]
`c5eb645f2dc2c904`; D3 = D3(BJ)/PBE/40 Bohr via torch-dftd 0.5.3; UMA = `uma-s-1.1` (extensivity-safe,
not `uma-s-1`), fairchem-core 2.21.0. N, seed, T, CIF in every summary JSON.

---

## 5. Methods honesty

- **Numerical guards are recipe elements.** Two self-corrections are documented and retained: a spurious
  −50 weight-clip that manufactured a fake Class-I verdict for bare MPA-0 (discarded), and an eV/kJ unit
  mismatch in the Boltzmann weight that inflated exponents ×96 and tripped the exp(700) overflow cap
  spuriously (fixed with RT = 2.478 kJ/mol; after the fix no leg hits the cap — confirming the cap-firing
  was the bug, not deep physics). Verdict *directions* were unchanged; the quantitative weights are now
  physical.
- **Convergence.** kUPS: 25 000 ghost insertions, 12 blocks, K_H SEM ≈ 3.4 %; the +16 % is a systematic.
  native: 1–2×10⁶ insertions, ⟨W⟩ < 0.5 %. The analytical LJ tail was tested on native (moves K_H only
  +8 %), confirming the recipe's `TailCorrections: false` and ruling tail out as the parity gap driver —
  the Ar–Si term, not tail, was the 1.8×.

---

## 6. Open scientific questions

- **kUPS K_H +16 % — RESOLVED (shift convention).** Root-caused to kUPS's truncated (unshifted) LJ vs
  RASPA/native shifted; δ ≈ 0.43 kJ/mol reconciles K_H to 2.4 % and q_st to 0.09 kJ/mol (§2.5). Open
  only as an upstream documentation point: kUPS's default LJ is unshifted.
- **Ewald leg (4c, Si-CHA + CO₂) — DONE, read via the convention matrix.** TraPPE-CO₂ + framework
  charges + Ewald, 14 Å + tail. **q_st validated to 0.01 kJ/mol** (22.98 vs 22.99). For K_H, a fresh
  same-session native re-run (the 2.223 was the v0.4 *lock*; the re-run gives 2.215, reproducing it to
  0.3 %) toggles the conventions: **shift = +11.6 %, tail = +11.9 %** (the shift is *not* negligible at
  14 Å — earlier-draft correction). kUPS uses a truncated LJ and **does apply a tail** (tested:
  `tail_correction: false` drops K_H −16 %, q_st −0.49 — a "no-tail-in-Widom" hypothesis is *falsified*).
  So its +2.2 % vs the shifted lock is a **cancellation**: read on the same (truncated+tail) convention,
  **kUPS is −8.1 % vs native** (−11.7 % with tail also off). That residual is **q_st-invariant
  (Δq_st ≤ 0.3 kJ/mol) → a normalization-flavoured ⟨W⟩ scale** (volume/insertion-count), 4c-specific and
  larger than 6c's −2.4 %; contributors include a tail-implementation difference (kUPS +16.3 % vs native
  +11.9 %), the Ewald convention, and CO₂ orientation (`native_4c_convention_matrix.json`).
- **OMS blocking positive case.** The dramatic blocking demonstration (collapsing a divergent open-metal
  K_H to a physical value) needs a kUPS-representable OMS recipe with charges (1c Becker, or 2a UFF
  Cu + EPM2); the exotic 1a/1b/1d Buckingham forms are not representable in kUPS (LJ/Coulomb/Morse/MACE
  only).
- **WPB through kUPS's in-engine JAX MACE — DONE.** Loaded kUPS's `TojaxedMliap` export and scored
  identical geometries; agrees with the torch path to **max rel 2.1×10⁻⁷** (`jax_torch_parity.json`),
  independently confirming the model card's rtol = 1e-4. The governance conclusions carry to kUPS's JAX
  runtime exactly.

---

## 7. One-paragraph summary

On the locked positive control, audit found the recipe had *double-counted silicon* — Talu's Ar–O
cross-parameter already folds silicon's dispersion into oxygen, and the recipe added an explicit Ar–Si on
top — so its strict-band agreement was a fortuitous four-error cancellation; rebuilt clean (single-source oxygen-only,
converged convention, on Talu's calibration Olson geometry) it reproduces the corrected reference
(K_H = 0.200) exactly and cross-engine (native 0.200, RASPA3 0.199). Separately and intact, the engines
agree on the shared recipe to 1.1 % — the cross-engine parity result.
CuspAI's own Widom estimator was independently validated for the first time — its energies are right
(q_st to 0.22 kJ/mol) and its +16 % K_H was root-caused to a shifted-vs-truncated LJ convention
(δ ≈ 0.43 kJ/mol, reconciling both K_H and q_st to within the SEM). Pointed at the MLIP
CuspAI ships, governance discriminated two distinct failure modes (overlap over-binding vs missing
dispersion), screen-passed a D3(BJ)-corrected route on the single governed Si-CHA + Ar screen, and showed the deployment-class flagship (UMA) fails the
same way — with the governance harness proving its own validity for each engine before judging it.
*The published table is never the whole recipe; the unpublished conventions are load-bearing, and they
can be characterized, discriminated, and in some cases fixed.*
