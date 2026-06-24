# WPB numerical-guards errata (kept, not deleted)

Numerical guards are recipe elements: a wrong guard manufactures a wrong verdict. Two corrections were
made during WPB development. Both are documented here rather than silently overwritten, because the
self-correction is part of the audit trail.

## Erratum 1 — the spurious −50 lower clip (DISCARDED)
- **What happened.** An intermediate weight expression `exp(clip(−U/KT, −50, 700))` floored *all*
  repulsive-insertion weights to the same value `exp(−50)`. For bare MACE-MPA-0 (all U > 0) this made
  every insertion equal-weight, so `flagged_weight_fraction` collapsed to the **count** fraction
  (99/120 = **0.825**) and produced a fake "Class I REFUSE."
- **Why it was wrong.** The −50 floor is not physical; it equalizes weights that should differ by
  orders of magnitude. The real MACE-MPA-0 failure is Class II (no well), not Class I.
- **Fix.** Cap only the upper exponent (`exp(min(−U/RT, 700))`); let genuine underflow go to ~0.
- **Status.** The spurious 0.825 / fake-Class-I run is **discarded** but recorded here.

## Erratum 2 — the eV/kJ unit mismatch in the weight (FIXED)
- **What happened.** `w = exp(−U / KT)` used `KT = k_B·T` in **eV** (0.0257) while `U` is in
  **kJ/mol**. That divided by a number ~96× too small, inflating every exponent by ×96. The exp(700)
  cap then fired *spuriously* (e.g. MACE-MP-small min U −19.3 → exp(751) → capped at exp(700) ≈ 1e301),
  and `KH_proxy_all` reached absurd magnitudes (1e301, 1e82) that are not physical numbers.
- **Why it mattered.** The cap-firing was being interpreted (briefly) as "deep hallucination needing
  protection." In truth U = −19.3 kJ/mol needs only `exp(19.3/2.478) = exp(7.79) ≈ 2418` — no cap.
- **Fix.** `RT_kJ = k_B·EV_TO_KJ·T = R·T = 2.478 kJ/mol`; `w = exp(min(−U/RT_kJ, 700))`. After the fix
  **no insertion in any of the three legs hits the cap** (`n_exponent_capped = 0` everywhere),
  confirming the cap-firing was the unit bug, not physics.
- **Effect on conclusions.** Verdicts are unchanged in *direction* (small→REFUSE Class I,
  bare→REFUSE Class II, +D3→PASS) but the quantitative weights are now physical:
  small `flagged_weight_fraction` 1.0→**0.9948**, `KH_proxy_all` 8.5e301→**59.05**; the D3 leg
  `KH_proxy_all` 7.7e82→**0.331**. These corrected numbers are what the schema and report use.

## Lesson (on-thesis)
The exp(700) cap, the clip floor, and the RT units are themselves *load-bearing recipe elements*. An
ungoverned pipeline that got any of them wrong would publish a wrong verdict with no warning — the
same domain-of-validity failure class the project documents for force-field recipes, now in the
governance code itself. The guard is locked, stated, and regenerable.
