# SUPERSEDED 2026-06-18 (v0.4.1) — kept intact as the documented negative case

This directory is the **original locked v0.4 6c evidence**: RASPA3 3.0.29, 3 seeds × 100k, K_H = 0.20701
mol/(kg·bar), on the IZA-idealized MFI at 12 Å shifted / tail-off, with the **silicon double-count
recipe** (Talu Ar–O 93.0 + an active TraPPE-zeo Si self-parameter → an LB Ar–Si).

It is **superseded** but **not deleted** — it is the worked negative case the paper's F1 reframe rests on.

**Why rebuilt (v0.4.1):** this recipe "passed" against the then-locked reference 0.224 only by a
cancellation of four compounding errors:
1. non-calibration structure (IZA-idealized vs Talu's Olson geometry): ~−25%;
2. under-converged convention (12 Å tail-off vs converged 24 Å + tail): ~−25% more;
3. a mis-derived reference (0.224 — STP molar volume + wrong citation — vs the correct 0.200); and
4. a double-counted silicon (×1.82).
These multiplied back to ≈ the (wrong) reference. Removing all four and running the coherent
single-source pure-Talu oxygen-only recipe on Talu's own (Olson) geometry reproduces the **corrected**
reference (0.200) exactly, cross-engine (native + RASPA3).

**Current 6c evidence:** `evidence/v04_6c_olson_clean/`. See `V04_LOCKED_SPEC_CHANGELOG.md` v0.4.1.
