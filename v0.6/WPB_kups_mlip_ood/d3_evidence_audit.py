"""D3(BJ) evidence audit for the MACE-MPA-0 bare vs +D3 Si-CHA+Ar screen.

Verifies, from the committed per-insertion JSON only (no GPU rerun):
  - bare and +D3 scored identical geometries
  - global physisorption-well minimum sits at a non-overlap distance
  - hard-overlap configs stay repulsive under D3 (no overlap-driven artificial binding)
  - block estimate of min U and flagged-weight robustness
  - no exponent cap fired
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "wpb_medium-mpa-0_per_insertion.json")) as fh:
    bare = json.load(fh)
with open(os.path.join(HERE, "wpb_medium-mpa-0_D3_per_insertion.json")) as fh:
    d3 = json.load(fh)

if not (len(bare) == len(d3) == 120):
    raise SystemExit(f"expected 120 insertions in each file, got {len(bare)} / {len(d3)}")

# 1. identical geometries: min_dist and classical energy must match element-wise
geom_ok = all(
    abs(b["min_dist_A"] - d["min_dist_A"]) < 1e-9
    and abs(b["U_classical_kJ"] - d["U_classical_kJ"]) < 1e-6
    for b, d in zip(bare, d3, strict=True)
)
print("identical geometries (bare vs D3):", geom_ok)

# 2. global well minimum and its distance (hard-overlap threshold = 2.762 A = 0.80*sigma_min)
gmin = min(d3, key=lambda r: r["U_mace_kJ"])
print("D3 global min U_mace = %.3f kJ/mol at min_dist = %.3f A (i=%d)"
      % (gmin["U_mace_kJ"], gmin["min_dist_A"], gmin["i"]))
print("  -> non-overlap (min_dist > 2.762 A hard-overlap threshold):",
      gmin["min_dist_A"] > 2.762)

# 3. do any hard-overlap insertions become attractive (U<0) under D3?
overlap_binding = [r for r in d3 if r["flag_hard_overlap"] and r["U_mace_kJ"] < 0]
print("hard-overlap insertions that became attractive under D3:", len(overlap_binding))
worst_overlap = min((r for r in d3 if r["flag_hard_overlap"]),
                    key=lambda r: r["U_mace_kJ"])
print("  least-repulsive hard-overlap under D3: U_mace = %.2f kJ/mol at %.3f A"
      % (worst_overlap["U_mace_kJ"], worst_overlap["min_dist_A"]))

# 4. block estimate: 6 blocks of 20, min U and flagged-weight per block
print("block estimate (6 blocks of 20):")
for b0 in range(0, 120, 20):
    blk = d3[b0:b0 + 20]
    wt = sum(r["w_mace"] for r in blk)
    fw = sum(r["w_mace"] for r in blk if r["flagged"])
    bmin = min(r["U_mace_kJ"] for r in blk)
    frac = (fw / wt) if wt > 0 else 0.0
    print("  block %d-%d: min U = %7.3f  flagged-weight frac = %.4f"
          % (b0, b0 + 19, bmin, frac))

# 5. exponent cap audit
ncap_bare = sum(r["flag_exponent_capped"] for r in bare)
ncap_d3 = sum(r["flag_exponent_capped"] for r in d3)
print("n_exponent_capped: bare=%d  D3=%d" % (ncap_bare, ncap_d3))

# 6. D3 adds attraction everywhere (per-insertion U_mace lowered vs bare)?
lowered = sum(1 for b, d in zip(bare, d3, strict=True)
              if d["U_mace_kJ"] <= b["U_mace_kJ"] + 1e-6)
print("insertions where D3 lowered (or held) U vs bare: %d / 120" % lowered)

# 7. overall flagged-weight fraction (the Class-I discriminator)
tw = sum(r["w_mace"] for r in d3)
tfw = sum(r["w_mace"] for r in d3 if r["flagged"])
print("D3 overall flagged-weight fraction = %.5f (Class-I needs >= 0.5)"
      % ((tfw / tw) if tw > 0 else 0.0))
