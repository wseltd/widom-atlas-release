import json

p = "/home/onur/projects/widom-atlas-release/v0.6/WPB_kups_mlip_ood/uma_ur_scan_2026-08.json"
d = json.load(open(p))
rows = d["scan"]

print("gate1 (open-open) = %.4f kJ/mol" % d["gate1_open_open_kJ"])
print("nearest framework element:", d["nearest_framework_element"])
print("deepest committed insertion index:", d["deepest_insertion_index"])
print()

r = d["repeat_second_reference"]
print("=== REPEAT of the committed deepest insertion ===")
print("  committed        %.3f kJ/mol" % r["committed_U_kJ"])
print("  recomputed ref1  %.3f kJ/mol   (delta %.5f)" % (r["recomputed_U_ref1_kJ"], r["recomputed_U_ref1_kJ"] - r["committed_U_kJ"]))
print("  recomputed ref2  %.3f kJ/mol   (delta %.5f)" % (r["recomputed_U_ref2_kJ"], r["recomputed_U_ref2_kJ"] - r["committed_U_kJ"]))
print()

shifts = [x["U_ref1_kJ"] - x["U_ref2_kJ"] for x in rows]
print("=== reference sensitivity across the whole scan ===")
print("  U(ref1)-U(ref2): min %.4f  max %.4f  spread %.5f kJ/mol"
      % (min(shifts), max(shifts), max(shifts) - min(shifts)))
print()

byd = sorted(rows, key=lambda x: x["min_dist_A"])
deepest = min(rows, key=lambda x: x["U_ref1_kJ"])
print("=== deepest point on the continuous trajectory ===")
print("  U = %.2f kJ/mol at nearest host-guest distance %.3f A"
      % (deepest["U_ref1_kJ"], deepest["min_dist_A"]))
print("  (corrected hard-overlap threshold = 2.626 A -> deepest point is %s the overlap region)"
      % ("INSIDE" if deepest["min_dist_A"] < 2.626 else "OUTSIDE"))
print()

print("=== curve, ordered by nearest host-guest distance ===")
print("  min_dist(A)   U(kJ/mol)")
seen = set()
for x in byd:
    k = round(x["min_dist_A"], 2)
    if k in seen:
        continue
    seen.add(k)
    print("   %8.3f  %12.2f" % (x["min_dist_A"], x["U_ref1_kJ"]))

neg = [x for x in rows if x["U_ref1_kJ"] < -25.0]
print()
print("=== extent of the spurious basin ===")
print("  %d of %d scan points are below the -25 kJ/mol anomaly floor" % (len(neg), len(rows)))
if neg:
    print("  spanning nearest host-guest distances %.3f to %.3f A"
          % (min(x["min_dist_A"] for x in neg), max(x["min_dist_A"] for x in neg)))
    out = [x for x in neg if x["min_dist_A"] > 2.626]
    print("  of which %d lie OUTSIDE the corrected 2.626 A overlap threshold" % len(out))
