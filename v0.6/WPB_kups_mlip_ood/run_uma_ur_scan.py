import json, os
import numpy as np
from ase.io import read
from ase import Atoms

REPO = "/home/onur/projects/widom-atlas-release"
CIF = os.path.join(REPO, "v0.6/figures/CHA_iza.cif")
GUEST, T_K, KB_EV, EV_TO_KJ, SEED, CUTOFF = "Ar", 298.15, 8.617333262e-5, 96.48533212, 0, 6.0
MODEL, TASK = "uma-s-1p1", "omat"

frame = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell); inv = np.linalg.inv(cell)
fpos = frame.get_positions(); fsym = np.array(frame.get_chemical_symbols())
print("supercell: %d atoms, reps %s" % (len(frame), reps), flush=True)


def min_dist(cart):
    df = (fpos - cart) @ inv; df -= np.round(df)
    return float(np.linalg.norm(df @ cell, axis=1).min())


tok = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(tok):
    os.environ.setdefault("HF_TOKEN", open(tok).read().strip())
from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit(MODEL, device="cuda")
calc = FAIRChemCalculator(predictor, task_name=TASK)


def E_host_Ar(cart):
    at = frame.copy(); at += Atoms(GUEST, positions=[cart])
    at.info["charge"] = 0; at.info["spin"] = 1
    at.calc = calc
    return at.get_potential_energy()


g = np.linspace(0.05, 0.95, 12)
grid = sorted(((min_dist(np.array([x, y, z]) @ cell), np.array([x, y, z]) @ cell)
               for x in g for y in g for z in g), key=lambda t: t[0])
ref_cart = grid[-1][1]
ref2_cart = grid[-4][1]
e_ref = E_host_Ar(ref_cart)
e_ref2 = E_host_Ar(ref2_cart)
gate1 = (e_ref2 - e_ref) * EV_TO_KJ
print("open-pore ref at min-dist %.3f A; second open ref %.3f A; gate1 = %.4f kJ/mol"
      % (grid[-1][0], grid[-4][0], gate1), flush=True)

rng = np.random.default_rng(SEED)
pts = [rng.random(3) @ cell for _ in range(120)]
mds = [min_dist(c) for c in pts]

stored = json.load(open(os.path.join(REPO, "v0.6/WPB_kups_mlip_ood/wpb_uma_omat_per_insertion.json")))
deep = min(stored, key=lambda r: r["U_mace_kJ"])
i_deep = deep["i"]
print("deepest committed insertion: i=%d, min_dist=%.4f A, U=%.2f kJ/mol"
      % (i_deep, deep["min_dist_A"], deep["U_mace_kJ"]), flush=True)
assert abs(mds[i_deep] - deep["min_dist_A"]) < 1e-3, "geometry mismatch"
deep_cart = pts[i_deep]

df = (fpos - deep_cart) @ inv; df -= np.round(df)
dist_all = np.linalg.norm(df @ cell, axis=1)
j = int(np.argmin(dist_all))
nearest_atom = deep_cart + (df[j] @ cell)
print("nearest framework atom: %s at %.4f A" % (fsym[j], dist_all[j]), flush=True)

u = (deep_cart - nearest_atom)
u = u / np.linalg.norm(u)

rows = []
for r in np.linspace(1.20, 5.20, 61):
    cart = nearest_atom + u * r
    md = min_dist(cart)
    e = E_host_Ar(cart)
    rows.append({
        "r_along_A": float(r),
        "min_dist_A": float(md),
        "U_ref1_kJ": float((e - e_ref) * EV_TO_KJ),
        "U_ref2_kJ": float((e - e_ref2) * EV_TO_KJ),
    })
    print("  r=%.2f  min_dist=%.3f  U(ref1)=%12.2f  U(ref2)=%12.2f"
          % (r, md, rows[-1]["U_ref1_kJ"], rows[-1]["U_ref2_kJ"]), flush=True)

e_deep = E_host_Ar(deep_cart)
repeat = {
    "committed_U_kJ": deep["U_mace_kJ"],
    "recomputed_U_ref1_kJ": float((e_deep - e_ref) * EV_TO_KJ),
    "recomputed_U_ref2_kJ": float((e_deep - e_ref2) * EV_TO_KJ),
}
print("\nREPEAT at the committed deepest insertion:", json.dumps(repeat, indent=1), flush=True)

out = {
    "model": MODEL, "task": TASK, "T_K": T_K,
    "protocol": "continuous U(r) along the line from the nearest framework atom through the deepest committed insertion; same-graph open-pore reference",
    "ref1_min_dist_A": float(grid[-1][0]),
    "ref2_min_dist_A": float(grid[-4][0]),
    "gate1_open_open_kJ": float(gate1),
    "deepest_insertion_index": int(i_deep),
    "nearest_framework_element": str(fsym[j]),
    "repeat_second_reference": repeat,
    "scan": rows,
}
p = os.path.join(REPO, "v0.6/WPB_kups_mlip_ood/uma_ur_scan_2026-08.json")
json.dump(out, open(p, "w"), indent=1)
print("wrote", p)
