#!/usr/bin/env python3
"""B2: score fixed geometries with kUPS's in-engine JAX MACE export (TojaxedMliap), to compare against
the upstream torch mace_mp on the SAME geometries. Builds the AtomGraphInput directly (pos, Z, cell,
pbc, edge_index, cell_offsets) from an ASE neighbor list, bypassing the heavy kUPS graph machinery.
Non-periodic (big box) so cell_offsets are zero and edges are all pairs within the model cutoff.
Run in venv-kups."""
import json, numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # export was made in x64; match int64/float64 avals
import jax.numpy as jnp
from ase import Atoms
from ase.neighborlist import neighbor_list
from kups.potential.mliap.tojax import TojaxedMliap

ZIP = "~/models/kups-mace-jax/mace-mpa-0-medium_32.zip"
m = TojaxedMliap.from_zip_file(ZIP)
cutoff = float(np.asarray(m.cutoff.data).ravel()[0])
print(f"loaded export; cutoff={cutoff:.3f} A")

geoms = json.load(open("v0.6/WPB_kups_mlip_ood/jax_torch_testgeoms.json"))
BOX = 60.0
out = []
for g in geoms:
    sym = g["symbols"]; pos = np.array(g["positions"])
    at = Atoms(sym, positions=pos, cell=[BOX, BOX, BOX], pbc=False)
    i, j, S = neighbor_list("ijS", at, cutoff)        # edges within cutoff (both directions)
    Z = at.get_atomic_numbers()
    inp = {
        "pos": jnp.asarray(pos, float),
        "atomic_numbers": jnp.asarray(Z, jnp.int32),
        "cell": jnp.asarray(at.cell.array, float)[None],     # (1,3,3)
        "pbc": jnp.asarray(at.pbc, bool)[None],              # (1,3)
        "edge_index": jnp.asarray(np.vstack([i, j]), jnp.int32),   # (2,E)
        "cell_offsets": jnp.asarray(S, float),               # (E,3)
        "batch": jnp.zeros(len(Z), jnp.int32),
        "charge": jnp.zeros(1, float),
        "spin": jnp.zeros(1, float),
    }
    e = np.asarray(m.call(inp)).ravel()
    out.append(float(e[0]))
    print(f"  geom N={len(Z)} edges={len(i)} -> E_jax = {e[0]:.6f} eV")

json.dump(out, open("v0.6/WPB_kups_mlip_ood/jax_mace_energies.json", "w"))
print("wrote jax_mace_energies.json")
