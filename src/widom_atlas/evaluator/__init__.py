"""widom_atlas.evaluator — internal NumPy Widom-insertion evaluator (v0.4).

Why this exists: RASPA cannot natively export per-insertion samples
(``growData`` is overwritten each step), so the v0.4 architecture pivots to
an internal evaluator that:

1. Ingests a RASPA3 force-field bundle (via ``widom_atlas.ingest.raspa3_ff``)
2. Builds a periodic energy function with LJ + Wolf-summation Coulomb
3. Runs Widom insertions on a deterministic uniform grid (or stochastic, or both)
4. Produces per-insertion (E_total, E_LJ, E_coul) samples + the K_H / Q_ads scalars
5. Provides parity primitives so a RASPA3 GCMC reference can validate it

Sub-modules
-----------
- ``ff_loader``     : load a UserParameterFile JSON or RASPA3 bundle into runtime tables
- ``component``     : multi-site rigid gas geometries (CO2, N2, CH4 single-site, Kr, etc.)
- ``grid``          : deterministic uniform grid + stochastic sample generators
- ``energy``        : LJ + Wolf-Coulomb periodic energy with PBC minimum-image
- ``runner``        : orchestrates one evaluator run
- ``parity``        : RASPA3 reference parity + MOFX simin parity gate
"""
