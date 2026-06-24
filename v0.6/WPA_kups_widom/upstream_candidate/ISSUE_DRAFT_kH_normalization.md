# SUPERSEDED — see UPSTREAM_NOTE.md §2 (the +16 % was an LJ shift convention, not a normalization bug)

> Status: this earlier draft hypothesised a K_H *normalization* (volume / insertion-count) issue. That
> hypothesis was **tested and rejected.** Root-cause (held until the data was in, per review): the
> +16 % K_H is a **shifted-vs-truncated LJ convention** — kUPS's `lennard_jones_energy` is truncated
> (no `−U(r_c)` shift), the RASPA3 reference is shifted. Toggling the shift off in the independent
> estimator reproduces kUPS to +2.4 % on K_H and 0.09 kJ/mol on q_st (`../F2_rootcause_shift.json`).
> It is **not a bug** and not a normalization issue.

The actionable, friendly upstream point is now a one-line documentation suggestion (LJ default is
truncated/unshifted) plus an optional `shift: true` flag for exact RASPA round-tripping — written up in
**`UPSTREAM_NOTE.md` §2**. This file is retained only to record that the normalization hypothesis was
considered and falsified.
