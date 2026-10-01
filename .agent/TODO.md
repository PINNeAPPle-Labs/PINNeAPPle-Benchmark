# TODO — PINNeAPPle-Benchmark

## Near-term (mechanical, blocked on PINNeAPPle-Climate's running jobs)

- [ ] Once `02_physics_digital_twin`, `03_bumper_beam_transolver`, `04_terramechanics_robust` and
  `06_millennium` (all of it, including the now-finished `p_vs_np`) are done in `PINNeAPPle-Climate`, move
  them here the same way 05/07/08 were moved (see that move's PR #1 here, and
  `PINNeAPPle-Climate/.agent/TODO.md` for the checklist on that side). Update `experiments/README.md`'s
  table when each one lands.
- [ ] `03_bumper_beam_transolver` is ~9.8 GB on disk in Climate — before moving, decide what's actually
  worth committing (the packaged `training_data.hdf5` + small results, almost certainly not the raw
  per-run OpenRadioss `runs/Exp_*/*.rst`/`.out` files) and extend `experiments/.gitignore` accordingly.
- [ ] If `paperkit.py` gets improved in `PINNeAPPle-Climate`, copy the change here too (no sync exists —
  see `DECISIONS.md`).

## Real strategic work (not started)

- [ ] Build the actual benchmark protocol layer the strategy doc envisions: fixed train/val/test splits,
  standard metrics, a compute budget, and a single reproducible command per case (Benchmark 01: heat
  equation; 02: Navier–Stokes vs. OpenFOAM; 03: geometry-conditioned surrogate) — see
  `helm/docs/estrategia-infraestrutura-aberta.md` §3 for the originally proposed ordering and what each
  case needs from the core `PINNeAPPle` library (a unified `PhysicalProblem` abstraction, a
  `problem.backend(...)` switch, a formal Physics Dataset Standard — none of which exist yet per that
  doc's own gap analysis).
- [ ] Decide whether `PINNeAPPle-arena` (private) should read this repo's published results for a
  leaderboard, per the strategy doc — no integration exists yet.
