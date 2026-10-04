# TODO — PINNeAPPle-Benchmark

**Update 2026-10-02**: the "near-term, mechanical" section below is entirely done — all eight experiments
are now here with final papers. The "real strategic work" section is unchanged and still fully open.

## Near-term (done)

- [x] Moved `02`, `03` (code+papers; raw ~11 GB data stays local in `PINNeAPPle-Climate`, documented there),
  `04`, and `06_millennium` (all of it, including the now-finished `p_vs_np`).
- [x] `experiments/README.md`'s table updated with all eight experiments.
- [x] Two RFF-bandwidth bugs (04, 06/p_vs_np) found and fixed; documented in `experiments/README.md` and
  `.agent/DECISIONS.md`.

## Real strategic work (not started — unchanged from before)

- [ ] Build the actual benchmark protocol layer the strategy doc envisions: fixed train/val/test splits,
  standard metrics, a compute budget, and a single reproducible command per case (Benchmark 01: heat
  equation; 02: Navier–Stokes vs. OpenFOAM; 03: geometry-conditioned surrogate) — see
  `helm/docs/estrategia-infraestrutura-aberta.md` §3 for the originally proposed ordering and what each
  case needs from the core `PINNeAPPle` library (a unified `PhysicalProblem` abstraction, a
  `problem.backend(...)` switch, a formal Physics Dataset Standard — none of which exist yet per that
  doc's own gap analysis).
- [ ] Decide whether `PINNeAPPle-arena` (private) should read this repo's published results for a
  leaderboard, per the strategy doc — no integration exists yet.
- [ ] Decide whether the eight experiments now here should eventually be restructured to fit that future
  protocol layer, or kept as-is alongside it.
