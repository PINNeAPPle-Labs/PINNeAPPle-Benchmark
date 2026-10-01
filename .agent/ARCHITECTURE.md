# ARCHITECTURE — PINNeAPPle-Benchmark

This repo's own architecture is thin (a `README.md`, a `LICENSE`, and `experiments/`). The substantial
architecture is the **content of `experiments/`**, which is identical code to what's described in
`PINNeAPPle-Climate/.agent/ARCHITECTURE.md` (per-experiment `src/`/`results/`/`figures/`/`paper/` layout,
the `OUT`/`save()`/`log()` convention, `paperkit.py`'s `Paper` class) — read that file; it is not
re-explained here to avoid drift between two copies of the same description.

## What is specific to THIS repo

- **`experiments/.gitignore`**: excludes `*/data/` (raw/downloaded datasets — each experiment's own script
  regenerates them), `*/results_smoke/` (smoke-test runs, not the reported results), `*.pt` (model
  checkpoints — not needed to rebuild a paper; every `make_paper.py` reads `results.json` and a handful of
  small `.npz` files, never a `.pt`), `__pycache__/`, `*.pyc`, `.DS_Store`.
- **`experiments/paperkit/paperkit.py`** is a **vendored, independent copy** of the same file in
  `PINNeAPPle-Climate/experiments/paperkit/`. Not a git submodule, not a package dependency — deliberately
  duplicated so this public repo has zero dependency on the private `PINNeAPPle-Climate` repo (see
  `DECISIONS.md`). If `paperkit.py` is improved in one place, it needs to be copied to the other by hand;
  there is no sync mechanism.
- **`experiments/README.md`** is the authoritative, human-maintained index of what's in this repo and what
  is still pending — check it (not this file) for the current list of moved-in vs. still-pending
  experiments, since that list changes as jobs finish elsewhere.

## Future architecture (not built yet — from `helm/docs/estrategia-infraestrutura-aberta.md`)

The strategy document that led to this repo's creation describes a real benchmark suite this could grow
into: per-case fixed protocols (train/val/test splits, metrics, a compute budget), reference data checked
in or pointed to, and a harness in the private `PINNeAPPle` core (`pinneapple_arena`, `benchmark_suite`)
that this repo's scripts would call into — with `PINNeAPPle-arena` (a separate, private app repo) reading
this repo's published results for a leaderboard. **None of that harness/protocol layer exists in this repo
yet.** What exists is just moved-in finished experiments with ad-hoc per-experiment scripts, not a unified
benchmark protocol. Treat the strategy doc as the target, not the current state.
