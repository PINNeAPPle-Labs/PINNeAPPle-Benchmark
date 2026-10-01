# PROJECT_CONTEXT — PINNeAPPle-Benchmark

## What this repo is

A **new, public** repository (created 2026-09-28, `PINNeAPPle-Labs/PINNeAPPle-Benchmark` on GitHub) meant
to hold "public, reproducible Physics AI benchmarks built on PINNeAPPle" (see its `README.md`). It exists
because of a strategic recommendation, not an accident: the org-wide strategy document
`helm/docs/estrategia-infraestrutura-aberta.md` (2026-09-27, see that repo's own handoff) recommended one
new public repo specifically for a benchmark, separate from the private core `PINNeAPPle` library and from
the private `PINNeAPPle-arena` app (a leaderboard web app that would read this repo's published results,
per that strategy doc — not yet built).

## Current actual state vs. the strategic plan

**As of 2026-09-28 this repo is not yet the "benchmark" the strategy envisioned** (fixed protocols,
reference data, standard metrics per PDE case, one command to reproduce). It is, for now, just a landing
spot for finished research-experiment output moved out of `PINNeAPPle-Climate` because that repo isn't the
right home for non-climate work — see `experiments/README.md`. Three experiments have moved in
(`05_thermal_channel_twin`, `07_pdr_thesis_reproduction`, `08_supersonic_shock_train`); at least three more
(`02_physics_digital_twin`, `03_bumper_beam_transolver`, `04_terramechanics_robust`, and the rest of
`06_millennium`) are expected to follow once their still-running jobs finish (tracked in
`PINNeAPPle-Climate/.agent/TODO.md`).

## Purpose (near-term, honest)

1. Be a clean public home for finished PINNeAPPle research experiments and their papers.
2. Eventually become the actual benchmark suite described in `helm`'s strategy doc: fixed cases (heat
   equation, Navier–Stokes, geometry-conditioned surrogates), a protocol (splits, metrics, compute budget)
   per case, and a single command to reproduce each — none of that infrastructure exists yet.

## Target users

External researchers/engineers evaluating PINNeAPPle, and internal use as citable, versioned evidence for
papers/posts ("PINNeAPPle-Benchmark v1, case 02"), per the strategy doc's reasoning for why this needed to
be a separate public repo rather than living inside the private core library or the private arena app.

## Main technologies

Same stack as every experiment: Python 3, PyTorch, NumPy/SciPy, `reportlab`+`matplotlib` (via the vendored
`experiments/paperkit/paperkit.py`), plus whatever each individual experiment needs (`h5py`, `pysat`,
Docker for OpenRadioss, etc. — see `PINNeAPPle-Climate/.agent/ARCHITECTURE.md` for the per-experiment
detail, since the moved experiments' code is unchanged from there). Every experiment script imports the
sibling `PINNeAPPle` repo from source via `sys.path.insert(..., ROOT.parents[N] / "PINNeAPPle")` — **this
repo depends on `PINNeAPPle` being checked out as a sibling directory** (`pinneapple-labs/PINNeAPPle`), same
as `PINNeAPPle-Climate`.

## Repository structure (verified)

```
PINNeAPPle-Benchmark/
├── README.md, LICENSE (Apache-2.0, matching the PINNeAPPle core library)
└── experiments/
    ├── .gitignore            ← excludes */data/, */results_smoke/, *.pt (see DECISIONS.md for why)
    ├── README.md              ← the actual index of what's here and what's still to come
    ├── paperkit/paperkit.py   ← vendored copy (also exists, independently, in PINNeAPPle-Climate)
    ├── 05_thermal_channel_twin/
    ├── 07_pdr_thesis_reproduction/
    └── 08_supersonic_shock_train/
```

## How it's run / tested / deployed

Identical, per-experiment, to `PINNeAPPle-Climate`'s experiments (this is literally the same code, moved):
`cd experiments/NN_name && python3 src/main_script.py [stage ...]`; papers via `cd paper && python3
make_paper.py`. No test suite, no CI, no packaging. See `PINNeAPPle-Climate/.agent/ARCHITECTURE.md` for the
common per-experiment conventions (`OUT`/`save()`/`log()`, `SMOKE=1`, etc.) — not repeated here since
nothing changed in the move.
